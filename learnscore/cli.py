"""`python -m learnscore.cli run --dataset <name> --seed <k> [--config path]`

Runs the full pipeline for one (dataset, seed): load -> split/standardize -> fit point
predictor -> fit/train the conformity score (`cfg.method`) -> split-conformal calibrate ->
evaluate on the test split. `run_one` returns (scalar metrics, set geometry): the geometry is
a (n_test, n_directions, output_dim) array of boundary points around each test point's
center -- the actual shape of the calibrated set, usable directly by anyone downstream, not
just its coverage/volume. Prints the metrics as one JSON line; if --output-dir is given, also
appends the metrics to a CSV there and saves the geometry to an .npz file.

`cfg.method` picks one of two paths in `run_one`:
  - CLeS family ("icnn", "asymmetric_linear", "spline"): trained via prediction_set_loss on
    D_score, exactly as before.
  - baselines ("absolute_residual", "global_ellipsoid", "empirical_copula"): no training loop --
    just the frozen point predictor plus, if the baseline has one, a `.fit()` step on D_score's
    residuals (e.g. computing a covariance or empirical CDFs). Calibration and evaluation are
    identical either way, since both paths only need the model's `.score`/
    `.log_prediction_set_size` hooks.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from learnscore.baseline_absolute_residual import AbsoluteResidual
from learnscore.baseline_ellipsoid import GlobalEllipsoid
from learnscore.baseline_empirical_copula import EmpiricalCopula
from learnscore.config import RunConfig, load_config
from learnscore.calibrate import conformal_threshold
from learnscore.datasets import registry
from learnscore.datasets.loaders import load_dataset
from learnscore.datasets.splits import make_split
from learnscore.evaluate import evaluate_on_test
from learnscore.geometry import set_boundary
from learnscore.loss import LossConfig
from learnscore.model import ICNN
from learnscore.model_asymmetric_linear import AsymmetricLinear
from learnscore.model_mvcs import MVCS
from learnscore.model_spline import SplineScore
from learnscore.point_predictor import PointConfig, predict, train_point_predictor
from learnscore.seed import set_global_seed
from learnscore.train import TrainConfig, train_score_model

# CLeS family: trained via prediction_set_loss, one model_factory per method.
CLES_MODEL_FACTORIES = {
    "icnn": lambda cfg, input_dim, output_dim: lambda: ICNN(
        input_dim=input_dim, residual_dim=output_dim, encoder_hidden_dims=cfg.encoder_hidden_dims,
        icnn_hidden_dims=cfg.icnn_hidden_dims, epsilon=cfg.epsilon, qmc_base_seed=cfg.seed * 100_003),
    "asymmetric_linear": lambda cfg, input_dim, output_dim: lambda: AsymmetricLinear(
        input_dim=input_dim, residual_dim=output_dim, encoder_hidden_dims=cfg.encoder_hidden_dims,
        epsilon=cfg.epsilon),
    "spline": lambda cfg, input_dim, output_dim: lambda: SplineScore(
        input_dim=input_dim, residual_dim=output_dim, encoder_hidden_dims=cfg.encoder_hidden_dims,
        num_knots=cfg.num_knots),
    "mvcs": lambda cfg, input_dim, output_dim: lambda: MVCS(
        input_dim=input_dim, residual_dim=output_dim, encoder_hidden_dims=cfg.encoder_hidden_dims,
        p_init=cfg.mvcs_p_init),
}

# Baselines: no training loop, just a fit step (if any) on D_score residuals.
BASELINE_FACTORIES = {
    "absolute_residual": lambda cfg, output_dim: AbsoluteResidual(residual_dim=output_dim),
    "global_ellipsoid": lambda cfg, output_dim: GlobalEllipsoid(residual_dim=output_dim,
                                                                 ridge=cfg.ellipsoid_ridge),
    "empirical_copula": lambda cfg, output_dim: EmpiricalCopula(residual_dim=output_dim),
}

RESULT_COLUMNS = ["dataset", "seed", "alpha", "coverage", "mean_volume_original_scale",
                   "conformal_quantile", "selected_epoch", "training_seconds"]


def run_one(cfg: RunConfig) -> tuple[dict, np.ndarray | None]:
    set_global_seed(cfg.seed)

    spec = registry.get(cfg.dataset)
    X, Y = load_dataset(spec)
    split = make_split(X, Y, seed=cfg.seed, pred_frac=cfg.pred_frac, score_frac=cfg.score_frac,
                        cal_frac=cfg.cal_frac, test_frac=cfg.test_frac,
                        pred_val_frac=cfg.pred_val_frac, score_val_frac=cfg.score_val_frac)

    # frozen mean predictor mu_hat(x), fit on D_pred only
    point_cfg = PointConfig(hidden_dims=cfg.point_hidden_dims, output_dim=spec.output_dim,
                             learning_rate=cfg.point_learning_rate, max_epochs=cfg.point_max_epochs,
                             patience=cfg.point_patience, seed=cfg.seed)
    point_model = train_point_predictor(
        split.X_pred_std[split.pred_train_idx], split.Y_pred_std[split.pred_train_idx],
        split.X_pred_std[split.pred_val_idx], split.Y_pred_std[split.pred_val_idx],
        point_cfg, device=cfg.device)

    def augment(x_std: np.ndarray) -> np.ndarray:
        """[x, mu_hat(x)]: the ICNN's encoder sees the point prediction alongside the raw
        covariates. mu_hat is frozen, so this can be precomputed once per split."""
        return np.concatenate([x_std, predict(point_model, x_std, device=cfg.device)], axis=1)

    def residual(x_std: np.ndarray, y_std: np.ndarray) -> np.ndarray:
        return y_std - predict(point_model, x_std, device=cfg.device)

    x_score_aug = augment(split.X_score_std)
    r_score = residual(split.X_score_std, split.Y_score_std)
    x_cal_aug = augment(split.X_cal_std)
    r_cal = residual(split.X_cal_std, split.Y_cal_std)
    x_test_aug = augment(split.X_test_std)
    r_test = residual(split.X_test_std, split.Y_test_std)

    input_dim = x_score_aug.shape[1]

    if cfg.method in CLES_MODEL_FACTORIES:
        model_factory = CLES_MODEL_FACTORIES[cfg.method](cfg, input_dim, spec.output_dim)

        loss_cfg = LossConfig(alpha=cfg.alpha, tau_mode=cfg.tau_mode, tau_fixed=cfg.tau_fixed,
                               size_objective=cfg.size_objective, kappa_coverage=cfg.kappa_coverage,
                               lambda_miss=cfg.lambda_miss, lambda_cov=cfg.lambda_cov)
        train_cfg = TrainConfig(learning_rate=cfg.learning_rate, batch_size=cfg.batch_size,
                                 max_epochs=cfg.max_epochs, patience=cfg.patience,
                                 grad_clip_norm=cfg.grad_clip_norm, checkpoint_every=cfg.checkpoint_every,
                                 coverage_tolerance=cfg.coverage_tolerance,
                                 min_selection_epoch=cfg.min_selection_epoch, seed=cfg.seed)

        start = time.time()
        result = train_score_model(
            model_factory, x_score_aug[split.score_train_idx], r_score[split.score_train_idx],
            x_score_aug[split.score_val_idx], r_score[split.score_val_idx],
            loss_cfg, train_cfg, device=cfg.device)
        training_seconds = time.time() - start
        model = result.model
        model.eval()
        selected_epoch = result.selected_epoch
    elif cfg.method in BASELINE_FACTORIES:
        # No prediction_set_loss training loop: build the (frozen) baseline object directly and,
        # if it has one, run its fit step on D_score's residuals.
        start = time.time()
        model = BASELINE_FACTORIES[cfg.method](cfg, spec.output_dim)
        if hasattr(model, "fit"):
            model.fit(r_score)
        training_seconds = time.time() - start
        selected_epoch = None
    else:
        raise ValueError(f"unknown method {cfg.method!r}")

    with torch.no_grad():
        x_cal_t = torch.as_tensor(x_cal_aug, dtype=torch.float32, device=cfg.device)
        r_cal_t = torch.as_tensor(r_cal, dtype=torch.float32, device=cfg.device)
        cal_scores = model.score(x_cal_t, r_cal_t).cpu().numpy().reshape(-1)
    q_hat = conformal_threshold(cal_scores, cfg.alpha)

    eval_result = evaluate_on_test(model, x_test_aug, r_test, q_hat, split.y_scaler.std, device=cfg.device)

    # The actual shape of the calibrated set, not just its coverage/volume: for each test
    # point, a center (the point prediction) plus a cloud of boundary points around it, in
    # the original response units -- exact at d=1/d=2, a QMC boundary sample at d>2 (see
    # geometry.py). Kept out of the scalar `row` dict; main() saves it separately.
    centers_std = predict(point_model, split.X_test_std, device=cfg.device)
    centers = split.y_scaler.inverse_transform(centers_std)
    if math.isinf(q_hat):
        boundary = None
    else:
        x_test_t = torch.as_tensor(x_test_aug, dtype=torch.float32, device=cfg.device)
        offsets_std = set_boundary(model, x_test_t, q_hat, n_directions=cfg.boundary_directions,
                                    seed=cfg.seed).cpu().numpy()
        boundary = centers[:, None, :] + offsets_std * split.y_scaler.std

    row = {
        "dataset": cfg.dataset, "seed": cfg.seed, "alpha": cfg.alpha,
        "coverage": eval_result.coverage,
        "mean_volume_original_scale": eval_result.mean_volume_original_scale,
        "conformal_quantile": eval_result.conformal_quantile,
        "selected_epoch": selected_epoch,
        "training_seconds": training_seconds,
    }
    return row, boundary


def _append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate a conformity score method on a dataset.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir", default=None,
                         help="if given, appends the metrics row to <output-dir>/<dataset>.csv "
                              "and saves the calibrated set's geometry (per test point: a "
                              "center + a cloud of boundary points, in original response "
                              "units) to <output-dir>/<dataset>_<method>_seed<seed>_sets.npz")
    args = parser.parse_args()

    cfg = load_config(args.config, args.dataset, args.seed)
    row, boundary = run_one(cfg)
    print(json.dumps(row))
    if args.output_dir:
        out = Path(args.output_dir)
        _append_csv(out / f"{args.dataset}.csv", row)
        if boundary is not None:
            np.savez_compressed(out / f"{args.dataset}_{cfg.method}_seed{cfg.seed}_sets.npz",
                                 boundary=boundary)


if __name__ == "__main__":
    main()
