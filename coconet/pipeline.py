"""One (dataset, seed, method) cell of the benchmark.

    prepare_data   load -> four-way split -> preprocess (fit on D_pred) -> frozen point predictor
                   F(x) -> residuals r = y - F(x) on D_score (train / validation), D_cal, D_test
    run_<method>   fit the method on D_score, calibrate the conformal threshold on D_cal
    evaluate       joint coverage and prediction-set volume on D_test

Devices: CoCoNet_ICNN trains and measures its test volume on the GPU (the paper used a Quadro
RTX 6000); every other method runs on the CPU with 4 threads, which is how the paper's numbers
were produced.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from coconet.conformal.calibration import conformal_threshold
from coconet.conformal.ellipsoid import GlobalEllipsoidScore, LocalKnnEllipsoidScore
from coconet.conformal.empirical_copula import EmpiricalCopulaConfig, EmpiricalCopulaScore
from coconet.data.loaders import load_dataset
from coconet.data.preprocessing import Preprocessor
from coconet.data.splits import make_four_way_split, make_internal_validation_split, slice_xy
from coconet.models.icnn import (
    ICNN, IdentityFICNN, IdentityXFICNN, IdentityXICNN, NoContextICNN,
    PredictionAugmentedICNN, PredictionOnlyICNN,
)
from coconet.models.mlp import MLP
from coconet.models.spline import PredictionAugmentedSplineScore, compute_score_training_knots
from coconet.models.volume_estimator import log_volume_converged_nested
from coconet.training.lambda_search import LambdaSearchConfig, search_lambdas
from coconet.training.train_mvcs import MvcsTrainConfig, train_mvcs_fixed_center
from coconet.training.train_point import PointTrainConfig, train_point_predictor
from coconet.training.train_score import ScoreHyperparams, ScoreTrainConfig
from coconet.utils import get_logger, read_json, residual_sha, set_global_seed, sha256_file

logger = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"

ABSOLUTE_RESIDUAL = "absolute_residual"
LOCALLY_WEIGHTED = "locally_weighted"
EMPIRICAL_COPULA = "empirical_copula"
GLOBAL_ELLIPSOID = "global_ellipsoid"
LOCAL_ELLIPSOID = "local_ellipsoid"
MVCS = "mvcs"
SPLINE = "spline"
ICNN_METHOD = "icnn"
MULTIVARIATE_METHODS = (ABSOLUTE_RESIDUAL, EMPIRICAL_COPULA, GLOBAL_ELLIPSOID, LOCAL_ELLIPSOID,
                        MVCS, SPLINE, ICNN_METHOD)
UNIVARIATE_METHODS = (ABSOLUTE_RESIDUAL, LOCALLY_WEIGHTED, MVCS, SPLINE, ICNN_METHOD)
ALL_METHODS = (ABSOLUTE_RESIDUAL, LOCALLY_WEIGHTED, EMPIRICAL_COPULA, GLOBAL_ELLIPSOID,
               LOCAL_ELLIPSOID, MVCS, SPLINE, ICNN_METHOD)

# Encoder-input arms of the ICNN (appendix ablation). "xf" is the paper's CoCoNet_ICNN.
ICNN_ENCODER_INPUTS = {
    "xf": PredictionAugmentedICNN, "f": PredictionOnlyICNN, "x": ICNN, "none": NoContextICNN,
    "id_xf": IdentityXFICNN, "id_x": IdentityXICNN, "id_f": IdentityFICNN,
}


def methods_for(output_dim: int) -> tuple[str, ...]:
    return UNIVARIATE_METHODS if output_dim == 1 else MULTIVARIATE_METHODS


# --------------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------------
@dataclass
class PreparedData:
    dataset: str
    seed: int
    output_dim: int
    input_dim: int
    x_score: np.ndarray          # D_score, internal training part
    r_score: np.ndarray
    x_validation: np.ndarray     # D_score, internal validation part
    r_validation: np.ndarray
    x_cal: np.ndarray
    r_cal: np.ndarray
    x_test: np.ndarray
    r_test: np.ndarray
    target_std: np.ndarray       # (d,): standardized volume -> original-scale volume Jacobian
    point_model: MLP
    n_pred: int
    n_score: int
    n_cal: int
    n_test: int
    row_fingerprint: str         # hash of the split row indices
    residual_sha: str            # hash of the test residuals (proves the same F was used)
    max_rows: int | None
    point_predictor_file: str = ""


def _row_fingerprint(split) -> str:
    idx = np.concatenate([split.pred_idx, split.score_idx, split.cal_idx, split.test_idx]).astype(np.int64)
    return hashlib.sha256(np.ascontiguousarray(idx).tobytes()).hexdigest()[:16]


def load_point_predictor(path: Path) -> MLP:
    """An MLP saved by `save_point_predictor` (architecture + state dict)."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model = MLP(input_dim=int(blob["input_dim"]), hidden_dims=list(blob["hidden_dims"]),
                output_dim=int(blob["output_dim"]))
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model


def save_point_predictor(model: MLP, path: Path) -> None:
    layers = [m for m in model.net if isinstance(m, torch.nn.Linear)]
    torch.save({"input_dim": layers[0].in_features, "hidden_dims": [m.out_features for m in layers[:-1]],
                "output_dim": layers[-1].out_features,
                "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}, path)


def fit_point_predictor(x_pred: np.ndarray, y_pred: np.ndarray, cfg: dict, seed: int,
                        pred_val_frac: float) -> MLP:
    """The MLP point predictor: fit on D_pred's internal training part, early-stopped on its
    validation part, on the CPU."""
    split = make_internal_validation_split(np.arange(len(x_pred)), pred_val_frac, seed=seed)
    pp = cfg["point_predictor"]
    point_cfg = PointTrainConfig(
        hidden_dims=list(pp["hidden_dims"]), batch_size=min(pp["batch_size"], max(len(split.train_idx), 1)),
        learning_rate=pp["learning_rate"], weight_decay=pp["weight_decay"],
        max_epochs=pp["max_epochs"], patience=pp["patience"], min_delta=pp["min_delta"],
        seed=seed, output_dim=y_pred.shape[1],
    )
    result = train_point_predictor(x_pred[split.train_idx], y_pred[split.train_idx],
                                   x_pred[split.val_idx], y_pred[split.val_idx], point_cfg, device="cpu")
    result.model.eval()
    return result.model


def prepare_data(dataset: str, seed: int, cfg: dict, datasets_cfg: dict,
                 point_predictor_dir: Path | None = None) -> PreparedData:
    """Load, split, preprocess and compute the residuals of the frozen point predictor.

    With `point_predictor_dir`, the predictor <dir>/<dataset>__seed<seed>.pt is loaded and its
    sidecar's row fingerprint and residual hash are checked against this run (any mismatch is an
    error). Without it, the predictor is fitted here.
    """
    set_global_seed(seed)
    ds_cfg = datasets_cfg[dataset]
    max_rows = ds_cfg.get("max_rows")
    bundle = load_dataset(dataset, datasets_cfg, RAW_DIR, max_rows=max_rows,
                          subsample_seed=int(cfg["row_sample_seed"]) + int(seed))
    if getattr(bundle.y, "ndim", 2) == 1:
        bundle.y = bundle.y.to_frame(name=(getattr(bundle.y, "name", None) or "target"))
    output_dim = int(ds_cfg["output_dim"])
    if bundle.y.shape[1] != output_dim:
        raise ValueError(f"{dataset}: loaded y has {bundle.y.shape[1]} columns, expected output_dim={output_dim}")

    sp = cfg["splitting"]
    n_total = len(bundle.X)
    split = make_four_way_split(n_total, sp["pred_frac"], sp["score_frac"], sp["cal_frac"],
                                sp["test_frac"], seed=seed)
    X_pred, y_pred = slice_xy(bundle.X, bundle.y, split.pred_idx)
    X_score, y_score = slice_xy(bundle.X, bundle.y, split.score_idx)
    X_cal, y_cal = slice_xy(bundle.X, bundle.y, split.cal_idx)
    X_test, y_test = slice_xy(bundle.X, bundle.y, split.test_idx)

    pre = Preprocessor(categorical_columns=bundle.categorical_columns, log1p_target=False)
    pre.fit(X_pred, y_pred)
    Xp = {name: pre.transform_features(X) for name, X in
          (("pred", X_pred), ("score", X_score), ("cal", X_cal), ("test", X_test))}
    yp = {name: pre.transform_target(y).reshape(len(Xp[name]), output_dim) for name, y in
          (("pred", y_pred), ("score", y_score), ("cal", y_cal), ("test", y_test))}
    target_std = np.asarray(pre._target_std, dtype=float).reshape(-1)
    if target_std.shape[0] != output_dim:
        target_std = np.full(output_dim, float(target_std))
    fingerprint = _row_fingerprint(split)

    predictor_file = ""
    sidecar = None
    if point_predictor_dir is not None:
        pt = Path(point_predictor_dir) / f"{dataset}__seed{seed}.pt"
        js = pt.with_suffix(".json")
        if not pt.is_file() or not js.is_file():
            raise RuntimeError(f"frozen point predictor missing: {pt} (+ .json)")
        sidecar = read_json(js)
        if sidecar.get("pt_sha256") != sha256_file(pt):
            raise RuntimeError(f"{pt} does not match its sidecar's sha256")
        if sidecar.get("dataset") != dataset or int(sidecar.get("seed", -1)) != seed \
                or sidecar.get("max_rows") != max_rows or sidecar.get("row_fingerprint") != fingerprint:
            raise RuntimeError(f"{js} was fitted on other data (dataset/seed/max_rows/row split differ)")
        point_model = load_point_predictor(pt)
        predictor_file = str(pt)
    else:
        point_model = fit_point_predictor(Xp["pred"], yp["pred"], cfg, seed, sp["pred_val_frac"])

    with torch.no_grad():
        mu = {name: point_model(torch.as_tensor(Xp[name], dtype=torch.float32)).numpy().reshape(len(Xp[name]), output_dim)
              for name in ("score", "cal", "test")}
    r = {name: yp[name] - mu[name] for name in ("score", "cal", "test")}

    val = make_internal_validation_split(np.arange(len(Xp["score"])), sp["score_val_frac"], seed=seed)
    r_sha = residual_sha(r["test"])
    if sidecar is not None and sidecar.get("residual_sha") != r_sha:
        raise RuntimeError(f"{predictor_file}: this run's residuals ({r_sha}) differ from the saved "
                           f"ones ({sidecar.get('residual_sha')}); the predictor or the data changed")
    return PreparedData(
        dataset=dataset, seed=seed, output_dim=output_dim, input_dim=Xp["score"].shape[1],
        x_score=Xp["score"][val.train_idx], r_score=r["score"][val.train_idx],
        x_validation=Xp["score"][val.val_idx], r_validation=r["score"][val.val_idx],
        x_cal=Xp["cal"], r_cal=r["cal"], x_test=Xp["test"], r_test=r["test"],
        target_std=target_std, point_model=point_model,
        n_pred=len(Xp["pred"]), n_score=len(Xp["score"]), n_cal=len(Xp["cal"]), n_test=len(Xp["test"]),
        row_fingerprint=fingerprint, residual_sha=r_sha, max_rows=max_rows,
        point_predictor_file=predictor_file,
    )


# --------------------------------------------------------------------------------------------
# a fitted method
# --------------------------------------------------------------------------------------------
@dataclass
class MethodResult:
    method: str
    q_hat: float
    model: Any
    volume_fn: Callable[[np.ndarray], np.ndarray]           # x -> volume per row
    membership_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]   # (x, r) -> covered per row
    training_time_seconds: float
    parameter_count: int
    selected_epoch: int = -1
    extra: dict[str, Any] = field(default_factory=dict)     # tuned values, certificates, ...
    history: list[dict] = field(default_factory=list)


class _TorchVolumeFn:
    def __init__(self, model, q_hat: float) -> None:
        self.model, self.q_hat = model, q_hat

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x_t = torch.as_tensor(x, dtype=torch.float32)
        with torch.no_grad():
            return self.model.volume(x_t, self.q_hat).numpy().reshape(-1)


class _TorchMembershipFn:
    def __init__(self, model, q_hat: float) -> None:
        self.model, self.q_hat = model, q_hat

    def __call__(self, x: np.ndarray, r: np.ndarray) -> np.ndarray:
        x_t = torch.as_tensor(x, dtype=torch.float32)
        r_t = torch.as_tensor(r, dtype=torch.float32)
        with torch.no_grad():
            return self.model.score(x_t, r_t).numpy().reshape(-1) <= self.q_hat


class ConvergedIcnnVolumeFn:
    """Test-set volumes of an ICNN by the iteratively whitened estimator: per-point seeded
    (by test row index), budgets M and 2M from one pass, V(2M) reported with the certificate
    |V(2M)^(1/d) / V(M)^(1/d) - 1|."""

    MAX_RAYS_PER_CHUNK = 2_000_000

    def __init__(self, model, q_hat: float, cfg: dict, device: str) -> None:
        tv = cfg["icnn"]["test_volume"]
        self.model, self.q_hat, self.device = model, q_hat, device
        self.rounds = int(tv["rounds"])
        self.refine_dirs = int(tv["refine_directions"])
        budgets = tv["budget_by_dim"]
        d = int(model.residual_dim)
        self.m1 = int(budgets.get(d, budgets.get(str(d), budgets["default"])))
        self.tolerance = float(tv["convergence_tolerance"])
        self.certificate: dict = {}

    def _move(self, dev: str) -> None:
        self.model.to(dev)
        for pm in (getattr(self.model, "_frozen_predictor", None) or []):
            pm.to(dev)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        dev = self.device
        x_t = torch.as_tensor(x, dtype=torch.float32, device=dev)
        if dev != "cpu":
            self._move(dev)
        try:
            m1, m2 = self.m1, 2 * self.m1
            per = max(1, self.MAX_RAYS_PER_CHUNK // m2)
            v1, v2 = [], []
            with torch.no_grad():
                for i in range(0, len(x_t), per):
                    xb = x_t[i:i + per]
                    lv = log_volume_converged_nested(self.model, xb, self.q_hat, [m1, m2],
                                                     point_seeds=list(range(i, i + len(xb))),
                                                     rounds=self.rounds, refine_dirs=self.refine_dirs)
                    v1.append(torch.exp(lv[m1]).cpu().numpy().reshape(-1))
                    v2.append(torch.exp(lv[m2]).cpu().numpy().reshape(-1))
        finally:
            if dev != "cpu":
                self._move("cpu")
        v1 = np.concatenate(v1) if v1 else np.empty(0, dtype=float)
        v2 = np.concatenate(v2) if v2 else np.empty(0, dtype=float)
        d = int(self.model.residual_dim)
        ok = np.isfinite(v1) & np.isfinite(v2) & (v1 > 0) & (v2 > 0)
        change = float((np.mean(v2[ok]) / np.mean(v1[ok])) ** (1.0 / d) - 1.0) if ok.any() else float("nan")
        self.certificate = {"test_budget_M": m1, "test_budget_reported_M": m2,
                            "test_budget_root_change_pct": 100.0 * change,
                            "test_budget_converged": bool(abs(change) < self.tolerance),
                            "test_volume_device": dev,
                            "test_volume_device_name": (torch.cuda.get_device_name(0) if dev == "cuda" else "cpu")}
        return v2


# --------------------------------------------------------------------------------------------
# baselines
# --------------------------------------------------------------------------------------------
def run_absolute_residual(data: PreparedData, alpha: float) -> MethodResult:
    """L-infinity cube: s = max_j |r_j|."""
    t0 = time.time()
    q_hat = conformal_threshold(np.abs(data.r_cal).max(axis=1), alpha)
    d = data.output_dim
    return MethodResult(
        method=ABSOLUTE_RESIDUAL, q_hat=float(q_hat), model=None,
        volume_fn=lambda x: np.full(len(x), math.exp(d * math.log(2.0 * q_hat))),
        membership_fn=lambda x, r: np.all(np.abs(r) <= q_hat, axis=1),
        training_time_seconds=time.time() - t0, parameter_count=0,
    )


class _LocallyWeighted:
    def __init__(self, rho_model, q_hat: float, rho_floor: float) -> None:
        self.rho_model, self.q_hat, self.rho_floor = rho_model, float(q_hat), float(rho_floor)

    def rho(self, x: np.ndarray) -> np.ndarray:
        self.rho_model.eval()
        with torch.no_grad():
            raw = self.rho_model(torch.as_tensor(x, dtype=torch.float32)).numpy().reshape(-1)
        return np.clip(raw, self.rho_floor, None)

    def volume(self, x: np.ndarray) -> np.ndarray:
        return 2.0 * self.rho(x) * self.q_hat

    def membership(self, x: np.ndarray, r: np.ndarray) -> np.ndarray:
        return np.abs(np.asarray(r, dtype=float).reshape(-1)) <= self.rho(x) * self.q_hat


def run_locally_weighted(data: PreparedData, cfg: dict, alpha: float) -> MethodResult:
    """Locally weighted split conformal (Lei et al. 2018): a second MLP of the point
    predictor's configuration regresses |r| on x; the score is |r| / rho(x). Univariate only."""
    if data.output_dim != 1:
        raise ValueError("locally_weighted is a univariate baseline")
    t0 = time.time()
    inner = make_internal_validation_split(np.arange(len(data.x_score)), cfg["splitting"]["score_val_frac"], seed=0)
    abs_r = np.abs(np.asarray(data.r_score, dtype=float).reshape(-1))
    pp = cfg["point_predictor"]
    train_cfg = PointTrainConfig(
        hidden_dims=list(pp["hidden_dims"]), batch_size=min(pp["batch_size"], max(len(inner.train_idx), 1)),
        learning_rate=pp["learning_rate"], weight_decay=pp["weight_decay"],
        max_epochs=pp["max_epochs"], patience=pp["patience"], min_delta=pp["min_delta"], seed=0, output_dim=1,
    )
    rho_model = train_point_predictor(
        data.x_score[inner.train_idx], abs_r[inner.train_idx].reshape(-1, 1),
        data.x_score[inner.val_idx], abs_r[inner.val_idx].reshape(-1, 1), train_cfg, device="cpu").model
    rho_model.eval()
    train_time = time.time() - t0

    rho_floor = float(cfg["locally_weighted_conformal"]["rho_floor"])
    lw = _LocallyWeighted(rho_model, 0.0, rho_floor)
    cal_scores = np.abs(np.asarray(data.r_cal, dtype=float).reshape(-1)) / lw.rho(data.x_cal)
    q_hat = conformal_threshold(cal_scores, alpha)
    lw.q_hat = float(q_hat)
    return MethodResult(
        method=LOCALLY_WEIGHTED, q_hat=float(q_hat), model=rho_model, volume_fn=lw.volume,
        membership_fn=lw.membership, training_time_seconds=train_time,
        parameter_count=int(sum(p.numel() for p in rho_model.parameters())),
    )


def run_empirical_copula(data: PreparedData, cfg: dict, alpha: float) -> MethodResult:
    cc = cfg["empirical_copula"]
    config = EmpiricalCopulaConfig(
        beta=float(cc["beta"]), log_epsilon=float(cc["log_epsilon"]), normalized_scores=True,
        normalizer_hidden_dims=list(cc["normalizer_hidden_dims"]), normalizer_epochs=int(cc["normalizer_epochs"]),
        normalizer_batch_size=int(cc["normalizer_batch_size"]),
        normalizer_learning_rate=float(cc["normalizer_learning_rate"]),
        normalizer_weight_decay=float(cc["normalizer_weight_decay"]), normalizer_patience=int(cc["normalizer_patience"]),
    )
    t0 = time.time()
    score = EmpiricalCopulaScore(output_dim=data.output_dim, config=config)
    score.fit_normalizer(data.x_score, data.r_score, data.x_validation, data.r_validation)
    train_time = time.time() - t0
    score.calibrate(data.x_cal, data.r_cal, alpha)
    return MethodResult(
        method=EMPIRICAL_COPULA, q_hat=float(score.u), model=score,
        volume_fn=lambda x: np.asarray(score.volume(x), dtype=float).reshape(-1),
        membership_fn=lambda x, r: np.asarray(score.contains(x, r), dtype=bool).reshape(-1),
        training_time_seconds=train_time, parameter_count=score.parameter_count(),
        extra=score.state_dict_summary(),
    )


def run_ellipsoid(method: str, data: PreparedData, cfg: dict, alpha: float) -> MethodResult:
    ell = cfg["ellipsoid"]
    ridge = float(ell["covariance_ridge"])
    t0 = time.time()
    if method == GLOBAL_ELLIPSOID:
        model = GlobalEllipsoidScore(output_dim=data.output_dim, covariance_ridge=ridge)
    else:
        model = LocalKnnEllipsoidScore(
            output_dim=data.output_dim, covariance_ridge=ridge,
            local_weight=float(ell["local_weight"]), neighbor_fraction=float(ell["neighbor_fraction"]),
            num_neighbors=None, distance_metric=str(ell["distance_metric"]),
            standardize_context=bool(ell["standardize_context"]),
        )
    model.fit(data.x_score, data.r_score)
    train_time = time.time() - t0
    cal_scores = np.asarray(model.score(data.x_cal, data.r_cal), dtype=float).reshape(-1)
    q_hat = conformal_threshold(cal_scores, alpha)
    return MethodResult(
        method=method, q_hat=float(q_hat), model=model,
        volume_fn=lambda x: np.asarray(model.area(x, q_hat), dtype=float).reshape(-1),
        membership_fn=lambda x, r: np.asarray(model.score(x, r), dtype=float).reshape(-1) <= q_hat,
        training_time_seconds=train_time, parameter_count=int(model.n_params()),
        extra={"stabilization_retries": model.stabilization_retries,
               **({"num_neighbors": model.k} if method == LOCAL_ELLIPSOID else {})},
    )


def run_mvcs(data: PreparedData, cfg: dict, alpha: float, seed: int) -> MethodResult:
    """MVCS_conv: every grid candidate is trained on D_score and scored by its validation
    objective; the best one is the reported model."""
    mv = cfg["mvcs"]
    n = max(len(data.x_score), 1)

    def config(hidden_dim, n_layers, matrix_lr, p_lr, batch_size) -> MvcsTrainConfig:
        return MvcsTrainConfig(
            output_dim=data.output_dim, alpha=alpha, hidden_dim=int(hidden_dim), n_layers=int(n_layers),
            ridge=float(mv["ridge"]), p_init=float(mv["p_init"]), p_min=float(mv["p_min"]),
            order_statistic_convention=str(mv["order_statistic_convention"]),
            average_volume_term=bool(mv["average_volume_term"]),
            matrix_learning_rate=float(matrix_lr), p_learning_rate=float(p_lr),
            max_epochs=int(mv["max_epochs"]), batch_size=int(min(batch_size, n)), patience=int(mv["patience"]),
            grad_clip_norm=float(mv["grad_clip_norm"]), checkpoint_every=int(mv["checkpoint_every"]),
            inner_calibration_frac=float(mv["inner_calibration_frac"]),
            nonfinite_batch_tolerance=float(mv["nonfinite_batch_tolerance"]), seed=seed, device="cpu",
        )

    g = mv["grid"]
    t0 = time.time()
    best, best_obj, candidates = None, float("inf"), []
    for hd, nl, mlr, bs, plr in itertools.product(g["hidden_dim"], g["n_layers"], g["matrix_learning_rate"],
                                                  g["batch_size"], g["p_learning_rate"]):
        cand = {"hidden_dim": int(hd), "n_layers": int(nl), "matrix_learning_rate": float(mlr),
                "batch_size": int(bs), "p_learning_rate": float(plr)}
        try:
            res = train_mvcs_fixed_center(data.x_score, data.r_score, data.x_validation, data.r_validation,
                                          config(hd, nl, mlr, plr, bs))
            obj = float(res.validation_mvcs_objective)
        except Exception as exc:
            candidates.append({**cand, "validation_objective": float("nan"), "error": repr(exc)[:200]})
            continue
        candidates.append({**cand, "validation_objective": obj})
        if np.isfinite(obj) and obj < best_obj:
            best_obj, best = obj, cand
    if best is None:
        raise RuntimeError("mvcs: no grid candidate produced a finite validation objective")
    logger.info("mvcs tuned: %s (validation objective %.6g)", best, best_obj)

    result = train_mvcs_fixed_center(
        data.x_score, data.r_score, data.x_validation, data.r_validation,
        config(best["hidden_dim"], best["n_layers"], best["matrix_learning_rate"],
               best["p_learning_rate"], best["batch_size"]))
    train_time = time.time() - t0
    score = result.score
    q_hat = conformal_threshold(score.score(data.x_cal, data.r_cal), alpha)
    return MethodResult(
        method=MVCS, q_hat=float(q_hat), model=score,
        volume_fn=lambda x: np.asarray(score.volume(x, q_hat), dtype=float).reshape(-1),
        membership_fn=lambda x, r: np.asarray(score.contains(x, r, q_hat), dtype=bool).reshape(-1),
        training_time_seconds=train_time, parameter_count=int(score.n_params()),
        selected_epoch=result.selected_epoch,
        extra={"learned_p": float(score.p), "tuned": best, "tuning_candidates": candidates,
               "validation_objective": best_obj},
        history=list(result.training_history),
    )


# --------------------------------------------------------------------------------------------
# learned scores (CoCoNet)
# --------------------------------------------------------------------------------------------
def score_train_config(cfg: dict, seed: int, n_train: int, batch_size: int, patience: int) -> ScoreTrainConfig:
    sl = cfg["score_learning"]
    loss = sl["loss"]
    return ScoreTrainConfig(
        epsilon=float(sl["epsilon"]), grad_clip_norm=float(sl["grad_clip_norm"]),
        max_epochs=int(sl["max_epochs"]), patience=int(patience),
        batch_size=min(int(batch_size), max(n_train, 1)),
        coverage_tolerance=float(sl["coverage_tolerance"]), seed=seed,
        lambda_fallback_rule=str(sl["lambda_fallback_rule"]), lambda_search_mode="grid",
        gamma_grid_miss=tuple(float(v) for v in sl["gamma_grid_miss"]),
        gamma_grid_cov=tuple(float(v) for v in sl["gamma_grid_cov"]),
        kappa_miss=float(loss["kappa_miss"]), size_objective=str(loss["size_objective"]),
        train_tau_mode=str(loss["train_tau_mode"]), train_tau_fixed=float(loss["train_tau_fixed"]),
        score_space_miss_form=str(loss["score_space_miss_form"]),
        optimizer=str(sl["optimizer"]), checkpoint_selection_metric=str(sl["checkpoint_selection_metric"]),
        checkpoint_every=int(sl["checkpoint_every"]), min_selection_epoch=int(sl["min_selection_epoch"]),
    )


def score_hyperparams(cfg: dict, learning_rate: float, encoder_hidden_dims, num_knots=None,
                      icnn_hidden_dim=None) -> ScoreHyperparams:
    loss = cfg["score_learning"]["loss"]
    return ScoreHyperparams(learning_rate=float(learning_rate), lambda_cov=1.0, lambda_l2=float(loss["lambda_l2"]),
                            beta=float(loss["beta"]), num_knots=num_knots,
                            encoder_hidden_dims=tuple(int(v) for v in encoder_hidden_dims),
                            icnn_hidden_dim=icnn_hidden_dim)


def build_icnn(data: PreparedData, cfg: dict, encoder_hidden_dims, strong_convexity: float,
               encoder_input: str = "xf"):
    ic = cfg["icnn"]
    rf, sb = ic["root_finding"], ic["set_boundary"]
    cls = ICNN_ENCODER_INPUTS[encoder_input]
    extra = ({"point_predictor": data.point_model, "prediction_dim": data.output_dim}
             if encoder_input in ("xf", "f", "id_xf", "id_f") else {})
    return cls(
        **extra, input_dim=data.input_dim, encoder_hidden_dims=list(encoder_hidden_dims),
        icnn_hidden_dim=int(ic["icnn_hidden_dim"]), icnn_depth=int(ic["icnn_depth"]), icnn_hidden_dims=None,
        epsilon=float(strong_convexity),
        initial_bracket=rf["initial_bracket"], bracket_growth=rf["bracket_growth"],
        max_bracket_expansions=rf["max_bracket_expansions"], bisection_tol=rf["bisection_tol"],
        bisection_max_iter=rf["bisection_max_iter"],
        residual_dim=data.output_dim, activation=str(ic["activation"]), context_dependent_residual_maps=True,
        train_area_directions=int(ic["train_area_directions"]), eval_area_directions=int(ic["eval_area_directions"]),
        train_volume_directions=int(ic["train_volume_directions"]),
        eval_volume_directions=int(ic["eval_volume_directions"]),
        volume_whiten=bool(ic["volume_whiten"]), whiten_momentum=float(ic["whiten_momentum"]),
        whiten_shrink=float(ic["whiten_shrink"]), whiten_warmup_steps=int(ic["whiten_warmup_steps"]),
        set_boundary_initial_radius=sb["initial_radius"], set_boundary_bracket_growth=sb["bracket_growth"],
        set_boundary_max_radius=sb["max_radius"], set_boundary_tolerance=sb["tolerance"],
        set_boundary_bisection_steps=sb["bisection_steps"],
    )


def spline_knots(data: PreparedData, num_knots: int):
    kp, km = [], []
    for j in range(data.output_dim):
        a, b = compute_score_training_knots(data.r_score[:, j], num_knots)
        kp.append(a)
        km.append(b)
    return (kp[0], km[0]) if data.output_dim == 1 else (kp, km)


def build_spline(data: PreparedData, cfg: dict, encoder_hidden_dims, num_knots: int):
    kp, km = spline_knots(data, num_knots)
    sp = cfg["spline"]
    return PredictionAugmentedSplineScore(
        input_dim=data.input_dim, point_predictor=data.point_model, prediction_dim=data.output_dim,
        knots_plus=kp, knots_minus=km, hidden_dims=list(encoder_hidden_dims), epsilon=float(sp["epsilon"]),
        output_dim=data.output_dim, transformation_lr_multiplier=float(sp["transformation_lr_multiplier"]),
    )


def _train_learned_score(factory_builder, data: PreparedData, alpha: float, hp: ScoreHyperparams,
                         train_cfg: ScoreTrainConfig, device: str):
    """The gamma-grid search over the two loss weights (12 trainings); returns the winner."""
    search = search_lambdas(
        factory_builder, data.x_score.astype(np.float32), data.r_score.astype(np.float32),
        data.x_validation.astype(np.float32), data.r_validation.astype(np.float32),
        alpha, hp, train_cfg,
        LambdaSearchConfig(max_candidates=32, mode="grid", grid_miss=train_cfg.gamma_grid_miss,
                           grid_cov=train_cfg.gamma_grid_cov),
        device=device,
    )
    audit = search.summary_row()
    audit.update({"validation_coverage": float(search.best.final_val_real_coverage),
                  "validation_size": float(search.best.final_val_real_width),
                  "lambda_candidates": search.candidate_rows()})
    return search.best, audit


def _calibrate_torch(model, data: PreparedData, alpha: float) -> float:
    model.eval()
    with torch.no_grad():
        cal_scores = model.score(torch.as_tensor(data.x_cal, dtype=torch.float32),
                                 torch.as_tensor(data.r_cal, dtype=torch.float32)).numpy().reshape(-1)
    return conformal_threshold(cal_scores, alpha)


def run_spline(data: PreparedData, cfg: dict, alpha: float, seed: int, tuned: dict) -> MethodResult:
    """CoCoNet_Spline with the tuned (knots, lr, batch, encoder) selection; CPU."""
    n_train = len(data.x_score)
    hp = score_hyperparams(cfg, tuned["learning_rate"], tuned["encoder_hidden_dims"], num_knots=int(tuned["num_knots"]))
    train_cfg = score_train_config(cfg, seed, n_train, tuned["batch_size"], tuned["patience"])

    def factory_builder(hp_c: ScoreHyperparams):
        return lambda: build_spline(data, cfg, hp_c.encoder_hidden_dims or hp.encoder_hidden_dims, hp.num_knots)

    t0 = time.time()
    result, audit = _train_learned_score(factory_builder, data, alpha, hp, train_cfg, device="cpu")
    train_time = time.time() - t0
    model = result.model
    q_hat = _calibrate_torch(model, data, alpha)
    return MethodResult(
        method=SPLINE, q_hat=float(q_hat), model=model, volume_fn=_TorchVolumeFn(model, q_hat),
        membership_fn=_TorchMembershipFn(model, q_hat), training_time_seconds=train_time,
        parameter_count=int(sum(p.numel() for p in model.parameters())),
        selected_epoch=int(result.selected_epoch),
        extra={"tuned": tuned, "checkpoint_selection_rule": result.selection_rule, **audit},
        history=list(result.diagnostics_history),
    )


def run_icnn(data: PreparedData, cfg: dict, alpha: float, seed: int, tuned: dict, device: str,
             encoder_input: str = "xf") -> MethodResult:
    """CoCoNet_ICNN with the tuned (lr, batch, encoder, eps) selection; trains on `device`."""
    n_train = len(data.x_score)
    enc = [] if encoder_input.startswith("id_") else tuned["encoder_hidden_dims"]
    hp = score_hyperparams(cfg, tuned["learning_rate"], enc, icnn_hidden_dim=int(cfg["icnn"]["icnn_hidden_dim"]))
    train_cfg = score_train_config(cfg, seed, n_train, tuned["batch_size"], tuned["patience"])
    eps = float(tuned.get("strong_convexity", cfg["icnn"]["strong_convexity"]))

    def factory_builder(hp_c: ScoreHyperparams):
        return lambda: build_icnn(data, cfg, hp_c.encoder_hidden_dims or hp.encoder_hidden_dims, eps, encoder_input)

    t0 = time.time()
    result, audit = _train_learned_score(factory_builder, data, alpha, hp, train_cfg, device=device)
    train_time = time.time() - t0
    model = result.model
    q_hat = _calibrate_torch(model, data, alpha)
    volume_fn = ConvergedIcnnVolumeFn(model, q_hat, cfg, device)
    return MethodResult(
        method=ICNN_METHOD, q_hat=float(q_hat), model=model, volume_fn=volume_fn,
        membership_fn=_TorchMembershipFn(model, q_hat), training_time_seconds=train_time,
        parameter_count=int(sum(p.numel() for p in model.parameters())),
        selected_epoch=int(result.selected_epoch),
        extra={"tuned": tuned, "encoder_input": encoder_input, "strong_convexity": eps,
               "checkpoint_selection_rule": result.selection_rule, **audit},
        history=list(result.diagnostics_history),
    )


# --------------------------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------------------------
def evaluate_on_test(res: MethodResult, data: PreparedData, alpha: float) -> dict:
    """Joint test coverage and the mean prediction-set volume (original target scale). A cell
    with any non-finite, negative or zero test volume is marked invalid."""
    membership = res.membership_fn(data.x_test, data.r_test)
    v_std = np.asarray(res.volume_fn(data.x_test), dtype=float)
    d = data.output_dim
    invalid = ~np.isfinite(v_std) | (v_std <= 0)
    n_invalid = int(invalid.sum())
    v_orig = v_std * np.exp(float(np.sum(np.log(data.target_std))))
    row = {
        "dataset": data.dataset, "seed": data.seed, "method": res.method, "d": d, "alpha": alpha,
        "coverage": float(np.mean(membership)),
        "volume": float(np.mean(v_orig)) if n_invalid == 0 else float("nan"),
        "volume_standardized": float(np.mean(v_std)) if n_invalid == 0 else float("nan"),
        "size": float(np.mean(v_orig) ** (1.0 / d)) if n_invalid == 0 else float("nan"),
        "q_hat": res.q_hat, "n_test": int(len(v_std)), "n_invalid_volumes": n_invalid,
        "status": "success" if n_invalid == 0 else "invalid",
        "training_time_seconds": res.training_time_seconds, "parameter_count": res.parameter_count,
        "selected_epoch": res.selected_epoch,
        "n_pred": data.n_pred, "n_score": data.n_score, "n_cal": data.n_cal,
        "input_dim": data.input_dim, "max_rows": data.max_rows,
        "row_fingerprint": data.row_fingerprint, "residual_sha": data.residual_sha,
        "point_predictor_file": data.point_predictor_file,
    }
    cert = getattr(res.volume_fn, "certificate", None)
    if cert:
        row.update(cert)
    row.update({k: v for k, v in res.extra.items() if k not in ("lambda_candidates", "tuning_candidates")})
    return row


def run_method(method: str, data: PreparedData, cfg: dict, tuned: dict | None = None,
               device: str = "cpu", encoder_input: str = "xf") -> MethodResult:
    alpha = float(cfg["alpha"])
    set_global_seed(data.seed)
    if method == ABSOLUTE_RESIDUAL:
        return run_absolute_residual(data, alpha)
    if method == LOCALLY_WEIGHTED:
        return run_locally_weighted(data, cfg, alpha)
    if method == EMPIRICAL_COPULA:
        return run_empirical_copula(data, cfg, alpha)
    if method in (GLOBAL_ELLIPSOID, LOCAL_ELLIPSOID):
        return run_ellipsoid(method, data, cfg, alpha)
    if method == MVCS:
        return run_mvcs(data, cfg, alpha, data.seed)
    if method == SPLINE:
        return run_spline(data, cfg, alpha, data.seed, tuned)
    if method == ICNN_METHOD:
        return run_icnn(data, cfg, alpha, data.seed, tuned, device, encoder_input)
    raise ValueError(f"unknown method {method!r}; choose from {ALL_METHODS}")
