"""One (dataset, seed, method) cell.

    prepare_data   load -> split D_pred / D_score / D_cal / D_test -> preprocess (fit on D_pred)
                   -> frozen point predictor F(x) -> residuals r = y - F(x)
    run_method     fit the method on D_score, calibrate the conformal threshold on D_cal
    evaluate       joint coverage and mean set volume on D_test (original target scale)
"""

import hashlib
import itertools
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from coconet.conformal.calibration import conformal_threshold
from coconet.conformal.ellipsoid import GlobalEllipsoid, LocalEllipsoid
from coconet.conformal import pbs
from coconet.conformal.empirical_copula import EmpiricalCopula
from coconet.conformal.mvcs_multi import MultiNormMvcs
from coconet.data.loaders import load_dataset
from coconet.data.preprocessing import Preprocessor
from coconet.data.splits import four_way_split, train_val_split
from coconet.models.icnn import ICNN
from coconet.models.mlp import MLP
from coconet.models.spline import SplineScore, compute_knots
from coconet.models.volume_estimator import Gauge, estimate_log_volumes, log_volume_converged, pick_budget
from coconet.training.lambda_search import search_lambdas
from coconet.training.train_mvcs import MvcsConfig, train_mvcs
from coconet.training.train_point import PointTrainConfig, train_mlp
from coconet.training.train_score import TrainSettings
from coconet.utils import read_json, residual_sha, set_seed, sha256_file

REPO = Path(__file__).resolve().parents[1]
RAW_DIR = REPO / "data" / "raw"

MULTIVARIATE_METHODS = ("absolute_residual", "empirical_copula", "global_ellipsoid", "local_ellipsoid",
                        "mvcs", "spline", "icnn")
UNIVARIATE_METHODS = ("absolute_residual", "locally_weighted", "mvcs", "spline", "icnn")
# further baselines of the extended multivariate comparison
EXTRA_METHODS = ("mvcs_free", "mvcs_multi", "mvcs_multi_free", "pbs")
ALL_METHODS = MULTIVARIATE_METHODS + EXTRA_METHODS + ("locally_weighted",)


def methods_for(output_dim):
    return UNIVARIATE_METHODS if output_dim == 1 else MULTIVARIATE_METHODS + EXTRA_METHODS


# ------------------------------------------------------------------------------------------
# data
# ------------------------------------------------------------------------------------------
@dataclass
class Data:
    dataset: str
    seed: int
    d: int
    input_dim: int
    x_score: np.ndarray        # D_score, training part
    r_score: np.ndarray
    x_val: np.ndarray          # D_score, validation part
    r_val: np.ndarray
    x_cal: np.ndarray
    r_cal: np.ndarray
    x_test: np.ndarray
    r_test: np.ndarray
    target_std: np.ndarray     # per-output std used for standardization (volumes are rescaled by its product)
    point_model: MLP
    n_pred: int
    n_score: int
    n_cal: int
    n_test: int
    row_fingerprint: str       # hash of the split row indices
    residual_sha: str          # hash of the test residuals: same value = same data and predictor
    max_rows: int
    point_predictor_file: str = ""


def load_point_predictor(path):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model = MLP(int(blob["input_dim"]), list(blob["hidden_dims"]), int(blob["output_dim"]))
    model.load_state_dict(blob["state_dict"])
    return model.eval()


def save_point_predictor(model, path):
    linear = [m for m in model.net if isinstance(m, torch.nn.Linear)]
    torch.save({"input_dim": linear[0].in_features, "hidden_dims": [m.out_features for m in linear[:-1]],
                "output_dim": linear[-1].out_features,
                "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}, path)


def fit_point_predictor(x_pred, y_pred, cfg, seed):
    train_idx, val_idx = train_val_split(len(x_pred), cfg["split"]["pred_val"], seed)
    pp = cfg["point_predictor"]
    c = PointTrainConfig(hidden_dims=pp["hidden_dims"], output_dim=y_pred.shape[1],
                         batch_size=min(pp["batch_size"], max(len(train_idx), 1)),
                         learning_rate=pp["learning_rate"], weight_decay=pp["weight_decay"],
                         max_epochs=pp["max_epochs"], patience=pp["patience"], min_delta=pp["min_delta"], seed=seed)
    return train_mlp(x_pred[train_idx], y_pred[train_idx], x_pred[val_idx], y_pred[val_idx], c)


def prepare_data(dataset, seed, cfg, datasets_cfg, point_predictor_dir=None):
    """With `point_predictor_dir` the saved predictor <dir>/<dataset>__seed<S>.pt is loaded and
    the row fingerprint and residual hash in its .json are checked against this run; otherwise
    the predictor is fitted here."""
    set_seed(seed)
    ds = datasets_cfg[dataset]
    d = int(ds["output_dim"])
    max_rows = ds.get("max_rows")
    X, y, categorical = load_dataset(dataset, ds, RAW_DIR, max_rows, subsample_seed=int(cfg["row_sample_seed"]) + seed)
    if y.shape[1] != d:
        raise ValueError(f"{dataset}: {y.shape[1]} target columns, expected {d}")

    sp = cfg["split"]
    split = four_way_split(len(X), sp["pred"], sp["score"], sp["cal"], seed)
    idx = {"pred": split.pred, "score": split.score, "cal": split.cal, "test": split.test}
    Xs = {k: X.iloc[i].reset_index(drop=True) for k, i in idx.items()}
    ys = {k: y.iloc[i].reset_index(drop=True) for k, i in idx.items()}
    pre = Preprocessor(categorical).fit(Xs["pred"], ys["pred"])
    Xp = {k: pre.transform_features(v) for k, v in Xs.items()}
    yp = {k: pre.transform_target(v).reshape(len(Xp[k]), d) for k, v in ys.items()}
    fingerprint = hashlib.sha256(np.concatenate([split.pred, split.score, split.cal, split.test])
                                 .astype(np.int64).tobytes()).hexdigest()[:16]

    predictor_file, sidecar = "", None
    if point_predictor_dir is not None:
        pt = Path(point_predictor_dir) / f"{dataset}__seed{seed}.pt"
        js = pt.with_suffix(".json")
        if not pt.is_file() or not js.is_file():
            raise FileNotFoundError(f"point predictor missing: {pt} (+ .json)")
        sidecar = read_json(js)
        if sidecar["pt_sha256"] != sha256_file(pt):
            raise RuntimeError(f"{pt} does not match the sha256 in its sidecar")
        if sidecar["dataset"] != dataset or int(sidecar["seed"]) != seed or sidecar["max_rows"] != max_rows \
                or sidecar["row_fingerprint"] != fingerprint:
            raise RuntimeError(f"{js} belongs to other data (dataset, seed, row cap or split differ)")
        point_model = load_point_predictor(pt)
        predictor_file = str(pt)
    else:
        point_model = fit_point_predictor(Xp["pred"], yp["pred"], cfg, seed)

    with torch.no_grad():
        mu = {k: point_model(torch.as_tensor(Xp[k], dtype=torch.float32)).numpy().reshape(len(Xp[k]), d)
              for k in ("score", "cal", "test")}
    r = {k: yp[k] - mu[k] for k in mu}
    train_idx, val_idx = train_val_split(len(Xp["score"]), sp["score_val"], seed)
    r_sha = residual_sha(r["test"])
    if sidecar is not None and sidecar["residual_sha"] != r_sha:
        raise RuntimeError(f"{predictor_file}: residuals differ from the saved ones ({r_sha} vs "
                           f"{sidecar['residual_sha']}); the data or the predictor changed")
    return Data(dataset, seed, d, Xp["score"].shape[1],
                Xp["score"][train_idx], r["score"][train_idx], Xp["score"][val_idx], r["score"][val_idx],
                Xp["cal"], r["cal"], Xp["test"], r["test"],
                np.asarray(pre.target_std, dtype=float).reshape(-1), point_model,
                len(Xp["pred"]), len(Xp["score"]), len(Xp["cal"]), len(Xp["test"]),
                fingerprint, r_sha, max_rows, predictor_file)


# ------------------------------------------------------------------------------------------
# a fitted method
# ------------------------------------------------------------------------------------------
@dataclass
class Fitted:
    method: str
    q_hat: float
    model: object
    volume: callable            # x -> set volume per row (standardized target scale)
    contains: callable          # (x, r) -> membership per row
    train_seconds: float
    n_params: int
    selected_epoch: int = -1
    extra: dict = field(default_factory=dict)
    history: list = field(default_factory=list)
    gauge: callable = None      # device -> (score(rows, r), rows, tau): the region for the shared volume estimator


def _t(a):
    return torch.as_tensor(a, dtype=torch.float32)


def _row_indices(n, device):
    return torch.arange(n, dtype=torch.float32, device=device).unsqueeze(1)


def box_gauge(half_widths):
    """max_j |r_j| / h_j(x) for an axis-aligned box of half-widths h (rows = test-row indices)."""
    def make(device):
        h = torch.as_tensor(half_widths, dtype=torch.float32, device=device)
        return (lambda rows, r: (r.abs() / h[rows[:, 0].long()]).max(dim=1).values), _row_indices(len(h), device), 1.0
    return make


def run_absolute_residual(data, alpha):
    t0 = time.time()
    q = conformal_threshold(np.abs(data.r_cal).max(axis=1), alpha)
    d = data.d
    return Fitted("absolute_residual", q, None,
                  volume=lambda x: np.full(len(x), math.exp(d * math.log(2.0 * q))),
                  contains=lambda x, r: np.all(np.abs(r) <= q, axis=1),
                  train_seconds=time.time() - t0, n_params=0,
                  gauge=box_gauge(np.full((len(data.x_test), d), float(q))))


def run_locally_weighted(data, cfg, alpha):
    """Locally weighted split conformal (univariate): a second MLP regresses |r| on x, the score
    is |r| / rho(x). The MLP is fitted on the training part of D_score with its own
    validation carve-out (fixed seed 0)."""
    t0 = time.time()
    train_idx, val_idx = train_val_split(len(data.x_score), cfg["split"]["score_val"], seed=0)
    abs_r = np.abs(np.asarray(data.r_score, dtype=float).reshape(-1))
    pp = cfg["point_predictor"]
    c = PointTrainConfig(hidden_dims=pp["hidden_dims"], output_dim=1,
                         batch_size=min(pp["batch_size"], max(len(train_idx), 1)),
                         learning_rate=pp["learning_rate"], weight_decay=pp["weight_decay"],
                         max_epochs=pp["max_epochs"], patience=pp["patience"], min_delta=pp["min_delta"], seed=0)
    rho_model = train_mlp(data.x_score[train_idx], abs_r[train_idx].reshape(-1, 1),
                          data.x_score[val_idx], abs_r[val_idx].reshape(-1, 1), c)
    floor = float(cfg["locally_weighted"]["rho_floor"])

    def rho(x):
        with torch.no_grad():
            return np.clip(rho_model(_t(x)).numpy().reshape(-1), floor, None)

    train_seconds = time.time() - t0
    q = conformal_threshold(np.abs(np.asarray(data.r_cal, dtype=float).reshape(-1)) / rho(data.x_cal), alpha)
    return Fitted("locally_weighted", q, rho_model,
                  volume=lambda x: 2.0 * rho(x) * q,
                  contains=lambda x, r: np.abs(np.asarray(r, dtype=float).reshape(-1)) <= rho(x) * q,
                  train_seconds=train_seconds, n_params=sum(p.numel() for p in rho_model.parameters()))


def run_empirical_copula(data, cfg, alpha):
    c = cfg["empirical_copula"]
    t0 = time.time()
    model = EmpiricalCopula(data.d, c["beta"], c["log_epsilon"], c["normalizer"])
    model.fit(data.x_score, data.r_score, data.x_val, data.r_val)
    train_seconds = time.time() - t0
    model.calibrate(data.x_cal, data.r_cal, alpha)
    return Fitted("empirical_copula", float(model.u), model, volume=model.volume, contains=model.contains,
                  train_seconds=train_seconds, n_params=model.n_params(),
                  extra={"copula_u": model.u, "copula_q": model.q.tolist()},
                  gauge=box_gauge(model.half_widths(data.x_test)))


def run_ellipsoid(method, data, cfg, alpha):
    e = cfg["ellipsoid"]
    t0 = time.time()
    if method == "global_ellipsoid":
        model = GlobalEllipsoid(data.d, e["covariance_ridge"])
    else:
        model = LocalEllipsoid(data.d, e["covariance_ridge"], e["local_weight"], e["neighbor_fraction"])
    model.fit(data.x_score, data.r_score)
    train_seconds = time.time() - t0
    q = conformal_threshold(model.score(data.x_cal, data.r_cal), alpha)

    def gauge(device):
        """||L(x)^-1 r|| with the ellipsoid's own Cholesky factor L (rows = test-row indices)."""
        chol = model.chol if method == "global_ellipsoid" else model._chol(data.x_test)
        if chol.ndim == 2:
            chol = np.repeat(chol[None], len(data.x_test), axis=0)
        inv = torch.as_tensor(np.linalg.inv(chol), dtype=torch.float32, device=device)
        return (lambda rows, r: torch.einsum("nij,nj->ni", inv[rows[:, 0].long()], r).norm(dim=1)), \
            _row_indices(len(inv), device), q

    return Fitted(method, q, model, volume=lambda x: model.volume(x, q),
                  contains=lambda x, r: model.score(x, r) <= q,
                  train_seconds=train_seconds, n_params=model.n_params(),
                  extra={"num_neighbors": model.k} if method == "local_ellipsoid" else {}, gauge=gauge)


def run_mvcs(data, cfg, alpha, seed, convex=True):
    """Every grid candidate is trained on D_score and scored by its validation objective; the
    best one is the reported model. convex=False is the free-exponent variant."""
    m = cfg["mvcs"]
    g = m["grid"]
    n = max(len(data.x_score), 1)
    t0 = time.time()
    best, candidates = None, []
    for hd, nl, mlr, bs, plr in itertools.product(g["hidden_dim"], g["n_layers"], g["matrix_learning_rate"],
                                                  g["batch_size"], g["p_learning_rate"]):
        cand = {"hidden_dim": int(hd), "n_layers": int(nl), "matrix_learning_rate": float(mlr),
                "batch_size": int(bs), "p_learning_rate": float(plr)}
        c = MvcsConfig(output_dim=data.d, alpha=alpha, hidden_dim=int(hd), n_layers=int(nl), ridge=m["ridge"],
                       p_init=m["p_init"], matrix_learning_rate=float(mlr), p_learning_rate=float(plr),
                       batch_size=int(min(bs, n)), max_epochs=m["max_epochs"], patience=m["patience"],
                       grad_clip_norm=m["grad_clip_norm"], nonfinite_batch_tolerance=m["nonfinite_batch_tolerance"],
                       seed=seed, convex=convex, p_min=m["p_min"])
        try:
            score, epoch, objective, history = train_mvcs(data.x_score, data.r_score, data.x_val, data.r_val, c)
        except Exception as exc:
            candidates.append({**cand, "validation_objective": float("nan"), "error": repr(exc)[:200]})
            continue
        candidates.append({**cand, "validation_objective": objective})
        if np.isfinite(objective) and (best is None or objective < best[0]):
            best = (objective, cand, score, epoch, history)
    if best is None:
        raise RuntimeError("mvcs: no grid candidate reached a finite validation objective")
    objective, cand, score, epoch, history = best
    print(f"mvcs selected {cand} (validation objective {objective:.6g})")
    q = conformal_threshold(score.score(data.x_cal, data.r_cal), alpha)

    def gauge(device):
        net = score.model.to(device)
        return (lambda xt, rt: net.scores(xt, rt)), _t(data.x_test).to(device), q

    return Fitted("mvcs" if convex else "mvcs_free", q, score, volume=lambda x: score.volume(x, q),
                  contains=lambda x, r: score.contains(x, r, q),
                  train_seconds=time.time() - t0, n_params=score.n_params(), selected_epoch=epoch,
                  extra={"learned_p": score.p, "tuned": cand, "validation_objective": objective,
                         "tuning_candidates": candidates},
                  history=history, gauge=gauge)


def run_mvcs_multi(data, cfg, alpha, seed, convex):
    """Multi-norm MVCS with the authors' code: one global set centred at F(x) + mu."""
    t0 = time.time()
    model = MultiNormMvcs(convex).fit(data.r_score, alpha, seed)
    train_seconds = time.time() - t0
    q = conformal_threshold(model.scores(data.r_cal), alpha)
    v = model.volume(q, data.d)                       # the same for every x
    score = model.gauge(data.d)

    def gauge(device):
        return (lambda xt, rt: score(rt)), _t(data.x_test).to(device), q

    return Fitted("mvcs_multi" if convex else "mvcs_multi_free", q, model,
                  volume=lambda x: np.full(len(x), v),
                  contains=lambda x, r: model.scores(np.asarray(r, float)) <= q,
                  train_seconds=train_seconds, n_params=0,
                  extra={"exponents": model.exponents(), "centre": model.centre()},
                  gauge=gauge)


def run_pbs(data, cfg, alpha, seed):
    """PBS: every configuration of the grid is trained on D_score; the kept one has the smallest
    validation coverage error (ties: validation objective). Volume by change of variables."""
    c = cfg["pbs"]
    t0 = time.time()
    candidates, models = [], []
    for width, depth, batch in c["configurations"]:
        pc = pbs.PbsConfig(width=int(width), depth=int(depth), batch_size=int(batch), lr=float(c["learning_rate"]),
                           amort_lr=float(c["learning_rate"]), weight_decay=float(c["weight_decay"]),
                           grad_clip=float(c["grad_clip"]), amort_restart_epochs=int(c["amortizer_restart_epochs"]),
                           max_epochs=int(c["max_epochs"]), newton_tol=float(c["newton_tol"]),
                           newton_max_iter=int(c["newton_max_iter"]), n_val_u=int(c["n_val_u"]),
                           alpha_init=float(c["alpha_init"]), seed=seed)
        model, info = pbs.train(data.x_score, data.r_score, data.x_val, data.r_val, pc)
        print(f"pbs width={width} depth={depth} batch={batch}: val coverage error {info['val_coverage_error']:.4f} "
              f"val objective {info['val_loss']:+.5f}")
        candidates.append({"width": int(width), "depth": int(depth), "batch_size": int(batch),
                           "val_coverage_error": info["val_coverage_error"], "val_loss": info["val_loss"],
                           "steps": info["steps"], "alpha": info["alpha"]})
        models.append(model)
    order = sorted(range(len(candidates)), key=lambda i: (candidates[i]["val_coverage_error"], candidates[i]["val_loss"]))
    model, chosen = models[order[0]], candidates[order[0]]
    train_seconds = time.time() - t0
    q = conformal_threshold(pbs.scores(model, data.x_cal, data.r_cal), alpha)
    budgets = [int(b) for b in c["volume_budgets"]]

    def volume(x):
        lv = pbs.log_volume(model, x, q, budgets)
        volume.doubling = {b: float(np.mean(np.exp(lv[b].numpy()))) for b in budgets}
        return np.exp(lv[max(budgets)].numpy())

    return Fitted("pbs", q, model, volume=volume, contains=lambda x, r: pbs.scores(model, x, r) <= q,
                  train_seconds=train_seconds, n_params=sum(p.numel() for p in model.phi.parameters()),
                  extra={"selected": chosen, "tuning_candidates": candidates})


# ------------------------------------------------------------------------------------------
# learned scores
# ------------------------------------------------------------------------------------------
def train_settings(cfg, method, seed, n_train, learning_rate, batch_size):
    s = cfg["score_learning"]
    return TrainSettings(learning_rate=float(learning_rate), batch_size=min(int(batch_size), max(n_train, 1)),
                         max_epochs=int(cfg[method]["max_epochs"]), patience=int(s["patience"]),
                         grad_clip_norm=float(s["grad_clip_norm"]), coverage_tolerance=float(s["coverage_tolerance"]),
                         checkpoint_every=int(s["checkpoint_every"]), min_selection_epoch=int(s["min_selection_epoch"]),
                         tau=float(s["tau"]), kappa_cov=float(s["kappa_cov"]), seed=seed)


def build_icnn(data, cfg, encoder_hidden_dims, encoder_input="xf"):
    c = cfg["icnn"]
    uses_f = encoder_input in ("xf", "f", "id_xf", "id_f")
    return ICNN(data.input_dim, data.d, point_predictor=data.point_model if uses_f else None,
                encoder_input=encoder_input, encoder_hidden_dims=list(encoder_hidden_dims),
                hidden_dim=c["hidden_dim"], depth=c["depth"], epsilon=c["strong_convexity"],
                train_area_directions=c["train_area_directions"], eval_area_directions=c["eval_area_directions"],
                train_volume_directions=c["train_volume_directions"], eval_volume_directions=c["eval_volume_directions"],
                whiten_momentum=c["whiten"]["momentum"], whiten_shrink=c["whiten"]["shrink"],
                whiten_warmup_steps=c["whiten"]["warmup_steps"],
                root_finding=c["root_finding"], set_boundary=c["set_boundary"])


def build_spline(data, cfg, encoder_hidden_dims, num_knots):
    knots = [compute_knots(data.r_score[:, j], num_knots) for j in range(data.d)]
    return SplineScore(data.input_dim, data.d, data.point_model, [k[0] for k in knots], [k[1] for k in knots],
                       encoder_hidden_dims=list(encoder_hidden_dims), minimum_slope=cfg["spline"]["minimum_slope"])


def fit_learned_score(build, data, cfg, settings, device):
    """The gamma-grid search over the two loss weights; returns (winner, q_hat, extra)."""
    s = cfg["score_learning"]
    f32 = np.float32
    best, candidates = search_lambdas(build, data.x_score.astype(f32), data.r_score.astype(f32),
                                      data.x_val.astype(f32), data.r_val.astype(f32), float(cfg["alpha"]), settings,
                                      [float(g) for g in s["gamma_miss"]], [float(g) for g in s["gamma_cov"]], device)
    res = best["result"]
    model = res.model.eval()
    with torch.no_grad():
        q = conformal_threshold(model.score(_t(data.x_cal), _t(data.r_cal)).numpy().reshape(-1), float(cfg["alpha"]))
    extra = {k: best[k] for k in ("gamma_miss", "gamma_cov", "lambda_miss", "lambda_cov",
                                  "lambda_miss_ref", "lambda_cov_ref", "selection_rule")}
    extra.update({"validation_coverage": res.val_coverage, "validation_size": res.val_size,
                  "checkpoint_rule": res.selection_rule,
                  "lambda_candidates": [{k: v for k, v in c.items() if k != "result"} for c in candidates]})
    return res, q, extra


def run_spline(data, cfg, alpha, seed, tuned):
    settings = train_settings(cfg, "spline", seed, len(data.x_score), tuned["learning_rate"], tuned["batch_size"])
    t0 = time.time()
    res, q, extra = fit_learned_score(lambda: build_spline(data, cfg, tuned["encoder_hidden_dims"], tuned["num_knots"]),
                                      data, cfg, settings, "cpu")
    model = res.model

    def volume(x):
        with torch.no_grad():
            return torch.exp(model.log_volume(_t(x), q)).numpy().reshape(-1)

    def contains(x, r):
        with torch.no_grad():
            return model.score(_t(x), _t(r)).numpy().reshape(-1) <= q

    def gauge(device):
        model.to(device)
        return (lambda xt, rt: model.score(xt, rt)), _t(data.x_test).to(device), q

    return Fitted("spline", q, model, volume, contains, time.time() - t0,
                  sum(p.numel() for p in model.parameters()), res.selected_epoch,
                  extra={"tuned": tuned, **extra}, history=res.history, gauge=gauge)


class IcnnTestVolume:
    """Test-set volumes of an ICNN: per-point seeded whitening, budgets M and 2M from one pass.
    V(2M) is reported together with the doubling check |V(2M)^(1/d) / V(M)^(1/d) - 1|."""

    MAX_RAYS_PER_CHUNK = 2_000_000

    def __init__(self, model, q, cfg, device):
        self.model, self.q, self.device = model, q, device
        self.rounds = int(cfg["rounds"])
        self.refine_dirs = int(cfg["refine_directions"])
        budgets = cfg["budget_by_dim"]
        self.m = int(budgets.get(model.residual_dim, budgets["default"]))
        self.tolerance = float(cfg["convergence_tolerance"])
        self.certificate = {}

    def __call__(self, x):
        x_t = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        self.model.to(self.device)
        try:
            m1, m2 = self.m, 2 * self.m
            per = max(1, self.MAX_RAYS_PER_CHUNK // m2)
            v1, v2 = [], []
            with torch.no_grad():
                for i in range(0, len(x_t), per):
                    xb = x_t[i:i + per]
                    lv = log_volume_converged(self.model, xb, self.q, [m1, m2], point_seeds=list(range(i, i + len(xb))),
                                              rounds=self.rounds, refine_dirs=self.refine_dirs)
                    v1.append(torch.exp(lv[m1]).cpu().numpy().reshape(-1))
                    v2.append(torch.exp(lv[m2]).cpu().numpy().reshape(-1))
        finally:
            self.model.to("cpu")
        v1, v2 = np.concatenate(v1), np.concatenate(v2)
        ok = np.isfinite(v1) & np.isfinite(v2) & (v1 > 0) & (v2 > 0)
        d = self.model.residual_dim
        change = float((np.mean(v2[ok]) / np.mean(v1[ok])) ** (1.0 / d) - 1.0) if ok.any() else float("nan")
        self.certificate = {"test_budget_M": m1, "test_budget_reported_M": m2,
                            "test_budget_root_change_pct": 100.0 * change,
                            "test_budget_converged": bool(abs(change) < self.tolerance),
                            "test_volume_device": torch.cuda.get_device_name(0) if self.device == "cuda" else "cpu"}
        return v2


def run_icnn(data, cfg, alpha, seed, tuned, device, encoder_input="xf"):
    settings = train_settings(cfg, "icnn", seed, len(data.x_score), tuned["learning_rate"], tuned["batch_size"])
    t0 = time.time()
    res, q, extra = fit_learned_score(lambda: build_icnn(data, cfg, tuned["encoder_hidden_dims"], encoder_input),
                                      data, cfg, settings, device)
    model = res.model

    def contains(x, r):
        with torch.no_grad():
            return model.score(_t(x), _t(r)).numpy().reshape(-1) <= q

    return Fitted("icnn", q, model, IcnnTestVolume(model, q, cfg["icnn"]["test_volume"], device), contains,
                  time.time() - t0, sum(p.numel() for p in model.parameters()), res.selected_epoch,
                  extra={"tuned": tuned, "encoder_input": encoder_input, **extra}, history=res.history)


# ------------------------------------------------------------------------------------------
# dispatch and evaluation
# ------------------------------------------------------------------------------------------
def run_method(method, data, cfg, tuned=None, device="cpu", encoder_input="xf"):
    alpha = float(cfg["alpha"])
    set_seed(data.seed)
    if method == "absolute_residual":
        return run_absolute_residual(data, alpha)
    if method == "locally_weighted":
        return run_locally_weighted(data, cfg, alpha)
    if method == "empirical_copula":
        return run_empirical_copula(data, cfg, alpha)
    if method in ("global_ellipsoid", "local_ellipsoid"):
        return run_ellipsoid(method, data, cfg, alpha)
    if method in ("mvcs", "mvcs_free"):
        return run_mvcs(data, cfg, alpha, data.seed, convex=(method == "mvcs"))
    if method in ("mvcs_multi", "mvcs_multi_free"):
        return run_mvcs_multi(data, cfg, alpha, data.seed, convex=(method == "mvcs_multi"))
    if method == "pbs":
        return run_pbs(data, cfg, alpha, data.seed)
    if method == "spline":
        return run_spline(data, cfg, alpha, data.seed, tuned)
    if method == "icnn":
        return run_icnn(data, cfg, alpha, data.seed, tuned, device, encoder_input)
    raise ValueError(f"unknown method {method!r}")


def evaluate(fit, data, alpha):
    """Joint test coverage and mean test volume on the original target scale. A cell with a
    non-finite or non-positive volume is marked invalid."""
    covered = fit.contains(data.x_test, data.r_test)
    v_std = np.asarray(fit.volume(data.x_test), dtype=float)
    invalid = int((~np.isfinite(v_std) | (v_std <= 0)).sum())
    v = v_std * np.exp(float(np.sum(np.log(data.target_std))))
    d = data.d
    row = {"dataset": data.dataset, "seed": data.seed, "method": fit.method, "d": d, "alpha": alpha,
           "coverage": float(np.mean(covered)),
           "volume": float(np.mean(v)) if invalid == 0 else float("nan"),
           "volume_standardized": float(np.mean(v_std)) if invalid == 0 else float("nan"),
           "size": float(np.mean(v) ** (1.0 / d)) if invalid == 0 else float("nan"),
           "q_hat": fit.q_hat, "n_test": int(len(v_std)), "n_invalid_volumes": invalid,
           "status": "success" if invalid == 0 else "invalid",
           "train_seconds": fit.train_seconds, "n_params": fit.n_params, "selected_epoch": fit.selected_epoch,
           "n_pred": data.n_pred, "n_score": data.n_score, "n_cal": data.n_cal, "input_dim": data.input_dim,
           "max_rows": data.max_rows, "row_fingerprint": data.row_fingerprint, "residual_sha": data.residual_sha,
           "point_predictor_file": data.point_predictor_file}
    row.update(getattr(fit.volume, "certificate", {}))     # the ICNN doubling check
    if hasattr(fit.volume, "doubling"):                    # PBS: mean volume at each budget
        row["volume_by_budget_standardized"] = fit.volume.doubling
    row.update({k: v for k, v in fit.extra.items() if k not in ("lambda_candidates", "tuning_candidates")})
    return row


def estimator_volume(fit, data, cfg, device):
    """The region's volume measured with the shared Monte Carlo estimator instead of its closed
    form. Budgets start at 2^14 and 2^15 and double until the rule in pick_budget is met."""
    e = cfg["volume_estimator"]
    score, rows, tau = fit.gauge(device)
    gauge = Gauge(score, data.d, cfg["icnn"]["set_boundary"])
    reported, largest = int(e["reported_budget"]), int(e["max_budget"])
    top = reported
    while True:
        budgets = [b for b in (2 ** k for k in range(14, 40)) if b <= top]
        with torch.no_grad():
            lv = estimate_log_volumes(gauge, rows, tau, budgets, rounds=int(e["rounds"]),
                                      refine_dirs=int(e["refine_directions"]))
        mean_v = {b: float(np.mean(np.exp(lv[b]))) for b in budgets}
        budget, converged = pick_budget(mean_v, data.d, reported, float(e["tolerance_pct"]))
        if converged or top >= largest:
            break
        top *= 4 if top == reported else 2
    jac = float(np.exp(np.sum(np.log(data.target_std))))
    return {"volume": mean_v[budget] * jac, "size": (mean_v[budget] * jac) ** (1.0 / data.d),
            "volume_budget": budget, "volume_converged": converged,
            "volume_by_budget_standardized": mean_v}
