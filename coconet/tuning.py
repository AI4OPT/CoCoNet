"""Per-(dataset, seed) tuning of the optimizer settings of the two learned scores.

Both tuners train every candidate at the neutral loss weights (gamma_miss, gamma_cov) = (1, 1)
times the reference scales probed on that candidate's freshly initialised model, exactly as the
gamma search trains its (1, 1) candidate, and score it by real split-conformal coverage and
size on the validation split. Per stage the winner is the coverage-admissible candidate
(|coverage - 0.9| <= tolerance) with the smallest validation size; when no candidate is
admissible, the smallest undercoverage deficit, then the simpler candidate. Any failed
training stops the tuner. A configuration already trained by an earlier stage is reused.

    ICNN    learning rate -> batch size -> encoder width -> strong convexity   (paper: encoder only)
    spline  number of knots -> learning rate -> batch size -> encoder width
"""

from __future__ import annotations

import math
import time
from dataclasses import replace

import numpy as np

from coconet.pipeline import PreparedData, build_icnn, build_spline, score_hyperparams, score_train_config
from coconet.training.lambda_search import _probe_reference_scales
from coconet.training.train_score import ScoreTrainingError, train_one_config
from coconet.utils import cuda_device_name, get_logger

logger = get_logger(__name__)

NEUTRAL_GAMMA = (1.0, 1.0)


def train_at_neutral_gamma(factory, data: PreparedData, alpha: float, hp, train_cfg, device: str):
    """Reference probe on the initialised model, then one training at lambda = 1 x reference."""
    ref = _probe_reference_scales(factory, data.x_score.astype(np.float32), data.r_score.astype(np.float32),
                                  alpha, train_cfg, device)
    lm_ref, lc_ref = float(ref.lambda_ref), float(ref.lambda_cov_ref)
    if not (math.isfinite(lm_ref) and math.isfinite(lc_ref)):
        raise ScoreTrainingError(f"non-finite reference scales (lambda_miss_ref={lm_ref}, lambda_cov_ref={lc_ref})")
    lam_miss, lam_cov = NEUTRAL_GAMMA[0] * lm_ref, NEUTRAL_GAMMA[1] * lc_ref
    res = train_one_config(factory, data.x_score.astype(np.float32), data.r_score.astype(np.float32),
                           data.x_validation.astype(np.float32), data.r_validation.astype(np.float32),
                           alpha, replace(hp, lambda_cov=lam_cov),
                           replace(train_cfg, lambda_miss=lam_miss, lambda_miss_gamma=NEUTRAL_GAMMA[0]), device)
    return res, {"lambda_miss_ref": lm_ref, "lambda_cov_ref": lc_ref, "lambda_miss": lam_miss, "lambda_cov": lam_cov}


def encoder_complexity_key(encoder) -> tuple:
    dims = tuple(int(v) for v in encoder)
    return (sum(dims), len(dims), dims)


def choose(cands: list[dict], target: float, simplicity=None) -> tuple[dict, str]:
    """Coverage-admissible min size. Without an admissible candidate: with `simplicity`, the
    smallest undercoverage deficit then the simpler candidate (encoder stages); without it, the
    smallest coverage gap then the smaller size (the other ICNN stages)."""
    adm = [r for r in cands if r["coverage_admissible"]]
    if adm:
        return min(adm, key=lambda r: r["val_volume"]), "coverage_admissible_min_calibrated_validation_volume"
    if simplicity is None:
        return min(cands, key=lambda r: (round(abs(r["val_coverage"] - target), 6), r["val_volume"])), "fallback_min_coverage_gap"

    def key(r):
        c = float(r["val_coverage"])
        deficit = max(0.0, target - c) if np.isfinite(c) else float("inf")
        return (round(deficit, 9),) + tuple(simplicity(r))
    return min(cands, key=key), "fallback_undercoverage_simpler"


def tune_icnn(data: PreparedData, cfg: dict, seed: int, device: str, encoder_input: str = "xf") -> dict:
    """Coordinate descent lr -> batch -> encoder -> strong convexity (single-valued axes are not
    searched). Returns the selection dict that `run_icnn` consumes."""
    alpha = float(cfg["alpha"])
    tune = cfg["tuning"]["icnn"]
    tol = float(cfg["score_learning"]["coverage_tolerance"])
    target = 1.0 - alpha
    n_train = len(data.x_score)
    default_enc = tuple(int(v) for v in cfg["icnn"]["encoder_hidden_dims"])
    default_eps = float(cfg["icnn"]["strong_convexity"])
    pat = int(tune["patience"])
    icnn_hidden = int(cfg["icnn"]["icnn_hidden_dim"])
    bs_grid = sorted({min(int(b), n_train) for b in tune["batch_size"]})
    bs0 = min(256, n_train) if min(256, n_train) in bs_grid else bs_grid[-1]
    rows, trained = [], {}

    def run(lr, bs, enc, eps, stage) -> dict:
        enc = tuple(int(v) for v in enc)
        key = (float(lr), int(bs), pat, enc, float(eps))
        if key in trained:
            row = trained[key]
            row["stages"] += ";" + stage
            logger.info("[%s] lr=%s bs=%s enc=%s eps=%s -> reused", stage, lr, bs, list(enc), eps)
            return row
        t0 = time.perf_counter()
        hp = score_hyperparams(cfg, lr, () if encoder_input.startswith("id_") else enc, icnn_hidden_dim=icnn_hidden)
        tcfg = score_train_config(cfg, seed, n_train, bs, pat)
        try:
            res, lam = train_at_neutral_gamma(lambda: build_icnn(data, cfg, hp.encoder_hidden_dims, eps, encoder_input),
                                              data, alpha, hp, tcfg, device)
        except Exception as exc:
            raise SystemExit(f"TUNING STOPPED at stage {stage!r}: lr={lr} bs={bs} enc={list(enc)} eps={eps} "
                             f"failed -> {type(exc).__name__}: {str(exc)[:300]}") from exc
        cov, vol = float(res.final_val_real_coverage), float(res.final_val_real_width)
        row = {"stage": stage, "stages": stage, "learning_rate": float(lr), "batch_size": int(bs), "patience": pat,
               "encoder_hidden_dims": list(enc), "strong_convexity": float(eps), **lam,
               "val_coverage": cov, "val_volume": vol, "coverage_admissible": bool(abs(cov - target) <= tol),
               "seconds": time.perf_counter() - t0}
        rows.append(row)
        trained[key] = row
        logger.info("[%s] lr=%s bs=%s enc=%s eps=%s -> val_cov=%.4f val_vol=%.6g (%.0fs)", stage, lr, bs,
                    list(enc), eps, cov, vol, row["seconds"])
        return row

    rules = {}
    lr_c = [run(lr, bs0, default_enc, default_eps, "lr") for lr in tune["learning_rate"]]
    w, rules["learning_rate"] = choose(lr_c, target)
    lr_best = w["learning_rate"]
    bs_c = [run(lr_best, bs, default_enc, default_eps, "batch_size") for bs in bs_grid]
    w, rules["batch_size"] = choose(bs_c, target)
    bs_best = w["batch_size"]
    enc_best, enc_grid = list(default_enc), [tuple(int(v) for v in e) for e in tune["encoder_hidden_dims"]]
    if len(enc_grid) > 1:
        enc_c = [run(lr_best, bs_best, enc, default_eps, "encoder") for enc in enc_grid]
        w, rules["encoder_hidden_dims"] = choose(enc_c, target, lambda r: encoder_complexity_key(r["encoder_hidden_dims"]))
        enc_best = list(w["encoder_hidden_dims"])
    eps_best, eps_grid = default_eps, [float(v) for v in tune["strong_convexity"]]
    if len(eps_grid) > 1:
        eps_c = [run(lr_best, bs_best, enc_best, eps, "strong_convexity") for eps in eps_grid]
        w, rules["strong_convexity"] = choose(eps_c, target)
        eps_best = float(w["strong_convexity"])
    sel_row = trained[(float(lr_best), int(bs_best), pat, tuple(enc_best), float(eps_best))]
    return {"dataset": data.dataset, "seed": seed, "method": "icnn", "encoder_input": encoder_input,
            "learning_rate": float(lr_best), "batch_size": int(bs_best), "patience": pat,
            "encoder_hidden_dims": [int(v) for v in enc_best], "strong_convexity": float(eps_best),
            "stage_rules": rules, "tuning_gamma": list(NEUTRAL_GAMMA),
            "tuning_lambda_miss_ref": sel_row["lambda_miss_ref"], "tuning_lambda_cov_ref": sel_row["lambda_cov_ref"],
            "coverage_tolerance": tol, "device": device, "device_name": cuda_device_name() if device == "cuda" else "cpu",
            "n_trainings": len(rows), "row_fingerprint": data.row_fingerprint, "residual_sha": data.residual_sha,
            "max_rows": data.max_rows, "candidates": rows}


def tune_spline(data: PreparedData, cfg: dict, seed: int) -> dict:
    """Coordinate descent knots -> lr -> batch -> encoder on the CPU."""
    alpha = float(cfg["alpha"])
    tune = cfg["tuning"]["spline"]
    tol = float(cfg["score_learning"]["coverage_tolerance"])
    target = 1.0 - alpha
    n_train = len(data.x_score)
    pat = int(tune["patience"])
    knots_grid = [int(k) for k in tune["num_knots"]]
    lr_grid = [float(v) for v in tune["learning_rate"]]
    bs_grid = sorted({min(int(b), n_train) for b in tune["batch_size"]})
    enc_grid = [tuple(int(v) for v in e) for e in tune["encoder_hidden_dims"]]
    rows, trained = [], {}

    def run(lr, bs, enc, nk, stage) -> dict:
        key = (float(lr), int(bs), tuple(int(v) for v in enc), int(nk))
        if key in trained:
            row = trained[key]
            row["stages"] += ";" + stage
            logger.info("[%s] knots=%s lr=%s bs=%s enc=%s -> reused", stage, nk, lr, bs, list(enc))
            return row
        t0 = time.perf_counter()
        hp = score_hyperparams(cfg, lr, enc, num_knots=int(nk))
        tcfg = score_train_config(cfg, seed, n_train, bs, pat)
        try:
            res, lam = train_at_neutral_gamma(lambda: build_spline(data, cfg, enc, int(nk)), data, alpha, hp, tcfg, "cpu")
        except Exception as exc:
            raise SystemExit(f"TUNING STOPPED at stage {stage!r}: knots={nk} lr={lr} bs={bs} enc={list(enc)} "
                             f"failed -> {type(exc).__name__}: {str(exc)[:300]}") from exc
        cov, vol = float(res.final_val_real_coverage), float(res.final_val_real_width)
        row = {"stage": stage, "stages": stage, "num_knots": int(nk), "learning_rate": float(lr), "batch_size": int(bs),
               "patience": pat, "encoder_hidden_dims": [int(v) for v in enc], **lam,
               "val_coverage": cov, "val_volume": vol, "coverage_admissible": bool(abs(cov - target) <= tol),
               "seconds": time.perf_counter() - t0}
        rows.append(row)
        trained[key] = row
        logger.info("[%s] knots=%s lr=%s bs=%s enc=%s -> val_cov=%.4f val_vol=%.6g (%.0fs)", stage, nk, lr, bs,
                    list(enc), cov, vol, row["seconds"])
        return row

    rules = {}
    lr0, enc0, bs0 = lr_grid[0], enc_grid[0], bs_grid[-1]
    cands = [run(lr0, bs0, enc0, nk, "num_knots") for nk in knots_grid]
    w, rules["num_knots"] = choose(cands, target, lambda r: (r["num_knots"],))
    nk_best = w["num_knots"]
    lr_best = lr0
    if len(lr_grid) > 1:
        cands = [run(lr, bs0, enc0, nk_best, "lr") for lr in lr_grid]
        w, rules["learning_rate"] = choose(cands, target, lambda r: (r["learning_rate"] != lr0, r["learning_rate"]))
        lr_best = w["learning_rate"]
    bs_best = bs0
    if len(bs_grid) > 1:
        cands = [run(lr_best, bs, enc0, nk_best, "batch_size") for bs in bs_grid]
        w, rules["batch_size"] = choose(cands, target, lambda r: (r["batch_size"] != bs0, r["batch_size"]))
        bs_best = w["batch_size"]
    enc_best = list(enc0)
    if len(enc_grid) > 1:
        cands = [run(lr_best, bs_best, enc, nk_best, "encoder") for enc in enc_grid]
        w, rules["encoder_hidden_dims"] = choose(cands, target, lambda r: encoder_complexity_key(r["encoder_hidden_dims"]))
        enc_best = list(w["encoder_hidden_dims"])
    return {"dataset": data.dataset, "seed": seed, "method": "spline", "num_knots": int(nk_best),
            "learning_rate": float(lr_best), "batch_size": int(bs_best), "patience": pat,
            "encoder_hidden_dims": [int(v) for v in enc_best], "stage_rules": rules,
            "tuning_gamma": list(NEUTRAL_GAMMA), "coverage_tolerance": tol, "device": "cpu",
            "n_trainings": len(rows), "row_fingerprint": data.row_fingerprint, "residual_sha": data.residual_sha,
            "max_rows": data.max_rows, "candidates": rows}
