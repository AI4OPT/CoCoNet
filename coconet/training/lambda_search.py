"""Grid search over the two loss weights of a learned score.

    lambda_miss = gamma_miss * lambda_miss_ref,    lambda_cov = gamma_cov * lambda_cov_ref

with the reference scales probed once on the initialised model. Every pair of the two gamma
grids is trained (never gamma_miss = gamma_cov = 0). The winner is the candidate whose
calibrated validation coverage is within the tolerance of 1 - alpha with the smallest
validation size; if none qualifies, the smallest undercoverage, then the pair closest to (1, 1)
in log-gamma distance.
"""

import gc
import math

import torch

from coconet.training.loss import reference_scales
from coconet.training.train_score import TrainingFailed, train_score


def _fallback_key(c, target):
    deficit = max(0.0, target - c["val_coverage"]) if math.isfinite(c["val_coverage"]) else math.inf
    g_miss, g_cov = c["gamma_miss"], c["gamma_cov"]
    distance = abs(math.log(g_miss)) + abs(math.log(g_cov)) if g_miss > 0 and g_cov > 0 else math.inf
    return (round(deficit, 9), round(distance, 12), round(g_miss, 12), round(g_cov, 12))


def select_best(candidates, alpha, tolerance):
    usable = [c for c in candidates if not c["failed"]]
    if not usable:
        raise TrainingFailed("every lambda candidate failed to train")
    target = 1.0 - alpha
    admissible = [c for c in usable if abs(c["val_coverage"] - target) <= tolerance]
    if admissible:
        return min(admissible, key=lambda c: c["val_size"]), "coverage_admissible_min_size"
    return min(usable, key=lambda c: _fallback_key(c, target)), "fallback_undercoverage_gamma_distance"


def probe_reference_scales(build_model, X, r, alpha, settings, device):
    torch.manual_seed(settings.seed)
    model = build_model().to(device)
    lm_ref, lc_ref = reference_scales(model, X, r, alpha, settings.tau, settings.kappa_cov, settings.batch_size, settings.seed, device)
    if not (math.isfinite(lm_ref) and math.isfinite(lc_ref)):
        raise TrainingFailed(f"non-finite reference scales ({lm_ref}, {lc_ref})")
    return lm_ref, lc_ref


def search_lambdas(build_model, X, r, X_val, r_val, alpha, settings, gamma_miss_grid, gamma_cov_grid, device="cpu"):
    """Returns (winning candidate dict with its TrainedScore under "result", all candidate dicts)."""
    lm_ref, lc_ref = probe_reference_scales(build_model, X, r, alpha, settings, device)
    print(f"reference scales: lambda_miss_ref={lm_ref:.6g} lambda_cov_ref={lc_ref:.6g}")
    candidates = []
    for g_miss in gamma_miss_grid:
        for g_cov in gamma_cov_grid:
            if g_miss == 0.0 and g_cov == 0.0:
                continue
            cand = {"gamma_miss": float(g_miss), "gamma_cov": float(g_cov),
                    "lambda_miss": float(g_miss) * lm_ref, "lambda_cov": float(g_cov) * lc_ref,
                    "lambda_miss_ref": lm_ref, "lambda_cov_ref": lc_ref,
                    "val_coverage": float("nan"), "val_size": float("inf"),
                    "failed": False, "error": "", "result": None}
            try:
                res = train_score(build_model, X, r, X_val, r_val, alpha, settings, cand["lambda_miss"], cand["lambda_cov"], device)
                cand.update(val_coverage=res.val_coverage, val_size=res.val_size, result=res)
            except TrainingFailed as exc:
                cand.update(failed=True, error=str(exc)[:300])
            candidates.append(cand)
            print(f"gamma_miss={g_miss:<6g} gamma_cov={g_cov:<6g} -> val_coverage={cand['val_coverage']:.4f} "
                  f"val_size={cand['val_size']:.5g}{'  FAILED' if cand['failed'] else ''}")
            # keep only the current best model in memory
            if any(not c["failed"] for c in candidates):
                best, _ = select_best(candidates, alpha, settings.coverage_tolerance)
                for c in candidates:
                    if c is not best and c["result"] is not None:
                        c["result"].model = None
                gc.collect()
    best, rule = select_best(candidates, alpha, settings.coverage_tolerance)
    best["selection_rule"] = rule
    return best, candidates
