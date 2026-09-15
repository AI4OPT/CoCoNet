"""Search over the two loss weights of a learned score.

    lambda_miss = gamma_miss * lambda_miss_ref,    lambda_cov = gamma_cov * lambda_cov_ref

with the reference scales probed once on the initialised model (`reference_scale`). Two
layouts: "grid" trains every pair of gamma_grid_miss x gamma_grid_cov (the paper's real-data
protocol, 4 x 3 = 12 candidates) and "staged" does coordinate descent over one gamma grid (used
by the synthetic experiment). The (0, 0) corner is never trained.

Selection: among candidates whose calibrated validation coverage is within
`coverage_tolerance` of 1 - alpha, the smallest validation size. If none is admissible: the
smallest undercoverage deficit, then the pair closest to (1, 1) in log-gamma distance.
"""

from __future__ import annotations

import ctypes
import gc
import math
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable

import numpy as np
import torch

from coconet.training.prediction_set_loss import (
    PRODUCTION_GAMMA_GRID_FULL,
    PRODUCTION_GAMMA_GRID_STAGE1,
    LossConfig,
    production_boundary_expansion,
    production_stage2_gammas,
    reference_scale,
)
from coconet.training.train_score import (
    ScoreHyperparams,
    ScoreTrainConfig,
    ScoreTrainingError,
    ScoreTrainResult,
    train_one_config,
)
from coconet.utils import get_logger

logger = get_logger(__name__)

NEUTRAL_GAMMA = 1.0

GAP_FALLBACK_RULE = "coverage_gap_min_width"
UNDERCOVERAGE_FALLBACK_RULE = "undercoverage_gamma_distance"
FALLBACK_RULES = (GAP_FALLBACK_RULE, UNDERCOVERAGE_FALLBACK_RULE)
DEFAULT_FALLBACK_RULE = UNDERCOVERAGE_FALLBACK_RULE


def complexity_rank_key(hp: Any) -> tuple:
    """Simpler-is-smaller tie-break over hyperparameters."""
    encoder_dims = tuple(getattr(hp, "encoder_hidden_dims", None) or ())
    icnn_hidden_dim = getattr(hp, "icnn_hidden_dim", None) or 0
    num_knots = getattr(hp, "num_knots", None) or 0
    lambda_l2 = getattr(hp, "lambda_l2", 0.0) or 0.0
    return (len(encoder_dims), sum(encoder_dims), icnn_hidden_dim, num_knots, -lambda_l2)


def _gamma_distance(c: "LambdaCandidate") -> float:
    """|log gamma_miss| + |log gamma_cov|; infinite when a weight is zero."""
    if not (c.gamma_miss > 0 and c.gamma_cov > 0):
        return math.inf
    return abs(math.log(c.gamma_miss)) + abs(math.log(c.gamma_cov))


def _undercoverage_fallback_key(c: "LambdaCandidate", r: Any, target: float) -> tuple:
    cov = float(c.val_real_coverage)
    deficit = max(0.0, target - cov) if math.isfinite(cov) else math.inf
    hp = getattr(r, "hyperparams", None)
    return (round(deficit, 9), round(_gamma_distance(c), 12),
            complexity_rank_key(hp) if hp is not None else (),
            round(float(c.gamma_miss), 12), round(float(c.gamma_cov), 12))


@dataclass
class LambdaSearchConfig:
    stage1_gammas: tuple[float, ...] = PRODUCTION_GAMMA_GRID_STAGE1
    enable_stage2: bool = True
    enable_boundary_expansion: bool = True
    enable_joint_refinement: bool = True
    max_candidates: int = 48
    fixed_gamma_cov: float | None = None
    # Staged search only: rank candidates at this shorter budget, then retrain the winner.
    probe_max_epochs: int | None = None
    mode: str = "staged"
    grid_miss: tuple[float, ...] = ()
    grid_cov: tuple[float, ...] = ()


@dataclass
class LambdaCandidate:
    gamma_miss: float
    gamma_cov: float
    lambda_miss: float
    lambda_cov: float
    stage: str
    val_real_coverage: float
    val_real_width: float
    coverage_admissible: bool
    failed: bool = False
    error: str = ""

    def as_row(self) -> dict[str, Any]:
        return {
            "gamma_miss": self.gamma_miss, "gamma_cov": self.gamma_cov,
            "lambda_miss": self.lambda_miss, "lambda_cov": self.lambda_cov,
            "stage": self.stage, "val_real_coverage": self.val_real_coverage,
            "val_real_width": self.val_real_width,
            "coverage_admissible": self.coverage_admissible,
            "failed": self.failed, "error": self.error,
        }


@dataclass
class LambdaSearchResult:
    best: ScoreTrainResult
    gamma_miss: float
    gamma_cov: float
    lambda_miss_ref: float
    lambda_cov_ref: float
    lambda_miss: float
    lambda_cov: float
    reference_size_term: float
    reference_miss_term: float
    reference_coverage_term: float
    size_objective: str
    miss_type: str
    selection_rule: str
    boundary_hit: str
    candidates: list[LambdaCandidate] = field(default_factory=list)
    n_trained: int = 0
    n_failed: int = 0
    coordinate_winner: tuple[float, float] = (float("nan"), float("nan"))
    joint_refinement_winner: tuple[float, float] = (float("nan"), float("nan"))
    joint_refinement_changed: bool = False
    search_runtime_seconds: float = float("nan")

    def summary_row(self) -> dict[str, Any]:
        return {
            "gamma_miss": self.gamma_miss, "lambda_miss_ref": self.lambda_miss_ref,
            "lambda_miss": self.lambda_miss,
            "gamma_cov": self.gamma_cov, "lambda_cov_ref": self.lambda_cov_ref,
            "lambda_cov": self.lambda_cov,
            "lambda_reference_size_term": self.reference_size_term,
            "lambda_reference_miss_term": self.reference_miss_term,
            "lambda_reference_coverage_term": self.reference_coverage_term,
            "lambda_reference_size_objective": self.size_objective,
            "lambda_search_miss_type": self.miss_type,
            "lambda_selection_rule": self.selection_rule,
            "lambda_boundary_hit": self.boundary_hit,
            "lambda_n_candidates": self.n_trained,
            "lambda_n_failed": self.n_failed,
            "lambda_candidates_evaluated": ";".join(
                f"({c.gamma_miss:g},{c.gamma_cov:g})" for c in self.candidates
            ),
            "lambda_coordinate_winner": f"({self.coordinate_winner[0]:g},{self.coordinate_winner[1]:g})",
            "lambda_joint_refinement_winner": f"({self.joint_refinement_winner[0]:g},{self.joint_refinement_winner[1]:g})",
            "lambda_joint_refinement_changed": self.joint_refinement_changed,
            "tuning_runtime_seconds": self.search_runtime_seconds,
        }

    def candidate_rows(self, **extra) -> list[dict]:
        return [{**extra, **c.as_row(),
                 "lambda_miss_ref": self.lambda_miss_ref, "lambda_cov_ref": self.lambda_cov_ref,
                 "s_ref": self.reference_size_term, "m_ref": self.reference_miss_term,
                 "c_ref": self.reference_coverage_term,
                 "is_selected": (c.gamma_miss == self.gamma_miss
                                 and c.gamma_cov == self.gamma_cov)}
                for c in self.candidates]


def select_best_candidate(
    scored: list[tuple[LambdaCandidate, ScoreTrainResult]], alpha: float, tolerance: float,
    fallback_rule: str = DEFAULT_FALLBACK_RULE,
) -> tuple[LambdaCandidate, ScoreTrainResult, str]:
    """Coverage-admissible minimum size, else the configured fallback rule."""
    if fallback_rule not in FALLBACK_RULES:
        raise ValueError(f"unknown lambda fallback_rule {fallback_rule!r}; expected one of {FALLBACK_RULES}")
    usable = [(c, r) for c, r in scored if not c.failed]
    if not usable:
        raise ScoreTrainingError("every lambda candidate failed to train")
    non_degenerate = [(c, r) for c, r in usable
                      if not (np.isclose(c.gamma_miss, 0.0) and np.isclose(c.gamma_cov, 0.0))]
    if non_degenerate:
        usable = non_degenerate
    target = 1.0 - alpha
    admissible = [(c, r) for c, r in usable if abs(c.val_real_coverage - target) <= tolerance]
    if admissible:
        c, r = min(admissible, key=lambda cr: cr[0].val_real_width)
        return c, r, "coverage_admissible_min_width"
    if fallback_rule == GAP_FALLBACK_RULE:
        c, r = min(usable, key=lambda cr: (abs(cr[0].val_real_coverage - target), cr[0].val_real_width))
        return c, r, "fallback_min_coverage_gap"
    c, r = min(usable, key=lambda cr: _undercoverage_fallback_key(cr[0], cr[1], target))
    return c, r, "fallback_undercoverage_gamma_distance"


def _grid_neighbourhood(value: float) -> tuple[float, ...]:
    grid = PRODUCTION_GAMMA_GRID_FULL
    pos = next((i for i, g in enumerate(grid) if np.isclose(g, value)), None)
    if pos is None:
        return (float(value),)
    lo = max(pos - 1, 0)
    hi = min(pos + 1, len(grid) - 1)
    return tuple(grid[lo:hi + 1])


def _probe_reference_scales(
    model_factory: Callable[[], torch.nn.Module], X_train: np.ndarray, r_train: np.ndarray,
    alpha: float, config: ScoreTrainConfig, device: str,
):
    """One reference-scale probe on the freshly initialised model (same seed as training)."""
    torch.manual_seed(config.seed)
    model = model_factory().to(device)
    probe_cfg = LossConfig(
        miss_type=config.miss_type,
        lambda_miss=0.0,
        kappa_miss=config.kappa_miss,
        score_space_miss_form=config.score_space_miss_form,
        size_objective=config.size_objective,
        size_reference_scale=config.size_reference_scale,
        train_tau_mode=config.train_tau_mode,
        train_tau_fixed=config.train_tau_fixed,
        coverage_temperature_mode=config.coverage_temperature_mode,
        coverage_temperature_multiplier=config.coverage_temperature_multiplier,
    )
    return reference_scale(model, X_train, r_train, alpha, probe_cfg,
                           batch_size=config.batch_size, seed=config.seed, device=device)


def search_lambdas(
    model_factory_builder: Callable[[ScoreHyperparams], Callable[[], torch.nn.Module]],
    X_train: np.ndarray,
    r_train: np.ndarray,
    X_val: np.ndarray,
    r_val: np.ndarray,
    alpha: float,
    hp: ScoreHyperparams,
    config: ScoreTrainConfig,
    search: LambdaSearchConfig | None = None,
    device: str = "cpu",
) -> LambdaSearchResult:
    search = search or LambdaSearchConfig()
    if search.mode not in ("staged", "grid"):
        raise ValueError(f"unknown lambda search mode {search.mode!r}")
    if search.mode == "grid" and (not search.grid_miss or not search.grid_cov):
        raise ValueError("lambda search mode 'grid' needs non-empty grid_miss and grid_cov")
    if search.mode == "grid" and search.fixed_gamma_cov is not None:
        raise ValueError("lambda search mode 'grid' and fixed_gamma_cov are mutually exclusive")
    fallback_rule = str(getattr(config, "lambda_fallback_rule", DEFAULT_FALLBACK_RULE))
    if fallback_rule not in FALLBACK_RULES:
        raise ValueError(f"unknown lambda_fallback_rule {fallback_rule!r}; expected one of {FALLBACK_RULES}")
    logger.info("lambda search: fallback rule = %s (used only when no candidate is coverage-admissible)",
                fallback_rule)
    t_search = time.perf_counter()
    ref = _probe_reference_scales(
        lambda: model_factory_builder(hp)(), X_train, r_train, alpha, config, device)
    lambda_miss_ref = float(ref.lambda_ref)
    lambda_cov_ref = float(ref.lambda_cov_ref)
    if not (math.isfinite(lambda_miss_ref) and math.isfinite(lambda_cov_ref)):
        raise ScoreTrainingError(
            f"non-finite reference scales (lambda_miss_ref={lambda_miss_ref}, "
            f"lambda_cov_ref={lambda_cov_ref}); refusing to build a lambda grid from them")
    logger.info(
        "lambda reference scales under size_objective=%s miss_type=%s: S_ref=%.6g M_ref=%.6g "
        "C_ref=%.6g -> lambda_miss_ref=%.6g lambda_cov_ref=%.6g",
        config.size_objective, config.miss_type, ref.s_ref, ref.m_ref, ref.c_ref,
        lambda_miss_ref, lambda_cov_ref,
    )

    X_val_search, r_val_search = X_val, r_val

    scored: list[tuple[LambdaCandidate, ScoreTrainResult]] = []
    seen: set[tuple[float, float]] = set()
    target = 1.0 - alpha
    n_failed = 0

    def _train(gamma_miss: float, gamma_cov: float, stage: str) -> LambdaCandidate | None:
        nonlocal n_failed
        key = (round(float(gamma_miss), 12), round(float(gamma_cov), 12))
        if key in seen or len(scored) >= search.max_candidates:
            return None
        seen.add(key)
        if key == (0.0, 0.0):
            logger.info("[%s] skipping the joint (gamma_miss=0, gamma_cov=0) corner: disallowed", stage)
            return None
        lam_miss = float(gamma_miss) * lambda_miss_ref
        lam_cov = float(gamma_cov) * lambda_cov_ref
        cand_hp = replace(hp, lambda_cov=lam_cov)
        cand_cfg = replace(config, lambda_miss=lam_miss, lambda_miss_gamma=float(gamma_miss),
                           **({"max_epochs": int(search.probe_max_epochs)}
                              if search.probe_max_epochs else {}))
        try:
            result = train_one_config(
                model_factory_builder(cand_hp), X_train, r_train, X_val_search, r_val_search,
                alpha, cand_hp, cand_cfg, device,
            )
        except ScoreTrainingError as exc:
            n_failed += 1
            cand = LambdaCandidate(
                gamma_miss=float(gamma_miss), gamma_cov=float(gamma_cov),
                lambda_miss=lam_miss, lambda_cov=lam_cov, stage=stage,
                val_real_coverage=float("nan"), val_real_width=float("inf"),
                coverage_admissible=False, failed=True, error=str(exc)[:300],
            )
            logger.warning("lambda candidate (gm=%g, gc=%g) failed: %s", gamma_miss, gamma_cov, exc)
            scored.append((cand, None))  # type: ignore[arg-type]
            return cand
        cand = LambdaCandidate(
            gamma_miss=float(gamma_miss), gamma_cov=float(gamma_cov),
            lambda_miss=lam_miss, lambda_cov=lam_cov, stage=stage,
            val_real_coverage=float(result.final_val_real_coverage),
            val_real_width=float(result.final_val_real_width),
            coverage_admissible=bool(
                abs(result.final_val_real_coverage - target) <= config.coverage_tolerance),
        )
        logger.info(
            "[%s] gamma_miss=%-6g gamma_cov=%-6g -> val_cov=%.4f val_width=%.5f%s",
            stage, gamma_miss, gamma_cov, cand.val_real_coverage, cand.val_real_width,
            "" if cand.coverage_admissible else "  (not coverage-admissible)",
        )
        scored.append((cand, result))
        _release_losers()
        return cand

    def _release_losers() -> None:
        """Free every non-winning candidate's model; only the current best is kept live."""
        usable = [(c, r) for c, r in scored if not c.failed and r is not None]
        if not usable:
            return
        best_c, _, _ = select_best_candidate(usable, alpha, config.coverage_tolerance, fallback_rule=fallback_rule)
        for c, r in scored:
            if r is None or c is best_c:
                continue
            r.model = None
            for attr in ("train_width_history", "train_cov_history", "train_l2_history",
                          "train_total_history", "val_width_history",
                          "val_smoothed_coverage_history", "val_total_history",
                          "diagnostics_history", "checkpoint_history"):
                if getattr(r, attr, None):
                    setattr(r, attr, [])
        gc.collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass

    def _best_on(axis: str, fixed: float) -> float:
        subset = [
            (c, r) for c, r in scored
            if not c.failed
            and np.isclose(c.gamma_cov if axis == "miss" else c.gamma_miss, fixed)
        ]
        if not subset:
            raise ScoreTrainingError(f"no successful candidate on the {axis} axis")
        c, _, _ = select_best_candidate(subset, alpha, config.coverage_tolerance, fallback_rule=fallback_rule)
        return c.gamma_miss if axis == "miss" else c.gamma_cov

    cov_is_fixed = search.fixed_gamma_cov is not None
    cov_pin = float(search.fixed_gamma_cov) if cov_is_fixed else NEUTRAL_GAMMA

    if search.mode == "grid":
        logger.info("lambda search: FIXED GRID gamma_miss %s x gamma_cov %s", list(search.grid_miss), list(search.grid_cov))
        for g_miss in search.grid_miss:
            for g_cov in search.grid_cov:
                _train(g_miss, g_cov, "grid")
        best_miss = best_cov = float("nan")
        boundary_hit = "none"
    else:
        for g in search.stage1_gammas:
            _train(g, cov_pin, "stage1_miss")
        best_miss = _best_on("miss", cov_pin)

        if cov_is_fixed:
            best_cov = cov_pin
        else:
            for g in search.stage1_gammas:
                _train(best_miss, g, "stage1_cov")
            best_cov = _best_on("cov", best_miss)

        if search.enable_stage2:
            done_miss = [c.gamma_miss for c, _ in scored if np.isclose(c.gamma_cov, best_cov)]
            for g in production_stage2_gammas(best_miss, done_miss):
                _train(g, best_cov, "stage2_miss")
            best_miss = _best_on("miss", best_cov)

            if not cov_is_fixed:
                done_cov = [c.gamma_cov for c, _ in scored if np.isclose(c.gamma_miss, best_miss)]
                for g in production_stage2_gammas(best_cov, done_cov):
                    _train(best_miss, g, "stage2_cov")
                best_cov = _best_on("cov", best_miss)

        boundary_hit = "none"
        if search.enable_boundary_expansion:
            for _ in range(2):
                done_miss = [c.gamma_miss for c, _ in scored if np.isclose(c.gamma_cov, best_cov)]
                extra = production_boundary_expansion(best_miss, done_miss)
                if not extra:
                    break
                boundary_hit = "miss_lower" if extra[0] < best_miss else "miss_upper"
                for g in extra:
                    _train(g, best_cov, "expand_miss")
                new_best = _best_on("miss", best_cov)
                if np.isclose(new_best, best_miss):
                    break
                best_miss = new_best
            if not cov_is_fixed:
                for _ in range(2):
                    done_cov = [c.gamma_cov for c, _ in scored if np.isclose(c.gamma_miss, best_miss)]
                    extra = production_boundary_expansion(best_cov, done_cov)
                    if not extra:
                        break
                    boundary_hit = ("cov_lower" if extra[0] < best_cov else "cov_upper") if \
                        boundary_hit == "none" else boundary_hit + "+cov"
                    for g in extra:
                        _train(best_miss, g, "expand_cov")
                    new_best = _best_on("cov", best_miss)
                    if np.isclose(new_best, best_cov):
                        break
                    best_cov = new_best

    coordinate_winner = (best_miss, best_cov)

    joint_changed = False
    if search.enable_joint_refinement and not cov_is_fixed and search.mode != "grid":
        n_miss = _grid_neighbourhood(best_miss)
        n_cov = _grid_neighbourhood(best_cov)
        logger.info("joint refinement around (gm=%g, gc=%g): %s x %s",
                    best_miss, best_cov, list(n_miss), list(n_cov))
        for g_miss in n_miss:
            for g_cov in n_cov:
                _train(g_miss, g_cov, "joint_refine")
        cand, _, _ = select_best_candidate([(c, r) for c, r in scored], alpha,
                                           config.coverage_tolerance, fallback_rule=fallback_rule)
        joint_changed = not (np.isclose(cand.gamma_miss, best_miss)
                             and np.isclose(cand.gamma_cov, best_cov))
        if joint_changed:
            logger.info("joint refinement MOVED the winner: (%g, %g) -> (%g, %g) -- the two "
                        "weights interact here", best_miss, best_cov,
                        cand.gamma_miss, cand.gamma_cov)
        best_miss, best_cov = cand.gamma_miss, cand.gamma_cov

    best_cand, best_result, rule = select_best_candidate(
        [(c, r) for c, r in scored], alpha, config.coverage_tolerance, fallback_rule=fallback_rule)
    if best_result is None or getattr(best_result, "model", None) is None:
        raise ScoreTrainingError(
            "the selected lambda candidate has no model -- the loser-pruning in "
            "_release_losers dropped the winner. This is a bug, not a training failure.")
    if len(scored) >= search.max_candidates:
        logger.warning("lambda search hit max_candidates=%d; the reported winner is the best of "
                       "the candidates actually trained", search.max_candidates)

    if search.probe_max_epochs:
        logger.info("lambda search: probe_max_epochs=%d used for exploration -- committing "
                    "winner (gm=%g, gc=%g) to a full-budget retrain at max_epochs=%d",
                    search.probe_max_epochs, best_cand.gamma_miss, best_cand.gamma_cov,
                    config.max_epochs)
        commit_hp = replace(hp, lambda_cov=best_cand.lambda_cov)
        commit_cfg = replace(config, lambda_miss=best_cand.lambda_miss,
                             lambda_miss_gamma=best_cand.gamma_miss)
        best_result = train_one_config(
            model_factory_builder(commit_hp), X_train, r_train, X_val_search, r_val_search,
            alpha, commit_hp, commit_cfg, device)

    return LambdaSearchResult(
        best=best_result,
        gamma_miss=best_cand.gamma_miss, gamma_cov=best_cand.gamma_cov,
        lambda_miss_ref=lambda_miss_ref, lambda_cov_ref=lambda_cov_ref,
        lambda_miss=best_cand.lambda_miss, lambda_cov=best_cand.lambda_cov,
        reference_size_term=float(ref.s_ref), reference_miss_term=float(ref.m_ref),
        reference_coverage_term=float(ref.c_ref),
        size_objective=config.size_objective, miss_type=config.miss_type,
        selection_rule=rule, boundary_hit=boundary_hit,
        candidates=[c for c, _ in scored], n_trained=len(scored), n_failed=n_failed,
        coordinate_winner=coordinate_winner,
        joint_refinement_winner=(best_cand.gamma_miss, best_cand.gamma_cov),
        joint_refinement_changed=bool(joint_changed),
        search_runtime_seconds=time.perf_counter() - t_search,
    )
