"""Load and merge YAML config: defaults, then a per-dataset override block."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class RunConfig:
    dataset: str
    seed: int
    # icnn | asymmetric_linear | spline | mvcs | absolute_residual | global_ellipsoid | empirical_copula
    method: str = "icnn"
    alpha: float = 0.10
    tau_mode: str = "fixed"
    tau_fixed: float = 1.0
    size_objective: str = "root_mean_volume"
    kappa_coverage: float = 10.0
    lambda_miss: float = 1.0
    lambda_cov: float = 1.0

    encoder_hidden_dims: list[int] = field(default_factory=lambda: [64, 32])
    icnn_hidden_dims: list[int] = field(default_factory=lambda: [64, 64])
    epsilon: float = 1e-3
    num_knots: int = 2  # spline method only
    ellipsoid_ridge: float = 1e-6  # global_ellipsoid method only
    mvcs_p_init: float = 2.0  # mvcs method only

    boundary_directions: int = 64  # points traced per test set, for the geometry output

    point_hidden_dims: list[int] = field(default_factory=lambda: [64, 32])
    point_learning_rate: float = 1e-3
    point_max_epochs: int = 300
    point_patience: int = 20

    learning_rate: float = 5e-4
    batch_size: int = 64
    max_epochs: int = 300
    patience: int = 80
    grad_clip_norm: float = 5.0
    checkpoint_every: int = 5
    coverage_tolerance: float = 0.02
    min_selection_epoch: int = 10

    pred_frac: float = 0.25
    score_frac: float = 0.30
    cal_frac: float = 0.25
    test_frac: float = 0.20
    pred_val_frac: float = 0.20
    score_val_frac: float = 0.20

    device: str = "cpu"


def load_config(config_path: str | Path, dataset: str, seed: int) -> RunConfig:
    """The config file's top-level keys are the base values; its `dataset_overrides.<name>`
    block (if present) overrides them for that dataset only."""
    raw = yaml.safe_load(Path(config_path).read_text()) or {}
    overrides = raw.get("dataset_overrides", {}).get(dataset, {})
    merged = {**raw, **overrides, "dataset": dataset, "seed": seed}
    known = {f.name for f in RunConfig.__dataclass_fields__.values()}
    return RunConfig(**{k: v for k, v in merged.items() if k in known})
