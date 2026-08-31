"""Cross-seed aggregation.

`root_mean_volume = (mean_seed[mean_volume])^(1/d)` -- the mean is taken FIRST, across seeds
and test points, and only then raised to 1/d. This is not the same number as
`mean(volume^(1/d))`: raising to a fractional power is concave, so by Jensen's inequality the
two orderings can disagree, and only the former is the quantity the paper reports.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AggregateResult:
    n_seeds: int
    mean_coverage: float
    mean_volume_original_scale: float
    root_mean_volume: float


def aggregate_seeds(coverages: list[float], mean_volumes: list[float], output_dim: int) -> AggregateResult:
    mean_cov = sum(coverages) / len(coverages)
    mean_vol = sum(mean_volumes) / len(mean_volumes)
    return AggregateResult(
        n_seeds=len(coverages),
        mean_coverage=mean_cov,
        mean_volume_original_scale=mean_vol,
        root_mean_volume=mean_vol ** (1.0 / output_dim),
    )
