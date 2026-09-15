"""Aggregate results/ into the paper's tables and compare them with the recorded paper cells.

    python make_table.py                      # results/ -> tables/*.csv + *.tex
    python make_table.py --compare            # also diff every cell against reference/paper_cells.csv

Per (dataset, method) over the seeds: size = (mean_s V_s)^(1/d) with its population SD over
V_s^(1/d) (at d = 1 the plain interval length), coverage mean and population SD. Methods are
ranked by size within a dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
METHOD_LABELS = {
    "absolute_residual": "Abs. Res.", "locally_weighted": "LW-SC", "empirical_copula": "Emp. Cop.",
    "global_ellipsoid": "G-Ellip.", "local_ellipsoid": "L-Ellip.", "mvcs": "MVCS_conv",
    "spline": "CoCoNet_Spline", "icnn": "CoCoNet_ICNN",
}
MULTIVARIATE_ORDER = ["absolute_residual", "empirical_copula", "global_ellipsoid", "local_ellipsoid",
                      "mvcs", "spline", "icnn"]
UNIVARIATE_ORDER = ["absolute_residual", "locally_weighted", "mvcs", "spline", "icnn"]


def load_rows(results_dir: Path) -> pd.DataFrame:
    rows = [json.loads(p.read_text()) for p in sorted(results_dir.glob("*/*/seed*/row.json"))]
    if not rows:
        raise SystemExit(f"no results under {results_dir}")
    df = pd.DataFrame(rows)
    return df[df["status"] == "success"].copy()


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (ds, m), g in df.groupby(["dataset", "method"]):
        d = int(g["d"].iloc[0])
        v = g["volume"].to_numpy(dtype=float)
        root = v ** (1.0 / d)
        cov = g["coverage"].to_numpy(dtype=float)
        out.append({"dataset": ds, "method": m, "d": d, "n_seeds": len(g),
                    "size": float(np.mean(v) ** (1.0 / d)), "size_sd": float(np.std(root, ddof=0)),
                    "coverage": float(cov.mean()), "coverage_sd": float(np.std(cov, ddof=0))})
    agg = pd.DataFrame(out)
    agg["rank"] = agg.groupby("dataset")["size"].rank(method="min")
    return agg


def to_latex(agg: pd.DataFrame, datasets_cfg: dict, order: list[str]) -> str:
    lines = [r"\begin{tabular}{l c " + " ".join("c" for _ in order) + "}", r"\toprule",
             "Dataset & $p$ & " + " & ".join(METHOD_LABELS[m] for m in order) + r" \\", r"\midrule"]
    for ds in [k for k in datasets_cfg if k in set(agg["dataset"])]:
        g = agg[agg["dataset"] == ds].set_index("method")
        best = g["size"].idxmin() if len(g) else None
        cells = []
        for m in order:
            if m not in g.index:
                cells.append("--")
                continue
            r = g.loc[m]
            size = f"{r['size']:.2f} \\pm {r['size_sd']:.2f}"
            if m == best:
                size = r"\mathbf{" + size + "}"
            cells.append(f"${size}$ / ${r['coverage']:.3f} \\pm {r['coverage_sd']:.3f}$")
        lines.append(f"{datasets_cfg[ds]['paper_name']} & {int(g['d'].iloc[0])} & " + " & ".join(cells) + r" \\")
    ranks = agg[agg["method"].isin(order)].groupby("method")["rank"].mean()
    lines.append(r"\midrule")
    lines.append("Mean rank & & " + " & ".join(f"{ranks.get(m, float('nan')):.2f}" for m in order) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def compare(df: pd.DataFrame, paper: pd.DataFrame) -> pd.DataFrame:
    """Per cell: this run's volume and coverage next to the paper's, with relative differences
    (an ICNN ablation arm is compared with the paper's cell of the same arm)."""
    key = ["dataset", "seed", "method", "arm"]
    df = df.copy()
    df["arm"] = df["encoder_input"].fillna("xf") if "encoder_input" in df.columns else "xf"
    merged = df[key + ["volume", "coverage"]].merge(paper[key + ["volume", "coverage"]], on=key,
                                                    suffixes=("", "_paper"), how="left")
    merged["volume_rel_diff"] = merged["volume"] / merged["volume_paper"] - 1.0
    merged["coverage_diff"] = merged["coverage"] - merged["coverage_paper"]
    return merged


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default=str(REPO / "results"))
    p.add_argument("--out", default=str(REPO / "tables"))
    p.add_argument("--datasets-config", default=str(REPO / "configs" / "datasets.yaml"))
    p.add_argument("--compare", action="store_true", help="diff against reference/paper_cells.csv")
    a = p.parse_args()
    import yaml
    datasets_cfg = yaml.safe_load(open(a.datasets_config))
    df = load_rows(Path(a.results))
    agg = aggregate(df)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out / "aggregated.csv", index=False)
    multi = agg[agg["d"] > 1]
    uni = agg[agg["d"] == 1]
    if len(multi):
        (out / "multivariate.tex").write_text(to_latex(multi, datasets_cfg, MULTIVARIATE_ORDER))
    if len(uni):
        (out / "univariate.tex").write_text(to_latex(uni, datasets_cfg, UNIVARIATE_ORDER))
    print(agg.to_string(index=False))
    if a.compare:
        paper = pd.read_csv(REPO / "reference" / "paper_cells.csv")
        cmp = compare(df, paper)
        cmp.to_csv(out / "comparison_with_paper.csv", index=False)
        print(cmp.to_string(index=False))
        print(f"max |volume rel diff| = {cmp['volume_rel_diff'].abs().max():.3g}, "
              f"max |coverage diff| = {cmp['coverage_diff'].abs().max():.3g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
