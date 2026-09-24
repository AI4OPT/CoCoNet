"""Aggregate results/ into tables.

    python make_table.py     # tables/aggregated.csv, multivariate.tex, extended.tex, univariate.tex

Per (dataset, method): size = (mean_s V_s)^(1/d) over the seeds with the population SD of
V_s^(1/d), and mean coverage with its SD. Within each table, methods are ranked by size per
dataset and the mean rank is taken over the datasets where every method of the table has a value.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from coconet.pipeline import MULTIVARIATE_METHODS, UNIVARIATE_METHODS  # noqa: E402

LABELS = {"absolute_residual": "Abs. Res.", "locally_weighted": "LW-SC", "empirical_copula": "Emp. Cop.",
          "global_ellipsoid": "G-Ellip.", "local_ellipsoid": "L-Ellip.", "mvcs": "MVCS",
          "mvcs_free": "MVCS$_{\\mathrm{free}\\ q}$", "mvcs_multi": "MVCS$_{\\mathrm{multi},q\\geq1}$",
          "mvcs_multi_free": "MVCS$_{\\mathrm{multi},\\mathrm{free}\\ q}$", "pbs": "PBS",
          "spline": "CoCoNet_Spline", "icnn": "CoCoNet_ICNN"}
MULTIVARIATE, UNIVARIATE = list(MULTIVARIATE_METHODS), list(UNIVARIATE_METHODS)
EXTENDED = ["mvcs", "mvcs_free", "mvcs_multi", "mvcs_multi_free", "pbs", "spline", "icnn"]
EXTENDED_LABELS = {**LABELS, "mvcs": "MVCS$_{q\\geq1}$"}


def load_rows(results_dir):
    rows = [json.loads(p.read_text()) for p in sorted(results_dir.glob("*/*/seed*/row.json"))]
    if not rows:
        raise SystemExit(f"no results under {results_dir}")
    df = pd.DataFrame(rows)
    df = df[df["status"] == "success"].copy()
    if df.empty:
        raise SystemExit("no successful runs")
    return df


def aggregate(df):
    out = []
    for (ds, m), g in df.groupby(["dataset", "method"]):
        d = int(g["d"].iloc[0])
        v = g["volume"].to_numpy(dtype=float)
        cov = g["coverage"].to_numpy(dtype=float)
        out.append({"dataset": ds, "method": m, "d": d, "n_seeds": len(g),
                    "size": float(np.mean(v) ** (1.0 / d)), "size_sd": float(np.std(v ** (1.0 / d))),
                    "coverage": float(cov.mean()), "coverage_sd": float(np.std(cov))})
    return pd.DataFrame(out)


def to_latex(agg, datasets, order, labels=LABELS):
    agg = agg[agg["method"].isin(order)]
    complete = agg.groupby("dataset")["method"].nunique() == len(order)
    ranks = (agg[agg["dataset"].isin(complete[complete].index)]
             .assign(rank=lambda t: t.groupby("dataset")["size"].rank(method="min"))
             .groupby("method")["rank"].mean())
    lines = [r"\begin{tabular}{l c " + " ".join("c" for _ in order) + "}", r"\toprule",
             "Dataset & $d$ & " + " & ".join(labels[m] for m in order) + r" \\", r"\midrule"]
    for ds in [k for k in datasets if k in set(agg["dataset"])]:
        g = agg[agg["dataset"] == ds].set_index("method")
        best = g["size"].idxmin()
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
        lines.append(f"{datasets[ds]['name']} & {int(g['d'].iloc[0])} & " + " & ".join(cells) + r" \\")
    lines += [r"\midrule", "Mean rank & & " + " & ".join(f"{ranks.get(m, float('nan')):.2f}" for m in order) + r" \\",
              r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default=str(REPO / "results"))
    p.add_argument("--out", default=str(REPO / "tables"))
    p.add_argument("--datasets", default=str(REPO / "configs" / "datasets.yaml"))
    a = p.parse_args()
    datasets = yaml.safe_load(open(a.datasets))
    agg = aggregate(load_rows(Path(a.results)))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out / "aggregated.csv", index=False)
    multi, uni = agg[agg["d"] > 1], agg[agg["d"] == 1]
    if len(multi):
        (out / "multivariate.tex").write_text(to_latex(multi, datasets, MULTIVARIATE))
        if multi["method"].isin(EXTENDED[1:5]).any():
            (out / "extended.tex").write_text(to_latex(multi, datasets, EXTENDED, EXTENDED_LABELS))
    if len(uni):
        (out / "univariate.tex").write_text(to_latex(uni, datasets, UNIVARIATE))
    print(agg.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
