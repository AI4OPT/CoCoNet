"""The synthetic-shapes figure: calibrated 90% regions of four methods against the true region.

    python plot_synthetic.py                                   # results_synthetic -> results_synthetic/synthetic_regions.pdf
    python plot_synthetic.py --seeds 42 42 42 42 --out regions.png

Reads the region.npz files written by synthetic.py, so run these first for the seeds you plot:

    python synthetic.py --shape <shape> --method {icnn,mvcs,spline,local_ellipsoid}

One panel per shape. Each method's region is the level set {score = threshold} of its own score at
x = (0, 0); gray points are the test residuals and the dashed gold outline is the true region.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from coconet.data.synthetic import Shape  # noqa: E402

SHAPES = ["ellipse", "rectangle", "triangle", "trapezoid"]
TITLES = {"ellipse": "Ellipsoid", "rectangle": "Rectangle", "triangle": "Triangle", "trapezoid": "Trapezoid"}
SEEDS = [43, 43, 43, 44]                    # one seed per shape, in the order of SHAPES
METHODS = ["icnn", "mvcs", "spline", "local_ellipsoid"]

STYLE = {
    "icnn": dict(color="#332288", linewidth=5.2, linestyle="-", zorder=5, label=r"CoCoNet$_{\mathrm{ICNN}}$"),
    "mvcs": dict(color="#661100", linewidth=5.2, linestyle="--", zorder=5, label="MVCS"),
    "spline": dict(color="#88CCEE", linewidth=5.2, linestyle="-.", zorder=5, label=r"CoCoNet$_{\mathrm{Spline}}$"),
    "local_ellipsoid": dict(color="#AA4499", linewidth=5.0, linestyle=(0, (5, 2)), zorder=5, label="L-Ellip."),
}
TRUE = dict(color="#DDCC77", linewidth=4.5, linestyle=(0, (6, 2)), zorder=2, label="Minimum set")
TRUE_FILL = dict(color="#DDCC77", alpha=0.18, zorder=0)
RESID = dict(s=22, color="#999999", alpha=0.6, linewidths=0, label="Test residuals")
FONT = 22
AX_LIM, AX_TICKS = (-2.25, 2.25), [-2, -1, 0, 1, 2]
LEFT, RIGHT, TOP, BOTTOM = 0.055, 0.995, 0.99, 0.13
FIG_H, WSPACE = 6.0, 0.04


def no_label(style):
    return {k: v for k, v in style.items() if k != "label"}


def outline(poly, n=400):
    """Closed outline with n evenly spaced points plus the corners."""
    q = np.vstack([poly, poly[:1]])
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))])
    t = np.union1d(np.linspace(0, s[-1], n), s)
    return np.column_stack([np.interp(t, s, q[:, 0]), np.interp(t, s, q[:, 1])])


def level_curves(g, score, level):
    f = plt.figure()
    cs = f.add_subplot().contour(g, g, score, levels=[level])
    segs = [s for s in cs.allsegs[0] if len(s) > 1]
    plt.close(f)
    return segs


def draw(ax, shape, seed, results):
    regions = {}
    for m in METHODS:
        f = results / shape / m / f"seed{seed}" / "region.npz"
        if not f.exists():
            sys.exit(f"missing {f}; run: python synthetic.py --shape {shape} --method {m} --seeds {seed}")
        regions[m] = np.load(f)
    resid = regions[METHODS[0]]["residuals"]
    ax.scatter(resid[:, 0], resid[:, 1], **no_label(RESID))
    true = outline(Shape(shape).optimal_region())
    ax.fill(true[:, 0], true[:, 1], **TRUE_FILL)
    ax.plot(true[:, 0], true[:, 1], **no_label(TRUE))
    for m in METHODS:
        b = regions[m]
        for seg in level_curves(b["grid"], b["score"], float(b["threshold"])):
            ax.plot(seg[:, 0], seg[:, 1], **no_label(STYLE[m]))
    ax.set(xlim=AX_LIM, ylim=AX_LIM, xticks=AX_TICKS, yticks=AX_TICKS)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$r_1$", fontsize=FONT)
    ax.tick_params(axis="both", labelsize=FONT)
    ax.text(0.025, 0.975, TITLES[shape], transform=ax.transAxes, ha="left", va="top", fontsize=FONT,
            fontweight="bold", zorder=20,
            bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="0.7", linewidth=0.7, alpha=0.95))


def plot(results, seeds=SEEDS):
    """The figure for one seed per shape (order of SHAPES), from the region.npz files under results."""
    results = Path(results)
    fig_w = len(SHAPES) * FIG_H * (TOP - BOTTOM) / (RIGHT - LEFT)
    fig, axes = plt.subplots(1, len(SHAPES), figsize=(fig_w, FIG_H), sharex=True, sharey=True, squeeze=False,
                             gridspec_kw={"wspace": WSPACE})
    for j, (shape, seed) in enumerate(zip(SHAPES, seeds)):
        ax = axes[0, j]
        draw(ax, shape, seed, results)
        if j == 0:
            ax.set_ylabel(r"$r_2$", fontsize=FONT)
        else:
            ax.tick_params(axis="y", left=False, labelleft=False)

    handles = [axes[0, 0].scatter([], [], **RESID), Line2D([], [], **TRUE)]
    handles += [Line2D([], [], **STYLE[m]) for m in METHODS]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=len(handles), frameon=False,
               fontsize=FONT)
    fig.subplots_adjust(left=LEFT, right=RIGHT, top=TOP, bottom=BOTTOM, wspace=WSPACE)
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default=str(REPO / "results_synthetic"), help="the --output-dir of synthetic.py")
    p.add_argument("--seeds", nargs=4, type=int, default=SEEDS, help="one seed per shape: ellipse rectangle triangle trapezoid")
    p.add_argument("--out", default=None, help="default: <results>/synthetic_regions.pdf")
    args = p.parse_args()
    fig = plot(args.results, args.seeds)
    out = Path(args.out) if args.out else Path(args.results) / "synthetic_regions.pdf"
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
