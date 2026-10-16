"""Synthetic shapes with a known optimal 90% region (ellipse, rectangle, triangle, trapezoid).

    python synthetic.py --shape triangle --method icnn            # seeds 42-46
    python synthetic.py --shape triangle --method spline --seeds 42
    python synthetic.py --shape triangle --method spline_l1 --seeds 42   # reuses the spline's tuning

Per (shape, seed): generate the data (coconet/data/synthetic.py), train and calibrate the method
with the settings of the `synthetic` block in configs/experiment.yaml, and score it on the test
split: coverage, V/V* (mean region area over the area of the true region) and IoU with the true
region (on a grid, averaged over 25 inputs). Writes <output-dir>/<shape>/<method>/seed<S>/row.json
and region.npz (the score at x = (0, 0) on the grid and its threshold, for plot_synthetic.py).
"""

import argparse
import copy
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from coconet import pipeline  # noqa: E402
from coconet.data.synthetic import SHAPES, ExactMean, Shape, polygon_area  # noqa: E402
from coconet.tuning import tune_spline  # noqa: E402
from coconet.utils import load_yaml, read_json, residual_sha, set_seed, write_json  # noqa: E402

METHODS = ("absolute_residual", "empirical_copula", "global_ellipsoid", "local_ellipsoid", "mvcs", "spline", "spline_l1",
           "icnn")


def make_data(shape, seed, s):
    """The four splits, drawn as one stream and sliced (train, validation, calibration, test)."""
    set_seed(seed)
    sizes = (s["n_train"], s["n_val"], s["n_cal"], s["n_test"])
    n = sum(sizes)
    x, y, r = shape.generate(n, np.random.default_rng(20260821 + seed))
    b = np.cumsum((0,) + sizes)
    part = lambda a, i: np.ascontiguousarray(a[b[i]:b[i + 1]], dtype=np.float32)  # noqa: E731
    return pipeline.Data(
        dataset=shape.name, seed=seed, d=2, input_dim=2,
        x_score=part(x, 0), r_score=part(r, 0), x_val=part(x, 1), r_val=part(r, 1),
        x_cal=part(x, 2), r_cal=part(r, 2), x_test=part(x, 3), r_test=part(r, 3),
        target_std=np.ones(2), point_model=ExactMean(), n_pred=0, n_score=sizes[0], n_cal=sizes[2],
        n_test=sizes[3], row_fingerprint=f"synthetic_{shape.name}_n{n}_seed{seed}",
        residual_sha=residual_sha(part(r, 3)), max_rows=None)


def grid_score(method, fit, x, R):
    """The method's own conformity score at input x for every residual row of R, and its threshold."""
    X = np.repeat(np.asarray(x, dtype=np.float32).reshape(1, -1), len(R), axis=0)
    if method in ("icnn", "spline", "spline_l1"):
        with torch.no_grad():
            s = fit.model.score(torch.as_tensor(X), torch.as_tensor(R)).numpy()
    elif method in ("mvcs", "global_ellipsoid", "local_ellipsoid"):
        s = fit.model.score(X, R)
    elif method == "absolute_residual":
        s = np.max(np.abs(R), axis=1)
    else:                                                      # empirical copula: box score, threshold 1
        return np.max(np.abs(R) / fit.model.half_widths(X[:1])[0][None, :], axis=1), 1.0
    return np.asarray(s).reshape(-1), fit.q_hat


def iou_with_truth(method, fit, shape, s):
    """Intersection over union of the method's region and the true region on a grid, averaged over
    the representative input (0, 0) and 24 random inputs. Also returns the grid axis and the score
    and threshold at (0, 0), which plot_synthetic.py draws."""
    from matplotlib.path import Path as MplPath
    truth = shape.optimal_region()
    ext = 1.25 * np.abs(shape.outer_region()).max()
    g = np.linspace(-ext, ext, int(s["iou_grid"]))
    G1, G2 = np.meshgrid(g, g)
    R = np.stack([G1.ravel(), G2.ravel()], 1).astype(np.float32)
    in_truth = MplPath(truth).contains_points(np.stack([G1.ravel(), G2.ravel()], 1))
    xs = np.vstack([np.zeros(2), np.random.default_rng(0).uniform(-1, 1, size=(int(s["iou_inputs"]) - 1, 2))])
    ious, at_origin = [], None
    for x in xs:
        score, q = grid_score(method, fit, x, R)
        inside = score <= q
        ious.append(float((inside & in_truth).sum() / max((inside | in_truth).sum(), 1)))
        if at_origin is None:
            at_origin = (score.reshape(G1.shape), q)
    return float(np.mean(ious)), g, at_origin


def run(args, cfg):
    s = cfg["synthetic"]
    cfg = copy.deepcopy(cfg)
    cfg["score_learning"]["gamma_cov"] = s["gamma_cov"]
    shape = Shape(args.shape)
    v_star = polygon_area(shape.optimal_region())
    device = "cuda" if args.method == "icnn" and torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4 if args.method == "icnn" else int(s["threads"]))
    for seed in args.seeds or s["seeds"]:
        out = Path(args.output_dir) / args.shape / args.method / f"seed{seed}"
        out.mkdir(parents=True, exist_ok=True)
        print(f"=== {args.shape} / {args.method} / seed {seed} ===")
        data = make_data(shape, seed, s)
        tuned = None
        if args.method == "icnn":
            tuned = dict(s["icnn"])                         # fixed, not tuned
        elif args.method in ("spline", "spline_l1"):                # the l_1 spline reuses the spline's tuning
            path = Path(args.output_dir) / "tuning" / "spline" / f"{args.shape}__seed{seed}.json"
            if not path.exists():
                sel = tune_spline(data, cfg, seed)
                path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(sel.pop("candidates")).to_csv(path.with_name(path.stem + "__candidates.csv"), index=False)
                write_json(path, sel)
            tuned = read_json(path)
            if tuned["residual_sha"] != data.residual_sha:
                sys.exit(f"{path} was tuned on other data; use another --output-dir")
        fit = pipeline.run_method(args.method, data, cfg, tuned, device)
        row = pipeline.evaluate(fit, data, float(cfg["alpha"]))
        iou, g, (score, q) = iou_with_truth(args.method, fit, shape, s)
        row.update({"shape": args.shape, "v_star": v_star, "v_over_v_star": row["volume"] / v_star, "iou": iou})
        np.savez_compressed(out / "region.npz", grid=g.astype(np.float32), score=score.astype(np.float32),
                            threshold=np.float32(q), residuals=data.r_test)
        write_json(out / "row.json", row)                    # last: a run with row.json is complete
        print(f"RESULT {args.shape} {args.method} seed={seed} coverage={row['coverage']:.4f} "
              f"V/V*={row['v_over_v_star']:.4f} IoU={row['iou']:.4f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shape", required=True, choices=sorted(SHAPES))
    p.add_argument("--method", required=True, choices=METHODS)
    p.add_argument("--seeds", nargs="+", type=int, default=None, help="default: the config's synthetic seeds")
    p.add_argument("--config", default=str(REPO / "configs" / "experiment.yaml"))
    p.add_argument("--output-dir", default=str(REPO / "results_synthetic"))
    args = p.parse_args()
    run(args, load_yaml(args.config))
    return 0


if __name__ == "__main__":
    sys.exit(main())
