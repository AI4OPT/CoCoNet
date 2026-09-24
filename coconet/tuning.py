"""Per-(dataset, seed) tuning of the learned scores' settings, by coordinate descent over the
grids in the config. Every candidate is trained once at the neutral loss weights
(gamma_miss = gamma_cov = 1) and scored by calibrated coverage and size on the validation
split; per stage the winner is the coverage-admissible candidate with the smallest size, else
the smallest undercoverage and then the simpler setting.

    icnn     learning rate -> batch size -> encoder width
    spline   knots -> learning rate -> batch size -> encoder width
"""

import time

import numpy as np

from coconet.pipeline import build_icnn, build_spline, train_settings
from coconet.training.lambda_search import probe_reference_scales
from coconet.training.train_score import train_score


def _neutral_training(build, data, cfg, settings, device):
    f32 = np.float32
    alpha = float(cfg["alpha"])
    x, r = data.x_score.astype(f32), data.r_score.astype(f32)
    lm, lc = probe_reference_scales(build, x, r, alpha, settings, device)
    res = train_score(build, x, r, data.x_val.astype(f32), data.r_val.astype(f32), alpha, settings, lm, lc, device)
    return res.val_coverage, res.val_size


def choose(rows, target, tolerance, simpler=None):
    """Coverage-admissible candidate with the smallest size. Without one: with `simpler`, the
    smallest undercoverage and then the simpler setting; otherwise the smallest coverage gap and
    then the smaller size."""
    admissible = [r for r in rows if abs(r["val_coverage"] - target) <= tolerance]
    if admissible:
        return min(admissible, key=lambda r: r["val_size"])
    if simpler is None:
        return min(rows, key=lambda r: (round(abs(r["val_coverage"] - target), 6), r["val_size"]))

    def key(r):
        c = r["val_coverage"]
        return (round(max(0.0, target - c) if np.isfinite(c) else float("inf"), 9),) + tuple(simpler(r))
    return min(rows, key=key)


def encoder_simplicity(row):
    dims = tuple(row["encoder_hidden_dims"])
    return (sum(dims), len(dims), dims)


class Tuner:
    def __init__(self, data, cfg, seed, device):
        self.data, self.cfg, self.seed, self.device = data, cfg, seed, device
        self.target = 1.0 - float(cfg["alpha"])
        self.tolerance = float(cfg["score_learning"]["coverage_tolerance"])
        self.n_train = len(data.x_score)
        self.rows, self.done = [], {}

    def run(self, stage, build, **setting):
        key = tuple(sorted((k, str(v)) for k, v in setting.items()))
        if key in self.done:
            return self.done[key]
        t0 = time.perf_counter()
        s = train_settings(self.cfg, self.seed, self.n_train, setting["learning_rate"], setting["batch_size"])
        cov, size = _neutral_training(build, self.data, self.cfg, s, self.device)
        row = {"stage": stage, **setting, "val_coverage": cov, "val_size": size, "seconds": time.perf_counter() - t0}
        print(f"[{stage}] {setting} -> val_coverage={cov:.4f} val_size={size:.6g} ({row['seconds']:.0f}s)")
        self.rows.append(row)
        self.done[key] = row
        return row

    def batch_grid(self, grid):
        return sorted({min(int(b), self.n_train) for b in grid})


def tune_icnn(data, cfg, seed, device, encoder_input="xf"):
    g = cfg["tuning"]["icnn"]
    t = Tuner(data, cfg, seed, device)
    enc0 = list(cfg["icnn"]["encoder_hidden_dims"])
    bs_grid = t.batch_grid(g["batch_size"])
    bs0 = bs_grid[-1]

    def run(stage, lr, bs, enc):
        return t.run(stage, lambda: build_icnn(data, cfg, enc, encoder_input),
                     learning_rate=float(lr), batch_size=int(bs), encoder_hidden_dims=list(enc))

    lr = choose([run("learning_rate", lr, bs0, enc0) for lr in g["learning_rate"]], t.target, t.tolerance)["learning_rate"]
    bs = choose([run("batch_size", lr, bs, enc0) for bs in bs_grid], t.target, t.tolerance)["batch_size"]
    enc = choose([run("encoder", lr, bs, e) for e in g["encoder_hidden_dims"]], t.target, t.tolerance,
                 encoder_simplicity)["encoder_hidden_dims"]
    return {"dataset": data.dataset, "seed": seed, "method": "icnn", "encoder_input": encoder_input,
            "learning_rate": lr, "batch_size": bs, "encoder_hidden_dims": enc,
            "row_fingerprint": data.row_fingerprint, "residual_sha": data.residual_sha, "candidates": t.rows}


def tune_spline(data, cfg, seed):
    g = cfg["tuning"]["spline"]
    t = Tuner(data, cfg, seed, "cpu")
    lr0, enc0 = float(g["learning_rate"][0]), list(g["encoder_hidden_dims"][0])
    bs_grid = t.batch_grid(g["batch_size"])
    bs0 = bs_grid[-1]

    def run(stage, knots, lr, bs, enc):
        return t.run(stage, lambda: build_spline(data, cfg, enc, knots),
                     num_knots=int(knots), learning_rate=float(lr), batch_size=int(bs), encoder_hidden_dims=list(enc))

    knots = choose([run("num_knots", k, lr0, bs0, enc0) for k in g["num_knots"]], t.target, t.tolerance,
                   lambda r: (r["num_knots"],))["num_knots"]
    lr = choose([run("learning_rate", knots, lr, bs0, enc0) for lr in g["learning_rate"]], t.target, t.tolerance,
                lambda r: (r["learning_rate"] != lr0, r["learning_rate"]))["learning_rate"]
    bs = choose([run("batch_size", knots, lr, bs, enc0) for bs in bs_grid], t.target, t.tolerance,
                lambda r: (r["batch_size"] != bs0, r["batch_size"]))["batch_size"]
    enc = choose([run("encoder", knots, lr, bs, e) for e in g["encoder_hidden_dims"]], t.target, t.tolerance,
                 encoder_simplicity)["encoder_hidden_dims"]
    return {"dataset": data.dataset, "seed": seed, "method": "spline", "num_knots": knots,
            "learning_rate": lr, "batch_size": bs, "encoder_hidden_dims": enc,
            "row_fingerprint": data.row_fingerprint, "residual_sha": data.residual_sha, "candidates": t.rows}
