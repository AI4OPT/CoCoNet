"""Run one method on one dataset for one or more seeds, with the paper's configuration.

    python run.py --dataset enb --method icnn --config configs/paper.yaml
    python run.py --dataset enb --method spline --seeds 0 1 2 3 4

Steps per (dataset, seed): load the frozen point predictor from reference/point_predictors
(or fit it with --fit-point-predictor), load the tuned optimizer settings of the learned scores
from reference/tuning (or tune them with --tune), train and calibrate the method, evaluate on
the test split, write results/<dataset>/<method>/seed<S>/row.json.

Devices: --method icnn needs a GPU (the paper: Quadro RTX 6000) and a CPU count of 4;
everything else runs on the CPU with 4 threads. The slurm/ scripts set both up.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import pandas as pd  # noqa: E402
import torch  # noqa: E402

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from coconet import pipeline  # noqa: E402
from coconet.tuning import tune_icnn, tune_spline  # noqa: E402
from coconet.utils import get_logger, load_yaml, read_json, write_json  # noqa: E402

logger = get_logger("run")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="a key of configs/datasets.yaml")
    p.add_argument("--method", required=True, choices=pipeline.ALL_METHODS)
    p.add_argument("--config", default=str(REPO / "configs" / "paper.yaml"))
    p.add_argument("--datasets-config", default=str(REPO / "configs" / "datasets.yaml"))
    p.add_argument("--seeds", nargs="+", type=int, default=None, help="default: the config's seeds")
    p.add_argument("--point-predictor-dir", default=str(REPO / "reference" / "point_predictors"),
                   help="frozen point predictors <dataset>__seed<S>.pt (+ .json)")
    p.add_argument("--fit-point-predictor", action="store_true",
                   help="fit the point predictor here instead of loading it (CPU, 4 threads)")
    p.add_argument("--tuning-dir", default=str(REPO / "reference" / "tuning"),
                   help="tuned selections <tuning-dir>/{icnn,spline}/<dataset>__seed<S>.json")
    p.add_argument("--tune", action="store_true",
                   help="run the tuner for icnn/spline and write its selection into --tuning-dir "
                        "before training (skipped when a selection already exists)")
    p.add_argument("--encoder-input", default="xf", choices=sorted(pipeline.ICNN_ENCODER_INPUTS),
                   help="icnn only: the encoder-input ablation arm (xf = the paper's CoCoNet_ICNN)")
    p.add_argument("--output-dir", default=str(REPO / "results"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.config)
    datasets_cfg = load_yaml(args.datasets_config)
    if args.dataset not in datasets_cfg:
        raise SystemExit(f"unknown dataset {args.dataset!r}; see configs/datasets.yaml")
    output_dim = int(datasets_cfg[args.dataset]["output_dim"])
    if args.method not in pipeline.methods_for(output_dim):
        raise SystemExit(f"{args.method} does not apply at d={output_dim}; methods: {pipeline.methods_for(output_dim)}")
    seeds = args.seeds if args.seeds is not None else list(cfg["seeds"])

    device = "cpu"
    if args.method == pipeline.ICNN_METHOD:
        if not torch.cuda.is_available():
            raise SystemExit("icnn needs a GPU (the paper used a Quadro RTX 6000)")
        device = "cuda"
        logger.info("icnn device: %s", torch.cuda.get_device_name(0))
    elif torch.cuda.is_available():
        logger.warning("a GPU is visible but %s runs on the CPU; hide it (CUDA_VISIBLE_DEVICES=) to "
                       "reproduce the paper's CPU numbers exactly", args.method)
    logger.info("torch threads: %d (the paper's CPU numbers need 4)", torch.get_num_threads())

    failed = 0
    for seed in seeds:
        out = Path(args.output_dir) / args.dataset / (
            args.method if args.encoder_input == "xf" else f"{args.method}_{args.encoder_input}") / f"seed{seed}"
        out.mkdir(parents=True, exist_ok=True)
        logger.info("=== %s / %s / seed %d ===", args.dataset, args.method, seed)
        data = pipeline.prepare_data(args.dataset, seed, cfg, datasets_cfg,
                                     None if args.fit_point_predictor else Path(args.point_predictor_dir))
        logger.info("d=%d input_dim=%d n_pred=%d n_score=%d n_cal=%d n_test=%d residual_sha=%s",
                    data.output_dim, data.input_dim, data.n_pred, data.n_score, data.n_cal, data.n_test,
                    data.residual_sha)

        tuned = None
        if args.method in (pipeline.ICNN_METHOD, pipeline.SPLINE):
            sel_path = Path(args.tuning_dir) / args.method / f"{args.dataset}__seed{seed}.json"
            if not sel_path.exists() and args.tune:
                logger.info("tuning %s ...", args.method)
                sel = (tune_icnn(data, cfg, seed, device, args.encoder_input) if args.method == pipeline.ICNN_METHOD
                       else tune_spline(data, cfg, seed))
                pd.DataFrame(sel.pop("candidates")).to_csv(sel_path.with_name(sel_path.stem + "__candidates.csv"), index=False)
                write_json(sel_path, sel)
                logger.info("wrote %s", sel_path)
            if not sel_path.exists():
                raise SystemExit(f"no tuned selection at {sel_path}; pass --tune to create it")
            tuned = read_json(sel_path)
            for key in ("row_fingerprint", "residual_sha"):
                if tuned.get(key) not in (None, "", getattr(data, key)):
                    raise SystemExit(f"{sel_path} was tuned on other data ({key} differs)")
            logger.info("tuned selection: %s", {k: tuned[k] for k in ("learning_rate", "batch_size", "encoder_hidden_dims")
                                                if k in tuned} | {k: tuned[k] for k in ("num_knots", "strong_convexity") if k in tuned})

        try:
            res = pipeline.run_method(args.method, data, cfg, tuned, device, args.encoder_input)
            row = pipeline.evaluate_on_test(res, data, float(cfg["alpha"]))
        except Exception as exc:
            logger.exception("%s seed %d FAILED: %r", args.method, seed, exc)
            row = {"dataset": args.dataset, "seed": seed, "method": args.method, "d": data.output_dim,
                   "status": "failed", "error": repr(exc)[:500]}
            failed += 1
            write_json(out / "row.json", row)
            continue
        write_json(out / "row.json", row)
        if res.history:
            pd.DataFrame(res.history).to_csv(out / "training_history.csv", index=False)
        cands = res.extra.get("lambda_candidates") or res.extra.get("tuning_candidates")
        if cands:
            pd.DataFrame(cands).to_csv(out / "candidates.csv", index=False)
        if isinstance(res.model, torch.nn.Module):
            torch.save(res.model.state_dict(), out / "model_state.pt")
        elif hasattr(res.model, "model") and isinstance(getattr(res.model, "model"), torch.nn.Module):
            torch.save(res.model.model.state_dict(), out / "model_state.pt")
        logger.info("RESULT %s %s seed=%d status=%s coverage=%.4f size=%.6g volume=%.6g",
                    args.dataset, args.method, seed, row["status"], row["coverage"], row["size"], row["volume"])
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
