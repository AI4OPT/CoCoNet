"""Run one method on one dataset for one or more seeds.

    python run.py --dataset enb --method icnn
    python run.py --dataset enb --method spline --seeds 0 1

Per (dataset, seed): load the saved point predictor (or fit it with --fit-point-predictor), load
the tuned settings of icnn / spline (or tune them with --tune), train and calibrate the method,
evaluate on the test split and write results/<dataset>/<method>/seed<S>/row.json.

icnn needs a GPU; every other method runs on the CPU with 4 threads. Set CUDA_VISIBLE_DEVICES=
for the CPU methods so torch does not pick up a GPU.
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from coconet import pipeline  # noqa: E402
from coconet.models.icnn import ENCODER_INPUTS  # noqa: E402
from coconet.tuning import tune_icnn, tune_spline  # noqa: E402
from coconet.utils import load_yaml, read_json, write_json  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="a key of configs/datasets.yaml")
    p.add_argument("--method", required=True, choices=pipeline.ALL_METHODS)
    p.add_argument("--seeds", nargs="+", type=int, default=None, help="default: the config's seeds")
    p.add_argument("--config", default=str(REPO / "configs" / "experiment.yaml"))
    p.add_argument("--datasets", default=str(REPO / "configs" / "datasets.yaml"))
    p.add_argument("--point-predictor-dir", default=str(REPO / "reference" / "point_predictors"))
    p.add_argument("--fit-point-predictor", action="store_true", help="fit F(x) here instead of loading it")
    p.add_argument("--tuning-dir", default=str(REPO / "reference" / "tuning"))
    p.add_argument("--tune", action="store_true",
                   help="icnn / spline: tune the settings and save them to --tuning-dir when no file exists yet")
    p.add_argument("--encoder-input", default="xf", choices=ENCODER_INPUTS,
                   help="icnn only: what the encoder sees (xf = [x, F(x)], the default model)")
    p.add_argument("--max-rows", type=int, default=None,
                   help="row cap in place of the dataset's (the encoder ablation uses 10000 with the files in reference/rows10000)")
    p.add_argument("--volume", default="closed_form", choices=("closed_form", "estimator"),
                   help="multivariate methods other than icnn and pbs: report the closed-form volume, or measure the "
                        "region with the shared Monte Carlo estimator (as the tables do)")
    p.add_argument("--output-dir", default=str(REPO / "results"))
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    datasets = load_yaml(args.datasets)
    if args.dataset not in datasets:
        sys.exit(f"unknown dataset {args.dataset!r}; see configs/datasets.yaml")
    if args.max_rows is not None:
        datasets[args.dataset]["max_rows"] = args.max_rows
    d = int(datasets[args.dataset]["output_dim"])
    if args.method not in pipeline.methods_for(d):
        sys.exit(f"{args.method} does not apply at d={d}; choose from {pipeline.methods_for(d)}")
    seeds = args.seeds if args.seeds is not None else list(cfg["seeds"])

    device = "cpu"
    if args.method == "icnn":
        if not torch.cuda.is_available():
            sys.exit("icnn needs a GPU but torch sees none")
        device = "cuda"
        print("device:", torch.cuda.get_device_name(0))
    elif torch.cuda.is_available() and args.volume != "estimator":
        print(f"warning: a GPU is visible but {args.method} runs on the CPU; hide it with CUDA_VISIBLE_DEVICES=")
    print("torch threads:", torch.get_num_threads())

    rows, failed = [], 0
    for seed in seeds:
        method_dir = args.method if args.encoder_input == "xf" else f"{args.method}_{args.encoder_input}"
        out = Path(args.output_dir) / args.dataset / method_dir / f"seed{seed}"
        out.mkdir(parents=True, exist_ok=True)
        print(f"=== {args.dataset} / {method_dir} / seed {seed} ===")
        data = pipeline.prepare_data(args.dataset, seed, cfg, datasets,
                                     None if args.fit_point_predictor else Path(args.point_predictor_dir))
        print(f"d={data.d} input_dim={data.input_dim} n_pred={data.n_pred} n_score={data.n_score} "
              f"n_cal={data.n_cal} n_test={data.n_test} residual_sha={data.residual_sha}")

        tuned = None
        if args.method in ("icnn", "spline"):
            path = Path(args.tuning_dir) / args.method / f"{args.dataset}__seed{seed}.json"
            if not path.exists() and args.tune:
                sel = (tune_icnn(data, cfg, seed, device, args.encoder_input) if args.method == "icnn"
                       else tune_spline(data, cfg, seed))
                path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(sel.pop("candidates")).to_csv(path.with_name(path.stem + "__candidates.csv"), index=False)
                write_json(path, sel)
            if not path.exists():
                sys.exit(f"no tuned settings at {path}; pass --tune to create them")
            tuned = read_json(path)
            for key in ("row_fingerprint", "residual_sha"):
                if tuned[key] != getattr(data, key):
                    sys.exit(f"{path} was tuned on other data ({key} differs)")
            print("tuned settings:", {k: tuned[k] for k in ("learning_rate", "batch_size", "encoder_hidden_dims", "num_knots")
                                       if k in tuned})

        try:
            fit = pipeline.run_method(args.method, data, cfg, tuned, device, args.encoder_input)
            row = pipeline.evaluate(fit, data, float(cfg["alpha"]))
            if args.volume == "estimator" and fit.gauge is not None and data.d > 1:
                vdev = "cuda" if torch.cuda.is_available() else "cpu"
                print("volume estimator on", torch.cuda.get_device_name(0) if vdev == "cuda" else "cpu")
                est = pipeline.estimator_volume(fit, data, cfg, vdev)
                row.update({"volume_closed_form": row["volume"], "size_closed_form": row["size"], **est})
        except Exception as exc:
            print(f"FAILED: {exc!r}")
            row = {"dataset": args.dataset, "seed": seed, "method": method_dir, "d": data.d,
                   "status": "failed", "error": repr(exc)[:500]}
            write_json(out / "row.json", row)
            failed += 1
            continue
        row["method"] = method_dir          # an ablation arm is its own method in the tables
        write_json(out / "row.json", row)
        if fit.history:
            pd.DataFrame(fit.history).to_csv(out / "training_history.csv", index=False)
        cands = fit.extra.get("lambda_candidates") or fit.extra.get("tuning_candidates")
        if cands:
            pd.DataFrame(cands).to_csv(out / "candidates.csv", index=False)
        if isinstance(fit.model, torch.nn.Module):
            torch.save(fit.model.state_dict(), out / "model_state.pt")
        elif isinstance(getattr(fit.model, "model", None), torch.nn.Module):
            torch.save(fit.model.model.state_dict(), out / "model_state.pt")
        print(f"RESULT {args.dataset} {method_dir} seed={seed} status={row['status']} "
              f"coverage={row['coverage']:.4f} size={row['size']:.6g} volume={row['volume']:.6g}")
        rows.append(row)

    rows = [r for r in rows if r["status"] == "success"]
    if len(rows) > 1:
        vol = np.array([r["volume"] for r in rows])
        cov = np.array([r["coverage"] for r in rows])
        print(f"AVERAGE over seeds {[r['seed'] for r in rows]}: size {vol.mean() ** (1.0 / d):.4f} "
              f"+- {np.std(vol ** (1.0 / d)):.4f}  coverage {cov.mean():.4f} +- {np.std(cov):.4f}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
