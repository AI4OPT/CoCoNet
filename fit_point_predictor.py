"""Fit the frozen MLP point predictor of a dataset for every seed and save it (optional: run.py fits
it on the fly; saved predictors can be passed to run.py with --point-predictor-dir).

    python fit_point_predictor.py --dataset enb --out my_predictors

Runs on the CPU with 4 threads (the thread count changes the last bits of the weights; the
reported numbers used predictors fitted this way). Writes <out>/<dataset>__seed<S>.pt and a .json with the
split fingerprint and residual hash that run.py checks before using a predictor. Existing files
are kept unless --force is given.
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from coconet import pipeline  # noqa: E402
from coconet.utils import load_yaml, sha256_file, write_json  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--config", default=str(REPO / "configs" / "experiment.yaml"))
    p.add_argument("--datasets", default=str(REPO / "configs" / "datasets.yaml"))
    p.add_argument("--max-rows", type=int, default=None, help="row cap in place of the dataset's, as in run.py")
    p.add_argument("--out", required=True)
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    cfg, datasets = load_yaml(a.config), load_yaml(a.datasets)
    if a.max_rows is not None:
        datasets[a.dataset]["max_rows"] = a.max_rows
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print("torch threads:", torch.get_num_threads())
    for seed in (a.seeds if a.seeds is not None else list(cfg["seeds"])):
        pt, js = out / f"{a.dataset}__seed{seed}.pt", out / f"{a.dataset}__seed{seed}.json"
        if pt.exists() and js.exists() and not a.force:
            print(f"{a.dataset} seed {seed}: exists, skipped")
            continue
        data = pipeline.prepare_data(a.dataset, seed, cfg, datasets, point_predictor_dir=None)
        pipeline.save_point_predictor(data.point_model, pt)
        write_json(js, {"dataset": a.dataset, "seed": seed, "max_rows": data.max_rows,
                        "row_fingerprint": data.row_fingerprint, "residual_sha": data.residual_sha,
                        "pt_sha256": sha256_file(pt), "torch_threads": torch.get_num_threads(),
                        "torch_version": torch.__version__})
        print(f"{a.dataset} seed {seed}: residual_sha={data.residual_sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
