# Convex Uncertainty Sets for Multivariate Regression (CoCoNet)

Code, data and fitted artifacts to reproduce the tables of the paper: 25 multivariate datasets
(main table), 6 univariate datasets and the encoder-input ablation (appendix), at 90% target
coverage over seeds 0-4.

Two learned convex conformity scores (CoCoNet_ICNN and CoCoNet_Spline) are compared with five
baselines (absolute residual, empirical copula, global and local ellipsoid, convex MVCS; LW-SC
replaces the multivariate baselines at d = 1). Every method receives the residuals of one
frozen MLP point predictor per (dataset, seed), is calibrated by split conformal prediction on
D_cal and is scored by joint coverage and (mean volume)^(1/d) on D_test.

## Layout

```
run.py                   one method on one dataset (all seeds), writes results/<dataset>/<method>/seed<S>/row.json
fit_point_predictor.py   fit and freeze the MLP point predictor of a dataset (reference/point_predictors has the paper's)
make_table.py            results/ -> tables/ (aggregated.csv, multivariate.tex, univariate.tex); --compare diffs against the paper
configs/paper.yaml       every setting of the paper's protocol
configs/datasets.yaml    the 31 datasets: loader, target columns, row cap, paper row name
coconet/pipeline.py      prepare_data (split, preprocessing, residuals) and the seven run_<method> functions
coconet/tuning.py        per-(dataset, seed) tuning of the learned scores' optimizer settings
coconet/models/          icnn.py, spline.py (the two scores), mlp.py, volume_estimator.py
coconet/training/        the size + miss + coverage loss, the gamma-grid search, training loops, checkpoint selection
coconet/conformal/       calibration, ellipsoid, empirical_copula, mvcs
coconet/data/            loaders, preprocessing, splits
data/raw/                raw dataset files (UCI/OpenML files are downloaded on first use if missing)
reference/               the paper's frozen point predictors, tuned selections and per-cell numbers (paper_cells.csv)
slurm/                   PACE job scripts
```

## Setup

Python 3.13 with `pip install -r requirements.txt` (the paper used torch 2.7.1+cu126, numpy 2.3.1,
pandas 2.3.1, scipy 1.16.0, scikit-learn 1.7.0).

## Running

One method on one dataset, all five seeds, with the paper's configuration:

```
python run.py --dataset enb --method icnn --config configs/paper.yaml          # GPU
python run.py --dataset enb --method spline --config configs/paper.yaml        # CPU
python run.py --dataset enb --method mvcs --seeds 0 1                          # CPU, two seeds
python run.py --dataset concrete --method locally_weighted                     # univariate baseline
```

Methods: `absolute_residual`, `empirical_copula`, `global_ellipsoid`, `local_ellipsoid`, `mvcs`,
`spline`, `icnn` (d >= 2) and `absolute_residual`, `locally_weighted`, `mvcs`, `spline`, `icnn`
(d = 1). Datasets: the keys of `configs/datasets.yaml`.

By default a run loads the paper's frozen point predictor and tuned selection for that
(dataset, seed) from `reference/` and checks that this run's data split and residuals are the
ones they were made with. To regenerate them instead:

```
python fit_point_predictor.py --dataset enb --out my_predictors                 # CPU, 4 threads
python run.py --dataset enb --method icnn --point-predictor-dir my_predictors --tuning-dir my_tuning --tune
```

The encoder-input ablation runs the ICNN with `--encoder-input {f,x,none,id_xf,id_x,id_f}`
(`xf` is the paper's CoCoNet_ICNN); the paper reused the production tuning, so pass the default
`--tuning-dir`.

On PACE:

```
sbatch slurm/run_cpu.sbatch enb spline        # any CPU method
sbatch slurm/run_icnn.sbatch enb              # CoCoNet_ICNN on a Quadro RTX 6000
bash slurm/submit_all.sh                      # every (dataset, method) of the tables
```

Tables:

```
python make_table.py             # tables/aggregated.csv, tables/multivariate.tex, tables/univariate.tex
python make_table.py --compare   # plus tables/comparison_with_paper.csv against reference/paper_cells.csv
```

## Reproducing the paper's numbers

The numbers depend on the device, so the paper's device assignment must be kept:

- CoCoNet_ICNN trains and measures its test volume on the GPU. The paper used a Quadro RTX 6000;
  on that card a rerun reproduces the recorded volumes to about 1e-5 relative. Another GPU model
  changes the floating-point results and, through the checkpoint and gamma selection steps, can
  change the selected model.
- Everything else (point predictor, baselines, CoCoNet_Spline, MVCS) runs on the CPU with 4
  threads and no GPU visible; those numbers are bit-reproducible. The thread count matters: the
  frozen point predictor's residuals are reproduced only with `OMP_NUM_THREADS=4`.

`reference/paper_cells.csv` lists, for every (dataset, seed, method) of the tables, the volume
and coverage that entered the paper; `make_table.py --compare` diffs a rerun against it. A rerun
of every method on ENB seed 0 (ICNN on a Quadro RTX 6000, the rest on 4 CPU threads), of the
global ellipsoid on SGEMM seed 1 and of the univariate baselines on Concrete seed 0 reproduced
the recorded volumes and coverages exactly, and refitting the point predictors of ENB and SGEMM
reproduced the recorded residuals bit for bit.

## Protocol (configs/paper.yaml)

Split 25/30/25/20 (D_pred / D_score / D_cal / D_test) from one seeded permutation; large datasets
are first capped to a seeded random sample (`max_rows` in `configs/datasets.yaml`). Features are
standardized / one-hot encoded and targets standardized with statistics of D_pred only; volumes
are reported on the original target scale.

The learned scores minimise (E[V])^(1/d) + lambda_miss * E[(S - 1)_+] + lambda_cov * (E[sigmoid(10 (1 - S))] - 0.9)^2
on D_score with the threshold fixed at 1, Adam, gradient clipping at 5, at most 400 epochs with
patience 80. The checkpoint is chosen every 5 epochs by calibrated validation coverage and size.
The two weights are gamma multiples of reference scales probed on the initialised model, searched
over gamma_miss in {0.01, 0.03, 1, 10} x gamma_cov in {0, 0.1, 1}; the winner is the
coverage-admissible candidate with the smallest validation size. The ICNN's encoder width
({[64,32], [256,128]}) and the spline's knots {2,4,6,8}, batch size {32,64,128} and encoder width
are tuned per (dataset, seed) at the neutral weights. The ICNN's test volume is the converged
whitened estimator at 2^15 directions (2^14 at d = 1) with a doubling certificate.

MVCS_conv is the fixed-centre convex (p >= 1) MVCS of Braun et al. with the authors' grid (60
candidates, selected by validation objective); the ellipsoids, the copula and LW-SC follow their
papers with the settings listed in the config.
