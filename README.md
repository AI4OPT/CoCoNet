CLeS

A minimal, from-scratch implementation of multivariate conformal prediction with a *learned*
conformity score, trained with the "tau1" loss and calibrated by split conformal, plus a set
of comparison methods.

This is a cleaned-down extraction of a larger research codebase: no experiment-tracking
machinery, no synthetic data, no automatic hyperparameter search, no loss-formulation
ablations. Every dataset used ships in `data/`.

## The methods

For covariates `x` and target `y`, every method here fits a frozen mean predictor `mu_hat(x)`
and works with the residual `r = y - mu_hat(x)`. A conformity score `S(x, r)` measures how
atypical `r` is for `x`; the calibrated prediction set at a test point is
`{y : S(x, y - mu_hat(x)) <= q_hat}`, with `q_hat` fixed by the classical split-conformal
order statistic on held-out data (`learnscore/calibrate.py`) -- this gives a finite-sample
coverage guarantee independent of how well the score fit.

**CLeS family** (`icnn`, `asymmetric_linear`, `spline`) -- three convex conformity scores,
each trained by minimizing the same loss:

```
L = L_size + lambda_miss * L_miss + lambda_cov * L_cov
```

at a threshold `tau` that's either fixed at 1 or the batch's empirical `(1 - alpha)` score
quantile (`tau_mode` in the config). `L_size` is the set's volume, `L_miss` a scale-invariant
hinge on points outside the set, `L_cov` pulls smoothed coverage toward `1 - alpha`. See
`learnscore/loss.py` and `learnscore/train.py`.

- `learnscore/model.py` -- the ICNN score. Convex by construction (nonnegative combinations
  of convex, nondecreasing layers); no closed-form inverse, so its width/area/volume come
  from root-finding (`learnscore/volume.py`).
- `learnscore/model_asymmetric_linear.py` -- a per-coordinate asymmetric-linear score; the
  induced set is an exact axis-aligned hyperrectangle, no root-finding needed.
- `learnscore/model_spline.py` -- a per-coordinate convex piecewise-linear score; also an
  exact hyperrectangle, inverted in closed form.

**MVCS** (`mvcs`) -- a different paper's method (Braun, Aolaritei, Jordan & Bach,
arXiv:2503.19068), included as a comparison: a learned p-norm ball reshaped by a
covariate-dependent linear map, with an exact closed-form volume. Trained the same way as the
CLeS family (`learnscore/model_mvcs.py`; see its docstring for the simplifications relative
to the paper's own three-stage training procedure).

**Baselines** (`absolute_residual`, `global_ellipsoid`, `empirical_copula`) -- no
gradient-based training at all, just a `.fit()` on the calibration-set residuals (nothing to
fit for `absolute_residual`): a fixed hypercube, an ellipsoid from one global covariance
matrix, and an axis-aligned box from per-coordinate empirical CDFs, respectively. See
`learnscore/baseline_*.py`.

Set `method:` in the config (or `--method` isn't a flag -- it's a config key, since most runs
fix it for a whole sweep) to choose among the seven.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python -m learnscore.cli run --dataset concrete --seed 0 --config configs/default.yaml
```

Prints a JSON line with `coverage`, `mean_volume_original_scale`, `conformal_quantile`, and
`selected_epoch`. Pass `--output-dir results/` to also append the row to
`results/<dataset>.csv` **and** save the calibrated set's actual geometry to
`results/<dataset>_<method>_seed<seed>_sets.npz` -- for every test point, a center (the point
prediction) plus a cloud of boundary points around it, in the original response units, so the
set itself (not just its coverage/volume summary) is usable downstream: exact at
`residual_dim` 1 (the two interval endpoints) and 2 (a polygon), a QMC boundary sample above
that. See `learnscore/geometry.py`.

Every dataset in `learnscore/datasets/registry.py` is runnable this way; run
`python -c "from learnscore.datasets.registry import REGISTRY; print(sorted(REGISTRY))"`
for the full list.

## Configuration

`configs/default.yaml` holds global defaults (`method`, architecture, optimizer, split
fractions, `alpha`, `tau_mode`) plus a `dataset_overrides.<name>` block for hyperparameters
that were tuned per dataset in the original study (encoder width, learning rate, batch size,
the loss weights `lambda_miss`/`lambda_cov`). Only `concrete`, `enb`, and `oes10` have
overrides checked in here, recovered from the original repo's tuning artifacts -- add more
datasets' tuned values the same way, or just run with the global defaults.

The one setting that defines the tau1 method itself: `tau_mode: fixed` with `tau_fixed: 1.0`
reproduces the paper's tau=1 setting; `tau_mode: quantile` trains against the batch's own
empirical `(1 - alpha)` quantile instead.

## Datasets

`data/<name>/` ships the raw file(s) each dataset's loader reads -- see
`learnscore/datasets/loaders.py`. A few dimension-variant families (`argoverse2_d2`/`_d4`,
`qm9_d2..d8`, `w300_d2..d16`) share one underlying table across variants rather than
duplicating it. Datasets are UCI/OpenML/Mulan-derived real regression tasks; no synthetic
data is included. See `data/README.md` for per-dataset provenance.

## Reproducing paper_results_tau1

`tests/test_reproduction.py` (marked `slow`) trains the ICNN on `concrete`, `enb`, and
`oes10` at seed 0 with the exact recorded hyperparameters and checks the result against
`tests/reference_metrics.json`, snapshotted from the original study's `raw_metrics.csv`.
This is an independent rewrite, not a byte-for-byte port, so exact numeric reproduction
isn't the bar -- the test instead checks coverage lands within 0.10 of the recorded value
(both near the 90% nominal target) and volume lands within a factor of 3.

Training takes more than a minute of compute, so run it as a batch job rather than
interactively:

```bash
sbatch scripts/run_tests.sbatch
```

Fast unit tests (`test_loss.py`, `test_calibration.py`, `test_model.py`, `test_mvcs.py`,
`test_baselines.py`) check the loss formula, the split-conformal order statistic, and each
method's score/volume math directly on tiny synthetic tensors, and run in seconds.

## Layout

```
learnscore/
  model.py                    the ICNN score + root-finding-based width/area/volume
  model_asymmetric_linear.py  the asymmetric-linear score (CLeS family)
  model_spline.py              the convex spline score (CLeS family)
  model_mvcs.py                 the MVCS baseline (Braun et al.)
  baseline_absolute_residual.py \
  baseline_ellipsoid.py          } the three fit-only baselines
  baseline_empirical_copula.py  /
  volume.py                    generic root-finding volume/width/area, used by model.py
  geometry.py                  the calibrated set's boundary, for any method
  loss.py                      the tau1 loss (CLeS family + MVCS)
  train.py                     training loop + checkpoint selection
  calibrate.py                 split-conformal threshold
  evaluate.py                  test-set coverage/volume
  aggregate.py                 cross-seed aggregation, (mean V)^(1/d)
  point_predictor.py           frozen mean regressor mu_hat(x)
  datasets/                    registry, loaders, splitting/standardization
  cli.py                       entry point, method dispatch
configs/                       default.yaml -- global + per-dataset hyperparameters
data/                          one folder per dataset
tests/                         unit tests + the paper-reproduction check
```
