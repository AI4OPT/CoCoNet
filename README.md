# CoCoNet

Learned convex prediction regions for multivariate regression with split-conformal coverage,
plus the baselines they are compared with. This repository contains the code, the raw data and
the fitted point predictors needed to rerun every experiment: 30 multivariate datasets, 6
univariate datasets, an extended comparison with further baselines and an encoder-input ablation,
at 90% target coverage over seeds 0-4.

![Overview](figures/overview.png)

An encoder maps the input x and the point prediction F(x) to a representation that conditions
a conformity score which is convex in the residual r = y - F(x). A calibration set gives the
conformal threshold; the sublevel set of the score at that threshold is the prediction region.

All methods receive the residuals of the same frozen MLP point predictor per (dataset, seed),
are calibrated by split conformal prediction on D_cal and are scored on D_test by joint coverage
and (mean volume)^(1/d).

| Method | `--method` | Region |
|---|---|---|
| CoCoNet_ICNN | `icnn` | sublevel set of an input-convex neural network score |
| CoCoNet_Spline | `spline` | rotated box from convex piecewise-linear scores |
| Abs. Res. | `absolute_residual` | cube, s = max_j \|r_j\| |
| Emp. Cop. | `empirical_copula` | axis-aligned box from the empirical copula of normalized scores |
| G-Ellip. | `global_ellipsoid` | one Mahalanobis ellipsoid |
| L-Ellip. | `local_ellipsoid` | k-nearest-neighbour local covariance ellipsoid |
| MVCS | `mvcs` | convex (p >= 1) minimum-volume covering set |
| LW-SC | `locally_weighted` | locally weighted split-conformal interval (univariate only) |

The extended multivariate comparison adds:

| Method | `--method` | Region |
|---|---|---|
| MVCS, free q | `mvcs_free` | the same set with a free exponent (q >= 0.1, not convex when q < 1) |
| MVCS multi-norm, q >= 1 | `mvcs_multi` | one global set, a scaled q_j-norm ball per orthant after a rotation, every q_j >= 1 |
| MVCS multi-norm, free q | `mvcs_multi_free` | the same with free exponents |
| PBS | `pbs` | pullback of a Euclidean ball through a conditional neural optimal-transport map |

The two multi-norm variants run the MVCS authors' own code; fetch it once with
`bash scripts/get_mvcs.sh` (it clones their public repository at a fixed commit into
`third_party/`).

## Layout

```
run.py                  one method on one dataset (all seeds) -> results/<dataset>/<method>/seed<S>/row.json
fit_point_predictor.py  fit and save the MLP point predictor of a dataset
make_table.py           results/ -> tables/ (aggregated.csv, multivariate.tex, univariate.tex)
configs/experiment.yaml every setting of the experiments
configs/datasets.yaml   the 36 datasets: loader, target columns, row cap
scripts/get_mvcs.sh     fetches the MVCS authors' code used by the multi-norm baselines
coconet/pipeline.py     data preparation, one run_<method> per method, evaluation
coconet/tuning.py       per-(dataset, seed) tuning of the learned scores
coconet/models/         icnn.py, spline.py (the two scores), mlp.py, volume_estimator.py
coconet/training/       loss, lambda grid search, training loops, checkpoint selection
coconet/conformal/      calibration, ellipsoids, empirical copula, MVCS (single and multi-norm), PBS
coconet/data/           loaders, preprocessing, splits
data/raw/               raw dataset files (UCI / OpenML files are downloaded on first use if missing)
reference/              fitted point predictors (.pt + .json) and tuned settings (.json) per (dataset, seed);
                        rows10000/ holds the 10,000-row versions used by the encoder ablation
```

## Setup

Python 3.13 with `pip install -r requirements.txt`. The reported numbers were produced with
torch 2.7.1+cu126, numpy 2.3.1, pandas 2.3.1, scipy 1.16.0 and scikit-learn 1.7.0.

## Datasets

`n` is the number of rows in the source data, `d_x` the number of raw features, `d` the
response dimension and "rows used" the seeded random subsample the experiments ran on
(`max_rows` in `configs/datasets.yaml`). Loaders drop rows with a missing target where the source has any; missing features are
median-imputed from D_pred.

### Multivariate

| Key | Label | Source | n | d_x | d | Rows used |
|---|---|---|---:|---:|---:|---:|
| `argoverse2_2d` | Argoverse2-2D | Argoverse 2 motion forecasting: displacement of the focal agent 6 s ahead from summaries of its 5 s history | 24,988 | 28 | 2 | 20,000 |
| `argoverse2_motion` | Argoverse2 Motion | same source, 3 s horizon | 24,988 | 28 | 2 | 20,000 |
| `bias_corr_temp` | Bias-Corr. Temp. | UCI 514: next-day (Tmax, Tmin) from present-day readings and forecasts | 7,723 | 23 | 2 | all |
| `enb` | ENB | UCI Energy Efficiency (OpenML 41478): heating and cooling load | 768 | 8 | 2 | all |
| `geographical_music` | Geographical Music | UCI 315: (latitude, longitude) of a track from 68 audio features | 1,059 | 68 | 2 | all |
| `casp` | CASP | processed CASP arrays, two targets | 45,730 | 8 | 2 | 20,000 |
| `online_retail_ii` | Online Retail II | UCI 502: (Quantity, Price) of an invoice line from country and date | 1,067,371 | 6 | 2 | 20,000 |
| `qm9_2d` | QM9-2D | QM9: first 2 of 12 molecular properties (mu, alpha) | 130,831 | 22 | 2 | 20,000 |
| `household_power` | Household Power | UCI 235: three sub-metering readings from four whole-house quantities | 2,049,280 | 4 | 3 | 20,000 |
| `scpf` | SCPF | See Click Predict Fix (OpenML 41487): views, votes, comments of an issue report | 1,137 | 23 | 3 | all |
| `argoverse2_4d` | Argoverse2-4D | displacements at 3 s and 6 s | 24,988 | 28 | 4 | 20,000 |
| `qm8` | QM8 | QM8 spectra: the four RI-CC2 quantities from composition and Coulomb-matrix eigenvalues | 21,786 | 14 | 4 | 20,000 |
| `qm9_4d` | QM9-4D | QM9: first 4 properties (+ homo, lumo) | 130,831 | 22 | 4 | 20,000 |
| `sgemm` | SGEMM | UCI 440: four repeated run times of a GPU kernel configuration | 241,600 | 14 | 4 | 20,000 |
| `beijing_aq_6d` | Beijing AQ-6D | UCI 501: six pollutants of one station-hour from its meteorology | 383,980 | 11 | 6 | 20,000 |
| `communities_crime` | Communities Crime | UCI 211: six per-capita crime rates of a community | 2,215 | 124 | 6 | all |
| `atp1d` | ATP1D | Mulan ATP1D (OpenML 41475): next-day minimum price of six flight/fare combinations from price-quote history | 337 | 411 | 6 | all |
| `qm9_6d` | QM9-6D | QM9: first 6 properties (+ r2, zpve) | 130,831 | 22 | 6 | 20,000 |
| `sarcos` | SARCOS | SARCOS inverse dynamics: 7 joint torques from the 21-D arm state | 48,933 | 21 | 7 | 20,000 |
| `argoverse2_8d` | Argoverse2-8D | displacements at 1.5, 3, 4.5 and 6 s | 24,988 | 28 | 8 | 20,000 |
| `qm9_8d` | QM9-8D | QM9: first 8 properties (+ Cv, U0) | 130,831 | 22 | 8 | 20,000 |
| `beijing_aq_12d` | Beijing AQ-12D | UCI 501: PM2.5 of all 12 stations at one hour from every station's meteorology (hours with all 12 readings) | 28,663 | 76 | 12 | 20,000 |
| `knee_contact_force` | Knee Contact Force | Zenodo 7253458: 12 knee contact-force peaks of a walking trial | 2,786 | 9 | 12 | all |
| `wq` | WQ | Mulan water quality (OpenML 41491): 14 species and indicator concentrations from 16 physico-chemical measurements | 1,060 | 16 | 14 | all |
| `qm7b` | QM7b | QM7b: the 14 molecular properties from Coulomb-matrix eigenvalues and element counts | 7,211 | 29 | 14 | all |
| `oes10` | OES10 | OES 2010: 16 occupation employment counts per metropolitan area | 403 | 298 | 16 | all |
| `oes97` | OES97 | OES 1997 | 334 | 263 | 16 | all |
| `bdg2_fox` | BDG2-Fox | Building Data Genome 2, site Fox: hourly electricity of 16 buildings from site weather and calendar features | 17,537 | 12 | 16 | all |
| `bdg2_eagle` | BDG2-Eagle | same, site Eagle | 17,105 | 12 | 16 | all |
| `bdg2_lamb` | BDG2-Lamb | same, site Lamb | 17,544 | 12 | 16 | all |

The Argoverse 2 and QM9 tasks of different d are cut from one table each; the columns that are
targets of another variant are removed from the features. QM9 properties are taken in the order
mu, alpha, homo, lumo, r2, zpve, Cv, U0 (then gap, U, H, G, unused).

### Univariate

| Key | Label | Source | n | d_x | Rows used |
|---|---|---|---:|---:|---:|
| `beijing_pm25` | Beijing PM2.5 | UCI 381: hourly PM2.5 from concurrent meteorology | 41,757 | 11 | 10,000 |
| `bike_sharing` | Bike Sharing | UCI 275: hourly rental count | 17,379 | 12 | 10,000 |
| `ccpp` | CCPP | UCI 294: net power output of a combined-cycle plant | 9,568 | 4 | all |
| `concrete` | Concrete | UCI concrete compressive strength | 1,030 | 8 | all |
| `kin8nm` | Kin8nm | OpenML 189: forward kinematics of an 8-link arm | 8,192 | 8 | all |
| `protein` | Protein | UCI 265 (CASP): RMSD of a protein structure | 45,730 | 9 | 10,000 |

## Running

One method on one dataset, all five seeds:

```
python run.py --dataset enb --method icnn                          # GPU
python run.py --dataset enb --method spline                        # CPU
python run.py --dataset enb --method mvcs --seeds 0 1              # CPU, two seeds
python run.py --dataset concrete --method locally_weighted         # univariate baseline
```

`python run.py --help` lists the remaining flags (output directory, config files, fitting
the point predictor in place of loading it). Multivariate datasets take `absolute_residual`, `empirical_copula`, `global_ellipsoid`,
`local_ellipsoid`, `mvcs`, `spline`, `icnn` and the four extended baselines; univariate ones
`absolute_residual`, `locally_weighted`, `mvcs`, `spline`, `icnn`.

**Volumes.** Each baseline reports the closed-form volume of its region by default. The
multivariate tables instead measure every region with the same Monte Carlo estimator the ICNN
uses (per test point, groups of 30 points, 2^15 directions, doubling until the size changes by
less than 0.1%). Add `--volume estimator` to do the same; it uses a GPU when one is visible. PBS
always reports its own change-of-variables volume, and univariate intervals are exact.

By default a run loads the saved point predictor and tuned settings of that (dataset, seed)
from `reference/` and checks that this run's data split and residuals are the ones they were
made with. To regenerate them instead:

```
python fit_point_predictor.py --dataset enb --out my_predictors
python run.py --dataset enb --method icnn --point-predictor-dir my_predictors --tuning-dir my_tuning --tune
```

The encoder-input ablation runs the ICNN with `--encoder-input {xf,f,x,none,id_xf,id_x,id_f}`
(`xf` is the default model; the `id_` arms feed the raw context without an encoder MLP; `none`
learns one global set) on ENB, Bias-Corr. Temp., Online Retail II, SCPF, QM8, Communities Crime,
QM9-8D, Knee Contact Force, QM7b and OES97. Every arm reuses the tuned settings of the default
model. Datasets with more than 10,000 rows are capped at 10,000 for the ablation, with their own
predictors and settings:

```
python run.py --dataset qm8 --method icnn --encoder-input x --max-rows 10000 \
    --point-predictor-dir reference/rows10000/point_predictors --tuning-dir reference/rows10000/tuning
```

Both scripts set `OMP_NUM_THREADS=4` unless it is already set. For the CPU methods hide any GPU
with `CUDA_VISIBLE_DEVICES=`. A cluster job needs one node, 4 cores, 8 GB per core and, for
`icnn`, one GPU. `icnn` was run on a Quadro RTX 6000; other GPU models give slightly different
numbers because the checkpoint and lambda selections react to last-bit differences.

Tables:

```
python make_table.py      # tables/aggregated.csv, tables/multivariate.tex, tables/univariate.tex
```

## Protocol (configs/experiment.yaml)

Split 25/30/25/20 (D_pred / D_score / D_cal / D_test) from one seeded permutation; large
datasets are first capped to a seeded random sample. Features are standardized / one-hot
encoded and targets standardized with statistics of D_pred only; volumes are reported on the
original target scale.

The learned scores minimise (mean V)^(1/d) + lambda_miss * mean[(S - 1)_+] + lambda_cov *
(mean sigmoid(10 (1 - S)) - 0.9)^2 on D_score with the threshold fixed at 1, Adam, gradient
clipping at 5, at most 400 epochs (ICNN) or 300 epochs (spline) with patience 80. The checkpoint is chosen every 5 epochs by
calibrated validation coverage and size. The two weights are gamma multiples of reference
scales measured on the initialised model, searched over gamma_miss in {0.01, 0.03, 1, 10} x
gamma_cov in {0, 0.1, 1}; the winner is the coverage-admissible candidate with the smallest
validation size. The ICNN's encoder width ({[64,32], [256,128]}) and the spline's knots
{2,4,6,8}, batch size {32,64,128} and encoder width are tuned per (dataset, seed) at the
neutral weights. The ICNN's test volume is a whitened Monte Carlo estimate at 2^16 directions
(2^15 at d = 1) with a doubling check.

MVCS uses its authors' hyperparameter grid (60 candidates, selected by validation objective),
with p >= 1 (`mvcs`) or p = max(|p|, 0.1) (`mvcs_free`). The multi-norm variants use the
authors' code and defaults with two splits; the convex one projects every exponent onto q >= 1
after each step. PBS trains the five network configurations in the config and keeps the one
with the smallest validation coverage error of its plug-in sets. The ellipsoids, the copula and
LW-SC use the settings listed in the config.
