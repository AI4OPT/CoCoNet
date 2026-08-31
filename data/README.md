# Data

Real regression datasets, mostly UCI/OpenML/Mulan-derived. No synthetic data. Each folder is
named after the dataset(s) that read it; see `learnscore/datasets/registry.py` for the exact
file each dataset loads, and `learnscore/datasets/loaders.py` for how it's parsed.

A few folders are shared by more than one registry entry, because the underlying table is one
source and only which columns are the target differs:

| folder | used by |
|---|---|
| `argoverse2_multidim/` | `argoverse2_d2`, `argoverse2_d4` |
| `qm9/` | `qm9_d2`, `qm9_d4`, `qm9_d6`, `qm9_d8` |
| `300w/` | `w300_d2`, `w300_d4`, `w300_d8`, `w300_d16` |
| `protein/CASP.csv` vs. `casp/CASP.npz` | `protein` (univariate) and `casp` (bivariate) read the same underlying CASP study through two different upstream packagings -- they are not duplicates of each other. |

`household_power_consumption/household_power_consumption.txt.gz` is gzipped (the original is
~127MB, over typical git hosting limits for a single file); loaders.py reads it directly.

Everything here is public research data (UCI Machine Learning Repository, OpenML, the Mulan
multi-target regression collection, and a small number of individually-cited sources -- see
each loader's docstring in `learnscore/datasets/loaders.py`), included for reproducibility.
