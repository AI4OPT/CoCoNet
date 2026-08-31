"""Per-dataset loaders.

Every loader takes a `DatasetSpec` (registry.py) and returns raw (X, Y) as numpy float
arrays -- categorical columns already one-hot encoded, missing feature values already
median-imputed. (Simplification vs. the original repo: imputation/encoding there was fit on
D_pred only; here it is done once, globally, at load time, since this repo ships the data
locally and keeps the split/standardize boundary in splits.py instead.)

All files are read from local disk under data/ -- no download machinery, since the data ships
with this repo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from learnscore.datasets.registry import DatasetSpec, QM9_CANONICAL_ORDER

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


class DatasetLoadError(RuntimeError):
    pass


def _path(spec: DatasetSpec, key: str) -> Path:
    return DATA_DIR / spec.files[key]


def _finalize(X: pd.DataFrame, y: pd.DataFrame, categorical_columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """One-hot encode categoricals, median-impute numeric NaNs, return (X, Y) as float arrays."""
    cat_cols = [c for c in categorical_columns if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]

    num = X[num_cols].apply(pd.to_numeric, errors="coerce")
    num = num.fillna(num.median(numeric_only=True))
    if cat_cols:
        cat = pd.get_dummies(X[cat_cols].astype(str), columns=cat_cols)
        X_final = pd.concat([num, cat], axis=1)
    else:
        X_final = num
    # Drop globally-constant columns (raw UCI/Mulan files occasionally ship one, e.g. naval
    # propulsion's two fixed sensor readings; one-hot dummies for a rare category can also come
    # out constant). Standardization in splits.py fits on D_pred alone and would otherwise
    # divide by ~0 whenever such a column happens to be constant within that subset too.
    constant_cols = [c for c in X_final.columns if X_final[c].nunique(dropna=True) <= 1]
    if constant_cols:
        X_final = X_final.drop(columns=constant_cols)

    Y = y.to_numpy(dtype=float)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    n_missing_y = int(np.isnan(Y).sum())
    if n_missing_y:
        raise DatasetLoadError(f"target contains {n_missing_y} missing value(s) after loading")
    return X_final.to_numpy(dtype=float), Y


def _split_xy(df: pd.DataFrame, target_columns: list[str], drop_columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = [c for c in target_columns if c not in df.columns]
    if missing:
        raise DatasetLoadError(f"target column(s) {missing} not found. Columns: {list(df.columns)}")
    y = df[target_columns]
    X = df.drop(columns=list(dict.fromkeys(target_columns + drop_columns)), errors="ignore")
    return X, y


# --------------------------------------------------------------------------
# concrete: UCI xls, single target column
# --------------------------------------------------------------------------
def _load_uci_xls(spec: DatasetSpec) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_excel(_path(spec, "xls"))
    X, y = _split_xy(df, [spec.target_column], spec.drop_columns)
    return _finalize(X, y, spec.categorical_columns)


# --------------------------------------------------------------------------
# kin8nm, protein, bike_sharing: a plain local CSV, single target column
# --------------------------------------------------------------------------
def _load_flat_csv_single(spec: DatasetSpec) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(_path(spec, "csv"))
    X, y = _split_xy(df, [spec.target_column], spec.drop_columns)
    return _finalize(X, y, spec.categorical_columns)


# --------------------------------------------------------------------------
# enb, sgemm, rf1/rf2, scm1d/scm20d, sf1/sf2, scpf, wq, atp1d, osales: a plain local CSV
# (openml-fetched or UCI multi-target file), multiple target columns
# --------------------------------------------------------------------------
def _load_flat_csv_multi(spec: DatasetSpec) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(_path(spec, "csv"))
    X, y = _split_xy(df, spec.target_columns, spec.drop_columns)
    if y.shape[1] != spec.output_dim:
        raise DatasetLoadError(f"{spec.name}: output_dim={spec.output_dim} != {y.shape[1]} target columns")
    return _finalize(X, y, spec.categorical_columns)


# --------------------------------------------------------------------------
# naval_propulsion: whitespace-delimited, no header, target/features by column index
# --------------------------------------------------------------------------
def _load_naval_propulsion(spec: DatasetSpec) -> tuple[np.ndarray, np.ndarray]:
    raw = pd.read_csv(_path(spec, "txt"), sep=r"\s+", header=None)
    if raw.shape[1] != 18:
        raise DatasetLoadError(f"{spec.name}: expected 18 whitespace columns, got {raw.shape[1]}")
    target_idx = spec.extra["target_column_index"]
    feature_idx = spec.extra["feature_column_indices"]
    y = raw.iloc[:, [target_idx]]
    y.columns = ["target"]
    X = raw.iloc[:, feature_idx].copy()
    X.columns = [f"feat_{i}" for i in feature_idx]
    return _finalize(X, y, [])


# --------------------------------------------------------------------------
# casp: MVCS authors' npz (X, Y arrays), used verbatim
# --------------------------------------------------------------------------
def _load_npz_xy(spec: DatasetSpec) -> tuple[np.ndarray, np.ndarray]:
    path = _path(spec, "npz")
    with np.load(path, allow_pickle=False) as z:
        if "X" not in z or "Y" not in z:
            raise DatasetLoadError(f"{spec.name}: expected arrays 'X' and 'Y', found {list(z.keys())}")
        X = np.asarray(z["X"], dtype=float)
        Y = np.asarray(z["Y"], dtype=float)
    if len(X) != len(Y):
        raise DatasetLoadError(f"{spec.name}: X has {len(X)} rows, Y has {len(Y)}")
    return X, Y


# --------------------------------------------------------------------------
# taxi: MVCS benchmark file -- numeric columns only, last output_dim columns are Y
# --------------------------------------------------------------------------
def _load_taxi(spec: DatasetSpec) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(_path(spec, "csv"))
    df = df.select_dtypes(include=[np.number])
    n_expected = spec.extra.get("expected_rows")
    if n_expected is not None and len(df) != n_expected:
        raise DatasetLoadError(f"{spec.name}: expected {n_expected} rows, got {len(df)}")
    d = spec.output_dim
    if df.shape[1] <= d:
        raise DatasetLoadError(f"{spec.name}: only {df.shape[1]} numeric column(s), need > output_dim={d}")
    y = df.iloc[:, -d:]
    X = df.iloc[:, :-d]
    expected_x = spec.extra.get("expected_x_dim")
    if expected_x is not None and X.shape[1] != expected_x:
        raise DatasetLoadError(f"{spec.name}: expected {expected_x} covariates, got {X.shape[1]}")
    return _finalize(X, y, [])


# --------------------------------------------------------------------------
# geographical_music: headerless UCI txt, last 2 columns are (latitude, longitude)
# --------------------------------------------------------------------------
def _load_geographical_music(spec: DatasetSpec) -> tuple[np.ndarray, np.ndarray]:
    raw = pd.read_csv(_path(spec, "txt"), header=None)
    n_features = raw.shape[1] - len(spec.target_columns)
    raw.columns = [f"audio_feature_{i + 1:03d}" for i in range(n_features)] + spec.target_columns
    X, y = _split_xy(raw, spec.target_columns, [])
    return _finalize(X, y, [])


# --------------------------------------------------------------------------
# argoverse2_motion_forecasting, argoverse2_d2, argoverse2_d4: prepared CSV with every future
# offset column present. LEAKAGE GUARD: drop every future_* column, not just this variant's
# target, so a variant never sees another variant's answer as a feature.
# --------------------------------------------------------------------------
def _load_argoverse2(spec: DatasetSpec) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(_path(spec, "csv"))
    future_cols = [c for c in df.columns if c.startswith("future_") and c not in spec.target_columns]
    X, y = _split_xy(df, spec.target_columns, spec.drop_columns + future_cols)
    return _finalize(X, y, spec.categorical_columns)


# --------------------------------------------------------------------------
# sarcos: two MATLAB arrays (train + test), concatenated; first n_input_columns are X,
# target_indices select which torques become Y
# --------------------------------------------------------------------------
def _load_sarcos(spec: DatasetSpec) -> tuple[np.ndarray, np.ndarray]:
    from scipy.io import loadmat

    arrays = []
    for key_path, mat_key in (("train", "sarcos_inv"), ("test", "sarcos_inv_test")):
        arr = np.asarray(loadmat(_path(spec, key_path))[mat_key], dtype=float)
        if arr.shape[1] != 28:
            raise DatasetLoadError(f"{spec.name}: expected 28 columns, got {arr.shape[1]}")
        arrays.append(arr)
    data = np.vstack(arrays)
    n_in = spec.extra["n_input_columns"]
    targets = spec.extra["target_indices"]
    return data[:, :n_in], data[:, targets]


# --------------------------------------------------------------------------
# qm8_excited_states: paired spectra (whitespace) + geometry (XYZ) files -> Coulomb-matrix
# eigenvalue features + elemental composition; RI-CC2 block only (see registry.py notes)
# --------------------------------------------------------------------------
_QM8_Z = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}
_QM8_MAX_ATOMS = 26


def _coulomb_eigenvalues(symbols, coords, n_keep: int) -> np.ndarray:
    z = np.array([_QM8_Z[s] for s in symbols], dtype=float)
    xyz = np.asarray(coords, dtype=float)
    d = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        M = np.where(d > 0, np.outer(z, z) / np.where(d > 0, d, 1.0), 0.0)
    np.fill_diagonal(M, 0.5 * z ** 2.4)
    ev = np.linalg.eigvalsh(M)[::-1]
    out = np.zeros(_QM8_MAX_ATOMS, dtype=float)
    out[: len(ev)] = ev
    return out[:n_keep]


def _parse_xyz_multi(text: str):
    lines = text.splitlines()
    i, n = 0, len(lines)
    while i < n:
        if not lines[i].strip():
            i += 1
            continue
        natoms = int(lines[i].split()[0])
        syms, xyz = [], []
        for k in range(i + 2, i + 2 + natoms):
            parts = lines[k].split()
            syms.append(parts[0])
            xyz.append([float(v.replace("*^", "e")) for v in parts[1:4]])
        yield syms, xyz
        i += natoms + 2


def _load_qm8(spec: DatasetSpec) -> tuple[np.ndarray, np.ndarray]:
    spec_text = _path(spec, "spectra").read_text(encoding="utf-8", errors="replace")
    xyz_text = _path(spec, "xyz").read_text(encoding="utf-8", errors="replace")

    rows = [l.split() for l in spec_text.splitlines() if l.strip() and not l.strip().startswith("#")]
    rows = [r for r in rows if len(r) == 17]
    if not rows:
        raise DatasetLoadError(f"{spec.name}: no 17-field data rows found")
    Y_all = np.array([[float(v) for v in r[1:]] for r in rows], dtype=float)

    mols = list(_parse_xyz_multi(xyz_text))
    if len(mols) != len(rows):
        raise DatasetLoadError(f"{spec.name}: {len(rows)} spectra rows but {len(mols)} geometries")

    n_keep = spec.extra["n_eigenvalues"]
    eig = np.vstack([_coulomb_eigenvalues(sy, xy, n_keep) for sy, xy in mols])
    comp = np.array(
        [[sum(1 for s in sy if s == e) for e in ("C", "H", "N", "O", "F")] + [len(sy)] for sy, _ in mols],
        dtype=float,
    )
    X = np.hstack([comp, eig])

    col_index = {name: i for i, name in enumerate(spec.extra["all_target_columns"])}
    Y = Y_all[:, [col_index[c] for c in spec.target_columns]]
    return X, Y


# --------------------------------------------------------------------------
# communities_crime_unnormalized: UCI 211, headerless, fixed column layout, "?" missing marker
# --------------------------------------------------------------------------
_CCU_COLUMNS = [
    "communityname", "State", "countyCode", "communityCode", "fold", "pop", "perHoush",
    "pctBlack", "pctWhite", "pctAsian", "pctHisp", "pct12-21", "pct12-29", "pct16-24",
    "pct65up", "persUrban", "pctUrban", "medIncome", "pctWwage", "pctWfarm", "pctWdiv",
    "pctWsocsec", "pctPubAsst", "pctRetire", "medFamIncome", "perCapInc", "whitePerCap",
    "blackPerCap", "NAperCap", "asianPerCap", "otherPerCap", "hispPerCap", "persPoverty",
    "pctPoverty", "pctLowEdu", "pctNotHSgrad", "pctCollGrad", "pctUnemploy", "pctEmploy",
    "pctEmployMfg", "pctEmployProfServ", "pctOccupManu", "pctOccupMgmt", "pctMaleDivorc",
    "pctMaleNevMar", "pctFemDivorc", "pctAllDivorc", "persPerFam", "pct2Par", "pctKids2Par",
    "pctKids-4w2Par", "pct12-17w2Par", "pctWorkMom-6", "pctWorkMom-18", "kidsBornNevrMarr",
    "pctKidsBornNevrMarr", "numForeignBorn", "pctFgnImmig-3", "pctFgnImmig-5", "pctFgnImmig-8",
    "pctFgnImmig-10", "pctImmig-3", "pctImmig-5", "pctImmig-8", "pctImmig-10",
    "pctSpeakOnlyEng", "pctNotSpeakEng", "pctLargHousFam", "pctLargHous", "persPerOccupHous",
    "persPerOwnOccup", "persPerRenterOccup", "pctPersOwnOccup", "pctPopDenseHous",
    "pctSmallHousUnits", "medNumBedrm", "houseVacant", "pctHousOccup", "pctHousOwnerOccup",
    "pctVacantBoarded", "pctVacant6up", "medYrHousBuilt", "pctHousWOphone", "pctHousWOplumb",
    "ownHousLowQ", "ownHousMed", "ownHousUperQ", "ownHousQrange", "rentLowQ", "rentMed",
    "rentUpperQ", "rentQrange", "medGrossRent", "medRentpctHousInc", "medOwnCostpct",
    "medOwnCostPctWO", "persEmergShelt", "persHomeless", "pctForeignBorn", "pctBornStateResid",
    "pctSameHouse-5", "pctSameCounty-5", "pctSameState-5", "numPolice", "policePerPop",
    "policeField", "policeFieldPerPop", "policeCalls", "policCallPerPop", "policCallPerOffic",
    "policePerPop2", "racialMatch", "pctPolicWhite", "pctPolicBlack", "pctPolicHisp",
    "pctPolicAsian", "pctPolicMinority", "officDrugUnits", "numDiffDrugsSeiz", "policAveOT",
    "landArea", "popDensity", "pctUsePubTrans", "policCarsAvail", "policOperBudget",
    "pctPolicPatrol", "gangUnit", "pctOfficDrugUnit", "policBudgetPerPop", "murders",
    "murdPerPop", "rapes", "rapesPerPop", "robberies", "robbbPerPop", "assaults",
    "assaultPerPop", "burglaries", "burglPerPop", "larcenies", "larcPerPop", "autoTheft",
    "autoTheftPerPop", "arsons", "arsonsPerPop", "violentPerPop", "nonViolPerPop",
]


def _load_communities_crime_unnormalized(spec: DatasetSpec) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(_path(spec, "txt"), header=None, names=_CCU_COLUMNS, na_values="?", encoding="latin-1")
    if df.shape[1] != len(_CCU_COLUMNS):
        raise DatasetLoadError(f"{spec.name}: expected {len(_CCU_COLUMNS)} columns, got {df.shape[1]}")
    df = df.dropna(subset=spec.target_columns).reset_index(drop=True)
    X, y = _split_xy(df, spec.target_columns, spec.drop_columns)
    return _finalize(X, y, [])


# --------------------------------------------------------------------------
# oes10, oes97: ARFF (Mulan MTR occupational-employment data), all-numeric
# --------------------------------------------------------------------------
def _load_oes_arff(spec: DatasetSpec) -> tuple[np.ndarray, np.ndarray]:
    from scipy.io import arff as scipy_arff

    data, _meta = scipy_arff.loadarff(_path(spec, "arff"))
    df = pd.DataFrame(data).astype(float)
    X, y = _split_xy(df, spec.target_columns, [])
    return _finalize(X, y, [])


# --------------------------------------------------------------------------
# bias_correction_temperature: local CSV (already extracted), drop constant columns
# --------------------------------------------------------------------------
def _load_bias_correction(spec: DatasetSpec) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(_path(spec, "csv"))
    df = df.dropna(subset=spec.target_columns).reset_index(drop=True)
    date_col = spec.extra["date_column"]
    drop_cols = spec.drop_columns + [date_col]
    X, y = _split_xy(df, spec.target_columns, drop_cols)
    constant_cols = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    if constant_cols:
        X = X.drop(columns=constant_cols)
    return _finalize(X, y, spec.categorical_columns)


# --------------------------------------------------------------------------
# online_retail_ii: 2 Excel sheets concatenated, InvoiceDate decomposed into 5 numeric parts
# --------------------------------------------------------------------------
def _load_online_retail_ii(spec: DatasetSpec) -> tuple[np.ndarray, np.ndarray]:
    sheet_names = spec.extra["sheet_names"]
    sheets = pd.read_excel(_path(spec, "xlsx"), sheet_name=sheet_names, engine="openpyxl")
    df = pd.concat([sheets[s] for s in sheet_names], ignore_index=True)

    dt = pd.to_datetime(df["InvoiceDate"])
    df["InvoiceYear"] = dt.dt.year
    df["InvoiceMonth"] = dt.dt.month
    df["InvoiceDay"] = dt.dt.day
    df["InvoiceHour"] = dt.dt.hour
    df["InvoiceDayOfWeek"] = dt.dt.dayofweek

    df = df.dropna(subset=spec.target_columns).reset_index(drop=True)
    X, y = _split_xy(df, spec.target_columns, spec.drop_columns)
    return _finalize(X, y, spec.categorical_columns)


# --------------------------------------------------------------------------
# household_power_consumption: gzipped semicolon-delimited UCI file, "?" missing marker
# --------------------------------------------------------------------------
def _load_household_power(spec: DatasetSpec) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(_path(spec, "gz"), sep=";", na_values="?", low_memory=False)
    X, y = _split_xy(df, spec.target_columns, spec.drop_columns)
    X = X.apply(pd.to_numeric, errors="coerce")
    y = y.apply(pd.to_numeric, errors="coerce")
    keep = ~(X.isna().all(axis=1) & y.isna().all(axis=1))
    X, y = X.loc[keep].reset_index(drop=True), y.loc[keep].reset_index(drop=True)
    y = y.dropna()
    X = X.loc[y.index]
    return _finalize(X, y, [])


# --------------------------------------------------------------------------
# qm9_d2/d4/d6/d8: nested multivariate regression, one shared prepared table
# --------------------------------------------------------------------------
def _load_qm9(spec: DatasetSpec) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(_path(spec, "csv"))
    target_dim = spec.extra["target_dim"]
    targets = QM9_CANONICAL_ORDER[:target_dim]
    # LEAKAGE GUARD: drop every canonical-order property column (not just this variant's
    # targets), since each one is a target for some other qm9_d* variant.
    property_cols = [c for c in QM9_CANONICAL_ORDER if c in df.columns]
    X, y = _split_xy(df, targets, property_cols)
    return _finalize(X, y, [])


# --------------------------------------------------------------------------
# w300_d2/d4/d8/d16: nested multivariate regression, one shared prepared table
# --------------------------------------------------------------------------
def _load_w300(spec: DatasetSpec) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(_path(spec, "csv"))
    target_dim = spec.extra["target_dim"]
    n_landmarks = target_dim // 2
    targets = [c for rank in range(1, n_landmarks + 1) for c in (f"lm{rank:02d}_x", f"lm{rank:02d}_y")]
    landmark_cols = [c for c in df.columns if c.startswith("lm") and (c.endswith("_x") or c.endswith("_y"))]
    X, y = _split_xy(df, targets, landmark_cols + ["subset"])
    return _finalize(X, y, [])


_DISPATCH = {
    "uci_xls": _load_uci_xls,
    "flat_csv_single": _load_flat_csv_single,
    "flat_csv_multi": _load_flat_csv_multi,
    "naval_propulsion": _load_naval_propulsion,
    "npz_xy": _load_npz_xy,
    "taxi": _load_taxi,
    "geographical_music": _load_geographical_music,
    "argoverse2": _load_argoverse2,
    "sarcos": _load_sarcos,
    "qm8": _load_qm8,
    "communities_crime_unnormalized": _load_communities_crime_unnormalized,
    "oes_arff": _load_oes_arff,
    "bias_correction": _load_bias_correction,
    "online_retail_ii": _load_online_retail_ii,
    "household_power": _load_household_power,
    "qm9": _load_qm9,
    "w300": _load_w300,
}


def load_dataset(spec: DatasetSpec) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch to the right loader and sanity-check the result."""
    if spec.kind not in _DISPATCH:
        raise DatasetLoadError(f"No loader implemented for kind='{spec.kind}' ({spec.name})")
    X, Y = _DISPATCH[spec.kind](spec)
    X, Y = np.asarray(X, dtype=float), np.asarray(Y, dtype=float)
    if len(X) != len(Y):
        raise DatasetLoadError(f"{spec.name}: X has {len(X)} rows, Y has {len(Y)}")
    if len(X) == 0:
        raise DatasetLoadError(f"{spec.name}: loaded zero rows")
    if Y.shape[1] != spec.output_dim:
        raise DatasetLoadError(f"{spec.name}: expected output_dim={spec.output_dim}, got Y.shape[1]={Y.shape[1]}")
    if not np.all(np.isfinite(X)):
        raise DatasetLoadError(f"{spec.name}: non-finite values in X after loading/imputation")
    return X, Y
