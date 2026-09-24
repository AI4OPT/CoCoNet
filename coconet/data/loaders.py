"""Dataset loaders. `load_dataset` reads the entry of configs/datasets.yaml, calls the loader
named by its `loader` key and applies the row cap (a seeded random sample). Every loader returns
raw features X (DataFrame), targets y (DataFrame, one column per output) and the categorical
feature names. Raw files live in data/raw; the UCI and OpenML ones are downloaded if missing."""

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from coconet.data.download import download_and_extract, download_file, zip_member


def load_dataset(name, cfg, raw_dir, max_rows=None, subsample_seed=None):
    X, y, categorical = LOADERS[cfg["loader"]](cfg, Path(raw_dir))
    if len(X) != len(y) or len(X) == 0:
        raise ValueError(f"{name}: bad shapes X {X.shape} y {y.shape}")
    if int(np.asarray(y.isna()).sum()) > 0:
        raise ValueError(f"{name}: target has missing values")
    if max_rows is not None and len(X) > max_rows:
        rng = np.random.default_rng(int(subsample_seed))
        rows = np.sort(rng.choice(len(X), size=max_rows, replace=False))
        X = X.iloc[rows].reset_index(drop=True)
        y = y.iloc[rows].reset_index(drop=True)
    return X, y, list(categorical)


def _split_targets(cfg, df):
    targets = list(cfg["target_columns"])
    missing = [c for c in targets if c not in df.columns]
    if missing:
        raise ValueError(f"target columns {missing} not found; columns: {list(df.columns)}")
    df = df.dropna(subset=targets).reset_index(drop=True)
    y = df[targets].astype(float)
    X = df.drop(columns=targets + [c for c in cfg.get("drop_columns", []) if c in df.columns])
    return X, y, cfg.get("categorical_columns", [])


def _raw_file(cfg, raw_dir):
    """The raw table: downloaded from `url` (a zip when `zip_member` is set) into data/raw."""
    if "zip_member" in cfg:
        return download_and_extract(cfg["url"], raw_dir, cfg["zip_member"], cfg.get("file"))
    return download_file(cfg["url"], raw_dir / cfg["file"])


def _read_table(path, cfg):
    if path.suffix in (".xls", ".xlsx"):
        return pd.read_excel(path, sheet_name=cfg.get("sheet", 0), engine="xlrd" if path.suffix == ".xls" else None)
    return pd.read_csv(path, sep=cfg.get("sep", ","))


def load_table(cfg, raw_dir):
    """A CSV / Excel file with the targets and features as columns."""
    return _split_targets(cfg, _read_table(_raw_file(cfg, raw_dir), cfg))


def load_household_power(cfg, raw_dir):
    """UCI 235; "?" marks missing values, rows with nothing measured are dropped."""
    df = pd.read_csv(_raw_file(cfg, raw_dir), sep=";", na_values="?", low_memory=False)
    targets = list(cfg["target_columns"])
    y = df[targets].astype(float)
    X = df.drop(columns=targets + cfg["drop_columns"]).astype(float)
    keep = ~(X.isna().all(axis=1) & y.isna().all(axis=1))
    return X.loc[keep].reset_index(drop=True), y.loc[keep].reset_index(drop=True), []


def load_openml(cfg, raw_dir):
    from sklearn.datasets import fetch_openml

    cache = raw_dir / "openml_cache"
    cache.mkdir(parents=True, exist_ok=True)
    data = fetch_openml(data_id=cfg["openml_id"], as_frame=True, data_home=str(cache))
    y = data.target.copy()
    y = y.to_frame() if isinstance(y, pd.Series) else y
    if list(y.columns) != list(cfg["target_columns"]):
        raise ValueError(f"OpenML {cfg['openml_id']} targets are {list(y.columns)}")
    X = data.data.copy()
    numeric = [c for c in X.columns if c not in cfg.get("categorical_columns", [])]
    X[numeric] = X[numeric].astype(float)
    return X, y.astype(float), cfg.get("categorical_columns", [])


def _beijing_multisite(cfg, raw_dir):
    """UCI 501: the 12 station files, concatenated, with a timestamp column."""
    outer = download_file(cfg["url"], raw_dir / cfg["file"])
    inner = zipfile.ZipFile(io.BytesIO(zip_member(outer, cfg["zip_member"])))
    names = sorted(n for n in inner.namelist() if n.endswith(".csv"))
    if len(names) != 12:
        raise ValueError(f"expected 12 station files, found {len(names)}")
    df = pd.concat([pd.read_csv(inner.open(n)) for n in names], ignore_index=True)
    df["_ts"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
    return df


def load_beijing_aq_6d(cfg, raw_dir):
    """One row per (station, hour): the six pollutants from the concurrent meteorology."""
    df = _beijing_multisite(cfg, raw_dir)
    targets = list(cfg["target_columns"])
    df = df.dropna(subset=targets).reset_index(drop=True)
    X = df.drop(columns=targets + cfg["drop_columns"] + ["_ts"])
    return X, df[targets].astype(float), cfg["categorical_columns"]


def load_beijing_aq_12d(cfg, raw_dir):
    """One row per hour: PM2.5 of the 12 stations from every station's meteorology."""
    df = _beijing_multisite(cfg, raw_dir)
    meteo = list(cfg["meteo_columns"])
    stations = sorted(df["station"].unique())
    wide = df.pivot_table(index="_ts", columns="station", values="PM2.5", aggfunc="first").dropna(axis=0, how="any")
    ts = wide.index
    time_cols = df[df["_ts"].isin(ts)].drop_duplicates(subset="_ts").set_index("_ts")[["year", "month", "day", "hour"]].loc[ts]
    frames, categorical = [], []
    for st in stations:
        part = df[(df["station"] == st) & (df["_ts"].isin(ts))].set_index("_ts")[meteo].reindex(ts)
        part.columns = [f"{st}__{c}" for c in meteo]
        frames.append(part)
        if f"{st}__wd" in part.columns:
            categorical.append(f"{st}__wd")
    X = pd.concat([time_cols] + frames, axis=1).reset_index(drop=True)
    y = wide.loc[ts].reset_index(drop=True)
    y.columns = [f"pm25_{c}" for c in y.columns]
    return X, y, categorical


def load_online_retail(cfg, raw_dir):
    """UCI 502, both yearly sheets; the invoice date becomes five calendar features."""
    path = _raw_file(cfg, raw_dir)
    sheets = pd.read_excel(path, sheet_name=cfg["sheets"], engine="openpyxl")
    df = pd.concat([sheets[s] for s in cfg["sheets"]], ignore_index=True)
    dt = pd.to_datetime(df["InvoiceDate"])
    df["InvoiceYear"], df["InvoiceMonth"], df["InvoiceDay"] = dt.dt.year, dt.dt.month, dt.dt.day
    df["InvoiceHour"], df["InvoiceDayOfWeek"] = dt.dt.hour, dt.dt.dayofweek
    X, y, categorical = _split_targets(cfg, df)
    if int(X.isna().sum().sum()):
        raise ValueError("unexpected missing feature values")
    return X, y, categorical


def _drop_constant(X):
    return X.drop(columns=[c for c in X.columns if X[c].nunique(dropna=True) <= 1])


def load_bias_corr_temp(cfg, raw_dir):
    """UCI 514; rows without a target or a parseable date are dropped, `station` is categorical."""
    raw = zip_member(download_file(cfg["url"], raw_dir / cfg["file"]), cfg["zip_member"])
    df = pd.read_csv(io.BytesIO(raw))
    targets = list(cfg["target_columns"])
    df = df.dropna(subset=targets).reset_index(drop=True)
    parsed = pd.to_datetime(df["Date"], errors="coerce")
    df = df.loc[parsed.notna().to_numpy()].reset_index(drop=True)
    drop = set(targets) | {"Date"} | {c for c in df.columns if c == "index" or c.startswith("Unnamed:")}
    X = _drop_constant(df[[c for c in df.columns if c not in drop]].copy())
    y = df[targets].astype("float32")
    return X, y, [c for c in cfg["categorical_columns"] if c in X.columns]


def load_geographical_music(cfg, raw_dir):
    """UCI 315: (latitude, longitude) from 68 audio features; the file has no header."""
    raw = zip_member(download_file(cfg["url"], raw_dir / cfg["file"]), cfg["zip_member"])
    df = pd.read_csv(io.BytesIO(raw), header=None)
    targets = list(cfg["target_columns"])
    features = [f"audio_feature_{i + 1:03d}" for i in range(df.shape[1] - len(targets))]
    df.columns = features + targets
    df = df.dropna(subset=targets).reset_index(drop=True)
    return _drop_constant(df[features].copy()), df[targets].astype("float32"), []


COMMUNITIES_COLUMNS = [
    "communityname", "State", "countyCode", "communityCode", "fold", "pop", "perHoush", "pctBlack",
    "pctWhite", "pctAsian", "pctHisp", "pct12-21", "pct12-29", "pct16-24", "pct65up", "persUrban",
    "pctUrban", "medIncome", "pctWwage", "pctWfarm", "pctWdiv", "pctWsocsec", "pctPubAsst",
    "pctRetire", "medFamIncome", "perCapInc", "whitePerCap", "blackPerCap", "NAperCap",
    "asianPerCap", "otherPerCap", "hispPerCap", "persPoverty", "pctPoverty", "pctLowEdu",
    "pctNotHSgrad", "pctCollGrad", "pctUnemploy", "pctEmploy", "pctEmployMfg", "pctEmployProfServ",
    "pctOccupManu", "pctOccupMgmt", "pctMaleDivorc", "pctMaleNevMar", "pctFemDivorc", "pctAllDivorc",
    "persPerFam", "pct2Par", "pctKids2Par", "pctKids-4w2Par", "pct12-17w2Par", "pctWorkMom-6",
    "pctWorkMom-18", "kidsBornNevrMarr", "pctKidsBornNevrMarr", "numForeignBorn", "pctFgnImmig-3",
    "pctFgnImmig-5", "pctFgnImmig-8", "pctFgnImmig-10", "pctImmig-3", "pctImmig-5", "pctImmig-8",
    "pctImmig-10", "pctSpeakOnlyEng", "pctNotSpeakEng", "pctLargHousFam", "pctLargHous",
    "persPerOccupHous", "persPerOwnOccup", "persPerRenterOccup", "pctPersOwnOccup",
    "pctPopDenseHous", "pctSmallHousUnits", "medNumBedrm", "houseVacant", "pctHousOccup",
    "pctHousOwnerOccup", "pctVacantBoarded", "pctVacant6up", "medYrHousBuilt", "pctHousWOphone",
    "pctHousWOplumb", "ownHousLowQ", "ownHousMed", "ownHousUperQ", "ownHousQrange", "rentLowQ",
    "rentMed", "rentUpperQ", "rentQrange", "medGrossRent", "medRentpctHousInc", "medOwnCostpct",
    "medOwnCostPctWO", "persEmergShelt", "persHomeless", "pctForeignBorn", "pctBornStateResid",
    "pctSameHouse-5", "pctSameCounty-5", "pctSameState-5", "numPolice", "policePerPop",
    "policeField", "policeFieldPerPop", "policeCalls", "policCallPerPop", "policCallPerOffic",
    "policePerPop2", "racialMatch", "pctPolicWhite", "pctPolicBlack", "pctPolicHisp",
    "pctPolicAsian", "pctPolicMinority", "officDrugUnits", "numDiffDrugsSeiz", "policAveOT",
    "landArea", "popDensity", "pctUsePubTrans", "policCarsAvail", "policOperBudget",
    "pctPolicPatrol", "gangUnit", "pctOfficDrugUnit", "policBudgetPerPop", "murders",
    "murdPerPop", "rapes", "rapesPerPop", "robberies", "robbbPerPop", "assaults", "assaultPerPop",
    "burglaries", "burglPerPop", "larcenies", "larcPerPop", "autoTheft", "autoTheftPerPop",
    "arsons", "arsonsPerPop", "violentPerPop", "nonViolPerPop",
]


def load_communities_crime(cfg, raw_dir):
    """UCI 211 (no header, "?" missing): six per-capita crime rates; the other crime counts
    and the identifier columns are dropped from the features."""
    raw = zip_member(download_file(cfg["url"], raw_dir / cfg["file"]), cfg["zip_member"])
    df = pd.read_csv(io.BytesIO(raw), header=None, names=COMMUNITIES_COLUMNS, na_values="?", encoding="latin-1")
    return _split_targets(cfg, df)


def load_oes(cfg, raw_dir):
    """OES 1997 / 2010 from the OpenML-hosted ARFF file."""
    from scipy.io import arff

    data, _ = arff.loadarff(download_file(cfg["url"], raw_dir / cfg["file"]))
    df = pd.DataFrame(data).astype(float)
    targets = list(cfg["target_columns"])
    return df.drop(columns=targets), df[targets], []


QM8_CHARGE = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}
QM8_MAX_ATOMS = 26


def _coulomb_eigenvalues(symbols, coords, n_keep):
    z = np.array([QM8_CHARGE[s] for s in symbols], dtype=float)
    xyz = np.asarray(coords, dtype=float)
    dist = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        M = np.where(dist > 0, np.outer(z, z) / np.where(dist > 0, dist, 1.0), 0.0)
    np.fill_diagonal(M, 0.5 * z ** 2.4)
    ev = np.linalg.eigvalsh(M)[::-1]
    out = np.zeros(QM8_MAX_ATOMS)
    out[:len(ev)] = ev
    return out[:n_keep]


def _parse_xyz(text):
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        n = int(lines[i].split()[0])
        symbols, coords = [], []
        for k in range(i + 2, i + 2 + n):
            parts = lines[k].split()
            symbols.append(parts[0])
            coords.append([float(v.replace("*^", "e")) for v in parts[1:4]])
        yield symbols, coords
        i += n + 2


def load_qm8(cfg, raw_dir):
    """QM8: four excited-state quantities from element counts and Coulomb-matrix eigenvalues."""
    spec = download_file(cfg["url"], raw_dir / cfg["file"]).read_text(encoding="utf-8", errors="replace")
    xyz = download_file(cfg["xyz_url"], raw_dir / cfg["xyz_file"]).read_text(encoding="utf-8", errors="replace")
    rows = [l.split() for l in spec.splitlines() if l.strip() and not l.strip().startswith("#")]
    rows = [r for r in rows if len(r) == 17]
    Y = np.array([[float(v) for v in r[1:]] for r in rows])
    mols = list(_parse_xyz(xyz))
    if len(mols) != len(rows):
        raise ValueError(f"{len(rows)} spectra rows but {len(mols)} geometries")
    n_keep = int(cfg["n_eigenvalues"])
    eig = np.vstack([_coulomb_eigenvalues(s, c, n_keep) for s, c in mols])
    comp = np.array([[sum(1 for a in s if a == e) for e in ("C", "H", "N", "O", "F")] + [len(s)] for s, _ in mols], dtype=float)
    X = pd.DataFrame(np.hstack([comp, eig]),
                     columns=["n_C", "n_H", "n_N", "n_O", "n_F", "n_atoms"] + [f"cm_eig_{i:02d}" for i in range(eig.shape[1])])
    col = {name: i for i, name in enumerate(cfg["all_target_columns"])}
    y = pd.DataFrame({c: Y[:, col[c]] for c in cfg["target_columns"]})
    return X, y, []


QM9_PROPERTIES = ["mu", "alpha", "homo", "lumo", "r2", "zpve", "Cv", "U0", "gap", "U", "H", "G"]


def load_qm9(cfg, raw_dir):
    """QM9: the first `target_dim` properties (in the order above); all 12 leave the features."""
    df = pd.read_csv(raw_dir / cfg["file"])
    targets = QM9_PROPERTIES[:int(cfg["target_dim"])]
    present = [c for c in QM9_PROPERTIES if c in df.columns]
    return df[[c for c in df.columns if c not in present]].copy(), df[targets].astype("float64"), []


def load_casp(cfg, raw_dir):
    """The processed CASP arrays (npz with X and Y)."""
    with np.load(raw_dir / cfg["file"], allow_pickle=False) as z:
        X, Y = np.asarray(z["X"], dtype=float), np.asarray(z["Y"], dtype=float)
    return (pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])]),
            pd.DataFrame(Y, columns=[f"y{i}" for i in range(Y.shape[1])]), [])


def load_argoverse2(cfg, raw_dir):
    """Future displacements of the focal agent from summaries of its history. The future_*
    columns that are not targets of this variant are removed from the features."""
    df = pd.read_csv(raw_dir / cfg["file"])
    targets = list(cfg["target_columns"])
    df = df.dropna(subset=targets).reset_index(drop=True)
    drop = set(targets) | set(cfg["drop_columns"]) | {c for c in df.columns if c == "index" or c.startswith("Unnamed:")}
    drop |= {c for c in df.columns if c.startswith("future_") and c not in targets}
    X = _drop_constant(df[[c for c in df.columns if c not in drop]].copy())
    y = df[targets].astype("float32")
    return X, y, [c for c in cfg["categorical_columns"] if c in X.columns]


def load_sarcos(cfg, raw_dir):
    """SARCOS inverse dynamics: 7 joint torques from the 21 state variables; the shipped
    train and test arrays are concatenated."""
    from scipy.io import loadmat

    parts = [np.asarray(loadmat(raw_dir / "sarcos" / f)[k], dtype=float)
             for f, k in (("sarcos_inv.mat", "sarcos_inv"), ("sarcos_inv_test.mat", "sarcos_inv_test"))]
    data = np.vstack(parts)
    joints = [f"joint{i + 1}" for i in range(7)]
    xcols = [f"{q}_{j}" for q in ("position", "velocity", "acceleration") for j in joints]
    X = pd.DataFrame(data[:, :21], columns=xcols)
    y = pd.DataFrame(data[:, 21:28], columns=[f"torque_{j}" for j in joints])
    return X, y, []


def load_csv(cfg, raw_dir):
    """A plain CSV shipped in data/raw with the targets named in the config."""
    return _split_targets(cfg, pd.read_csv(raw_dir / cfg["file"]))


LOADERS = {
    "table": load_table,
    "household_power": load_household_power,
    "openml": load_openml,
    "beijing_aq_6d": load_beijing_aq_6d,
    "beijing_aq_12d": load_beijing_aq_12d,
    "online_retail": load_online_retail,
    "bias_corr_temp": load_bias_corr_temp,
    "geographical_music": load_geographical_music,
    "communities_crime": load_communities_crime,
    "oes": load_oes,
    "qm8": load_qm8,
    "qm9": load_qm9,
    "casp": load_casp,
    "argoverse2": load_argoverse2,
    "sarcos": load_sarcos,
    "csv": load_csv,
}
