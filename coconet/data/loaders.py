"""Loaders for the datasets of the paper. `load_dataset` reads configs/datasets.yaml, calls the
loader named by the entry's `loader` key and applies the row cap (a seeded random sample
without replacement). Every loader returns raw, unpreprocessed features X (DataFrame), the
target y (Series at d = 1, DataFrame otherwise) and the list of categorical columns.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from coconet.data.download import DownloadError, download_and_extract_zip, download_file
from coconet.utils import get_logger

logger = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]


class DatasetLoadError(RuntimeError):
    pass


@dataclass
class DatasetBundle:
    name: str
    X: pd.DataFrame
    y: pd.Series | pd.DataFrame
    categorical_columns: list[str]
    log1p_target_available: bool
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.X) != len(self.y):
            raise DatasetLoadError(
                f"{self.name}: X has {len(self.X)} rows but y has {len(self.y)} rows"
            )
        if len(self.X) == 0:
            raise DatasetLoadError(f"{self.name}: loaded zero rows")
        n_missing_y = int(np.asarray(self.y.isna()).sum())
        if n_missing_y > 0:
            raise DatasetLoadError(f"{self.name}: target contains {n_missing_y} missing values")
        for col in self.categorical_columns:
            if col not in self.X.columns:
                raise DatasetLoadError(f"{self.name}: categorical column '{col}' not found in X")


def load_dataset(
    name: str,
    datasets_config: dict[str, Any],
    raw_dir: Path,
    max_rows: int | None = None,
    subsample_seed: int | None = None,
) -> DatasetBundle:
    """Load `name` and, when it has more than `max_rows` rows, keep a seeded random sample of
    `max_rows` rows (without replacement, in file order)."""
    if name not in datasets_config:
        raise DatasetLoadError(f"No configuration entry for dataset '{name}' in datasets.yaml")
    cfg = datasets_config[name]
    raw_dir = Path(raw_dir)
    loader = LOADERS.get(cfg.get("loader"))
    if loader is None:
        raise DatasetLoadError(f"{name}: unknown loader {cfg.get('loader')!r}; known: {sorted(LOADERS)}")

    bundle = loader(name, cfg, raw_dir)

    if max_rows is not None and len(bundle.X) > max_rows:
        if subsample_seed is None:
            raise DatasetLoadError(f"{name}: a row cap needs subsample_seed")
        rng = np.random.default_rng(int(subsample_seed))
        positions = np.sort(rng.choice(len(bundle.X), size=max_rows, replace=False))
        idx = bundle.X.index[positions]
        logger.info("[row-cap] seeded random sample (seed=%d, without replacement) of %s: %d -> %d rows",
                    int(subsample_seed), name, len(bundle.X), max_rows)
        bundle.X = bundle.X.loc[idx].reset_index(drop=True)
        bundle.y = bundle.y.loc[idx].reset_index(drop=True)
        if bundle.meta and bundle.meta.get("dates") is not None:
            bundle.meta["dates"] = np.asarray(bundle.meta["dates"])[positions]

    return bundle


def _check_targets(name: str, cfg: dict, df: pd.DataFrame) -> list[str]:
    target_columns = list(cfg.get("target_columns", []))
    missing = [c for c in target_columns if c not in df.columns]
    if missing:
        raise DatasetLoadError(f"{name}: target columns {missing} not found. Columns: {list(df.columns)}")
    output_dim = cfg.get("output_dim")
    if output_dim is not None and len(target_columns) != output_dim:
        raise DatasetLoadError(f"{name}: output_dim={output_dim} does not match len(target_columns)={len(target_columns)}")
    return target_columns


# --------------------------------------------------------------------------
# univariate
# --------------------------------------------------------------------------
def _load_concrete(name: str, cfg: dict[str, Any], raw_dir: Path) -> DatasetBundle:
    dest = raw_dir / cfg["raw_filename"]
    try:
        download_file(cfg["url"], dest)
        df = pd.read_excel(dest, engine="xlrd" if dest.suffix == ".xls" else None)
    except (DownloadError, Exception) as exc:
        raise DatasetLoadError(f"{name}: failed to load - {exc}") from exc

    target_col = cfg["target_column"]
    if target_col not in df.columns:
        raise DatasetLoadError(
            f"{name}: expected target column '{target_col}' not found. Columns: {list(df.columns)}"
        )
    y = df[target_col].astype(float)
    X = df.drop(columns=[target_col] + cfg.get("drop_columns", []))
    return DatasetBundle(
        name=name, X=X, y=y, categorical_columns=cfg.get("categorical_columns", []),
        log1p_target_available=cfg.get("log1p_target", False),
        meta={"source_url": cfg["url"]},
    )


def _load_ccpp(name: str, cfg: dict[str, Any], raw_dir: Path) -> DatasetBundle:
    """Only Sheet1 is read: the workbook ships 5 shuffled copies of the same 9568 rows."""
    try:
        extracted = download_and_extract_zip(cfg["url"], raw_dir, cfg["zip_inner_file"])
        df = pd.read_excel(extracted, sheet_name=cfg.get("sheet_name", "Sheet1"))
    except (DownloadError, Exception) as exc:
        raise DatasetLoadError(f"{name}: failed to load - {exc}") from exc

    target_col = cfg["target_column"]
    if target_col not in df.columns:
        raise DatasetLoadError(
            f"{name}: expected target column '{target_col}' not found. Columns: {list(df.columns)}"
        )
    y = df[target_col].astype(float)
    X = df.drop(columns=[target_col] + cfg.get("drop_columns", []))
    return DatasetBundle(
        name=name, X=X, y=y, categorical_columns=cfg.get("categorical_columns", []),
        log1p_target_available=cfg.get("log1p_target", False),
        meta={"source_url": cfg["url"], "sheet_name": cfg.get("sheet_name", "Sheet1")},
    )


def _load_beijing_pm25(name: str, cfg: dict[str, Any], raw_dir: Path) -> DatasetBundle:
    """PM2.5 from the concurrent meteorological readings; rows without a reading are dropped."""
    try:
        extracted = download_and_extract_zip(cfg["url"], raw_dir, cfg["zip_inner_file"])
        df = pd.read_csv(extracted, na_values=["NA"])
    except (DownloadError, Exception) as exc:
        raise DatasetLoadError(f"{name}: failed to load - {exc}") from exc

    target_col = cfg["target_column"]
    if target_col not in df.columns:
        raise DatasetLoadError(
            f"{name}: expected target column '{target_col}' not found. Columns: {list(df.columns)}"
        )
    n_before = len(df)
    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    n_dropped = n_before - len(df)

    y = df[target_col].astype(float)
    drop_cols = [target_col] + cfg.get("drop_columns", [])
    X = df.drop(columns=drop_cols)
    return DatasetBundle(
        name=name, X=X, y=y, categorical_columns=cfg.get("categorical_columns", []),
        log1p_target_available=cfg.get("log1p_target", False),
        meta={"source_url": cfg["url"], "n_rows_dropped_missing_target": int(n_dropped)},
    )


def _load_openml_univariate(name: str, cfg: dict[str, Any], raw_dir: Path) -> DatasetBundle:
    from sklearn.datasets import fetch_openml

    cache_dir = raw_dir / "openml_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        data = fetch_openml(data_id=cfg["openml_data_id"], as_frame=True, data_home=str(cache_dir))
    except Exception as exc:
        raise DatasetLoadError(f"{name}: fetch_openml failed - {exc}") from exc

    X = data.data.copy()
    y = data.target.astype(float)
    y.name = cfg["target_column"]
    return DatasetBundle(
        name=name, X=X, y=y, categorical_columns=cfg.get("categorical_columns", []),
        log1p_target_available=cfg.get("log1p_target", False),
        meta={"source": "openml", "data_id": cfg["openml_data_id"]},
    )


def _load_bike_sharing(name: str, cfg: dict[str, Any], raw_dir: Path) -> DatasetBundle:
    try:
        zip_path = raw_dir / Path(cfg["url"]).name
        download_file(cfg["url"], zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            match = [n for n in names if Path(n).name == cfg["zip_inner_file"]]
            if not match:
                raise DatasetLoadError(f"{name}: '{cfg['zip_inner_file']}' not in archive: {names}")
            with zf.open(match[0]) as f:
                df = pd.read_csv(io.BytesIO(f.read()))
    except (DownloadError, zipfile.BadZipFile) as exc:
        raise DatasetLoadError(f"{name}: failed to load - {exc}") from exc

    target_col = cfg["target_column"]
    drop_cols = [c for c in cfg.get("drop_columns", []) if c in df.columns]
    date_col = cfg.get("date_column")
    date_series = df[date_col].copy() if date_col and date_col in df.columns else None

    if target_col not in df.columns:
        raise DatasetLoadError(f"{name}: target column '{target_col}' missing")
    y = df[target_col].astype(float)
    X = df.drop(columns=[target_col] + drop_cols)

    meta: dict[str, Any] = {}
    if date_series is not None:
        meta["date_column_preserved"] = date_series

    return DatasetBundle(
        name=name, X=X, y=y, categorical_columns=cfg.get("categorical_columns", []),
        log1p_target_available=cfg.get("log1p_target", False),
        meta=meta,
    )


def _load_protein(name: str, cfg: dict[str, Any], raw_dir: Path) -> DatasetBundle:
    dest = raw_dir / cfg["raw_filename"]
    try:
        download_file(cfg["url"], dest)
        df = pd.read_csv(dest)
    except (DownloadError, Exception) as exc:
        raise DatasetLoadError(f"{name}: failed to load - {exc}") from exc

    target_col = cfg["target_column"]
    if target_col not in df.columns:
        raise DatasetLoadError(f"{name}: target column '{target_col}' missing. Columns: {list(df.columns)}")

    expected_n = cfg.get("expected_n_rows")
    if expected_n is not None and len(df) != expected_n:
        logger.warning(
            "%s: expected %d rows per UCI documentation, found %d", name, expected_n, len(df)
        )

    y = df[target_col].astype(float)
    X = df.drop(columns=[target_col])
    return DatasetBundle(
        name=name, X=X, y=y, categorical_columns=cfg.get("categorical_columns", []),
        log1p_target_available=cfg.get("log1p_target", False),
        meta={"expected_n_rows": expected_n, "actual_n_rows": len(df)},
    )


# --------------------------------------------------------------------------
# multivariate: UCI zips
# --------------------------------------------------------------------------
def _load_sgemm(name: str, cfg: dict[str, Any], raw_dir: Path) -> DatasetBundle:
    """4 repeated timings of the same GPU kernel configuration as a 4-D target."""
    try:
        extracted = download_and_extract_zip(cfg["url"], raw_dir, cfg["zip_inner_file"])
        df = pd.read_csv(extracted)
    except (DownloadError, Exception) as exc:
        raise DatasetLoadError(f"{name}: failed to load - {exc}") from exc

    target_columns = _check_targets(name, cfg, df)
    y = df[target_columns].astype(float)
    X = df.drop(columns=target_columns + cfg.get("drop_columns", []))
    return DatasetBundle(
        name=name, X=X, y=y, categorical_columns=cfg.get("categorical_columns", []),
        log1p_target_available=cfg.get("log1p_target", False),
        meta={"source_url": cfg["url"], "output_dim": len(target_columns)},
    )


def _load_household_power(name: str, cfg: dict[str, Any], raw_dir: Path) -> DatasetBundle:
    """3 sub-metering readings from the 4 concurrent whole-house quantities; "?" is missing."""
    try:
        extracted = download_and_extract_zip(cfg["url"], raw_dir, cfg["zip_inner_file"])
        df = pd.read_csv(extracted, sep=";", na_values=cfg.get("missing_value_marker", "?"), low_memory=False)
    except (DownloadError, Exception) as exc:
        raise DatasetLoadError(f"{name}: failed to load - {exc}") from exc

    target_columns = _check_targets(name, cfg, df)
    drop_cols = target_columns + cfg.get("drop_columns", [])
    y = df[target_columns].astype(float)
    X = df.drop(columns=drop_cols).astype(float)
    keep = ~(X.isna().all(axis=1) & y.isna().all(axis=1))
    X, y = X.loc[keep].reset_index(drop=True), y.loc[keep].reset_index(drop=True)
    return DatasetBundle(
        name=name, X=X, y=y, categorical_columns=cfg.get("categorical_columns", []),
        log1p_target_available=cfg.get("log1p_target", False),
        meta={"source_url": cfg["url"], "output_dim": len(target_columns)},
    )


def _load_beijing_multisite_raw(cfg: dict[str, Any], raw_dir: Path, name: str) -> pd.DataFrame:
    """The 12 per-station CSVs of UCI 501, concatenated, with a '_ts' timestamp column."""
    outer_dest = raw_dir / cfg["raw_filename"]
    try:
        download_file(cfg["url"], outer_dest)
        with zipfile.ZipFile(outer_dest) as outer_zf:
            nested_name = cfg["nested_zip_name"]
            inner_matches = [n for n in outer_zf.namelist() if Path(n).name == nested_name]
            if not inner_matches:
                raise DatasetLoadError(
                    f"{name}: nested zip '{nested_name}' not found in archive. "
                    f"Available members: {outer_zf.namelist()}"
                )
            nested_bytes = outer_zf.read(inner_matches[0])
        with zipfile.ZipFile(io.BytesIO(nested_bytes)) as inner_zf:
            csv_names = sorted(n for n in inner_zf.namelist() if n.endswith(".csv"))
            expected_n = cfg.get("expected_n_stations", 12)
            if len(csv_names) != expected_n:
                raise DatasetLoadError(
                    f"{name}: expected {expected_n} station CSVs in the nested zip, found "
                    f"{len(csv_names)}: {csv_names}"
                )
            station_frames = [pd.read_csv(inner_zf.open(csv_name)) for csv_name in csv_names]
    except (DownloadError, zipfile.BadZipFile) as exc:
        raise DatasetLoadError(f"{name}: failed to load - {exc}") from exc

    df = pd.concat(station_frames, ignore_index=True)
    df["_ts"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
    return df


def _load_beijing_aq_6d(name: str, cfg: dict[str, Any], raw_dir: Path) -> DatasetBundle:
    """One row per (station, hour): the 6 pollutants from the concurrent meteorology."""
    df = _load_beijing_multisite_raw(cfg, raw_dir, name)

    target_columns = _check_targets(name, cfg, df)
    n_missing_target = int(df[target_columns].isna().any(axis=1).sum())
    if n_missing_target:
        logger.info("%s: dropping %d row(s) missing at least one of %s", name, n_missing_target, target_columns)
        df = df.dropna(subset=target_columns).reset_index(drop=True)

    dates = df["_ts"].to_numpy()
    y = df[target_columns].astype(float)
    drop_cols = target_columns + cfg.get("drop_columns", []) + ["_ts"]
    X = df.drop(columns=drop_cols)
    return DatasetBundle(
        name=name, X=X, y=y, categorical_columns=cfg.get("categorical_columns", []),
        log1p_target_available=cfg.get("log1p_target", False),
        meta={"source_url": cfg["url"], "output_dim": len(target_columns),
              "n_rows_dropped_missing_target": n_missing_target, "dates": dates,
              "date_column": None},
    )


def _load_beijing_aq_12d(name: str, cfg: dict[str, Any], raw_dir: Path) -> DatasetBundle:
    """One row per timestamp: the 12 stations' PM2.5 from every station's meteorology."""
    df = _load_beijing_multisite_raw(cfg, raw_dir, name)
    target_col = cfg["pm25_column"]
    meteo_cols = list(cfg["meteo_columns"])
    stations = sorted(df["station"].unique())
    expected_n = cfg.get("expected_n_stations", 12)
    if len(stations) != expected_n:
        raise DatasetLoadError(f"{name}: expected {expected_n} stations, found {len(stations)}: {stations}")

    pm25_wide = df.pivot_table(index="_ts", columns="station", values=target_col, aggfunc="first")
    pm25_wide = pm25_wide.dropna(axis=0, how="any")  # keep only timestamps valid at EVERY station
    keep_ts = pm25_wide.index

    time_cols = df[df["_ts"].isin(keep_ts)].drop_duplicates(subset="_ts").set_index("_ts")[
        ["year", "month", "day", "hour"]].loc[keep_ts]

    meteo_frames = []
    categorical_columns = []
    for station in stations:
        station_df = df[(df["station"] == station) & (df["_ts"].isin(keep_ts))].set_index("_ts")
        station_meteo = station_df[meteo_cols].reindex(keep_ts)
        station_meteo.columns = [f"{station}__{c}" for c in meteo_cols]
        meteo_frames.append(station_meteo)
        wd_col = f"{station}__wd"
        if wd_col in station_meteo.columns:
            categorical_columns.append(wd_col)

    X = pd.concat([time_cols] + meteo_frames, axis=1).reset_index(drop=True)
    y = pm25_wide.loc[keep_ts].reset_index(drop=True)
    y.columns = [f"pm25_{c}" for c in y.columns]
    dates = np.asarray(keep_ts)

    output_dim = cfg.get("output_dim")
    if output_dim is not None and y.shape[1] != output_dim:
        raise DatasetLoadError(f"{name}: output_dim={output_dim} does not match pivoted station count {y.shape[1]}")

    n_timestamps_total = df["_ts"].nunique()
    return DatasetBundle(
        name=name, X=X, y=y, categorical_columns=categorical_columns,
        log1p_target_available=cfg.get("log1p_target", False),
        meta={"source_url": cfg["url"], "output_dim": y.shape[1], "n_timestamps_kept": len(keep_ts),
              "n_timestamps_dropped": n_timestamps_total - len(keep_ts),
              "stations": stations, "dates": dates, "date_column": None},
    )


def _load_online_retail_ii(name: str, cfg: dict[str, Any], raw_dir: Path) -> DatasetBundle:
    """(Quantity, Price) from Country and 5 calendar parts of the invoice date; both sheets."""
    try:
        extracted = download_and_extract_zip(cfg["url"], raw_dir, cfg["zip_inner_file"])
        sheet_names = cfg["sheet_names"]
        sheets = pd.read_excel(extracted, sheet_name=sheet_names, engine="openpyxl")
        df = pd.concat([sheets[s] for s in sheet_names], ignore_index=True)
    except (DownloadError, Exception) as exc:
        raise DatasetLoadError(f"{name}: failed to load - {exc}") from exc

    target_columns = _check_targets(name, cfg, df)

    date_col = "InvoiceDate"
    if date_col not in df.columns:
        raise DatasetLoadError(f"{name}: expected date column '{date_col}' not found. Columns: {list(df.columns)}")
    dt = pd.to_datetime(df[date_col])
    df["InvoiceYear"] = dt.dt.year
    df["InvoiceMonth"] = dt.dt.month
    df["InvoiceDay"] = dt.dt.day
    df["InvoiceHour"] = dt.dt.hour
    df["InvoiceDayOfWeek"] = dt.dt.dayofweek

    n_missing_target = int(df[target_columns].isna().any(axis=1).sum())
    if n_missing_target:
        logger.info("%s: dropping %d row(s) missing at least one of %s", name, n_missing_target, target_columns)
        df = df.dropna(subset=target_columns).reset_index(drop=True)

    y = df[target_columns].astype(float)
    drop_columns = [c for c in cfg.get("drop_columns", []) if c in df.columns]
    X = df.drop(columns=target_columns + drop_columns)
    n_missing_feature = int(X.isna().sum().sum())
    if n_missing_feature:
        raise DatasetLoadError(
            f"{name}: unexpected {n_missing_feature} missing feature value(s) after dropping the "
            f"documented-missing columns -- investigate before silently imputing"
        )

    return DatasetBundle(
        name=name, X=X, y=y, categorical_columns=cfg.get("categorical_columns", []),
        log1p_target_available=cfg.get("log1p_target", False),
        meta={"source_url": cfg["url"], "output_dim": len(target_columns),
              "n_rows_dropped_missing_target": n_missing_target, "sheets_used": sheet_names},
    )


def _load_bias_correction_temperature(name: str, cfg: dict[str, Any], raw_dir: Path) -> DatasetBundle:
    """Next-day (Tmax, Tmin) from the file's own 23 predictors; rows without a target or a
    parseable date are dropped, `station` is categorical."""
    target_columns = list(cfg.get("target_columns", ["Next_Tmax", "Next_Tmin"]))
    date_column = cfg.get("date_column", "Date")

    local = cfg.get("data_path")
    if local:
        csv_path = Path(local)
        if not csv_path.exists():
            raise DatasetLoadError(f"{name}: data_path '{csv_path}' does not exist")
        df = pd.read_csv(csv_path)
    else:
        try:
            zip_path = raw_dir / cfg.get("raw_filename", Path(cfg["url"]).name)
            download_file(cfg["url"], zip_path)
            with zipfile.ZipFile(zip_path) as zf:
                names = zf.namelist()
                match = [n for n in names if Path(n).name == cfg["zip_inner_file"]]
                if not match:
                    raise DatasetLoadError(f"{name}: '{cfg['zip_inner_file']}' not in archive: {names}")
                with zf.open(match[0]) as fh:
                    df = pd.read_csv(io.BytesIO(fh.read()))
        except (DownloadError, zipfile.BadZipFile) as exc:
            raise DatasetLoadError(f"{name}: failed to load - {exc}") from exc

    missing_targets = [c for c in target_columns if c not in df.columns]
    if missing_targets:
        raise DatasetLoadError(f"{name}: target column(s) {missing_targets} not found")

    n_before = len(df)
    df = df.dropna(subset=target_columns).reset_index(drop=True)
    n_dropped = n_before - len(df)

    if date_column not in df.columns:
        raise DatasetLoadError(f"{name}: date column '{date_column}' not found")
    parsed = pd.to_datetime(df[date_column], errors="coerce")
    n_bad_date = int(parsed.isna().sum())
    if n_bad_date:
        keep = parsed.notna().to_numpy()
        df = df.loc[keep].reset_index(drop=True)
        parsed = parsed.loc[keep].reset_index(drop=True)
    dates = parsed

    index_like = [c for c in df.columns if c in ("Unnamed: 0", "index") or c.startswith("Unnamed:")]
    drop_cols = set(target_columns) | {date_column} | set(index_like) | set(cfg.get("drop_columns", []))
    feature_columns = [c for c in df.columns if c not in drop_cols]

    X = df[feature_columns].copy()
    constant_cols = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    if constant_cols:
        X = X.drop(columns=constant_cols)
    y = df[target_columns].astype("float32")

    categorical_columns = [c for c in cfg.get("categorical_columns", []) if c in X.columns]

    return DatasetBundle(
        name=name, X=X, y=y, categorical_columns=categorical_columns,
        log1p_target_available=bool(cfg.get("log1p_target", False)),
        meta={
            "dates": dates, "date_column": date_column, "target_columns": target_columns,
            "n_rows_dropped_missing_target": int(n_dropped),
            "n_rows_dropped_missing_date": n_bad_date,
            "constant_columns_dropped": constant_cols,
            "original_feature_names": list(X.columns),
            "n_stations": int(df["station"].nunique()) if "station" in df.columns else None,
            "n_unique_dates": int(dates.nunique()),
        },
    )


def _load_geographical_music(name: str, cfg: dict[str, Any], raw_dir: Path) -> DatasetBundle:
    """(latitude, longitude) from 68 audio features; the file has no header."""
    target_columns = list(cfg.get("target_columns", ["latitude", "longitude"]))
    if len(target_columns) != 2:
        raise DatasetLoadError(f"{name}: expected exactly 2 target columns, got {target_columns}")

    local = cfg.get("data_path")
    if local:
        txt_path = Path(local)
        if not txt_path.exists():
            raise DatasetLoadError(f"{name}: data_path '{txt_path}' does not exist")
        raw = pd.read_csv(txt_path, header=None)
    else:
        try:
            zip_path = raw_dir / cfg.get("raw_filename", Path(cfg["url"]).name)
            download_file(cfg["url"], zip_path)
            with zipfile.ZipFile(zip_path) as zf:
                names = [n for n in zf.namelist() if not n.startswith("__MACOSX")]
                match = [n for n in names if Path(n).name == cfg["zip_inner_file"]]
                if not match:
                    raise DatasetLoadError(
                        f"{name}: '{cfg['zip_inner_file']}' not in archive: {names}")
                with zf.open(match[0]) as fh:
                    raw = pd.read_csv(io.BytesIO(fh.read()), header=None)
        except (DownloadError, zipfile.BadZipFile) as exc:
            raise DatasetLoadError(f"{name}: failed to load - {exc}") from exc

    if raw.shape[1] < 3:
        raise DatasetLoadError(f"{name}: expected >=3 columns, got shape {raw.shape}")

    n_features = raw.shape[1] - len(target_columns)
    feature_names = [f"audio_feature_{i + 1:03d}" for i in range(n_features)]
    raw.columns = feature_names + target_columns

    n_before = len(raw)
    raw = raw.dropna(subset=target_columns).reset_index(drop=True)
    n_dropped = n_before - len(raw)

    X = raw[feature_names].copy()
    constant_cols = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    if constant_cols:
        X = X.drop(columns=constant_cols)
    y = raw[target_columns].astype("float32")

    return DatasetBundle(
        name=name, X=X, y=y, categorical_columns=[],
        log1p_target_available=bool(cfg.get("log1p_target", False)),
        meta={
            "dates": None, "date_column": None, "target_columns": target_columns,
            "n_rows_dropped_missing_target": int(n_dropped),
            "n_rows_dropped_missing_date": 0,
            "constant_columns_dropped": constant_cols,
            "original_feature_names": list(X.columns),
            "n_stations": None, "n_unique_dates": None,
        },
    )


_COMMUNITIES_CRIME_UNNORM_COLUMNS = [
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

_COMMUNITIES_GOAL_VARIABLES = [
    "murders", "murdPerPop", "rapes", "rapesPerPop", "robberies", "robbbPerPop", "assaults",
    "assaultPerPop", "burglaries", "burglPerPop", "larcenies", "larcPerPop", "autoTheft",
    "autoTheftPerPop", "arsons", "arsonsPerPop", "violentPerPop", "nonViolPerPop",
]


def _load_communities_crime_unnormalized(name, cfg, raw_dir):
    """UCI 211 (no header, "?" missing): the six per-capita crime rates as targets; every
    other goal variable and the identifier columns are dropped from the features."""
    dest = raw_dir / cfg["raw_filename"]
    try:
        download_file(cfg["url"], dest)
        if cfg.get("zip_inner_file"):
            with zipfile.ZipFile(dest) as z:
                with z.open(cfg["zip_inner_file"]) as fh:
                    df = pd.read_csv(fh, header=None,
                                     names=_COMMUNITIES_CRIME_UNNORM_COLUMNS,
                                     na_values=cfg.get("missing_value_marker", "?"),
                                     encoding="latin-1")
        else:
            df = pd.read_csv(dest, header=None, names=_COMMUNITIES_CRIME_UNNORM_COLUMNS,
                             na_values=cfg.get("missing_value_marker", "?"), encoding="latin-1")
    except (DownloadError, Exception) as exc:
        raise DatasetLoadError(f"{name}: failed to load - {exc}") from exc

    if df.shape[1] != len(_COMMUNITIES_CRIME_UNNORM_COLUMNS):
        raise DatasetLoadError(
            f"{name}: expected {len(_COMMUNITIES_CRIME_UNNORM_COLUMNS)} columns, got {df.shape[1]}")

    target_columns = list(cfg.get("target_columns", []))
    missing = [c for c in target_columns if c not in df.columns]
    if missing:
        raise DatasetLoadError(f"{name}: target columns missing: {missing}")
    output_dim = cfg.get("output_dim")
    if output_dim is not None and len(target_columns) != output_dim:
        raise DatasetLoadError(
            f"{name}: output_dim={output_dim} != len(target_columns)={len(target_columns)}")

    n_before = len(df)
    df = df.dropna(subset=target_columns).reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped:
        logger.info("%s: dropping %d row(s) with a missing target", name, n_dropped)

    drop_cols = [c for c in cfg.get("drop_columns", []) if c in df.columns]
    leaked = set(target_columns) & set(drop_cols)
    if leaked:
        raise DatasetLoadError(f"{name}: {sorted(leaked)} listed as both target and dropped")
    residual_targets = [c for c in _COMMUNITIES_GOAL_VARIABLES
                        if c not in target_columns and c not in drop_cols]
    if residual_targets:
        raise DatasetLoadError(
            f"{name}: goal variables {residual_targets} would be used as FEATURES; "
            "add them to drop_columns")

    y = df[target_columns].astype(float)
    X = df.drop(columns=target_columns + drop_cols)
    return DatasetBundle(
        name=name, X=X, y=y, categorical_columns=cfg.get("categorical_columns", []),
        log1p_target_available=cfg.get("log1p_target", False),
        meta={"source_url": cfg["url"], "output_dim": len(target_columns),
              "n_rows_dropped_missing_target": n_dropped},
    )


# --------------------------------------------------------------------------
# multivariate: OpenML / ARFF
# --------------------------------------------------------------------------
def _load_openml_multi_target(name: str, cfg: dict[str, Any], raw_dir: Path) -> DatasetBundle:
    from sklearn.datasets import fetch_openml

    target_columns = list(cfg.get("target_columns", []))
    if not target_columns:
        raise DatasetLoadError(f"{name}: target_columns must be a non-empty list in configs/datasets.yaml")
    expected_output_dim = cfg.get("output_dim")
    if expected_output_dim is not None and len(target_columns) != expected_output_dim:
        raise DatasetLoadError(
            f"{name}: configured output_dim={expected_output_dim} does not match "
            f"len(target_columns)={len(target_columns)}"
        )

    cache_dir = raw_dir / "openml_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        data = fetch_openml(data_id=cfg["openml_data_id"], as_frame=True, data_home=str(cache_dir))
    except Exception as exc:
        raise DatasetLoadError(f"{name}: fetch_openml failed - {exc}") from exc

    y = data.target.copy()
    if list(y.columns) != target_columns:
        raise DatasetLoadError(
            f"{name}: openml target columns {list(y.columns)} do not match configured "
            f"target_columns {target_columns} -- update configs/datasets.yaml if the openml "
            f"dataset version has changed column names/order"
        )
    y = y.astype(float)

    categorical_cols = set(cfg.get("categorical_columns", []))
    X = data.data.copy()
    numeric_cols = [c for c in X.columns if c not in categorical_cols]
    X[numeric_cols] = X[numeric_cols].astype(float)

    return DatasetBundle(
        name=name, X=X, y=y, categorical_columns=cfg.get("categorical_columns", []),
        log1p_target_available=cfg.get("log1p_target", False),
        meta={"source": "openml", "data_id": cfg["openml_data_id"], "output_dim": len(target_columns)},
    )


def _load_oes(name: str, cfg: dict[str, Any], raw_dir: Path) -> DatasetBundle:
    """oes97 / oes10 from the OpenML-hosted ARFF file (fetch_openml's metadata is broken)."""
    from scipy.io import arff as scipy_arff

    target_columns = list(cfg.get("target_columns", []))
    if not target_columns:
        raise DatasetLoadError(f"{name}: target_columns must be a non-empty list in configs/datasets.yaml")
    try:
        dest = raw_dir / cfg["raw_filename"]
        download_file(cfg["url"], dest)
        data, _meta = scipy_arff.loadarff(dest)
        df = pd.DataFrame(data)
    except (DownloadError, Exception) as exc:
        raise DatasetLoadError(f"{name}: failed to load - {exc}") from exc

    _check_targets(name, cfg, df)
    df = df.astype(float)
    y = df[target_columns].astype(float)
    X = df.drop(columns=target_columns)
    n_missing = int(X.isna().sum().sum()) + int(y.isna().sum().sum())
    if n_missing:
        raise DatasetLoadError(f"{name}: unexpected {n_missing} missing value(s) in a documented-complete dataset")

    return DatasetBundle(
        name=name, X=X, y=y, categorical_columns=cfg.get("categorical_columns", []),
        log1p_target_available=cfg.get("log1p_target", False),
        meta={"source_url": cfg["url"], "openml_data_id": cfg.get("openml_data_id"),
              "output_dim": len(target_columns)},
    )


# --------------------------------------------------------------------------
# multivariate: molecules
# --------------------------------------------------------------------------
_QM8_Z = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}
_QM8_MAX_ATOMS = 26
_QM8_N_EIGENVALUES = 8


def _coulomb_eigenvalues(symbols, coords, max_atoms=_QM8_MAX_ATOMS, n_keep=None):
    """Sorted Coulomb-matrix eigenvalues (Rupp et al. 2012), zero-padded to max_atoms."""
    z = np.array([_QM8_Z[sym] for sym in symbols], dtype=float)
    xyz = np.asarray(coords, dtype=float)
    d = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        M = np.where(d > 0, np.outer(z, z) / np.where(d > 0, d, 1.0), 0.0)
    np.fill_diagonal(M, 0.5 * z ** 2.4)
    ev = np.linalg.eigvalsh(M)[::-1]
    out = np.zeros(max_atoms, dtype=float)
    out[:len(ev)] = ev
    return out if n_keep is None else out[:int(n_keep)]


def _parse_xyz_multi(text):
    """Yield (symbols, coords) per molecule from a concatenated XYZ file."""
    lines = text.splitlines()
    i, n_lines = 0, len(lines)
    while i < n_lines:
        if not lines[i].strip():
            i += 1
            continue
        n = int(lines[i].split()[0])
        syms, xyz = [], []
        for k in range(i + 2, i + 2 + n):
            parts = lines[k].split()
            syms.append(parts[0])
            xyz.append([float(v.replace("*^", "e")) for v in parts[1:4]])
        yield syms, xyz
        i += n + 2


def _load_qm8_excited_states(name, cfg, raw_dir):
    """QM8: the four RI-CC2 quantities (E1, E2, f1, f2) from composition + 8 Coulomb eigenvalues."""
    spec_dest = raw_dir / cfg["raw_filename"]
    xyz_dest = raw_dir / cfg["xyz_raw_filename"]
    try:
        download_file(cfg["url"], spec_dest)
        download_file(cfg["xyz_url"], xyz_dest)
        spec_text = spec_dest.read_text(encoding="utf-8", errors="replace")
        xyz_text = xyz_dest.read_text(encoding="utf-8", errors="replace")
    except (DownloadError, Exception) as exc:
        raise DatasetLoadError(f"{name}: failed to load - {exc}") from exc

    rows = [l.split() for l in spec_text.splitlines()
            if l.strip() and not l.strip().startswith("#")]
    rows = [r for r in rows if len(r) == 17]
    if not rows:
        raise DatasetLoadError(f"{name}: no 17-field data rows found in {spec_dest}")
    Y_all = np.array([[float(v) for v in r[1:]] for r in rows], dtype=float)

    mols = list(_parse_xyz_multi(xyz_text))
    if len(mols) != len(rows):
        raise DatasetLoadError(
            f"{name}: {len(rows)} spectra rows but {len(mols)} geometries -- the two files are "
            "aligned positionally, so a length mismatch means they cannot be paired safely")

    bad = sorted({sym for syms, _ in mols for sym in syms} - set(_QM8_Z))
    if bad:
        raise DatasetLoadError(f"{name}: unexpected element(s) {bad}; no nuclear charge defined")

    n_keep = int(cfg.get("n_eigenvalues", _QM8_N_EIGENVALUES))
    if not 1 <= n_keep <= _QM8_MAX_ATOMS:
        raise DatasetLoadError(
            f"{name}: n_eigenvalues must be in [1, {_QM8_MAX_ATOMS}], got {n_keep}")
    eig = np.vstack([_coulomb_eigenvalues(sy, xy, n_keep=n_keep) for sy, xy in mols])
    comp = np.array([[sum(1 for s in sy if s == e) for e in ("C", "H", "N", "O", "F")]
                     + [len(sy)] for sy, _ in mols], dtype=float)
    X = pd.DataFrame(
        np.hstack([comp, eig]),
        columns=["n_C", "n_H", "n_N", "n_O", "n_F", "n_atoms"]
        + [f"cm_eig_{i:02d}" for i in range(eig.shape[1])],
    )

    col_index = {n: i for i, n in enumerate(cfg["all_target_columns"])}
    target_columns = list(cfg.get("target_columns", []))
    missing = [c for c in target_columns if c not in col_index]
    if missing:
        raise DatasetLoadError(f"{name}: unknown target column(s) {missing}")
    output_dim = cfg.get("output_dim")
    if output_dim is not None and len(target_columns) != output_dim:
        raise DatasetLoadError(
            f"{name}: output_dim={output_dim} != len(target_columns)={len(target_columns)}")
    y = pd.DataFrame({c: Y_all[:, col_index[c]] for c in target_columns})

    return DatasetBundle(
        name=name, X=X, y=y, categorical_columns=cfg.get("categorical_columns", []),
        log1p_target_available=cfg.get("log1p_target", False),
        meta={"source_url": cfg["url"], "geometry_url": cfg["xyz_url"],
              "output_dim": len(target_columns), "n_molecules": len(mols),
              "representation": "sorted Coulomb-matrix eigenvalues + composition",
              "n_eigenvalues": n_keep},
    )


# The QM9 properties in one fixed order; qm9_dN takes the first N as its targets and drops all 12
# from the features.
QM9_CANONICAL_ORDER = ["mu", "alpha", "homo", "lumo", "r2", "zpve", "Cv", "U0", "gap", "U", "H", "G"]


def _load_qm9(name: str, cfg: dict[str, Any], raw_dir: Path) -> DatasetBundle:
    target_dim = int(cfg["target_dim"])
    if target_dim not in (2, 4, 6, 8, 10, 12):
        raise DatasetLoadError(f"{name}: target_dim must be one of (2,4,6,8,10,12), "
                               f"got {target_dim}")
    csv_path = Path(cfg["data_path"])
    if not csv_path.is_absolute():
        csv_path = REPO_ROOT / csv_path
    if not csv_path.exists():
        raise DatasetLoadError(f"{name}: {csv_path} not found")
    df = pd.read_csv(csv_path)

    targets = list(QM9_CANONICAL_ORDER[:target_dim])
    missing = [c for c in targets if c not in df.columns]
    if missing:
        raise DatasetLoadError(f"{name}: target column(s) {missing} not in {csv_path}")
    property_cols = [c for c in QM9_CANONICAL_ORDER if c in df.columns]
    X = df[[c for c in df.columns if c not in property_cols]].copy()
    y = df[targets].astype("float64")
    return DatasetBundle(
        name=name, X=X, y=y, categorical_columns=[], log1p_target_available=False,
        meta={"dates": None, "date_column": None, "target_columns": targets,
              "n_rows_dropped_missing_target": 0, "n_rows_dropped_missing_date": 0,
              "constant_columns_dropped": [], "original_feature_names": list(X.columns),
              "n_stations": None, "n_unique_dates": None,
              "canonical_order": list(QM9_CANONICAL_ORDER), "target_dim": target_dim,
              "properties_dropped_from_features": len(property_cols)},
    )


# --------------------------------------------------------------------------
# multivariate: prepared tables and arrays
# --------------------------------------------------------------------------
def _load_mvcs_npz(name, cfg, raw_dir):
    """The MVCS authors' processed (X, Y) arrays, taken verbatim."""
    path = Path(cfg["npz_path"])
    if not path.is_absolute():
        path = Path(raw_dir) / path
    if not path.exists():
        raise DatasetLoadError(f"{name}: MVCS npz not found at {path}")
    with np.load(path, allow_pickle=False) as z:
        if "X" not in z or "Y" not in z:
            raise DatasetLoadError(f"{name}: expected arrays 'X' and 'Y', found {list(z.keys())}")
        X = np.asarray(z["X"], dtype=float)
        Y = np.asarray(z["Y"], dtype=float)
    if len(X) != len(Y):
        raise DatasetLoadError(f"{name}: X has {len(X)} rows, Y has {len(Y)}")
    Xdf = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    ycols = [f"y{i}" for i in range(Y.shape[1])]
    ydf = pd.DataFrame(Y, columns=ycols)
    y = ydf.iloc[:, 0] if Y.shape[1] == 1 else ydf
    return DatasetBundle(name=name, X=Xdf, y=y, categorical_columns=[],
                         log1p_target_available=False,
                         meta={"source": "MVCS authors processed_data",
                               "npz_path": str(path), "n": int(len(X)),
                               "input_dim": int(X.shape[1]), "output_dim": int(Y.shape[1])})


def _load_argoverse2(name: str, cfg: dict[str, Any], raw_dir: Path) -> DatasetBundle:
    """Future displacement(s) of the focal agent from kinematic summaries of its observed
    history. Every `future_*` column that is not a target of this variant is dropped."""
    target_columns = list(cfg.get("target_columns", ["future_dx", "future_dy"]))
    csv_path = Path(cfg["data_path"])
    if not csv_path.is_absolute():
        csv_path = REPO_ROOT / csv_path
    if not csv_path.exists():
        raise DatasetLoadError(f"{name}: prepared table '{csv_path}' not found")
    df = pd.read_csv(csv_path)

    missing = [c for c in target_columns if c not in df.columns]
    if missing:
        raise DatasetLoadError(f"{name}: target column(s) {missing} not found in {csv_path}")

    n_before = len(df)
    df = df.dropna(subset=target_columns).reset_index(drop=True)
    n_dropped = n_before - len(df)

    index_like = [c for c in df.columns if c == "index" or c.startswith("Unnamed:")]
    future_cols = [c for c in df.columns
                   if c.startswith("future_") and c not in target_columns]
    drop_cols = (set(target_columns) | set(index_like)
                 | set(cfg.get("drop_columns", [])) | set(future_cols))
    X = df[[c for c in df.columns if c not in drop_cols]].copy()
    constant_cols = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    if constant_cols:
        X = X.drop(columns=constant_cols)
    y = df[target_columns].astype("float32")

    categorical_columns = [c for c in cfg.get("categorical_columns", []) if c in X.columns]
    return DatasetBundle(
        name=name, X=X, y=y, categorical_columns=categorical_columns,
        log1p_target_available=bool(cfg.get("log1p_target", False)),
        meta={
            "dates": None, "date_column": None, "target_columns": target_columns,
            "n_rows_dropped_missing_target": int(n_dropped),
            "n_rows_dropped_missing_date": 0,
            "constant_columns_dropped": constant_cols,
            "original_feature_names": list(X.columns),
            "n_stations": None, "n_unique_dates": None,
            "future_columns_dropped": len(future_cols),
        },
    )


def _load_sarcos(name: str, cfg: dict[str, Any], raw_dir: Path) -> DatasetBundle:
    """SARCOS inverse dynamics: 7 joint torques from the 21-D kinematic state; the shipped
    train and test arrays are concatenated."""
    from scipy.io import loadmat

    base = Path(cfg.get("data_dir", "sarcos"))
    if not base.is_absolute():
        base = Path(raw_dir) / base
    parts = []
    for fname, key in (("sarcos_inv.mat", "sarcos_inv"),
                       ("sarcos_inv_test.mat", "sarcos_inv_test")):
        fp = base / fname
        if not fp.exists():
            raise DatasetLoadError(f"{name}: {fp} not found -- download from "
                                   f"https://gaussianprocess.org/gpml/data/{fname}")
        arr = np.asarray(loadmat(fp)[key], dtype=float)
        if arr.shape[1] != 28:
            raise DatasetLoadError(f"{name}: expected 28 columns in {fname}, got {arr.shape[1]}")
        parts.append(arr)
        if cfg.get("use_shipped_split", False):
            break
    data = np.vstack(parts)

    n_in = int(cfg.get("n_input_columns", 21))
    targets = list(cfg.get("target_indices", list(range(21, 28))))
    joints = ["joint%d" % (i + 1) for i in range(7)]
    xcols = ([f"position_{j}" for j in joints] + [f"velocity_{j}" for j in joints]
             + [f"acceleration_{j}" for j in joints])[:n_in]
    X = pd.DataFrame(data[:, :n_in], columns=xcols)
    ycols = [f"torque_joint{t - 20}" for t in targets]
    ydf = pd.DataFrame(data[:, targets], columns=ycols)
    y = ydf.iloc[:, 0] if len(targets) == 1 else ydf
    return DatasetBundle(
        name=name, X=X, y=y, categorical_columns=[], log1p_target_available=False,
        meta={"dates": None, "date_column": None, "target_columns": ycols,
              "n_rows_dropped_missing_target": 0, "n_rows_dropped_missing_date": 0,
              "constant_columns_dropped": [], "original_feature_names": list(X.columns),
              "n_stations": None, "n_unique_dates": None,
              "source": "gaussianprocess.org/gpml/data",
              "shipped_split_concatenated": not bool(cfg.get("use_shipped_split", False))},
    )


def _load_csv_targets(name, cfg, raw_dir):
    """A plain CSV with the target columns named in the config (knee_contact_force_d12, qm7b_d14)."""
    dest = Path(raw_dir) / cfg["raw_filename"]
    if not dest.exists():
        raise DatasetLoadError(f"{name}: {dest} not found")
    df = pd.read_csv(dest)
    t = list(cfg["target_columns"])
    missing = [c for c in t if c not in df.columns]
    if missing:
        raise DatasetLoadError(f"{name}: target columns missing from {dest}: {missing}")
    X = df.drop(columns=t + list(cfg.get("drop_columns", [])))
    return DatasetBundle(name=name, X=X, y=df[t].astype(float), categorical_columns=list(cfg.get("categorical_columns", [])),
                         log1p_target_available=False, meta={"source": cfg.get("source", "csv")})


LOADERS = {
    "concrete": _load_concrete,
    "ccpp": _load_ccpp,
    "beijing_pm25": _load_beijing_pm25,
    "openml_univariate": _load_openml_univariate,
    "bike_sharing": _load_bike_sharing,
    "protein": _load_protein,
    "sgemm": _load_sgemm,
    "household_power": _load_household_power,
    "beijing_aq_6d": _load_beijing_aq_6d,
    "beijing_aq_12d": _load_beijing_aq_12d,
    "online_retail_ii": _load_online_retail_ii,
    "bias_correction_temperature": _load_bias_correction_temperature,
    "geographical_music": _load_geographical_music,
    "communities_crime_unnormalized": _load_communities_crime_unnormalized,
    "openml_multi_target": _load_openml_multi_target,
    "oes": _load_oes,
    "qm8_excited_states": _load_qm8_excited_states,
    "qm9": _load_qm9,
    "mvcs_npz": _load_mvcs_npz,
    "argoverse2": _load_argoverse2,
    "sarcos": _load_sarcos,
    "csv_targets": _load_csv_targets,
}
