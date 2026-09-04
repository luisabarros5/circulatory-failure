from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd


DROP_COLS = {
    "patientunitstayid",
    "patientunitstay_id",
    "stay_id",
    "offset_min",
    "mortality",
    "falencia",
    "falencia_point",
    "falencia_any",
    "falencia_onset",
}

TIME_CANDIDATES = ("offset_min", "rel_charttime", "charttime", "time", "timestamp")


def setup_logger(log_path: Path | None = None, verbose: bool = True) -> logging.Logger:
    logger = logging.getLogger("temporal_ts2vec")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    if verbose:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def detect_id_col(columns: Iterable[str]) -> str:
    cols = set(columns)
    if "patientunitstayid" in cols:
        return "patientunitstayid"
    if "patientunitstay_id" in cols:
        return "patientunitstay_id"
    if "stay_id" in cols:
        return "stay_id"
    raise ValueError("Could not detect stay id column in input data.")


def detect_time_col(columns: Iterable[str]) -> str:
    cols = set(columns)
    for c in TIME_CANDIDATES:
        if c in cols:
            return c
    raise ValueError("Could not detect time column in input data.")


def select_feature_columns(columns: Iterable[str], drop_imputed: bool = True) -> list[str]:
    out: list[str] = []
    for c in columns:
        if c in DROP_COLS:
            continue
        if drop_imputed and c.endswith("_imputed"):
            continue
        out.append(c)
    return out


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_engineered_features(engineered_dir: Path, id_col: str, logger: logging.Logger | None = None) -> pd.DataFrame:
    paths = sorted(engineered_dir.glob("features_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No engineered feature files found in {engineered_dir}")
    frames = []
    for path in paths:
        df = pd.read_parquet(path)
        frames.append(df)
    df_all = pd.concat(frames, ignore_index=True)
    if id_col not in df_all.columns:
        if "patientunitstayid" in df_all.columns:
            df_all = df_all.rename(columns={"patientunitstayid": id_col})
        elif "stay_id" in df_all.columns:
            df_all = df_all.rename(columns={"stay_id": id_col})
        else:
            raise ValueError("Engineered features missing id column.")
    if logger:
        logger.info("Loaded engineered features: %d stays, %d columns", len(df_all), df_all.shape[1] - 1)
    return df_all
