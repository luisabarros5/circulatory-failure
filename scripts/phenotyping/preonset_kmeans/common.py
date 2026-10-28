from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


CLINICAL_FEATURES = [
    "heart_rate", "sbp", "mbp", "resp_rate", "temperature", "spo2",
    "glucose", "lab_50813", "lab_50983", "lab_50971", "lab_50912",
    "lab_51006", "lab_50878", "lab_50861", "lab_50885", "lab_50862",
    "lab_51222", "lab_51265", "lab_51274", "lab_51275",
]

FEATURE_LABELS = {
    "heart_rate": "Heart rate", "sbp": "Systolic blood pressure",
    "mbp": "Mean blood pressure", "resp_rate": "Respiratory rate",
    "temperature": "Temperature", "spo2": "SpO2", "glucose": "Glucose",
    "lab_50813": "Lactate", "lab_50983": "Sodium", "lab_50971": "Potassium",
    "lab_50912": "Creatinine", "lab_51006": "BUN", "lab_50878": "AST",
    "lab_50861": "ALT", "lab_50885": "Total bilirubin", "lab_50862": "Albumin",
    "lab_51222": "Hemoglobin", "lab_51265": "Platelets",
    "lab_51274": "Prothrombin time", "lab_51275": "Partial thromboplastin time",
}

LOG_FEATURES = {
    "glucose", "lab_50813", "lab_50912", "lab_51006", "lab_50878",
    "lab_50861", "lab_50885", "lab_51265", "lab_51274", "lab_51275",
}

NON_FEATURE = {"stay_id", "charttime", "rel_charttime", "intime", "outtime"}


def read_parquet(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Read these DuckDB-produced files without Arrow dataset discovery."""
    return pq.ParquetFile(path).read(columns=columns).to_pandas()


def write_json(path: Path, obj: dict) -> None:
    def normalize(value):
        if isinstance(value, dict):
            return {str(key): normalize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]
        if isinstance(value, np.generic):
            return value.item()
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(normalize(obj), indent=2, ensure_ascii=False, default=str) + "\n")


def clinical_columns(schema: list[str]) -> list[str]:
    out = []
    for col in schema:
        low = col.lower()
        if col in NON_FEATURE or low.endswith("_imputed") or "time_since_last" in low:
            continue
        if low.startswith(("target_", "failure_", "cf_", "falencia")):
            continue
        out.append(col)
    return out


def subject_split(subject_ids: np.ndarray, validation_fraction: float = 0.30) -> np.ndarray:
    """Stable split that never depends on input row order."""
    threshold = int(validation_fraction * 10_000)
    vals = []
    for subject_id in subject_ids:
        digest = hashlib.sha256(f"lucas-phenotype-v1:{int(subject_id)}".encode()).hexdigest()
        vals.append("validation" if int(digest[:8], 16) % 10_000 < threshold else "derivation")
    return np.asarray(vals, dtype=object)


def eta_squared(values: np.ndarray, labels: np.ndarray) -> float:
    values = np.asarray(values, float)
    labels = np.asarray(labels)
    ok = np.isfinite(values)
    values, labels = values[ok], labels[ok]
    if len(values) == 0:
        return float("nan")
    total = np.square(values - values.mean()).sum()
    if total == 0:
        return 0.0
    between = sum(
        len(values[labels == group]) * np.square(values[labels == group].mean() - values.mean())
        for group in np.unique(labels)
    )
    return float(between / total)


def smd_continuous(x0: pd.Series, x1: pd.Series) -> float:
    x0 = pd.to_numeric(x0, errors="coerce").dropna()
    x1 = pd.to_numeric(x1, errors="coerce").dropna()
    if len(x0) < 2 or len(x1) < 2:
        return float("nan")
    pooled = np.sqrt((x0.var(ddof=1) + x1.var(ddof=1)) / 2)
    return float((x1.mean() - x0.mean()) / pooled) if pooled > 0 else 0.0

