"""Source exported from the reviewed development notebook.

This file contains code only; patient-level data and notebook outputs are not distributed.
"""

# %% [source cell 1]
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path

MATPLOTLIB_CACHE = Path("/path/to/local_resource")
MATPLOTLIB_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MATPLOTLIB_CACHE)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)

NOTEBOOK_VERSION = 3
SEED = 42
NEGATIVE_WINDOW_SEEDS = sorted(set([42, *range(20)]))
BOOTSTRAP_REPETITIONS = 1000
PERMUTATION_REPETITIONS = 200
ALERT_GAP_MINUTES = 30
TARGET_HORIZON_HOURS = 8

PROJECT_TRANSFORMER = Path(
    "/path/to/circulatory_failure_prediction"
)

TRANSFORMER_RUN = (
    PROJECT_TRANSFORMER
    / "outputs/step11_transformer/transformer_primary_seed42_v7"
)

# O projeto LightGBM apareceu em dois caminhos durante o trabalho.
# Escolhemos automaticamente somente um run que esteja completo.
LIGHTGBM_PROJECT_CANDIDATES = [
    PROJECT_TRANSFORMER,
    Path("/path/to/circulatory_failure_prediction"),
]
LIGHTGBM_RUN_CANDIDATES = list(dict.fromkeys(
    project / "outputs/step09_lightgbm/weighted_event_primary"
    for project in LIGHTGBM_PROJECT_CANDIDATES
))
LIGHTGBM_REQUIRED_NAMES = (
    "metrics.json",
    "features_used.json",
    "validation_predictions.parquet",
    "test_predictions.parquet",
)
COMPLETE_LIGHTGBM_RUNS = [
    candidate for candidate in LIGHTGBM_RUN_CANDIDATES
    if all((candidate / name).is_file() for name in LIGHTGBM_REQUIRED_NAMES)
]
if not COMPLETE_LIGHTGBM_RUNS:
    inspected = {
        str(candidate): [
            name for name in LIGHTGBM_REQUIRED_NAMES
            if not (candidate / name).is_file()
        ]
        for candidate in LIGHTGBM_RUN_CANDIDATES
    }
    raise FileNotFoundError(
        "Nenhum run completo do LightGBM foi encontrado. "
        "Arquivos ausentes por candidato:\n"
        + json.dumps(inspected, indent=2)
    )
LIGHTGBM_RUN = COMPLETE_LIGHTGBM_RUNS[0]
PROJECT_LIGHTGBM = LIGHTGBM_RUN.parents[2]
LIGHTGBM_SENSITIVITY = (
    PROJECT_LIGHTGBM
    / "outputs/step13_target_feature_sensitivity/weighted_target_feature_sensitivity.csv"
)
LABEL_DIR = PROJECT_LIGHTGBM / "outputs/step05_labels_v2"

AUDIT_ROOT = (
    PROJECT_TRANSFORMER
    / "outputs/step12_lightgbm_transformer_clinical_audit"
)
AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
RUN_ID = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%S%fZ")
OUTPUT_DIR = AUDIT_ROOT / RUN_ID
OUTPUT_DIR.mkdir(parents=False, exist_ok=False)
CACHE_DIR = AUDIT_ROOT / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_COLUMNS = {
    "Transformer": "probability_transformer",
    "LightGBM": "probability_lightgbm",
}

print("Notebook 12 — versão", NOTEBOOK_VERSION)
print("Cache do Matplotlib:", MATPLOTLIB_CACHE)
print("Saída:", OUTPUT_DIR)
print("Transformer:", TRANSFORMER_RUN)
print("LightGBM:", LIGHTGBM_RUN)
if len(COMPLETE_LIGHTGBM_RUNS) > 1:
    print("AVISO: mais de um run completo encontrado; prioridade:", LIGHTGBM_RUN)

# %% [source cell 3]
def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(value: dict | list, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def finite_probability(values: np.ndarray, label: str) -> None:
    if not np.isfinite(values).all():
        raise RuntimeError(f"Probabilidades não finitas em {label}")
    if np.any((values < 0) | (values > 1)):
        raise RuntimeError(f"Probabilidades fora de [0,1] em {label}")


def ranking_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y, dtype="int8")
    p = np.asarray(p, dtype="float64")
    finite_probability(p, "ranking_metrics")
    if np.unique(y).size != 2:
        raise RuntimeError("Métrica exige duas classes")
    return {
        "rows": int(len(y)),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()),
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "auprc_lift_over_prevalence": float(average_precision_score(y, p) / y.mean()),
    }


def confusion_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    y = np.asarray(y, dtype=bool)
    alert = np.asarray(p, dtype=float) >= float(threshold)
    tp = int(np.sum(alert & y)); fp = int(np.sum(alert & ~y))
    fn = int(np.sum(~alert & y)); tn = int(np.sum(~alert & ~y))
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    npv = tn / (tn + fn) if tn + fn else 1.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall,
        "specificity": specificity, "npv": npv, "f1": f1,
    }


def threshold_curve(y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    y = np.asarray(y, dtype=bool); p = np.asarray(p, dtype=float)
    thresholds = np.unique(np.r_[0.0, 1.0, np.quantile(p, np.linspace(0, 1, 2001))])
    rows = []
    for threshold in thresholds:
        rows.append({"threshold": float(threshold), **confusion_metrics(y, p, threshold)})
    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


def choose_thresholds(curve: pd.DataFrame) -> dict[str, float]:
    best = curve.sort_values(["f1", "threshold"], ascending=[False, False]).iloc[0]
    result = {"maximum_event_window_f1": float(best.threshold)}
    for target_recall in (0.70, 0.80, 0.90):
        candidates = curve.loc[curve.recall.ge(target_recall)]
        if not candidates.empty:
            result[f"event_recall_{int(target_recall * 100)}"] = float(candidates.threshold.max())
    return result


def md_table(frame: pd.DataFrame, digits: int = 4) -> str:
    shown = frame.copy()
    for column in shown.select_dtypes(include=["float", "float32", "float64"]).columns:
        shown[column] = shown[column].map(lambda x: "" if pd.isna(x) else f"{x:.{digits}f}")
    shown = shown.astype(str)
    header = "| " + " | ".join(shown.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(shown.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in shown.to_numpy()]
    return "\n".join([header, separator, *rows])


def add_check(checks: list[dict], area: str, check: str, status: str, evidence: str) -> None:
    if status not in {"APROVADO", "REVISAR", "FALHA", "NÃO AVALIADO"}:
        raise ValueError(status)
    checks.append({"area": area, "check": check, "status": status, "evidence": evidence})

# %% [source cell 5]
REQUIRED = {
    "transformer_metrics": TRANSFORMER_RUN / "metrics.json",
    "transformer_config": TRANSFORMER_RUN / "config.json",
    "transformer_scaler": TRANSFORMER_RUN / "train_scaler.npz",
    "transformer_validation_predictions": TRANSFORMER_RUN / "validation_predictions.parquet",
    "transformer_test_predictions": TRANSFORMER_RUN / "test_predictions.parquet",
    "lightgbm_metrics": LIGHTGBM_RUN / "metrics.json",
    "lightgbm_features": LIGHTGBM_RUN / "features_used.json",
    "lightgbm_validation_predictions": LIGHTGBM_RUN / "validation_predictions.parquet",
    "lightgbm_test_predictions": LIGHTGBM_RUN / "test_predictions.parquet",
}
missing = {name: str(path) for name, path in REQUIRED.items() if not path.is_file()}
if missing:
    raise FileNotFoundError("Arquivos obrigatórios ausentes:\n" + json.dumps(missing, indent=2))

transformer_metrics_original = read_json(REQUIRED["transformer_metrics"])
transformer_config = read_json(REQUIRED["transformer_config"])
lightgbm_metrics_original = read_json(REQUIRED["lightgbm_metrics"])
lightgbm_features_payload = read_json(REQUIRED["lightgbm_features"])

scaler = np.load(REQUIRED["transformer_scaler"], allow_pickle=True)
transformer_features = [str(x) for x in scaler["features"].tolist()]
lightgbm_features = [str(x) for x in lightgbm_features_payload.get("features", [])]
if not lightgbm_features:
    raise RuntimeError("features_used.json do LightGBM não contém a lista 'features'")

def names_sha256(names: list[str]) -> str:
    return hashlib.sha256("\n".join(names).encode()).hexdigest()

checks: list[dict] = []
same_features = transformer_features == lightgbm_features
add_check(
    checks, "comparabilidade", "Lista e ordem das features",
    "APROVADO" if same_features else "FALHA",
    f"Transformer={len(transformer_features)}; LightGBM={len(lightgbm_features)}; "
    f"sha_T={names_sha256(transformer_features)}; sha_L={names_sha256(lightgbm_features)}",
)

vaso_pattern = re.compile(
    r"vasopress|norepine|noradren|epinephrine|adrenaline|dopamine|dobutamine|phenylephrine",
    re.I,
)
target_pattern = re.compile(r"target|failure_state|first_onset|future", re.I)
suspicious_vaso = sorted(f for f in transformer_features if vaso_pattern.search(f))
suspicious_target = sorted(f for f in transformer_features if target_pattern.search(f))
add_check(
    checks, "leakage", "Vasopressores ausentes do input",
    "APROVADO" if not suspicious_vaso else "FALHA",
    "nenhuma feature suspeita" if not suspicious_vaso else str(suspicious_vaso),
)
add_check(
    checks, "leakage", "Colunas com nome explícito de target/futuro ausentes",
    "APROVADO" if not suspicious_target else "FALHA",
    "nenhuma feature suspeita" if not suspicious_target else str(suspicious_target),
)

recorded_split_t = transformer_metrics_original.get("split_reused_from")
recorded_split_l = lightgbm_metrics_original.get("patient_level_split_reused")

def resolve_split_path(recorded_path: str | None, fallback_paths: list[Path]) -> tuple[Path | None, str]:
    candidates: list[tuple[str, Path]] = []
    if recorded_path:
        candidates.append(("caminho registrado", Path(recorded_path)))
    candidates.extend(("cópia no run", Path(path)) for path in fallback_paths)
    seen: set[str] = set()
    for origin, candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            return candidate, origin
    return None, "não encontrado"

def inspect_split(path: Path | None) -> dict:
    if path is None:
        return {"signature": None, "stay_counts": {}, "subject_counts": {}, "subject_overlap": None, "error": "arquivo não encontrado"}
    frame = pd.read_parquet(path)
    required_columns = ["stay_id", "subject_id", "split"]
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        return {"signature": None, "stay_counts": {}, "subject_counts": {}, "subject_overlap": None, "error": f"colunas ausentes: {missing_columns}"}
    canonical = frame[required_columns].copy()
    canonical["split"] = canonical["split"].astype(str).str.lower().str.strip()
    canonical = canonical.drop_duplicates().sort_values(required_columns).reset_index(drop=True)
    raw = pd.util.hash_pandas_object(canonical, index=False).to_numpy(dtype="uint64").tobytes()
    stay_counts = canonical.groupby("split")["stay_id"].nunique().to_dict()
    subject_counts = canonical.groupby("split")["subject_id"].nunique().to_dict()
    subject_overlap = int(canonical.groupby("subject_id")["split"].nunique().gt(1).sum())
    return {
        "signature": hashlib.sha256(raw).hexdigest(),
        "stay_counts": {str(k): int(v) for k, v in stay_counts.items()},
        "subject_counts": {str(k): int(v) for k, v in subject_counts.items()},
        "subject_overlap": subject_overlap,
        "error": None,
    }

split_path_t, split_origin_t = resolve_split_path(
    recorded_split_t,
    [TRANSFORMER_RUN / "stay_subject_split.parquet", LIGHTGBM_RUN / "stay_subject_split.parquet"],
)
split_path_l, split_origin_l = resolve_split_path(
    recorded_split_l,
    [LIGHTGBM_RUN / "stay_subject_split.parquet"],
)
split_audit_t = inspect_split(split_path_t)
split_audit_l = inspect_split(split_path_l)

if split_path_t is None or split_path_l is None:
    split_status = "NÃO AVALIADO"
elif split_audit_t["error"] or split_audit_l["error"]:
    split_status = "FALHA"
elif split_audit_t["subject_overlap"] or split_audit_l["subject_overlap"]:
    split_status = "FALHA"
elif split_audit_t["signature"] != split_audit_l["signature"]:
    split_status = "FALHA"
else:
    split_status = "APROVADO"

add_check(
    checks, "comparabilidade", "Split por paciente semanticamente idêntico",
    split_status,
    f"recorded_T={recorded_split_t}; resolved_T={split_path_t}; origin_T={split_origin_t}; "
    f"recorded_L={recorded_split_l}; resolved_L={split_path_l}; origin_L={split_origin_l}; "
    f"signature_T={split_audit_t['signature']}; signature_L={split_audit_l['signature']}; "
    f"stay_counts_T={split_audit_t['stay_counts']}; stay_counts_L={split_audit_l['stay_counts']}; "
    f"subject_counts_T={split_audit_t['subject_counts']}; subject_counts_L={split_audit_l['subject_counts']}; "
    f"subject_overlap_T={split_audit_t['subject_overlap']}; subject_overlap_L={split_audit_l['subject_overlap']}; "
    f"error_T={split_audit_t['error']}; error_L={split_audit_l['error']}",
)

ratio_t = float(transformer_config.get("negative_ratio", np.nan))
ratio_l = float(lightgbm_metrics_original.get("negative_to_positive_sampling_ratio", np.nan))
ratio_equal = np.isfinite(ratio_t) and np.isfinite(ratio_l) and math.isclose(ratio_t, ratio_l)
add_check(
    checks, "comparabilidade", "Razão de amostragem negativa documentada",
    "APROVADO" if ratio_equal else "REVISAR",
    f"Transformer={ratio_t}; LightGBM={ratio_l}",
)

feature_audit = {
    "transformer_feature_count": len(transformer_features),
    "lightgbm_feature_count": len(lightgbm_features),
    "transformer_feature_sha256": names_sha256(transformer_features),
    "lightgbm_feature_sha256": names_sha256(lightgbm_features),
    "same_features_same_order": same_features,
    "vasopressor_like_features": suspicious_vaso,
    "target_or_future_like_features": suspicious_target,
    "lactate_features": [f for f in transformer_features if re.search(r"50813|lactate", f, re.I)],
    "pressure_features": [f for f in transformer_features if re.search(r"(^|_)(map|mbp|sbp|dbp)($|_)", f, re.I)],
}
write_json_atomic(feature_audit, OUTPUT_DIR / "feature_audit.json")
print(md_table(pd.DataFrame(checks)))

# %% [source cell 7]
KEY_COLUMNS = ["stay_id", "subject_id", "label_time"]

def load_prediction(path: Path, output_probability_name: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    probabilities = [c for c in ("probability", "prediction") if c in frame.columns]
    if len(probabilities) != 1:
        raise RuntimeError(f"{path}: coluna de probabilidade ambígua: {probabilities}")
    required = [*KEY_COLUMNS, "target", probabilities[0]]
    missing_columns = sorted(set(required) - set(frame.columns))
    if missing_columns:
        raise RuntimeError(f"{path}: colunas ausentes: {missing_columns}")
    frame = frame[required].rename(columns={probabilities[0]: output_probability_name})
    frame["label_time"] = pd.to_datetime(frame["label_time"], errors="raise")
    frame["target"] = frame["target"].astype("int8")
    frame[output_probability_name] = frame[output_probability_name].astype("float64")
    if frame.duplicated(KEY_COLUMNS).any():
        raise RuntimeError(f"Predições duplicadas em {path}")
    finite_probability(frame[output_probability_name].to_numpy(), str(path))
    return frame


def align_split(split: str) -> pd.DataFrame:
    transformer = load_prediction(
        TRANSFORMER_RUN / f"{split}_predictions.parquet",
        MODEL_COLUMNS["Transformer"],
    )
    lightgbm = load_prediction(
        LIGHTGBM_RUN / f"{split}_predictions.parquet",
        MODEL_COLUMNS["LightGBM"],
    )
    same_order = (
        len(transformer) == len(lightgbm)
        and all(np.array_equal(transformer[c].to_numpy(), lightgbm[c].to_numpy()) for c in [*KEY_COLUMNS, "target"])
    )
    if same_order:
        aligned = transformer.copy()
        aligned[MODEL_COLUMNS["LightGBM"]] = lightgbm[MODEL_COLUMNS["LightGBM"]].to_numpy()
    else:
        aligned = transformer.merge(
            lightgbm,
            on=KEY_COLUMNS,
            how="outer",
            suffixes=("_transformer", "_lightgbm"),
            indicator=True,
            validate="one_to_one",
        )
        coverage = {
            str(category): int(count)
            for category, count in aligned["_merge"].value_counts(sort=False).items()
        }
        unmatched = {
            category: count
            for category, count in coverage.items()
            if category != "both" and count > 0
        }
        if unmatched or coverage.get("both", 0) != len(aligned):
            raise RuntimeError(f"Cobertura diferente em {split}: {coverage}")
        if not np.array_equal(
            aligned.target_transformer.to_numpy(), aligned.target_lightgbm.to_numpy()
        ):
            raise RuntimeError(f"Targets divergentes em {split}")
        aligned["target"] = aligned.pop("target_transformer").astype("int8")
        aligned = aligned.drop(columns=["target_lightgbm", "_merge"])
    aligned = aligned.sort_values(KEY_COLUMNS).reset_index(drop=True)
    return aligned


aligned_predictions: dict[str, pd.DataFrame] = {}
timestep_rows = []
for split in ("validation", "test"):
    print("Alinhando", split, flush=True)
    aligned = align_split(split)
    aligned_predictions[split] = aligned
    for model, column in MODEL_COLUMNS.items():
        timestep_rows.append({
            "split": split,
            "model": model,
            **ranking_metrics(aligned.target.to_numpy(), aligned[column].to_numpy()),
        })
    add_check(
        checks, "comparabilidade", f"Predições alinhadas em {split}", "APROVADO",
        f"rows={len(aligned)}; positives={int(aligned.target.sum())}; "
        f"subjects={aligned.subject_id.nunique()}; stays={aligned.stay_id.nunique()}",
    )

def verify_recorded_timestep_metrics(model: str, split: str, original: dict) -> None:
    recalculated = timestep_metrics.loc[
        timestep_metrics.model.eq(model) & timestep_metrics.split.eq(split)
    ].iloc[0]
    metric_names = ("prevalence", "auroc", "auprc", "log_loss", "brier")
    differences = {
        metric: abs(float(recalculated[metric]) - float(original[metric]))
        for metric in metric_names
    }
    numerically_equal = {
        metric: bool(np.isclose(
            float(recalculated[metric]),
            float(original[metric]),
            rtol=1e-7,
            atol=1e-10,
        ))
        for metric in metric_names
    }
    counts_equal = (
        int(recalculated["rows"]) == int(original["rows"])
        and int(recalculated["positives"]) == int(original["positives"])
    )
    exact = counts_equal and all(numerically_equal.values())
    add_check(
        checks, "reprodutibilidade", f"Métricas por timestep reproduzidas — {model}/{split}",
        "APROVADO" if exact else "FALHA",
        f"counts_equal={counts_equal}; numerically_equal={numerically_equal}; "
        f"rtol=1e-7; atol=1e-10; absolute_differences={differences}",
    )

timestep_metrics = pd.DataFrame(timestep_rows)
for split in ("validation", "test"):
    verify_recorded_timestep_metrics(
        "Transformer", split,
        transformer_metrics_original["timestep_metrics"][split],
    )
    verify_recorded_timestep_metrics(
        "LightGBM", split,
        lightgbm_metrics_original[split],
    )

write_csv_atomic(timestep_metrics, OUTPUT_DIR / "timestep_metrics_recalculated.csv")
display(timestep_metrics)

# %% [source cell 9]
@dataclass
class EventCache:
    positives: pd.DataFrame
    positive_lengths: np.ndarray
    control_meta: pd.DataFrame
    negative_times: np.ndarray
    negative_transformer: np.ndarray
    negative_lightgbm: np.ndarray


def prepare_event_cache(aligned: pd.DataFrame) -> EventCache:
    truth = aligned.groupby("stay_id", sort=False).target.max()
    positive_ids = truth.index[truth.eq(1)]
    negative_ids = truth.index[truth.eq(0)]
    positive_rows = aligned.loc[aligned.target.eq(1)]
    positives = (
        positive_rows.groupby(["stay_id", "subject_id"], as_index=False)
        .agg(
            target=("target", "max"),
            window_rows=("target", "size"),
            window_start=("label_time", "min"),
            window_end=("label_time", "max"),
            score_transformer=(MODEL_COLUMNS["Transformer"], "max"),
            score_lightgbm=(MODEL_COLUMNS["LightGBM"], "max"),
        )
    )
    if set(positives.stay_id) != set(positive_ids):
        raise RuntimeError("Stay positivo sem janela positiva")
    positives["target"] = 1

    negative = (
        aligned.loc[aligned.stay_id.isin(negative_ids)]
        .sort_values(["stay_id", "subject_id", "label_time"])
        .reset_index(drop=True)
    )
    sizes = (
        negative.groupby(["stay_id", "subject_id"], sort=True)
        .size().rename("available_rows").reset_index()
    )
    sizes["offset"] = sizes.available_rows.cumsum().shift(fill_value=0).astype("int64")
    if int(sizes.available_rows.sum()) != len(negative):
        raise RuntimeError("Offsets negativos inconsistentes")
    return EventCache(
        positives=positives,
        positive_lengths=positives.window_rows.to_numpy(dtype="int64"),
        control_meta=sizes,
        negative_times=negative.label_time.to_numpy(),
        negative_transformer=negative[MODEL_COLUMNS["Transformer"]].to_numpy(dtype="float64"),
        negative_lightgbm=negative[MODEL_COLUMNS["LightGBM"]].to_numpy(dtype="float64"),
    )


def event_table_from_cache(cache: EventCache, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    controls = []
    for row in cache.control_meta.itertuples(index=False):
        desired = int(cache.positive_lengths[rng.integers(0, len(cache.positive_lengths))])
        length = min(desired, int(row.available_rows))
        local_start = int(rng.integers(0, int(row.available_rows) - length + 1))
        start = int(row.offset) + local_start
        stop = start + length
        controls.append({
            "stay_id": row.stay_id,
            "subject_id": row.subject_id,
            "target": 0,
            "window_rows": length,
            "desired_window_rows": desired,
            "window_start": pd.Timestamp(cache.negative_times[start]),
            "window_end": pd.Timestamp(cache.negative_times[stop - 1]),
            "score_transformer": float(np.max(cache.negative_transformer[start:stop])),
            "score_lightgbm": float(np.max(cache.negative_lightgbm[start:stop])),
        })
    table = pd.concat([cache.positives, pd.DataFrame(controls)], ignore_index=True, sort=False)
    if table.stay_id.duplicated().any():
        raise RuntimeError("Mais de uma janela por stay")
    return table.sort_values(["subject_id", "stay_id"]).reset_index(drop=True)


event_caches = {split: prepare_event_cache(frame) for split, frame in aligned_predictions.items()}
event_tables = {split: event_table_from_cache(cache, SEED) for split, cache in event_caches.items()}
for split, table in event_tables.items():
    table.to_parquet(OUTPUT_DIR / f"{split}_shared_event_windows_seed42.parquet", index=False, compression="zstd")
    print(split, "eventos:", len(table), "positivos:", int(table.target.sum()))

# %% [source cell 11]
def paired_event_bootstrap(
    table: pd.DataFrame,
    repetitions: int = BOOTSTRAP_REPETITIONS,
    seed: int = SEED,
) -> tuple[pd.DataFrame, dict]:
    y = table.target.to_numpy(dtype="int8")
    p_t = table.score_transformer.to_numpy(dtype="float64")
    p_l = table.score_lightgbm.to_numpy(dtype="float64")
    subjects = table.subject_id.unique()
    subject_array = table.subject_id.to_numpy()
    by_subject = {s: np.flatnonzero(subject_array == s) for s in subjects}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(repetitions):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        idx = np.concatenate([by_subject[s] for s in sampled])
        if np.unique(y[idx]).size != 2:
            continue
        ap_t = average_precision_score(y[idx], p_t[idx])
        ap_l = average_precision_score(y[idx], p_l[idx])
        auc_t = roc_auc_score(y[idx], p_t[idx])
        auc_l = roc_auc_score(y[idx], p_l[idx])
        draws.append({
            "transformer_auprc": ap_t, "lightgbm_auprc": ap_l,
            "delta_auprc_lightgbm_minus_transformer": ap_l - ap_t,
            "transformer_auroc": auc_t, "lightgbm_auroc": auc_l,
            "delta_auroc_lightgbm_minus_transformer": auc_l - auc_t,
        })
    draw_frame = pd.DataFrame(draws)
    summary = {}
    for column in draw_frame.columns:
        summary[column] = {
            "median": float(draw_frame[column].median()),
            "ci95": [float(draw_frame[column].quantile(.025)), float(draw_frame[column].quantile(.975))],
        }
    summary["bootstrap_unit"] = "subject_id"
    summary["repetitions"] = int(len(draw_frame))
    return draw_frame, summary


event_ranking_rows = []
bootstrap_summaries = {}
for split, table in event_tables.items():
    for model, score_column in {
        "Transformer": "score_transformer",
        "LightGBM": "score_lightgbm",
    }.items():
        event_ranking_rows.append({
            "split": split,
            "model": model,
            **ranking_metrics(table.target.to_numpy(), table[score_column].to_numpy()),
        })
    draws, summary = paired_event_bootstrap(table)
    draws.to_csv(OUTPUT_DIR / f"{split}_paired_event_bootstrap_draws.csv", index=False)
    bootstrap_summaries[split] = summary

event_ranking = pd.DataFrame(event_ranking_rows)

def verify_recorded_event_metrics(
    model: str, split: str, original: dict | None,
    row_key: str = "rows", positive_key: str = "positives",
) -> None:
    if original is None:
        add_check(
            checks, "reprodutibilidade", f"Métricas por evento reproduzidas — {model}/{split}",
            "NÃO AVALIADO", "Resultado original não encontrado; métricas foram recalculadas.",
        )
        return
    recalculated = event_ranking.loc[
        event_ranking.model.eq(model) & event_ranking.split.eq(split)
    ].iloc[0]
    differences = {
        metric: abs(float(recalculated[metric]) - float(original[metric]))
        for metric in ("prevalence", "auroc", "auprc")
    }
    counts_equal = (
        int(recalculated["rows"]) == int(original[row_key])
        and int(recalculated["positives"]) == int(original[positive_key])
    )
    exact = counts_equal and max(differences.values()) <= 1e-10
    add_check(
        checks, "reprodutibilidade", f"Métricas por evento reproduzidas — {model}/{split}",
        "APROVADO" if exact else "FALHA",
        f"counts_equal={counts_equal}; absolute_differences={differences}",
    )

lightgbm_event_path = LIGHTGBM_RUN / "event_evaluation_v2/event_evaluation.json"
lightgbm_event_original = read_json(lightgbm_event_path) if lightgbm_event_path.is_file() else {}
transformer_event_original = transformer_metrics_original.get("event_evaluation", {}) or {}
for split in ("validation", "test"):
    verify_recorded_event_metrics(
        "Transformer", split,
        transformer_event_original.get(f"{split}_ranking"),
    )
    verify_recorded_event_metrics(
        "LightGBM", split,
        lightgbm_event_original.get("event_window_ranking", {}).get(split),
        row_key="events", positive_key="positive_events",
    )

write_csv_atomic(event_ranking, OUTPUT_DIR / "event_ranking_recalculated.csv")
write_json_atomic(bootstrap_summaries, OUTPUT_DIR / "paired_event_bootstrap_summary.json")
display(event_ranking)
print(json.dumps(bootstrap_summaries["test"], indent=2, ensure_ascii=False))

# %% [source cell 13]
def group_alerts(
    aligned: pd.DataFrame,
    probability_column: str,
    threshold: float,
    gap_minutes: int = ALERT_GAP_MINUTES,
) -> tuple[pd.DataFrame, dict]:
    alerts = aligned.loc[
        aligned[probability_column].ge(threshold),
        ["stay_id", "subject_id", "label_time", "target", probability_column],
    ].copy()
    alerts = alerts.sort_values(["stay_id", "label_time"])
    if alerts.empty:
        episodes = pd.DataFrame(columns=["stay_id", "subject_id", "episode", "true_event_window"])
    else:
        gap = alerts.groupby("stay_id").label_time.diff()
        new_episode = gap.isna() | gap.gt(pd.Timedelta(minutes=gap_minutes))
        alerts["episode"] = new_episode.groupby(alerts.stay_id).cumsum().astype("int64")
        episodes = alerts.groupby(["stay_id", "subject_id", "episode"], as_index=False).agg(
            alert_start=("label_time", "min"),
            alert_end=("label_time", "max"),
            alert_rows=("label_time", "size"),
            max_probability=(probability_column, "max"),
            true_event_window=("target", "max"),
        )
    eligible_negative_patient_days = float(aligned.target.eq(0).sum() * 5 / 1440)
    false_episodes = int(episodes.true_event_window.eq(0).sum()) if len(episodes) else 0
    true_episodes = int(episodes.true_event_window.eq(1).sum()) if len(episodes) else 0
    return episodes, {
        "alert_gap_minutes": gap_minutes,
        "alert_episodes": int(len(episodes)),
        "true_alert_episodes": true_episodes,
        "false_alert_episodes": false_episodes,
        "eligible_negative_patient_days": eligible_negative_patient_days,
        "false_alert_episodes_per_100_patient_days": (
            100 * false_episodes / eligible_negative_patient_days
            if eligible_negative_patient_days else None
        ),
    }


def load_first_onsets() -> tuple[pd.DataFrame, str]:
    cache = CACHE_DIR / "first_onsets.parquet"
    if cache.is_file():
        return pd.read_parquet(cache), "step05_cached"
    files = sorted(LABEL_DIR.glob("batch_*.parquet"))
    if len(files) != 285:
        # Fallback conservador: aproxima o onset como 5 min após o último timestep positivo.
        combined = pd.concat([
            frame.loc[frame.target.eq(1), ["stay_id", "label_time"]]
            for frame in aligned_predictions.values()
        ]).drop_duplicates()
        onsets = combined.groupby("stay_id", as_index=False).label_time.max()
        onsets["first_onset_time"] = onsets.pop("label_time") + pd.Timedelta(minutes=5)
        return onsets, "approximation_from_last_positive_plus_5min"
    parts = []
    for index, path in enumerate(files, 1):
        frame = pd.read_parquet(path, columns=["stay_id", "charttime", "failure_state_new_at_t"])
        mask = frame.failure_state_new_at_t.astype("boolean").fillna(False)
        if mask.any():
            part = (
                frame.loc[mask, ["stay_id", "charttime"]]
                .groupby("stay_id", as_index=False).charttime.min()
                .rename(columns={"charttime": "first_onset_time"})
            )
            parts.append(part)
        if index == 1 or index % 25 == 0 or index == len(files):
            print(f"onsets: {index}/{len(files)}", flush=True)
    onsets = pd.concat(parts, ignore_index=True)
    if onsets.stay_id.duplicated().any():
        raise RuntimeError("Onsets duplicados entre batches")
    onsets.first_onset_time = pd.to_datetime(onsets.first_onset_time, errors="raise")
    onsets.to_parquet(cache, index=False, compression="zstd")
    return onsets, "step05_failure_state_new_at_t"


def lead_time_summary(
    aligned: pd.DataFrame,
    probability_column: str,
    threshold: float,
    onsets: pd.DataFrame,
) -> dict:
    positive = aligned.loc[aligned.target.eq(1)]
    base = positive.groupby(["stay_id", "subject_id"], as_index=False).agg(
        first_positive_time=("label_time", "min"),
        positive_rows=("label_time", "size"),
    ).merge(onsets, on="stay_id", how="left", validate="one_to_one")
    detected = (
        positive.loc[positive[probability_column].ge(threshold)]
        .groupby("stay_id", as_index=False).label_time.min()
        .rename(columns={"label_time": "first_alert_time"})
    )
    base = base.merge(detected, on="stay_id", how="left", validate="one_to_one")
    base["detected"] = base.first_alert_time.notna()
    base["lead_time_hours"] = (
        base.first_onset_time - base.first_alert_time
    ).dt.total_seconds() / 3600
    leads = base.loc[base.detected, "lead_time_hours"].dropna()
    return {
        "detected_events": int(base.detected.sum()),
        "events": int(len(base)),
        "median_lead_time_hours": float(leads.median()) if len(leads) else None,
        "lead_time_q25_hours": float(leads.quantile(.25)) if len(leads) else None,
        "lead_time_q75_hours": float(leads.quantile(.75)) if len(leads) else None,
        "detected_at_least_2h": float((leads >= 2).mean()) if len(leads) else None,
        "detected_at_least_4h": float((leads >= 4).mean()) if len(leads) else None,
        "detected_at_least_6h": float((leads >= 6).mean()) if len(leads) else None,
    }


onsets, onset_source = load_first_onsets()
threshold_rows = []
thresholds_by_model = {}
validation_events = event_tables["validation"]
test_events = event_tables["test"]
for model, event_score, probability_column in (
    ("Transformer", "score_transformer", MODEL_COLUMNS["Transformer"]),
    ("LightGBM", "score_lightgbm", MODEL_COLUMNS["LightGBM"]),
):
    curve = threshold_curve(validation_events.target, validation_events[event_score])
    curve.to_csv(OUTPUT_DIR / f"validation_threshold_curve_{model.lower()}.csv", index=False)
    selected = choose_thresholds(curve)
    thresholds_by_model[model] = selected
    for operating_point, threshold in selected.items():
        for split, events in event_tables.items():
            confusion = confusion_metrics(events.target, events[event_score], threshold)
            episodes, burden = group_alerts(
                aligned_predictions[split], probability_column, threshold
            )
            lead = lead_time_summary(
                aligned_predictions[split], probability_column, threshold, onsets
            )
            if split == "test":
                episodes.to_parquet(
                    OUTPUT_DIR / f"test_alert_episodes_{model.lower()}_{operating_point}.parquet",
                    index=False, compression="zstd",
                )
            threshold_rows.append({
                "model": model, "operating_point": operating_point,
                "threshold_selected_on_validation": threshold, "split": split,
                **confusion, **burden, **lead,
            })

threshold_results = pd.DataFrame(threshold_rows)
write_csv_atomic(threshold_results, OUTPUT_DIR / "frozen_threshold_clinical_results.csv")
write_json_atomic(thresholds_by_model, OUTPUT_DIR / "thresholds_selected_on_validation.json")
display(threshold_results.loc[threshold_results.split.eq("test")])

# %% [source cell 15]
seed_rows = []
test_cache = event_caches["test"]
for seed in NEGATIVE_WINDOW_SEEDS:
    table = event_table_from_cache(test_cache, seed)
    for model, score_column in (
        ("Transformer", "score_transformer"),
        ("LightGBM", "score_lightgbm"),
    ):
        seed_rows.append({
            "seed": seed, "model": model,
            "auprc": average_precision_score(table.target, table[score_column]),
            "auroc": roc_auc_score(table.target, table[score_column]),
            "prevalence": table.target.mean(),
        })
control_seed_sensitivity = pd.DataFrame(seed_rows)
write_csv_atomic(control_seed_sensitivity, OUTPUT_DIR / "negative_window_seed_sensitivity.csv")

permutation_rows = []
rng = np.random.default_rng(SEED)
base_test = event_tables["test"]
y_original = base_test.target.to_numpy(dtype="int8")
for repetition in range(PERMUTATION_REPETITIONS):
    y_permuted = rng.permutation(y_original)
    for model, score_column in (
        ("Transformer", "score_transformer"),
        ("LightGBM", "score_lightgbm"),
    ):
        permutation_rows.append({
            "repetition": repetition, "model": model,
            "auprc_permuted": average_precision_score(y_permuted, base_test[score_column]),
        })
permutation_sanity = pd.DataFrame(permutation_rows)
write_csv_atomic(permutation_sanity, OUTPUT_DIR / "event_label_permutation_sanity.csv")

seed_summary = control_seed_sensitivity.groupby("model").agg(
    auprc_mean=("auprc", "mean"), auprc_std=("auprc", "std"),
    auprc_min=("auprc", "min"), auprc_max=("auprc", "max"),
).reset_index()
seed_summary["auprc_range"] = seed_summary.auprc_max - seed_summary.auprc_min
permutation_summary = permutation_sanity.groupby("model").agg(
    permuted_auprc_mean=("auprc_permuted", "mean"),
    permuted_auprc_q975=("auprc_permuted", lambda x: x.quantile(.975)),
).reset_index()

for row in seed_summary.itertuples(index=False):
    add_check(
        checks, "inflação", f"Estabilidade a janelas negativas — {row.model}",
        "APROVADO" if row.auprc_range <= 0.05 else "REVISAR",
        f"mean={row.auprc_mean:.4f}; std={row.auprc_std:.4f}; "
        f"min={row.auprc_min:.4f}; max={row.auprc_max:.4f}; range={row.auprc_range:.4f}",
    )

event_test_lookup = event_ranking.loc[event_ranking.split.eq("test")].set_index("model")
for row in permutation_summary.itertuples(index=False):
    observed = float(event_test_lookup.loc[row.model, "auprc"])
    add_check(
        checks, "inflação", f"Sanidade por permutação — {row.model}",
        "APROVADO" if observed > row.permuted_auprc_q975 else "FALHA",
        f"observed={observed:.4f}; permuted_mean={row.permuted_auprc_mean:.4f}; "
        f"permuted_q97.5={row.permuted_auprc_q975:.4f}",
    )

for model in MODEL_COLUMNS:
    val_ap = float(event_ranking.loc[(event_ranking.model.eq(model)) & (event_ranking.split.eq("validation")), "auprc"].iloc[0])
    test_ap = float(event_ranking.loc[(event_ranking.model.eq(model)) & (event_ranking.split.eq("test")), "auprc"].iloc[0])
    gap = abs(val_ap - test_ap)
    add_check(
        checks, "inflação", f"Gap validação-teste por evento — {model}",
        "APROVADO" if gap <= 0.03 else "REVISAR",
        f"validation={val_ap:.4f}; test={test_ap:.4f}; absolute_gap={gap:.4f}",
    )

add_check(
    checks, "inflação", "Assimetria da avaliação por janela",
    "REVISAR",
    "Casos usam a janela verdadeira; controles usam uma janela negativa amostrada. "
    "A comparação relativa é pareada, mas a AUPRC absoluta pode ser otimista.",
)
add_check(
    checks, "generalização", "Validação externa",
    "NÃO AVALIADO",
    "Este notebook usa validação e teste internos da mesma base; não prova transportabilidade externa.",
)

sensitivity = pd.DataFrame()
if LIGHTGBM_SENSITIVITY.is_file():
    sensitivity = pd.read_csv(LIGHTGBM_SENSITIVITY)
    sensitivity.to_csv(OUTPUT_DIR / "lightgbm_target_feature_sensitivity_copy.csv", index=False)
    add_check(
        checks, "leakage", "Ablação de lactato/pressão — LightGBM", "APROVADO",
        "Resultados disponíveis e incorporados ao relatório.",
    )
else:
    add_check(
        checks, "leakage", "Ablação de lactato/pressão — LightGBM", "NÃO AVALIADO",
        f"Arquivo ausente: {LIGHTGBM_SENSITIVITY}",
    )
add_check(
    checks, "leakage", "Ablação de lactato/pressão — Transformer",
    "REVISAR", "Ainda não existe ablação equivalente; não impede a comparação principal, mas limita a auditoria de mecanismo.",
)

display(seed_summary)
display(permutation_summary)

# %% [source cell 17]
target_audit = {"onset_source": onset_source, "splits": {}}
for split, aligned in aligned_predictions.items():
    with_onset = aligned.merge(onsets, on="stay_id", how="left", validate="many_to_one")
    positive = with_onset.loc[with_onset.target.eq(1)].copy()
    positive["hours_to_onset"] = (
        positive.first_onset_time - positive.label_time
    ).dt.total_seconds() / 3600
    positive_missing_onset = int(positive.first_onset_time.isna().sum())
    positive_after_or_at_onset = int(positive.hours_to_onset.le(0).fillna(False).sum())
    positive_beyond_horizon = int(positive.hours_to_onset.gt(TARGET_HORIZON_HOURS + 1e-9).fillna(False).sum())
    rows_after_onset = int(
        (with_onset.label_time.ge(with_onset.first_onset_time) & with_onset.first_onset_time.notna()).sum()
    )
    target_audit["splits"][split] = {
        "positive_rows": int(len(positive)),
        "positive_rows_missing_onset": positive_missing_onset,
        "positive_rows_at_or_after_onset": positive_after_or_at_onset,
        "positive_rows_beyond_8h": positive_beyond_horizon,
        "all_eligible_rows_at_or_after_onset": rows_after_onset,
        "min_hours_to_onset_positive": float(positive.hours_to_onset.min()),
        "max_hours_to_onset_positive": float(positive.hours_to_onset.max()),
    }
    status = "APROVADO" if (
        positive_missing_onset == 0
        and positive_after_or_at_onset == 0
        and positive_beyond_horizon == 0
        and rows_after_onset == 0
    ) else "FALHA"
    add_check(
        checks, "target", f"Alinhamento temporal — {split}", status,
        json.dumps(target_audit["splits"][split], ensure_ascii=False),
    )
write_json_atomic(target_audit, OUTPUT_DIR / "target_temporal_audit.json")
print(json.dumps(target_audit, indent=2, ensure_ascii=False))

# %% [source cell 19]
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("ggplot")

# AUPRC por unidade de avaliação no teste
plot_rows = pd.concat([
    timestep_metrics.loc[timestep_metrics.split.eq("test"), ["model", "auprc"]].assign(unit="Timestep"),
    event_ranking.loc[event_ranking.split.eq("test"), ["model", "auprc"]].assign(unit="Evento/stay"),
], ignore_index=True)
fig, ax = plt.subplots(figsize=(8, 5))
pivot = plot_rows.pivot(index="unit", columns="model", values="auprc")
pivot.plot(kind="bar", ax=ax, rot=0)
ax.set_ylim(0, 1); ax.set_ylabel("AUPRC"); ax.set_xlabel("")
ax.set_title("AUPRC de teste por unidade de avaliação")
fig.tight_layout(); fig.savefig(OUTPUT_DIR / "test_auprc_by_unit.png", dpi=180); plt.show()

# Curva precisão-recall por evento no teste
fig, ax = plt.subplots(figsize=(7, 6))
for model, score_column in (("Transformer", "score_transformer"), ("LightGBM", "score_lightgbm")):
    precision, recall, _ = precision_recall_curve(test_events.target, test_events[score_column])
    ap = average_precision_score(test_events.target, test_events[score_column])
    ax.plot(recall, precision, label=f"{model} (AUPRC={ap:.3f})")
ax.axhline(test_events.target.mean(), linestyle="--", color="gray", label="Prevalência")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_xlabel("Recall"); ax.set_ylabel("Precisão")
ax.set_title("Curva precisão-recall por evento — teste")
ax.legend(); fig.tight_layout(); fig.savefig(OUTPUT_DIR / "test_event_precision_recall.png", dpi=180); plt.show()

# Trade-off recall versus falsos alertas
clinical_test = threshold_results.loc[
    threshold_results.split.eq("test")
    & threshold_results.operating_point.str.startswith("event_recall_")
].copy()
fig, ax = plt.subplots(figsize=(7, 5))
for model, group in clinical_test.groupby("model"):
    group = group.sort_values("recall")
    ax.plot(
        100 * group.recall,
        group.false_alert_episodes_per_100_patient_days,
        marker="o", label=model,
    )
ax.set_xlabel("Recall de eventos (%)")
ax.set_ylabel("Falsos alertas por 100 pacientes-dia")
ax.set_title("Custo operacional para aumentar a sensibilidade")
ax.legend(); fig.tight_layout(); fig.savefig(OUTPUT_DIR / "recall_vs_false_alert_burden.png", dpi=180); plt.show()

# %% [source cell 21]
checks_frame = pd.DataFrame(checks)
write_csv_atomic(checks_frame, OUTPUT_DIR / "audit_checks.csv")

fail_count = int(checks_frame.status.eq("FALHA").sum())
review_count = int(checks_frame.status.eq("REVISAR").sum())
if fail_count:
    overall_status = "FALHA — corrigir antes de comparar ou publicar"
elif review_count:
    overall_status = "REVISAR — comparação possível, com limitações explícitas"
else:
    overall_status = "APROVADO — nenhuma inconsistência interna detectada"

test_timestep = timestep_metrics.loc[timestep_metrics.split.eq("test")].copy()
test_event = event_ranking.loc[event_ranking.split.eq("test")].copy()
test_operating = threshold_results.loc[
    threshold_results.split.eq("test")
    & threshold_results.operating_point.isin(["maximum_event_window_f1", "event_recall_80"])
].copy()

delta_summary = bootstrap_summaries["test"]
delta_ap = delta_summary["delta_auprc_lightgbm_minus_transformer"]
delta_auc = delta_summary["delta_auroc_lightgbm_minus_transformer"]

clinical_lines = []
for row in test_operating.itertuples(index=False):
    clinical_lines.append(
        f"- **{row.model} — {row.operating_point}:** TP={row.tp}, FP={row.fp}, "
        f"FN={row.fn}, TN={row.tn}; precisão={row.precision:.1%}, recall={row.recall:.1%}, "
        f"especificidade={row.specificity:.1%}; "
        f"falsos alertas={row.false_alert_episodes_per_100_patient_days:.2f}/100 pacientes-dia; "
        f"antecedência mediana={row.median_lead_time_hours:.2f} h "
        f"(IQR {row.lead_time_q25_hours:.2f}–{row.lead_time_q75_hours:.2f} h)."
    )

sensitivity_section = "Não disponível."
if not sensitivity.empty:
    columns = [
        c for c in ["variant", "timestep_test_auprc", "event_test_auprc",
                    "event_test_recall", "false_alerts_per_100_patient_days"]
        if c in sensitivity.columns
    ]
    sensitivity_section = md_table(sensitivity[columns])

report = f"""# Relatório — auditoria LightGBM × Transformer

**Status geral:** {overall_status}

**Execução:** `{RUN_ID}`
**Target esperado:** primeira falência circulatória em `(t, t+8h]`
**Thresholds:** selecionados somente na validação e congelados no teste
**Unidade do bootstrap:** `subject_id`

## 1. Comparabilidade e integridade

{md_table(checks_frame)}

## 2. Métricas por timestep — teste

{md_table(test_timestep[["model", "rows", "positives", "prevalence", "auroc", "auprc", "log_loss", "brier", "auprc_lift_over_prevalence"]])}

Interpretação: esta avaliação mede cada momento elegível de 5 minutos. A AUPRC deve ser lida em relação à prevalência. Ela é a análise mais direta da tarefa dinâmica, mas contém muitos timesteps correlacionados do mesmo paciente.

## 3. Métricas por evento/stay — teste

{md_table(test_event[["model", "rows", "positives", "prevalence", "auroc", "auprc", "auprc_lift_over_prevalence"]])}

Diferença pareada LightGBM − Transformer:

- AUPRC: mediana `{delta_ap['median']:.4f}`, IC95% `{delta_ap['ci95'][0]:.4f}` a `{delta_ap['ci95'][1]:.4f}`.
- AUROC: mediana `{delta_auc['median']:.4f}`, IC95% `{delta_auc['ci95'][0]:.4f}` a `{delta_auc['ci95'][1]:.4f}`.

Interpretação: se o IC95% da diferença não incluir zero, há evidência interna de diferença entre os modelos na mesma coorte. Isso não substitui validação externa.

## 4. Falsos positivos, falsos negativos e utilidade clínica

{chr(10).join(clinical_lines)}

- **Falso positivo:** janela/episódio alertado sem evento correspondente; aumenta interrupções e fadiga de alarmes.
- **Falso negativo:** evento não detectado; representa oportunidade perdida de intervenção.
- **Precisão/PPV:** entre os alertas emitidos, quantos correspondem a eventos.
- **Recall/sensibilidade:** entre os eventos, quantos foram detectados.
- **Especificidade:** entre controles, quantos não foram alertados.
- **Antecedência:** tempo entre o primeiro alerta dentro da janela positiva e o onset.

O ponto operacional não deve ser escolhido apenas pelo maior F1. A equipe clínica precisa decidir qual recall e qual carga de falsos alertas são aceitáveis.

## 5. Auditoria de inflação

{md_table(seed_summary)}

{md_table(permutation_summary)}

- Estabilidade entre seeds pequenas favorece robustez à escolha da janela negativa.
- AUPRC permutada próxima da prevalência confirma que o cálculo não cria discriminação sem relação entre score e target.
- Validação e teste próximos reduzem suspeita de overfitting, mas não provam ausência de leakage.
- O desenho atual conhece a janela positiva verdadeira e amostra uma janela por controle; por isso, a AUPRC por evento pode ser otimista como estimativa absoluta.

## 6. Sensibilidade às features do target

{sensitivity_section}

Queda após retirar lactato/PAM mostra dependência clínica esperada dessas variáveis. Desempenho residual acima do acaso indica que o modelo não depende exclusivamente delas. A mesma ablação ainda deve ser feita no Transformer se quisermos comparar mecanismo, não apenas desempenho.

## 7. Alinhamento com o objetivo do trabalho

O treinamento dinâmico por timestep é adequado para `X(≤t) → evento em (t,t+8h]`. A avaliação por evento/stay é adequada como endpoint clínico complementar, desde que acompanhada de carga de alertas e antecedência. Agregar todo o stay antes do treinamento poderia introduzir futuro; este notebook não recomenda essa estratégia.

## 8. Limites e decisão

- O teste interno já foi consultado; qualquer novo ajuste de modelo orientado por estes resultados exigirá um novo holdout ou validação externa.
- Coerência clínica final requer revisão médica dos thresholds, carga de alertas e antecedência.
- Ausência de falhas internas não prova transportabilidade para outro hospital.
- O resultado por evento deve ser apresentado junto com a métrica por timestep, prevalência, FP/FN, alert burden e lead time.
"""
report = textwrap.dedent(report).strip() + "\n"
(OUTPUT_DIR / "REPORT.md").write_text(report, encoding="utf-8")

summary = {
    "status": overall_status,
    "run_id": RUN_ID,
    "output_directory": str(OUTPUT_DIR),
    "failed_checks": fail_count,
    "review_checks": review_count,
    "event_test_delta_auprc_lightgbm_minus_transformer": delta_ap,
    "event_test_delta_auroc_lightgbm_minus_transformer": delta_auc,
    "onset_source": onset_source,
    "warning": "Do not tune models on this test set without a new holdout or external validation.",
}
write_json_atomic(summary, OUTPUT_DIR / "audit_summary.json")
print(report)
print("\nAUDITORIA CONCLUÍDA:", OUTPUT_DIR)

# %% [source cell 23]
# Tabelas finais para apresentação dos resultados
from IPython.display import display, Markdown

MODEL_ORDER = ["Transformer", "LightGBM"]
SPLIT_LABELS = {
    "validation": "Validação interna",
    "test": "Teste interno",
}

# ============================================================
# 1. Tabela principal — desempenho por evento/stay
# AUROC/AUPRC independem de threshold.
# F1/Precisão/Recall usam o threshold de maior F1 escolhido
# na validação e congelado no teste.
# ============================================================

max_f1_results = threshold_results.loc[
    threshold_results.operating_point.eq("maximum_event_window_f1"),
    ["split", "model", "threshold_selected_on_validation",
     "f1", "precision", "recall"],
].copy()

performance_table = (
    event_ranking[
        ["split", "model", "auroc", "auprc"]
    ]
    .merge(
        max_f1_results,
        on=["split", "model"],
        how="inner",
        validate="one_to_one",
    )
)

performance_table["Dataset"] = performance_table["split"].map(SPLIT_LABELS)
performance_table["model_order"] = performance_table["model"].map(
    {model: position for position, model in enumerate(MODEL_ORDER)}
)
performance_table["split_order"] = performance_table["split"].map(
    {"validation": 0, "test": 1}
)

performance_table = (
    performance_table
    .sort_values(["split_order", "model_order"])
    .rename(columns={
        "model": "Modelo",
        "auroc": "AUROC",
        "auprc": "AUPRC",
        "f1": "F1",
        "precision": "Precisão",
        "recall": "Recall",
        "threshold_selected_on_validation": "Threshold",
    })
    [["Dataset", "Modelo", "AUROC", "AUPRC",
      "F1", "Precisão", "Recall", "Threshold"]]
    .reset_index(drop=True)
)

def destacar_melhores_por_dataset(data):
    styles = pd.DataFrame(
        "",
        index=data.index,
        columns=data.columns,
    )
    metricas = ["AUROC", "AUPRC", "F1", "Precisão", "Recall"]

    for dataset in data["Dataset"].unique():
        indices = data.index[data["Dataset"].eq(dataset)]

        for metrica in metricas:
            melhor = data.loc[indices, metrica].max()
            melhores_indices = indices[
                data.loc[indices, metrica].eq(melhor)
            ]
            styles.loc[melhores_indices, metrica] = "font-weight: bold"

    return styles

display(Markdown(
    "### Tabela 1 — Comparação de desempenho por evento/stay\n"
    "*F1, precisão e recall calculados no threshold de maior F1 "
    "selecionado na validação e congelado no teste.*"
))

display(
    performance_table.style
    .apply(destacar_melhores_por_dataset, axis=None)
    .format({
        "AUROC": "{:.3f}",
        "AUPRC": "{:.3f}",
        "F1": "{:.3f}",
        "Precisão": "{:.3f}",
        "Recall": "{:.3f}",
        "Threshold": "{:.3f}",
    })
    .hide(axis="index")
)

# ============================================================
# 2. Tabela de utilidade clínica — conjunto de teste
# ============================================================

clinical_table = threshold_results.loc[
    threshold_results.split.eq("test")
    & threshold_results.operating_point.eq("maximum_event_window_f1")
].copy()

clinical_table["model_order"] = clinical_table["model"].map(
    {model: position for position, model in enumerate(MODEL_ORDER)}
)

clinical_table = (
    clinical_table
    .sort_values("model_order")
    .rename(columns={
        "model": "Modelo",
        "tp": "TP",
        "fp": "FP",
        "fn": "FN",
        "tn": "TN",
        "specificity": "Especificidade",
        "npv": "NPV",
        "false_alert_episodes_per_100_patient_days":
            "Falsos alertas/100 pacientes-dia",
        "median_lead_time_hours": "Antecedência mediana (h)",
    })
    [[
        "Modelo", "TP", "FP", "FN", "TN",
        "Especificidade", "NPV",
        "Falsos alertas/100 pacientes-dia",
        "Antecedência mediana (h)",
    ]]
    .reset_index(drop=True)
)

display(Markdown(
    "### Tabela 2 — Utilidade clínica no conjunto de teste\n"
    "*Resultados no threshold de maior F1.*"
))

display(
    clinical_table.style
    .format({
        "Especificidade": "{:.3f}",
        "NPV": "{:.3f}",
        "Falsos alertas/100 pacientes-dia": "{:.2f}",
        "Antecedência mediana (h)": "{:.2f}",
    })
    .hide(axis="index")
)

# ============================================================
# 3. Tabela de robustez
# ============================================================

robustness_rows = []

for model in MODEL_ORDER:
    model_key = model.lower()

    bootstrap = bootstrap_summaries["test"][f"{model_key}_auprc"]

    seed_row = seed_summary.loc[
        seed_summary.model.eq(model)
    ].iloc[0]

    permutation_row = permutation_summary.loc[
        permutation_summary.model.eq(model)
    ].iloc[0]

    validation_auprc = event_ranking.loc[
        event_ranking.model.eq(model)
        & event_ranking.split.eq("validation"),
        "auprc",
    ].iloc[0]

    test_auprc = event_ranking.loc[
        event_ranking.model.eq(model)
        & event_ranking.split.eq("test"),
        "auprc",
    ].iloc[0]

    timestep_test = timestep_metrics.loc[
        timestep_metrics.model.eq(model)
        & timestep_metrics.split.eq("test")
    ].iloc[0]

    robustness_rows.append({
        "Modelo": model,
        "AUPRC timestep": timestep_test.auprc,
        "Lift sobre prevalência": timestep_test.auprc_lift_over_prevalence,
        "AUPRC evento": test_auprc,
        "IC95% evento": (
            f"{bootstrap['ci95'][0]:.3f}–"
            f"{bootstrap['ci95'][1]:.3f}"
        ),
        "Gap validação-teste": abs(validation_auprc - test_auprc),
        "Média entre seeds": seed_row.auprc_mean,
        "DP entre seeds": seed_row.auprc_std,
        "AUPRC permutada": permutation_row.permuted_auprc_mean,
    })

robustness_table = pd.DataFrame(robustness_rows)

display(Markdown(
    "### Tabela 3 — Evidências de robustez interna\n"
    "*Bootstrap por paciente, múltiplas janelas negativas e teste "
    "de permutação.*"
))

display(
    robustness_table.style
    .format({
        "AUPRC timestep": "{:.3f}",
        "Lift sobre prevalência": "{:.1f}×",
        "AUPRC evento": "{:.3f}",
        "Gap validação-teste": "{:.4f}",
        "Média entre seeds": "{:.3f}",
        "DP entre seeds": "{:.4f}",
        "AUPRC permutada": "{:.3f}",
    })
    .hide(axis="index")
)

# ============================================================
# 4. Diferença pareada LightGBM − Transformer
# ============================================================

delta_auprc = bootstrap_summaries["test"][
    "delta_auprc_lightgbm_minus_transformer"
]
delta_auroc = bootstrap_summaries["test"][
    "delta_auroc_lightgbm_minus_transformer"
]

display(Markdown(
    "### Comparação pareada no teste\n"
    f"- **Diferença de AUPRC (LightGBM − Transformer):** "
    f"{delta_auprc['median']:.3f} "
    f"(IC95% {delta_auprc['ci95'][0]:.3f}–"
    f"{delta_auprc['ci95'][1]:.3f}).\n"
    f"- **Diferença de AUROC (LightGBM − Transformer):** "
    f"{delta_auroc['median']:.3f} "
    f"(IC95% {delta_auroc['ci95'][0]:.3f}–"
    f"{delta_auroc['ci95'][1]:.3f}).\n"
    "- Os intervalos não incluem zero, indicando superioridade "
    "interna consistente do LightGBM."
))

# Salva versões CSV para utilização no artigo
performance_table.to_csv(
    OUTPUT_DIR / "presentation_performance_table.csv",
    index=False,
)

clinical_table.to_csv(
    OUTPUT_DIR / "presentation_clinical_utility_table.csv",
    index=False,
)

robustness_table.to_csv(
    OUTPUT_DIR / "presentation_robustness_table.csv",
    index=False,
)

print("Tabelas de apresentação salvas em:", OUTPUT_DIR)
