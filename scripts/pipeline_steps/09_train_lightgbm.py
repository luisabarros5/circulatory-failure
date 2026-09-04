"""Source exported from the reviewed development notebook.

This file contains code only; patient-level data and notebook outputs are not distributed.
"""
from __future__ import annotations

# %% [source cell 2]
import gc
import hashlib
import json
import os
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import psycopg2
import pyarrow.parquet as pq

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)

PROJECT = Path("/path/to/circulatory_failure_prediction")

STEP7 = PROJECT / "outputs/step07_aligned_xy_v2"
STEP8 = PROJECT / "outputs/step08_validation_v2"
BATCH_DIR = STEP7 / "batches"
FEATURE_FILE = STEP7 / "feature_names.json"
MANIFEST7_FILE = STEP7 / "manifest_step7.json"
MANIFEST8_FILE = STEP8 / "manifest_step8.json"

STEP9 = PROJECT / "outputs/step09_lightgbm"
METADATA_DIR = PROJECT / "data/metadata"
SUBJECT_MAP_FILE = METADATA_DIR / "stay_subject_map.parquet"

TARGET = "target_first_onset_0_8h"
SEED = 42

print("Python:", sys.version.split()[0])
print("LightGBM:", lgb.__version__)
print("Pandas:", pd.__version__)
print("Projeto:", PROJECT)

# %% [source cell 4]
manifest7 = json.loads(MANIFEST7_FILE.read_text())
manifest8 = json.loads(MANIFEST8_FILE.read_text())
feature_registry = json.loads(FEATURE_FILE.read_text())

assert manifest7["status"] == "approved"
assert manifest8["status"] == "approved"
assert manifest8["forbidden_features_found"] == 0
assert manifest8["onset_and_post_onset_rows_found"] == 0
assert manifest8["duplicate_keys_found"] == 0
assert manifest8["unexplained_excluded_stays"] == 0
assert manifest7["target"] == TARGET

features = feature_registry["feature_names"]

assert len(features) == feature_registry["feature_count"] == 204
assert len(features) == len(set(features))
assert TARGET not in features

batch_files = sorted(BATCH_DIR.glob("batch_*.parquet"))

assert len(batch_files) == 285
assert [p.name for p in batch_files] == [
    f"batch_{i:04d}.parquet" for i in range(285)
]

schema = pq.read_schema(batch_files[0])
required = {"stay_id", "feature_time", "label_time", TARGET, *features}
missing = required - set(schema.names)

assert not missing, f"Colunas ausentes: {sorted(missing)}"

forbidden_terms = (
    "vasopress", "vasoactive", "vasoativo", "pressor",
    "norepine", "noradren", "epineph", "dopamin",
    "dobut", "milrin", "phenyleph", "inotrop",
    "onset", "post_onset", "failure_state",
)

forbidden_features = [
    name for name in features
    if any(term in name.lower() for term in forbidden_terms)
]

assert not forbidden_features, forbidden_features

print("Dataset:", manifest7["dataset_version"])
print("Linhas:", f'{manifest7["rows"]:,}')
print("Positivas:", f'{manifest7["positive_rows"]:,}')
print("Prevalência:", f'{manifest7["positive_prevalence"]:.4%}')
print("Features:", len(features))
print("Batches:", len(batch_files))
print("Alinhamento:", manifest8["alignment"])
print("Validação dos Steps 7 e 8: OK")

# %% [source cell 6]
METADATA_DIR.mkdir(parents=True, exist_ok=True)

if SUBJECT_MAP_FILE.exists():
    subject_map = pd.read_parquet(SUBJECT_MAP_FILE)
    print("Mapa existente carregado:", SUBJECT_MAP_FILE)
else:
    mimic_database_url = os.environ.get("MIMIC_DATABASE_URL")
    if not mimic_database_url:
        raise RuntimeError(
            "Configure MIMIC_DATABASE_URL securely outside the repository."
        )
    connection = psycopg2.connect(mimic_database_url)

    try:
        subject_map = pd.read_sql_query(
            """
            SELECT stay_id, subject_id
            FROM mimiciv_icu.icustays
            ORDER BY stay_id
            """,
            connection,
        )
    finally:
        connection.close()

    subject_map = subject_map.astype({
        "stay_id": "int64",
        "subject_id": "int64",
    })

    assert not subject_map["stay_id"].duplicated().any()

    subject_map.to_parquet(
        SUBJECT_MAP_FILE,
        index=False,
        compression="zstd",
    )

subject_map = subject_map[["stay_id", "subject_id"]].copy()
subject_map = subject_map.astype({
    "stay_id": "int64",
    "subject_id": "int64",
})

assert not subject_map["stay_id"].duplicated().any()
assert not subject_map.isna().any().any()

print("Pares stay–paciente:", f"{len(subject_map):,}")
print("Pacientes:", f'{subject_map["subject_id"].nunique():,}')

# %% [source cell 8]
parts = []

for number, path in enumerate(batch_files, start=1):
    frame = pd.read_parquet(path, columns=["stay_id", TARGET])

    stay_summary = (
        frame.groupby("stay_id", as_index=False)[TARGET]
        .max()
        .rename(columns={TARGET: "stay_positive"})
    )

    parts.append(stay_summary)

    if number % 50 == 0:
        print(f"Batches lidos: {number}/285")

stays = pd.concat(parts, ignore_index=True)
del parts
gc.collect()

assert not stays["stay_id"].duplicated().any()

stays = stays.merge(
    subject_map,
    on="stay_id",
    how="left",
    validate="one_to_one",
)

assert not stays["subject_id"].isna().any()

stays["subject_id"] = stays["subject_id"].astype("int64")

subjects = (
    stays.groupby("subject_id", as_index=False)["stay_positive"]
    .max()
    .rename(columns={"stay_positive": "subject_positive"})
)

rng = np.random.default_rng(SEED)
assignments = []

for positive_class, group in subjects.groupby("subject_positive"):
    ids = group["subject_id"].to_numpy(copy=True)
    rng.shuffle(ids)

    train_end = int(len(ids) * 0.70)
    validation_end = train_end + int(len(ids) * 0.15)

    assignments.extend([
        pd.DataFrame({
            "subject_id": ids[:train_end],
            "split": "train",
        }),
        pd.DataFrame({
            "subject_id": ids[train_end:validation_end],
            "split": "validation",
        }),
        pd.DataFrame({
            "subject_id": ids[validation_end:],
            "split": "test",
        }),
    ])

subject_splits = pd.concat(assignments, ignore_index=True)

assert not subject_splits["subject_id"].duplicated().any()

stay_splits = stays.merge(
    subject_splits,
    on="subject_id",
    how="left",
    validate="many_to_one",
)

sets = {
    split: set(
        subject_splits.loc[
            subject_splits["split"].eq(split),
            "subject_id",
        ]
    )
    for split in ("train", "validation", "test")
}

assert sets["train"].isdisjoint(sets["validation"])
assert sets["train"].isdisjoint(sets["test"])
assert sets["validation"].isdisjoint(sets["test"])

STEP9.mkdir(parents=True, exist_ok=True)

stay_splits.to_parquet(
    STEP9 / "stay_subject_split_seed42.parquet",
    index=False,
    compression="zstd",
)

display(
    stay_splits.groupby("split").agg(
        patients=("subject_id", "nunique"),
        stays=("stay_id", "nunique"),
        positive_stays=("stay_positive", "sum"),
    )
)

print("Split por paciente: OK")

# %% [source cell 10]
def stable_hash(frame, salt):
    keys = frame[
        ["stay_id", "label_time"]
    ].copy()

    keys["_sampling_salt"] = str(salt)

    return pd.util.hash_pandas_object(
        keys,
        index=False,
    ).to_numpy(dtype="uint64")


def stable_negative_sample(
    frame,
    fraction,
    seed,
):
    if fraction >= 1:
        return pd.Series(
            True,
            index=frame.index,
        )

    hashes = stable_hash(
        frame,
        salt=f"negative-sampling-{seed}",
    )

    threshold = np.uint64(
        fraction *
        np.iinfo(np.uint64).max
    )

    return pd.Series(
        hashes <= threshold,
        index=frame.index,
    )


def collect_split(
    split,
    max_rows,
    negative_fraction=1.0,
):
    allowed = stay_splits.loc[
        stay_splits["split"].eq(split),
        ["stay_id", "subject_id"],
    ].set_index("stay_id")["subject_id"]

    columns = [
        "stay_id",
        "label_time",
        TARGET,
        *features,
    ]

    reservoir = None

    split_seed = {
        "train": SEED + 11,
        "validation": SEED + 22,
        "test": SEED + 33,
    }[split]

    for batch_number, path in enumerate(batch_files):
        frame = pd.read_parquet(
            path,
            columns=columns,
        )

        frame = frame.loc[
            frame["stay_id"].isin(allowed.index)
        ].copy()

        if frame.empty:
            continue

        if split == "train" and negative_fraction < 1:
            sampled_negative = stable_negative_sample(
                frame,
                negative_fraction,
                SEED,
            )

            keep = (
                frame[TARGET].eq(1) |
                (
                    frame[TARGET].eq(0) &
                    sampled_negative
                )
            )

            frame = frame.loc[keep].copy()

        if frame.empty:
            continue

        priority = stable_hash(
            frame,
            salt=f"reservoir-{split}-{split_seed}",
        )

        frame["_sample_priority"] = priority

        if reservoir is None:
            reservoir = frame
        else:
            reservoir = pd.concat(
                [reservoir, frame],
                ignore_index=True,
            )

        if max_rows is not None and len(reservoir) > max_rows:
            reservoir = (
                reservoir
                .nsmallest(
                    max_rows,
                    "_sample_priority",
                )
                .reset_index(drop=True)
            )

        del frame
        gc.collect()

        if (batch_number + 1) % 50 == 0:
            print(
                f"{split}: "
                f"{batch_number + 1}/285 batches; "
                f"{len(reservoir):,} linhas selecionadas"
            )

    if reservoir is None or reservoir.empty:
        raise RuntimeError(
            f"Nenhuma linha encontrada para {split}"
        )

    if max_rows is not None:
        reservoir = (
            reservoir
            .nsmallest(
                max_rows,
                "_sample_priority",
            )
            .reset_index(drop=True)
        )

    reservoir["subject_id"] = (
        reservoir["stay_id"]
        .map(allowed)
    )

    if reservoir["subject_id"].isna().any():
        raise RuntimeError(
            f"Foram encontrados stays sem subject_id no split {split}"
        )

    reservoir["subject_id"] = (
        reservoir["subject_id"]
        .astype("int64")
    )

    y = reservoir.pop(TARGET).to_numpy(
        dtype="int8"
    )

    metadata = reservoir[
        [
            "stay_id",
            "subject_id",
            "label_time",
        ]
    ].copy()

    x = (
        reservoir[features]
        .replace([np.inf, -np.inf], np.nan)
        .astype("float32")
    )

    weights = np.ones(
        len(y),
        dtype="float32",
    )

    if split == "train" and negative_fraction < 1:
        weights[y == 0] = (
            1.0 / negative_fraction
        )

    if list(x.columns) != features:
        raise RuntimeError(
            "A ordem das features foi alterada"
        )

    if len(np.unique(y)) != 2:
        raise RuntimeError(
            f"O split {split} não contém as duas classes"
        )

    del reservoir
    gc.collect()

    return x, y, weights, metadata


def basic_metrics(
    y,
    probability,
):
    return {
        "rows": int(len(y)),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()),
        "auroc": float(
            roc_auc_score(
                y,
                probability,
            )
        ),
        "auprc": float(
            average_precision_score(
                y,
                probability,
            )
        ),
        "log_loss": float(
            log_loss(
                y,
                probability,
                labels=[0, 1],
            )
        ),
        "brier": float(
            brier_score_loss(
                y,
                probability,
            )
        ),
    }


def choose_threshold(
    y,
    probability,
):
    precision, recall, thresholds = (
        precision_recall_curve(
            y,
            probability,
        )
    )

    f1 = (
        2 *
        precision[:-1] *
        recall[:-1] /
        np.maximum(
            precision[:-1] + recall[:-1],
            1e-12,
        )
    )

    return float(
        thresholds[np.nanargmax(f1)]
    )


def threshold_metrics(
    y,
    probability,
    threshold,
):
    predicted = (
        probability >= threshold
    ).astype("int8")

    tn, fp, fn, tp = confusion_matrix(
        y,
        predicted,
        labels=[0, 1],
    ).ravel()

    sensitivity = (
        tp / (tp + fn)
        if tp + fn > 0
        else np.nan
    )

    specificity = (
        tn / (tn + fp)
        if tn + fp > 0
        else np.nan
    )

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else np.nan
    )

    npv = (
        tn / (tn + fn)
        if tn + fn > 0
        else np.nan
    )

    return {
        "threshold": float(threshold),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "npv": float(npv),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def grouped_metrics(
    y,
    probability,
    metadata,
    group_column,
):
    scored = metadata[
        [group_column]
    ].copy()

    scored["target"] = y
    scored["probability"] = probability

    grouped = (
        scored
        .groupby(
            group_column,
            as_index=False,
        )
        .agg(
            target=("target", "max"),
            probability=("probability", "max"),
        )
    )

    return basic_metrics(
        grouped["target"].to_numpy(
            dtype="int8"
        ),
        grouped["probability"].to_numpy(
            dtype="float64"
        ),
    )

# %% [source cell 12]
SMOKE_DIR = STEP9 / "smoke_v4"

if SMOKE_DIR.exists():
    raise FileExistsError(
        f"{SMOKE_DIR} já existe. Não vou sobrescrever."
    )

SMOKE_DIR.mkdir(parents=True)

negative_fraction = min(
    1.0,
    5 *
    manifest7["positive_rows"] /
    manifest7["negative_rows"],
)

print(
    "Fração de negativas no treino:",
    negative_fraction,
)

x_train, y_train, train_weights, train_meta = collect_split(
    split="train",
    max_rows=100_000,
    negative_fraction=negative_fraction,
)

x_validation, y_validation, _, validation_meta = collect_split(
    split="validation",
    max_rows=50_000,
)

x_test, y_test, _, test_meta = collect_split(
    split="test",
    max_rows=50_000,
)

weighted_prevalence = np.average(
    y_train,
    weights=train_weights,
)

print(
    "Treino:",
    x_train.shape,
    "prevalência bruta:",
    y_train.mean(),
)

print(
    "Prevalência ponderada do treino:",
    weighted_prevalence,
)

print(
    "Validação:",
    x_validation.shape,
    "prevalência:",
    y_validation.mean(),
)

print(
    "Teste:",
    x_test.shape,
    "prevalência:",
    y_test.mean(),
)

model = lgb.LGBMClassifier(
    objective="binary",
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=100,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    n_jobs=-1,
    random_state=SEED,
    verbosity=-1,
)

model.fit(
    x_train,
    y_train,
    sample_weight=train_weights,
    eval_set=[
        (
            x_validation,
            y_validation,
        )
    ],
    eval_metric=[
        "binary_logloss",
        "auc",
        "average_precision",
    ],
    callbacks=[
        lgb.early_stopping(20),
        lgb.log_evaluation(10),
    ],
)

validation_probability = model.predict_proba(
    x_validation,
    num_iteration=model.best_iteration_,
)[:, 1]

threshold = choose_threshold(
    y_validation,
    validation_probability,
)

test_probability = model.predict_proba(
    x_test,
    num_iteration=model.best_iteration_,
)[:, 1]

results = {
    "status": "smoke_completed",
    "target": TARGET,
    "alignment": manifest8["alignment"],
    "features": len(features),
    "best_boosting_iteration": int(
        model.best_iteration_
    ),
    "normalization": False,
    "missing_values": (
        "LightGBM native handling"
    ),
    "training_raw_prevalence": float(
        y_train.mean()
    ),
    "training_weighted_prevalence": float(
        weighted_prevalence
    ),
    "negative_sampling_fraction": float(
        negative_fraction
    ),
    "validation": {
        "timestep": basic_metrics(
            y_validation,
            validation_probability,
        ),
        "stay": grouped_metrics(
            y_validation,
            validation_probability,
            validation_meta,
            "stay_id",
        ),
        "patient": grouped_metrics(
            y_validation,
            validation_probability,
            validation_meta,
            "subject_id",
        ),
        "classification": threshold_metrics(
            y_validation,
            validation_probability,
            threshold,
        ),
    },
    "test": {
        "timestep": basic_metrics(
            y_test,
            test_probability,
        ),
        "stay": grouped_metrics(
            y_test,
            test_probability,
            test_meta,
            "stay_id",
        ),
        "patient": grouped_metrics(
            y_test,
            test_probability,
            test_meta,
            "subject_id",
        ),
        "classification": threshold_metrics(
            y_test,
            test_probability,
            threshold,
        ),
    },
}

joblib.dump(
    model,
    SMOKE_DIR / "model.joblib",
)

model.booster_.save_model(
    SMOKE_DIR / "model.txt",
    num_iteration=model.best_iteration_,
)

importance = pd.DataFrame({
    "feature": features,
    "gain": model.booster_.feature_importance(
        importance_type="gain"
    ),
    "split": model.booster_.feature_importance(
        importance_type="split"
    ),
}).sort_values(
    "gain",
    ascending=False,
)

importance.to_csv(
    SMOKE_DIR / "feature_importance.csv",
    index=False,
)

with open(
    SMOKE_DIR / "metrics.json",
    "w",
) as file:
    json.dump(
        results,
        file,
        indent=2,
    )

print(
    json.dumps(
        results,
        indent=2,
    )
)

display(
    importance.head(20)
)

# %% [source cell 14]
import shutil


def available_memory_bytes():
    values = {}

    with open("/proc/meminfo") as file:
        for line in file:
            name, value = line.split(":", 1)

            values[name] = int(
                value.strip().split()[0]
            ) * 1024

    return (
        values["MemTotal"],
        values["MemAvailable"],
    )


split_lookup = (
    stay_splits
    .set_index("stay_id")["split"]
)

counts = {
    split: {
        "rows": 0,
        "positives": 0,
        "negatives": 0,
        "stays": set(),
    }
    for split in (
        "train",
        "validation",
        "test",
    )
}

for batch_number, path in enumerate(
    batch_files,
    start=1,
):
    frame = pd.read_parquet(
        path,
        columns=[
            "stay_id",
            TARGET,
        ],
    )

    frame["split"] = (
        frame["stay_id"]
        .map(split_lookup)
    )

    if frame["split"].isna().any():
        raise RuntimeError(
            f"Existem linhas sem split em {path.name}"
        )

    for split in (
        "train",
        "validation",
        "test",
    ):
        selected = frame.loc[
            frame["split"].eq(split),
            [
                "stay_id",
                TARGET,
            ],
        ]

        positive_rows = int(
            selected[TARGET].sum()
        )

        counts[split]["rows"] += len(
            selected
        )

        counts[split]["positives"] += (
            positive_rows
        )

        counts[split]["negatives"] += (
            len(selected) -
            positive_rows
        )

        counts[split]["stays"].update(
            selected["stay_id"].unique()
        )

    del frame
    gc.collect()

    if batch_number % 50 == 0:
        print(
            f"Batches contados: "
            f"{batch_number}/285"
        )


summary_rows = []

for split, values in counts.items():
    summary_rows.append({
        "split": split,
        "rows": values["rows"],
        "positives": values["positives"],
        "negatives": values["negatives"],
        "prevalence": (
            values["positives"] /
            values["rows"]
        ),
        "stays": len(values["stays"]),
        "patients": int(
            stay_splits.loc[
                stay_splits["split"].eq(split),
                "subject_id",
            ].nunique()
        ),
    })

split_summary = pd.DataFrame(
    summary_rows
)

train_values = counts["train"]

estimated_sampled_train_rows = int(
    train_values["positives"] +
    (
        train_values["negatives"] *
        negative_fraction
    )
)

bytes_per_matrix_row = (
    len(features) *
    np.dtype("float32").itemsize
)

estimated_train_matrix_bytes = (
    estimated_sampled_train_rows *
    bytes_per_matrix_row
)

estimated_validation_matrix_bytes = (
    counts["validation"]["rows"] *
    bytes_per_matrix_row
)

estimated_test_matrix_bytes = (
    counts["test"]["rows"] *
    bytes_per_matrix_row
)

# Durante o fit, treino e validação ficam
# simultaneamente na memória. O fator adicional
# cobre DataFrames, Dataset interno do LightGBM,
# pesos, índices e estruturas auxiliares.
estimated_peak_bytes = int(
    (
        estimated_train_matrix_bytes +
        estimated_validation_matrix_bytes
    ) * 2.5
)

total_ram, available_ram = (
    available_memory_bytes()
)

disk = shutil.disk_usage(PROJECT)

resource_summary = {
    "feature_count": len(features),
    "estimated_sampled_train_rows": (
        estimated_sampled_train_rows
    ),
    "estimated_train_matrix_gb": (
        estimated_train_matrix_bytes /
        1024**3
    ),
    "estimated_validation_matrix_gb": (
        estimated_validation_matrix_bytes /
        1024**3
    ),
    "estimated_test_matrix_gb": (
        estimated_test_matrix_bytes /
        1024**3
    ),
    "estimated_peak_training_gb": (
        estimated_peak_bytes /
        1024**3
    ),
    "total_ram_gb": (
        total_ram /
        1024**3
    ),
    "available_ram_gb": (
        available_ram /
        1024**3
    ),
    "project_disk_free_gb": (
        disk.free /
        1024**3
    ),
    "estimated_peak_fraction_of_available_ram": (
        estimated_peak_bytes /
        available_ram
    ),
}

display(split_summary)

print(
    json.dumps(
        resource_summary,
        indent=2,
    )
)

if estimated_peak_bytes > (
    available_ram * 0.65
):
    print(
        "\nATENÇÃO: não carregar validação completa "
        "junto com o treino. Será necessário fazer "
        "validação e teste em batches."
    )
else:
    print(
        "\nA estimativa indica que o treinamento "
        "pode caber na memória disponível."
    )

# %% [source cell 16]
import time

import matplotlib.pyplot as plt


BENCHMARK_DIR = (
    STEP9 /
    "benchmark_medium_v1"
)

if BENCHMARK_DIR.exists():
    raise FileExistsError(
        f"{BENCHMARK_DIR} já existe. "
        "Não vou sobrescrever."
    )

BENCHMARK_DIR.mkdir(
    parents=True,
    exist_ok=False,
)

benchmark_started = time.perf_counter()

print(
    "Preparando 500.000 linhas "
    "representativas de treino..."
)

train_loading_started = time.perf_counter()

(
    x_benchmark_train,
    y_benchmark_train,
    benchmark_weights,
    benchmark_train_meta,
) = collect_split(
    split="train",
    max_rows=500_000,
    negative_fraction=negative_fraction,
)

train_loading_seconds = (
    time.perf_counter() -
    train_loading_started
)

print()
print(
    "Preparando 250.000 linhas "
    "representativas de validação..."
)

validation_loading_started = (
    time.perf_counter()
)

(
    x_benchmark_validation,
    y_benchmark_validation,
    _,
    benchmark_validation_meta,
) = collect_split(
    split="validation",
    max_rows=250_000,
    negative_fraction=1.0,
)

validation_loading_seconds = (
    time.perf_counter() -
    validation_loading_started
)

benchmark_weighted_prevalence = (
    np.average(
        y_benchmark_train,
        weights=benchmark_weights,
    )
)

print()
print(
    "Treino:",
    x_benchmark_train.shape,
)

print(
    "Prevalência bruta do treino:",
    f"{y_benchmark_train.mean():.4%}",
)

print(
    "Prevalência ponderada do treino:",
    f"{benchmark_weighted_prevalence:.4%}",
)

print(
    "Validação:",
    x_benchmark_validation.shape,
)

print(
    "Prevalência da validação:",
    f"{y_benchmark_validation.mean():.4%}",
)

benchmark_model = lgb.LGBMClassifier(
    objective="binary",
    metric=[
        "average_precision",
        "binary_logloss",
    ],
    n_estimators=500,
    learning_rate=0.03,
    num_leaves=63,
    max_depth=-1,
    min_child_samples=200,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=1.0,
    n_jobs=-1,
    random_state=SEED,
    deterministic=True,
    force_col_wise=True,
    verbosity=-1,
)

print()
print(
    "Iniciando benchmark do LightGBM..."
)

fit_started = time.perf_counter()

benchmark_model.fit(
    x_benchmark_train,
    y_benchmark_train,
    sample_weight=benchmark_weights,
    eval_set=[
        (
            x_benchmark_validation,
            y_benchmark_validation,
        )
    ],
    eval_metric=[
        "average_precision",
        "binary_logloss",
    ],
    callbacks=[
        lgb.early_stopping(
            stopping_rounds=50,
            first_metric_only=True,
            verbose=True,
        ),
        lgb.log_evaluation(
            period=10,
        ),
    ],
)

fit_seconds = (
    time.perf_counter() -
    fit_started
)

best_iteration = int(
    benchmark_model.best_iteration_
)

benchmark_probability = (
    benchmark_model.predict_proba(
        x_benchmark_validation,
        num_iteration=best_iteration,
    )[:, 1]
)

benchmark_metrics = basic_metrics(
    y_benchmark_validation,
    benchmark_probability,
)

evaluation_history = (
    benchmark_model.evals_result_[
        "valid_0"
    ]
)

available_metric_names = list(
    evaluation_history
)

average_precision_name = next(
    name
    for name in available_metric_names
    if name in {
        "average_precision",
        "average precision",
    }
)

binary_logloss_name = next(
    name
    for name in available_metric_names
    if name in {
        "binary_logloss",
        "binary logloss",
    }
)

average_precision_history = (
    evaluation_history[
        average_precision_name
    ]
)

binary_logloss_history = (
    evaluation_history[
        binary_logloss_name
    ]
)

iterations = np.arange(
    1,
    len(average_precision_history) + 1,
)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 4.5),
)

axes[0].plot(
    iterations,
    average_precision_history,
)

axes[0].axvline(
    best_iteration,
    linestyle="--",
    color="red",
    label=(
        f"Melhor rodada: "
        f"{best_iteration}"
    ),
)

axes[0].set_xlabel(
    "Boosting iteration"
)

axes[0].set_ylabel(
    "AUPRC"
)

axes[0].set_title(
    "AUPRC por rodada"
)

axes[0].legend()

axes[1].plot(
    iterations,
    binary_logloss_history,
)

axes[1].axvline(
    best_iteration,
    linestyle="--",
    color="red",
    label=(
        f"Melhor rodada: "
        f"{best_iteration}"
    ),
)

axes[1].set_xlabel(
    "Boosting iteration"
)

axes[1].set_ylabel(
    "Binary log-loss"
)

axes[1].set_title(
    "Loss por rodada"
)

axes[1].legend()

fig.tight_layout()

fig.savefig(
    BENCHMARK_DIR /
    "learning_curves.png",
    dpi=200,
    bbox_inches="tight",
)

plt.show()

full_training_rows = (
    estimated_sampled_train_rows
)

planned_validation_rows = 500_000

benchmark_workload = (
    len(y_benchmark_train) +
    len(y_benchmark_validation)
)

planned_workload = (
    full_training_rows +
    planned_validation_rows
)

workload_factor = (
    planned_workload /
    benchmark_workload
)

estimated_fit_seconds = (
    fit_seconds *
    workload_factor
)

# Acrescentamos uma margem porque o modelo completo
# pode precisar de mais rodadas e mais memória.
estimated_lower_seconds = (
    estimated_fit_seconds * 1.0
)

estimated_upper_seconds = (
    estimated_fit_seconds * 2.5
)

total_benchmark_seconds = (
    time.perf_counter() -
    benchmark_started
)

benchmark_results = {
    "status": "completed",
    "train_rows": int(
        len(y_benchmark_train)
    ),
    "validation_rows": int(
        len(y_benchmark_validation)
    ),
    "training_raw_prevalence": float(
        y_benchmark_train.mean()
    ),
    "training_weighted_prevalence": float(
        benchmark_weighted_prevalence
    ),
    "validation_prevalence": float(
        y_benchmark_validation.mean()
    ),
    "best_boosting_iteration": (
        best_iteration
    ),
    "train_loading_minutes": (
        train_loading_seconds / 60
    ),
    "validation_loading_minutes": (
        validation_loading_seconds / 60
    ),
    "fit_minutes": (
        fit_seconds / 60
    ),
    "total_benchmark_minutes": (
        total_benchmark_seconds / 60
    ),
    "estimated_full_fit_minutes_lower": (
        estimated_lower_seconds / 60
    ),
    "estimated_full_fit_minutes_upper": (
        estimated_upper_seconds / 60
    ),
    "validation_metrics": (
        benchmark_metrics
    ),
    "lightgbm_terms": {
        "training_steps": (
            "boosting iterations"
        ),
        "optimizer": (
            "gradient boosting; "
            "Adam is not used"
        ),
        "loss": "binary_logloss",
        "early_stopping_metric": (
            "average_precision"
        ),
    },
}

with open(
    BENCHMARK_DIR /
    "benchmark_metrics.json",
    "w",
) as file:
    json.dump(
        benchmark_results,
        file,
        indent=2,
    )

joblib.dump(
    benchmark_model,
    BENCHMARK_DIR /
    "benchmark_model.joblib",
)

print()
print(
    json.dumps(
        benchmark_results,
        indent=2,
    )
)

del (
    x_benchmark_train,
    y_benchmark_train,
    benchmark_weights,
    benchmark_train_meta,
    x_benchmark_validation,
    y_benchmark_validation,
    benchmark_validation_meta,
    benchmark_probability,
)

gc.collect()

# %% [source cell 18]
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


BENCHMARK_DIR = (
    STEP9 /
    "benchmark_medium_v1"
)

benchmark_model = joblib.load(
    BENCHMARK_DIR /
    "benchmark_model.joblib"
)

print(
    "Carregando novamente a amostra "
    "representativa de validação..."
)

(
    x_threshold_validation,
    y_threshold_validation,
    _,
    threshold_validation_meta,
) = collect_split(
    split="validation",
    max_rows=250_000,
    negative_fraction=1.0,
)

validation_probability = (
    benchmark_model.predict_proba(
        x_threshold_validation,
        num_iteration=(
            benchmark_model.best_iteration_
        ),
    )[:, 1]
)

precision_curve, recall_curve, thresholds = (
    precision_recall_curve(
        y_threshold_validation,
        validation_probability,
    )
)

f1_curve = (
    2 *
    precision_curve[:-1] *
    recall_curve[:-1] /
    np.maximum(
        precision_curve[:-1] +
        recall_curve[:-1],
        1e-12,
    )
)

best_threshold_index = int(
    np.nanargmax(f1_curve)
)

selected_threshold = float(
    thresholds[
        best_threshold_index
    ]
)

predicted_class = (
    validation_probability >=
    selected_threshold
).astype("int8")

tn, fp, fn, tp = confusion_matrix(
    y_threshold_validation,
    predicted_class,
    labels=[0, 1],
).ravel()

classification_results = {
    "threshold_selection": (
        "maximum F1 on validation"
    ),
    "threshold": selected_threshold,
    "accuracy": float(
        accuracy_score(
            y_threshold_validation,
            predicted_class,
        )
    ),
    "precision": float(
        precision_score(
            y_threshold_validation,
            predicted_class,
            zero_division=0,
        )
    ),
    "recall_sensitivity": float(
        recall_score(
            y_threshold_validation,
            predicted_class,
            zero_division=0,
        )
    ),
    "f1_score": float(
        f1_score(
            y_threshold_validation,
            predicted_class,
            zero_division=0,
        )
    ),
    "specificity": float(
        tn / (tn + fp)
    ),
    "negative_predictive_value": float(
        tn / (tn + fn)
    ),
    "confusion_matrix": {
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    },
}

with open(
    BENCHMARK_DIR /
    "classification_metrics.json",
    "w",
) as file:
    json.dump(
        classification_results,
        file,
        indent=2,
    )

fig, axis = plt.subplots(
    figsize=(5.5, 5),
)

ConfusionMatrixDisplay(
    confusion_matrix=np.array([
        [tn, fp],
        [fn, tp],
    ]),
    display_labels=[
        "Sem falência",
        "Falência",
    ],
).plot(
    ax=axis,
    values_format=",",
    colorbar=False,
)

axis.set_title(
    "Matriz de confusão — validação"
)

fig.tight_layout()

fig.savefig(
    BENCHMARK_DIR /
    "validation_confusion_matrix.png",
    dpi=200,
    bbox_inches="tight",
)

plt.show()

print(
    json.dumps(
        classification_results,
        indent=2,
    )
)

del (
    x_threshold_validation,
    y_threshold_validation,
    threshold_validation_meta,
    validation_probability,
    predicted_class,
)

gc.collect()

# %% [source cell 20]
import time

from datetime import datetime, timezone

import matplotlib.pyplot as plt


def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def collect_full_split(
    split,
    negative_fraction=1.0,
):
    allowed = stay_splits.loc[
        stay_splits["split"].eq(split),
        ["stay_id", "subject_id"],
    ].set_index("stay_id")["subject_id"]

    columns = [
        "stay_id",
        "label_time",
        TARGET,
        *features,
    ]

    parts = []
    accumulated_rows = 0

    for batch_number, path in enumerate(
        batch_files,
        start=1,
    ):
        frame = pd.read_parquet(
            path,
            columns=columns,
        )

        frame = frame.loc[
            frame["stay_id"].isin(
                allowed.index
            )
        ].copy()

        if frame.empty:
            continue

        if (
            split == "train"
            and negative_fraction < 1
        ):
            sampled_negative = (
                stable_negative_sample(
                    frame,
                    negative_fraction,
                    SEED,
                )
            )

            keep = (
                frame[TARGET].eq(1)
                |
                (
                    frame[TARGET].eq(0)
                    & sampled_negative
                )
            )

            frame = frame.loc[keep].copy()

        if not frame.empty:
            parts.append(frame)
            accumulated_rows += len(frame)

        if batch_number % 25 == 0:
            print(
                f"{split}: "
                f"{batch_number}/285 batches; "
                f"{accumulated_rows:,} linhas"
            )

        del frame
        gc.collect()

    if not parts:
        raise RuntimeError(
            f"Nenhuma linha encontrada para {split}"
        )

    print(
        f"{split}: consolidando "
        f"{accumulated_rows:,} linhas..."
    )

    data = pd.concat(
        parts,
        ignore_index=True,
    )

    del parts
    gc.collect()

    data["subject_id"] = (
        data["stay_id"]
        .map(allowed)
    )

    if data["subject_id"].isna().any():
        raise RuntimeError(
            f"Existem stays sem subject_id "
            f"no split {split}"
        )

    data["subject_id"] = (
        data["subject_id"]
        .astype("int64")
    )

    y = data.pop(TARGET).to_numpy(
        dtype="int8"
    )

    metadata = data[
        [
            "stay_id",
            "subject_id",
            "label_time",
        ]
    ].copy()

    x = (
        data[features]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .astype("float32")
    )

    weights = np.ones(
        len(y),
        dtype="float32",
    )

    if (
        split == "train"
        and negative_fraction < 1
    ):
        weights[y == 0] = (
            1.0 / negative_fraction
        )

    if list(x.columns) != features:
        raise RuntimeError(
            "A ordem das features foi alterada"
        )

    if len(np.unique(y)) != 2:
        raise RuntimeError(
            f"O split {split} não contém "
            "as duas classes"
        )

    del data
    gc.collect()

    print(
        f"{split} pronto: "
        f"{len(y):,} linhas; "
        f"prevalência {y.mean():.4%}"
    )

    return x, y, weights, metadata


def classification_at_threshold(
    y,
    probability,
    threshold,
):
    prediction = (
        probability >= threshold
    ).astype("int8")

    tn, fp, fn, tp = confusion_matrix(
        y,
        prediction,
        labels=[0, 1],
    ).ravel()

    precision = (
        tp / (tp + fp)
        if tp + fp
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if tn + fp
        else 0.0
    )

    npv = (
        tn / (tn + fn)
        if tn + fn
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall_sensitivity": float(recall),
        "f1_score": float(f1),
        "specificity": float(specificity),
        "negative_predictive_value": float(npv),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def select_validation_thresholds(
    y,
    probability,
):
    precision, recall, thresholds = (
        precision_recall_curve(
            y,
            probability,
        )
    )

    precision = precision[:-1]
    recall = recall[:-1]

    f1 = (
        2 * precision * recall
        / np.maximum(
            precision + recall,
            1e-12,
        )
    )

    best_f1_index = int(
        np.nanargmax(f1)
    )

    selected = {
        "maximum_f1": float(
            thresholds[best_f1_index]
        )
    }

    for target_recall in (
        0.60,
        0.70,
        0.80,
        0.90,
    ):
        candidates = np.flatnonzero(
            recall >= target_recall
        )

        if len(candidates) == 0:
            continue

        candidate_precision = (
            precision[candidates]
        )

        best_candidate = int(
            candidates[
                np.nanargmax(
                    candidate_precision
                )
            ]
        )

        selected[
            f"recall_{int(target_recall * 100)}"
        ] = float(
            thresholds[best_candidate]
        )

    return selected


def evaluate_thresholds(
    y,
    probability,
    threshold_dictionary,
):
    return {
        name: classification_at_threshold(
            y,
            probability,
            threshold,
        )
        for name, threshold
        in threshold_dictionary.items()
    }


def save_predictions(
    metadata,
    y,
    probability,
    output_file,
):
    predictions = metadata.copy()

    predictions["target"] = y

    predictions["probability"] = (
        probability.astype("float32")
    )

    predictions.to_parquet(
        output_file,
        index=False,
        compression="zstd",
    )

    del predictions
    gc.collect()


def plot_confusion_matrix(
    metrics,
    title,
    output_file,
):
    matrix = np.array([
        [
            metrics["true_negatives"],
            metrics["false_positives"],
        ],
        [
            metrics["false_negatives"],
            metrics["true_positives"],
        ],
    ])

    fig, axis = plt.subplots(
        figsize=(5.5, 5),
    )

    display_matrix = (
        ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=[
                "Sem falência",
                "Falência",
            ],
        )
    )

    display_matrix.plot(
        ax=axis,
        values_format=",",
        colorbar=False,
    )

    axis.set_title(title)

    fig.tight_layout()

    fig.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


manifest_hashes_before = {
    "step7": sha256_file(
        MANIFEST7_FILE
    ),
    "step8": sha256_file(
        MANIFEST8_FILE
    ),
}

assert sets["train"].isdisjoint(
    sets["validation"]
)

assert sets["train"].isdisjoint(
    sets["test"]
)

assert sets["validation"].isdisjoint(
    sets["test"]
)

run_id = datetime.now(
    timezone.utc
).strftime(
    "full_primary_%Y%m%dT%H%M%SZ"
)

FULL_DIR = STEP9 / run_id

FULL_DIR.mkdir(
    parents=True,
    exist_ok=False,
)

training_started = time.perf_counter()

config = {
    "run_id": run_id,
    "target": TARGET,
    "alignment": manifest8["alignment"],
    "seed": SEED,
    "feature_count": len(features),
    "patient_level_split": True,
    "split_fractions": {
        "train": 0.70,
        "validation": 0.15,
        "test": 0.15,
    },
    "training_negative_to_positive_ratio": 5.0,
    "negative_sampling_fraction": float(
        negative_fraction
    ),
    "negative_sampling_weight_correction": True,
    "normalization": False,
    "missing_values": (
        "LightGBM native NaN handling"
    ),
    "early_stopping_validation_rows": 500_000,
    "early_stopping_metric": (
        "average_precision"
    ),
    "early_stopping_rounds": 100,
    "lightgbm_parameters": {
        "objective": "binary",
        "metric": "average_precision",
        "n_estimators": 3000,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 200,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "n_jobs": -1,
        "random_state": SEED,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    },
}

with open(
    FULL_DIR / "config.json",
    "w",
) as file:
    json.dump(
        config,
        file,
        indent=2,
    )

stay_splits.to_parquet(
    FULL_DIR
    / "stay_subject_split.parquet",
    index=False,
    compression="zstd",
)

print("Execução:", run_id)
print("Saída:", FULL_DIR)
print()
print("1/6 — Carregando treino completo...")

load_train_started = time.perf_counter()

(
    x_full_train,
    y_full_train,
    full_train_weights,
    full_train_metadata,
) = collect_full_split(
    split="train",
    negative_fraction=negative_fraction,
)

load_train_minutes = (
    time.perf_counter()
    - load_train_started
) / 60

actual_training_rows = int(
    len(y_full_train)
)

training_raw_prevalence = float(
    y_full_train.mean()
)

training_weighted_prevalence = float(
    np.average(
        y_full_train,
        weights=full_train_weights,
    )
)

print()
print(
    "Prevalência bruta do treino:",
    f"{training_raw_prevalence:.4%}",
)

print(
    "Prevalência ponderada do treino:",
    f"{training_weighted_prevalence:.4%}",
)

print()
print(
    "2/6 — Carregando amostra de validação "
    "para early stopping..."
)

load_early_validation_started = (
    time.perf_counter()
)

(
    x_early_validation,
    y_early_validation,
    _,
    early_validation_metadata,
) = collect_split(
    split="validation",
    max_rows=500_000,
    negative_fraction=1.0,
)

load_early_validation_minutes = (
    time.perf_counter()
    - load_early_validation_started
) / 60

print()
print("3/6 — Treinando LightGBM...")

model = lgb.LGBMClassifier(
    **config["lightgbm_parameters"]
)

fit_started = time.perf_counter()

model.fit(
    x_full_train,
    y_full_train,
    sample_weight=full_train_weights,
    eval_set=[
        (
            x_early_validation,
            y_early_validation,
        )
    ],
    eval_metric="average_precision",
    callbacks=[
        lgb.early_stopping(
            stopping_rounds=100,
            first_metric_only=True,
            verbose=True,
        ),
        lgb.log_evaluation(
            period=25,
        ),
    ],
)

fit_minutes = (
    time.perf_counter()
    - fit_started
) / 60

best_iteration = int(
    model.best_iteration_
)

print()
print(
    "Melhor boosting iteration:",
    best_iteration,
)

joblib.dump(
    model,
    FULL_DIR / "model.joblib",
)

model.booster_.save_model(
    FULL_DIR / "model.txt",
    num_iteration=best_iteration,
)

importance = pd.DataFrame({
    "feature": features,
    "gain": model.booster_.feature_importance(
        importance_type="gain"
    ),
    "split": model.booster_.feature_importance(
        importance_type="split"
    ),
}).sort_values(
    "gain",
    ascending=False,
)

importance["gain_fraction"] = (
    importance["gain"]
    / importance["gain"].sum()
)

importance.to_csv(
    FULL_DIR / "feature_importance.csv",
    index=False,
)

del (
    x_full_train,
    y_full_train,
    full_train_weights,
    full_train_metadata,
    x_early_validation,
    y_early_validation,
    early_validation_metadata,
)

gc.collect()

print()
print(
    "4/6 — Avaliando validação completa..."
)

load_validation_started = (
    time.perf_counter()
)

(
    x_full_validation,
    y_full_validation,
    _,
    full_validation_metadata,
) = collect_full_split(
    split="validation",
    negative_fraction=1.0,
)

load_validation_minutes = (
    time.perf_counter()
    - load_validation_started
) / 60

validation_probability = (
    model.predict_proba(
        x_full_validation,
        num_iteration=best_iteration,
    )[:, 1]
)

thresholds_selected = (
    select_validation_thresholds(
        y_full_validation,
        validation_probability,
    )
)

validation_threshold_metrics = (
    evaluate_thresholds(
        y_full_validation,
        validation_probability,
        thresholds_selected,
    )
)

validation_ranking_metrics = (
    basic_metrics(
        y_full_validation,
        validation_probability,
    )
)

validation_stay_metrics = (
    grouped_metrics(
        y_full_validation,
        validation_probability,
        full_validation_metadata,
        "stay_id",
    )
)

validation_patient_metrics = (
    grouped_metrics(
        y_full_validation,
        validation_probability,
        full_validation_metadata,
        "subject_id",
    )
)

save_predictions(
    full_validation_metadata,
    y_full_validation,
    validation_probability,
    FULL_DIR
    / "validation_predictions.parquet",
)

validation_threshold_table = (
    pd.DataFrame(
        validation_threshold_metrics
    )
    .T
    .reset_index(
        names="operating_point"
    )
)

validation_threshold_table.to_csv(
    FULL_DIR
    / "validation_thresholds.csv",
    index=False,
)

plot_confusion_matrix(
    validation_threshold_metrics[
        "maximum_f1"
    ],
    (
        "Validação — threshold "
        "de máximo F1"
    ),
    FULL_DIR
    / "validation_confusion_matrix_max_f1.png",
)

if "recall_80" in validation_threshold_metrics:
    plot_confusion_matrix(
        validation_threshold_metrics[
            "recall_80"
        ],
        "Validação — recall-alvo de 80%",
        FULL_DIR
        / "validation_confusion_matrix_recall_80.png",
    )

del (
    x_full_validation,
    y_full_validation,
    full_validation_metadata,
    validation_probability,
)

gc.collect()

print()
print("5/6 — Avaliando teste completo...")

load_test_started = time.perf_counter()

(
    x_full_test,
    y_full_test,
    _,
    full_test_metadata,
) = collect_full_split(
    split="test",
    negative_fraction=1.0,
)

load_test_minutes = (
    time.perf_counter()
    - load_test_started
) / 60

test_probability = model.predict_proba(
    x_full_test,
    num_iteration=best_iteration,
)[:, 1]

test_threshold_metrics = (
    evaluate_thresholds(
        y_full_test,
        test_probability,
        thresholds_selected,
    )
)

test_ranking_metrics = basic_metrics(
    y_full_test,
    test_probability,
)

test_stay_metrics = grouped_metrics(
    y_full_test,
    test_probability,
    full_test_metadata,
    "stay_id",
)

test_patient_metrics = grouped_metrics(
    y_full_test,
    test_probability,
    full_test_metadata,
    "subject_id",
)

save_predictions(
    full_test_metadata,
    y_full_test,
    test_probability,
    FULL_DIR
    / "test_predictions.parquet",
)

test_threshold_table = (
    pd.DataFrame(
        test_threshold_metrics
    )
    .T
    .reset_index(
        names="operating_point"
    )
)

test_threshold_table.to_csv(
    FULL_DIR
    / "test_thresholds.csv",
    index=False,
)

plot_confusion_matrix(
    test_threshold_metrics[
        "maximum_f1"
    ],
    "Teste — threshold congelado de máximo F1",
    FULL_DIR
    / "test_confusion_matrix_max_f1.png",
)

if "recall_80" in test_threshold_metrics:
    plot_confusion_matrix(
        test_threshold_metrics[
            "recall_80"
        ],
        "Teste — threshold congelado de recall 80%",
        FULL_DIR
        / "test_confusion_matrix_recall_80.png",
    )

manifest_hashes_after = {
    "step7": sha256_file(
        MANIFEST7_FILE
    ),
    "step8": sha256_file(
        MANIFEST8_FILE
    ),
}

if (
    manifest_hashes_before
    != manifest_hashes_after
):
    raise RuntimeError(
        "Os manifestos dos Steps 7 ou 8 "
        "foram alterados"
    )

total_minutes = (
    time.perf_counter()
    - training_started
) / 60

results = {
    "status": "completed",
    "run_id": run_id,
    "completed_at_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "dataset_version": (
        manifest8[
            "validated_dataset_version"
        ]
    ),
    "target": TARGET,
    "alignment": manifest8["alignment"],
    "feature_count": len(features),
    "lightgbm_version": lgb.__version__,
    "best_boosting_iteration": (
        best_iteration
    ),
    "training_rows": actual_training_rows,
    "training_raw_prevalence": (
        training_raw_prevalence
    ),
    "training_weighted_prevalence": (
        training_weighted_prevalence
    ),
    "validation_rows": int(
        counts["validation"]["rows"]
    ),
    "test_rows": int(
        counts["test"]["rows"]
    ),
    "thresholds_selected_on_validation": (
        thresholds_selected
    ),
    "validation": {
        "ranking": (
            validation_ranking_metrics
        ),
        "thresholds": (
            validation_threshold_metrics
        ),
        "stay_exploratory": (
            validation_stay_metrics
        ),
        "patient_exploratory": (
            validation_patient_metrics
        ),
    },
    "test": {
        "ranking": test_ranking_metrics,
        "thresholds": (
            test_threshold_metrics
        ),
        "stay_exploratory": (
            test_stay_metrics
        ),
        "patient_exploratory": (
            test_patient_metrics
        ),
    },
    "timing_minutes": {
        "load_training": (
            load_train_minutes
        ),
        "load_early_validation": (
            load_early_validation_minutes
        ),
        "fit": fit_minutes,
        "load_full_validation": (
            load_validation_minutes
        ),
        "load_full_test": (
            load_test_minutes
        ),
        "total": total_minutes,
    },
    "manifest_hashes_before": (
        manifest_hashes_before
    ),
    "manifest_hashes_after": (
        manifest_hashes_after
    ),
    "config": config,
}

with open(
    FULL_DIR / "metrics.json",
    "w",
) as file:
    json.dump(
        results,
        file,
        indent=2,
    )

with open(
    FULL_DIR / "status.json",
    "w",
) as file:
    json.dump(
        {
            "status": "completed",
            "run_id": run_id,
            "metrics": str(
                FULL_DIR / "metrics.json"
            ),
        },
        file,
        indent=2,
    )

del (
    x_full_test,
    y_full_test,
    full_test_metadata,
    test_probability,
)

gc.collect()

print()
print("6/6 — TREINAMENTO COMPLETO FINALIZADO")
print("Resultados:", FULL_DIR)
print(
    "Tempo total:",
    f"{total_minutes:.2f} minutos",
)
print(
    "Melhor boosting iteration:",
    best_iteration,
)

print()
print("MÉTRICAS DE RANKING")
display(
    pd.DataFrame({
        "validation": (
            validation_ranking_metrics
        ),
        "test": test_ranking_metrics,
    })
)

print()
print("THRESHOLDS — VALIDAÇÃO")
display(
    validation_threshold_table
)

print()
print("THRESHOLDS CONGELADOS — TESTE")
display(
    test_threshold_table
)

print()
print("20 FEATURES MAIS IMPORTANTES")
display(
    importance.head(20)
)

# %% [source cell 22]
EPISODE_DIR = (
    FULL_DIR /
    "episode_evaluation"
)

EPISODE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

STEP5_DIR = (
    PROJECT /
    "outputs" /
    "step05_labels_v2"
)

STEP5_MANIFEST_FILE = (
    STEP5_DIR /
    "manifest_step5_eligibility_v2.json"
)

TEST_PREDICTIONS_FILE = (
    FULL_DIR /
    "test_predictions.parquet"
)

FULL_METRICS_FILE = (
    FULL_DIR /
    "metrics.json"
)

if not TEST_PREDICTIONS_FILE.exists():
    raise FileNotFoundError(
        TEST_PREDICTIONS_FILE
    )

if not FULL_METRICS_FILE.exists():
    raise FileNotFoundError(
        FULL_METRICS_FILE
    )

step5_manifest = json.loads(
    STEP5_MANIFEST_FILE.read_text()
)

full_metrics = json.loads(
    FULL_METRICS_FILE.read_text()
)

thresholds = full_metrics[
    "thresholds_selected_on_validation"
]

step5_files = sorted(
    STEP5_DIR.glob(
        "batch_*.parquet"
    )
)

if len(step5_files) != 285:
    raise RuntimeError(
        "Esperávamos 285 batches do Step 5; "
        f"encontramos {len(step5_files)}"
    )

print(
    "Carregando os primeiros onsets "
    "reais do Step 5..."
)

onset_parts = []

for batch_number, path in enumerate(
    step5_files,
    start=1,
):
    labels = pd.read_parquet(
        path,
        columns=[
            "stay_id",
            "charttime",
            "failure_state_new_at_t",
        ],
    )

    onset_rows = labels.loc[
        labels[
            "failure_state_new_at_t"
        ]
        .fillna(False)
        .astype(bool),
        [
            "stay_id",
            "charttime",
        ],
    ]

    if not onset_rows.empty:
        onset_parts.append(
            onset_rows.groupby(
                "stay_id",
                as_index=False,
            )["charttime"]
            .min()
            .rename(
                columns={
                    "charttime": "first_onset_time"
                }
            )
        )

    del labels, onset_rows
    gc.collect()

    if batch_number % 50 == 0:
        print(
            f"Step 5: "
            f"{batch_number}/285 batches"
        )

first_onsets = pd.concat(
    onset_parts,
    ignore_index=True,
)

del onset_parts
gc.collect()

if first_onsets[
    "stay_id"
].duplicated().any():
    duplicated = (
        first_onsets.loc[
            first_onsets[
                "stay_id"
            ].duplicated(
                keep=False
            ),
            "stay_id",
        ]
        .head()
        .tolist()
    )

    raise RuntimeError(
        "Um stay apareceu em mais de um "
        f"batch do Step 5: {duplicated}"
    )

first_onsets["first_onset_time"] = (
    pd.to_datetime(
        first_onsets[
            "first_onset_time"
        ]
    )
)

print(
    "Primeiros onsets encontrados:",
    f"{len(first_onsets):,}",
)

if (
    len(first_onsets)
    != step5_manifest[
        "stays_with_first_onset_sum"
    ]
):
    raise RuntimeError(
        "A contagem de onsets não corresponde "
        "ao manifesto do Step 5"
    )

print()
print(
    "Carregando as predições completas "
    "do conjunto de teste..."
)

predictions = pd.read_parquet(
    TEST_PREDICTIONS_FILE
)

required_prediction_columns = {
    "stay_id",
    "subject_id",
    "label_time",
    "target",
    "probability",
}

missing_columns = (
    required_prediction_columns
    - set(predictions.columns)
)

if missing_columns:
    raise RuntimeError(
        "Colunas ausentes nas predições: "
        f"{sorted(missing_columns)}"
    )

predictions["label_time"] = (
    pd.to_datetime(
        predictions["label_time"]
    )
)

predictions["target"] = (
    predictions["target"]
    .astype("int8")
)

if predictions[
    [
        "stay_id",
        "label_time",
    ]
].duplicated().any():
    raise RuntimeError(
        "Existem predições duplicadas por "
        "stay_id e label_time"
    )

stay_subject = (
    predictions[
        [
            "stay_id",
            "subject_id",
        ]
    ]
    .drop_duplicates()
)

if stay_subject[
    "stay_id"
].duplicated().any():
    raise RuntimeError(
        "Um stay foi associado a mais "
        "de um subject_id"
    )

stay_truth = (
    predictions
    .groupby(
        "stay_id"
    )["target"]
    .max()
    .astype("int8")
)

stay_max_probability = (
    predictions
    .groupby(
        "stay_id"
    )["probability"]
    .max()
)

positive_stays = set(
    stay_truth.loc[
        stay_truth.eq(1)
    ].index
)

negative_stays = set(
    stay_truth.loc[
        stay_truth.eq(0)
    ].index
)

positive_stay_subject = (
    stay_subject.loc[
        stay_subject[
            "stay_id"
        ].isin(
            positive_stays
        )
    ]
)

positive_subjects = set(
    positive_stay_subject[
        "subject_id"
    ]
)

test_onsets = (
    first_onsets.loc[
        first_onsets[
            "stay_id"
        ].isin(
            positive_stays
        )
    ]
    .merge(
        positive_stay_subject,
        on="stay_id",
        how="right",
        validate="one_to_one",
    )
)

if test_onsets[
    "first_onset_time"
].isna().any():
    missing = (
        test_onsets.loc[
            test_onsets[
                "first_onset_time"
            ].isna(),
            "stay_id",
        ]
        .head()
        .tolist()
    )

    raise RuntimeError(
        "Stays positivos sem onset no Step 5: "
        f"{missing}"
    )

positive_window_rows = (
    predictions.loc[
        predictions[
            "target"
        ].eq(1),
        [
            "stay_id",
            "subject_id",
            "label_time",
            "probability",
        ],
    ]
    .merge(
        test_onsets[
            [
                "stay_id",
                "first_onset_time",
            ]
        ],
        on="stay_id",
        how="left",
        validate="many_to_one",
    )
)

positive_window_rows[
    "lead_time_hours"
] = (
    (
        positive_window_rows[
            "first_onset_time"
        ]
        - positive_window_rows[
            "label_time"
        ]
    )
    .dt.total_seconds()
    / 3600
)

invalid_lead_time = (
    positive_window_rows[
        "lead_time_hours"
    ].le(0)
    |
    positive_window_rows[
        "lead_time_hours"
    ].gt(8.0001)
)

if invalid_lead_time.any():
    invalid_examples = (
        positive_window_rows.loc[
            invalid_lead_time,
            [
                "stay_id",
                "label_time",
                "first_onset_time",
                "lead_time_hours",
            ],
        ]
        .head()
    )

    raise RuntimeError(
        "Foram encontrados targets positivos "
        "fora da janela (0,8h]. Exemplos:\n"
        f"{invalid_examples}"
    )

print()
print(
    "Stays avaliados:",
    f"{stay_truth.size:,}",
)

print(
    "Stays com primeiro onset:",
    f"{len(positive_stays):,}",
)

print(
    "Stays sem onset:",
    f"{len(negative_stays):,}",
)

print(
    "Pacientes com primeiro onset:",
    f"{len(positive_subjects):,}",
)

episode_rows = []
stay_rows = []
lead_time_records = []

for operating_point, threshold in (
    thresholds.items()
):
    threshold = float(threshold)

    alerts_in_event_window = (
        positive_window_rows.loc[
            positive_window_rows[
                "probability"
            ].ge(threshold)
        ]
    )

    first_alerts = (
        alerts_in_event_window
        .groupby(
            [
                "stay_id",
                "subject_id",
                "first_onset_time",
            ],
            as_index=False,
        )["label_time"]
        .min()
        .rename(
            columns={
                "label_time": "first_alert_time"
            }
        )
    )

    first_alerts[
        "lead_time_hours"
    ] = (
        (
            first_alerts[
                "first_onset_time"
            ]
            - first_alerts[
                "first_alert_time"
            ]
        )
        .dt.total_seconds()
        / 3600
    )

    first_alerts[
        "operating_point"
    ] = operating_point

    first_alerts[
        "threshold"
    ] = threshold

    lead_time_records.append(
        first_alerts[
            [
                "operating_point",
                "threshold",
                "stay_id",
                "subject_id",
                "first_alert_time",
                "first_onset_time",
                "lead_time_hours",
            ]
        ]
    )

    detected_stays = set(
        first_alerts["stay_id"]
    )

    detected_subjects = set(
        first_alerts["subject_id"]
    )

    missed_stays = (
        positive_stays
        - detected_stays
    )

    negative_stay_scores = (
        stay_max_probability.loc[
            list(negative_stays)
        ]
    )

    false_alerted_negative_stays = int(
        negative_stay_scores
        .ge(threshold)
        .sum()
    )

    negative_stay_alert_rate = (
        false_alerted_negative_stays
        / len(negative_stays)
    )

    negative_stay_specificity = (
        1
        - negative_stay_alert_rate
    )

    if len(first_alerts) > 0:
        lead_times = (
            first_alerts[
                "lead_time_hours"
            ]
        )

        median_lead = float(
            lead_times.median()
        )

        q1_lead = float(
            lead_times.quantile(0.25)
        )

        q3_lead = float(
            lead_times.quantile(0.75)
        )

        alerted_at_least_2h = float(
            lead_times.ge(2).mean()
        )

        alerted_at_least_4h = float(
            lead_times.ge(4).mean()
        )

        alerted_at_least_6h = float(
            lead_times.ge(6).mean()
        )
    else:
        median_lead = np.nan
        q1_lead = np.nan
        q3_lead = np.nan
        alerted_at_least_2h = np.nan
        alerted_at_least_4h = np.nan
        alerted_at_least_6h = np.nan

    episode_rows.append({
        "operating_point": operating_point,
        "threshold": threshold,
        "positive_stays": int(
            len(positive_stays)
        ),
        "detected_episodes": int(
            len(detected_stays)
        ),
        "missed_episodes": int(
            len(missed_stays)
        ),
        "episode_recall": float(
            len(detected_stays)
            / len(positive_stays)
        ),
        "positive_patients": int(
            len(positive_subjects)
        ),
        "detected_positive_patients": int(
            len(detected_subjects)
        ),
        "patient_recall": float(
            len(detected_subjects)
            / len(positive_subjects)
        ),
        "median_first_alert_lead_hours": (
            median_lead
        ),
        "lead_time_q1_hours": q1_lead,
        "lead_time_q3_hours": q3_lead,
        "detected_at_least_2h_before": (
            alerted_at_least_2h
        ),
        "detected_at_least_4h_before": (
            alerted_at_least_4h
        ),
        "detected_at_least_6h_before": (
            alerted_at_least_6h
        ),
        "negative_stays": int(
            len(negative_stays)
        ),
        "negative_stays_with_any_alert": (
            false_alerted_negative_stays
        ),
        "negative_stay_alert_rate": float(
            negative_stay_alert_rate
        ),
        "negative_stay_specificity": float(
            negative_stay_specificity
        ),
    })

    stay_classification = (
        classification_at_threshold(
            stay_truth.to_numpy(
                dtype="int8"
            ),
            stay_max_probability.loc[
                stay_truth.index
            ].to_numpy(
                dtype="float64"
            ),
            threshold,
        )
    )

    stay_classification[
        "operating_point"
    ] = operating_point

    stay_rows.append(
        stay_classification
    )

episode_table = pd.DataFrame(
    episode_rows
)

stay_table = pd.DataFrame(
    stay_rows
)

all_lead_times = pd.concat(
    lead_time_records,
    ignore_index=True,
)

episode_table.to_csv(
    EPISODE_DIR /
    "episode_detection_test.csv",
    index=False,
)

stay_table.to_csv(
    EPISODE_DIR /
    "stay_max_probability_test_exploratory.csv",
    index=False,
)

all_lead_times.to_parquet(
    EPISODE_DIR /
    "detected_episode_lead_times.parquet",
    index=False,
    compression="zstd",
)

episode_summary = {
    "status": "completed",
    "run_id": full_metrics["run_id"],
    "evaluation_set": "test",
    "episode_definition": (
        "first circulatory failure onset per stay"
    ),
    "valid_alert_window": (
        "(first_onset - 8h, first_onset)"
    ),
    "episode_detection_rule": (
        "at least one prediction above the "
        "validation-selected threshold within "
        "the valid pre-onset window"
    ),
    "positive_stays": int(
        len(positive_stays)
    ),
    "negative_stays": int(
        len(negative_stays)
    ),
    "positive_patients": int(
        len(positive_subjects)
    ),
    "thresholds_frozen_from_validation": (
        thresholds
    ),
    "episode_results": (
        episode_table.to_dict(
            orient="records"
        )
    ),
    "stay_max_probability_results_exploratory": (
        stay_table.to_dict(
            orient="records"
        )
    ),
    "important_limitation": (
        "Negative-stay alert rate does not yet "
        "collapse repeated alerts into clinical "
        "alert episodes or patient-days."
    ),
}

with open(
    EPISODE_DIR /
    "episode_evaluation.json",
    "w",
) as file:
    json.dump(
        episode_summary,
        file,
        indent=2,
    )

fig, axes = plt.subplots(
    1,
    2,
    figsize=(13, 4.5),
)

axes[0].bar(
    episode_table[
        "operating_point"
    ],
    episode_table[
        "episode_recall"
    ],
)

axes[0].set_ylim(
    0,
    1,
)

axes[0].set_ylabel(
    "Episode recall"
)

axes[0].set_title(
    "Detecção do primeiro onset por stay"
)

axes[0].tick_params(
    axis="x",
    rotation=35,
)

axes[1].bar(
    episode_table[
        "operating_point"
    ],
    episode_table[
        "median_first_alert_lead_hours"
    ],
)

axes[1].set_ylim(
    0,
    8,
)

axes[1].set_ylabel(
    "Horas"
)

axes[1].set_title(
    "Antecedência mediana do primeiro alerta"
)

axes[1].tick_params(
    axis="x",
    rotation=35,
)

fig.tight_layout()

fig.savefig(
    EPISODE_DIR /
    "episode_recall_and_lead_time.png",
    dpi=200,
    bbox_inches="tight",
)

plt.show()

print()
print(
    "AVALIAÇÃO POR EPISÓDIO/STAY CONCLUÍDA"
)

print(
    "Resultados:",
    EPISODE_DIR,
)

print()
print(
    "RESULTADOS POR EPISÓDIO"
)

display(
    episode_table
)

print()
print(
    "AGREGAÇÃO EXPLORATÓRIA "
    "PELA MAIOR PROBABILIDADE DO STAY"
)

display(
    stay_table
)
