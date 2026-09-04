"""Source exported from the reviewed development notebook.

This file contains code only; patient-level data and notebook outputs are not distributed.
"""

# %% [source cell 2]
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import re
import shutil
import time
import warnings

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

warnings.filterwarnings("ignore", category=FutureWarning)

STEP_MINUTES = 5
EXPECTED_BATCHES = 285
EXPECTED_STAYS = 71054
EXPECTED_STEP5_ROWS = 72338551
STEP5_VERSION = "observed_onset_sufficient_without_full_8h_followup_v2"
DATASET_VERSION = "dataset_aligned_xy_v2"
MIN_FREE_GB = 5.0

PROJECT_DIR = Path(
    "/path/to/circulatory_failure_prediction"
)

CLEAN_DIR = Path(
    "/path/to/mimic_source_project/"
    "output_icu_clean_v2/raw_grid_all_icu_clean_v2"
)

LABEL_DIR = (
    PROJECT_DIR / "outputs" / "step05_labels_v2"
)

STEP5_MANIFEST_PATH = (
    LABEL_DIR / "manifest_step5_eligibility_v2.json"
)

OUTPUT_DIR = (
    PROJECT_DIR / "outputs" / "step07_aligned_xy_v2"
)

DATA_DIR = OUTPUT_DIR / "batches"
SUMMARY_DIR = OUTPUT_DIR / "summaries"

FEATURE_NAMES_PATH = (
    OUTPUT_DIR / "feature_names.json"
)

MANIFEST_PATH = (
    OUTPUT_DIR / "manifest_step7.json"
)

FEATURE_GUARD_PATH = (
    PROJECT_DIR / "scripts" / "feature_guard.py"
)

DATA_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

print("ICU-clean:", CLEAN_DIR)
print("Labels:", LABEL_DIR)
print("Saída Step 7:", OUTPUT_DIR)

# %% [source cell 4]
def canonical_batches(directory):
    return sorted(
        arquivo
        for arquivo in directory.glob("*.parquet")
        if re.fullmatch(
            r"batch_\d{4}\.parquet",
            arquivo.name,
        )
    )


clean_files = canonical_batches(CLEAN_DIR)
label_files = canonical_batches(LABEL_DIR)

clean_names = [arquivo.name for arquivo in clean_files]
label_names = [arquivo.name for arquivo in label_files]

if len(clean_files) != EXPECTED_BATCHES:
    raise RuntimeError(
        f"Esperávamos {EXPECTED_BATCHES} ICU-clean, "
        f"mas encontramos {len(clean_files)}."
    )

if len(label_files) != EXPECTED_BATCHES:
    raise RuntimeError(
        f"Esperávamos {EXPECTED_BATCHES} labels, "
        f"mas encontramos {len(label_files)}."
    )

expected_names = [
    f"batch_{indice:04d}.parquet"
    for indice in range(EXPECTED_BATCHES)
]

if clean_names != expected_names:
    raise RuntimeError(
        "A sequência ICU-clean não corresponde a 0000–0284."
    )

if label_names != expected_names:
    raise RuntimeError(
        "A sequência dos labels não corresponde a 0000–0284."
    )



if not STEP5_MANIFEST_PATH.exists():
    raise FileNotFoundError(
        f"Manifesto aprovado do Step 5 v2 não encontrado: {STEP5_MANIFEST_PATH}"
    )

step5_manifest_bytes = STEP5_MANIFEST_PATH.read_bytes()
step5_manifest = json.loads(step5_manifest_bytes.decode("utf-8"))
STEP5_MANIFEST_SHA256 = hashlib.sha256(step5_manifest_bytes).hexdigest()

expected_step5 = {
    "status": "approved",
    "step": 5,
    "version": STEP5_VERSION,
    "batches": EXPECTED_BATCHES,
    "stays": EXPECTED_STAYS,
    "rows": EXPECTED_STEP5_ROWS,
    "negative_requires_complete_known_8h_horizon": True,
    "positive_requires_complete_8h_followup_after_t": False,
}

step5_mismatches = {
    key: {"expected": expected, "observed": step5_manifest.get(key)}
    for key, expected in expected_step5.items()
    if step5_manifest.get(key) != expected
}

if step5_mismatches:
    raise RuntimeError(
        "O manifesto do Step 5 não corresponde à versão aprovada: "
        f"{step5_mismatches}"
    )

if not FEATURE_GUARD_PATH.exists():
    raise FileNotFoundError(
        f"Blacklist não encontrada: {FEATURE_GUARD_PATH}"
    )

spec = importlib.util.spec_from_file_location(
    "feature_guard_step7",
    FEATURE_GUARD_PATH,
)

if spec is None or spec.loader is None:
    raise RuntimeError(
        "Não foi possível carregar feature_guard.py."
    )

feature_guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(feature_guard)

free_gb = shutil.disk_usage(
    OUTPUT_DIR
).free / 1024**3

print("ICU-clean confirmados:", len(clean_files))
print("Labels confirmados:", len(label_files))
print("Primeiro batch:", clean_names[0])
print("Último batch:", clean_names[-1])
print("Espaço livre:", round(free_gb, 2), "GB")
print("Blacklist carregada:", FEATURE_GUARD_PATH)
print("Step 5 aprovado:", step5_manifest["version"])
print("SHA-256 do manifesto Step 5:", STEP5_MANIFEST_SHA256)

if free_gb < MIN_FREE_GB:
    raise RuntimeError(
        f"Há somente {free_gb:.2f} GB livres. "
        f"O mínimo configurado é {MIN_FREE_GB:.1f} GB."
    )

# %% [source cell 6]
KEY = ["stay_id", "charttime"]

TARGET_COLUMN = "target_first_onset_0_8h"
ELIGIBILITY_COLUMN = "eligible_for_training"

LABEL_REQUIRED_COLUMNS = [
    "stay_id",
    "charttime",
    "failure_state_new_at_t",
    TARGET_COLUMN,
    ELIGIBILITY_COLUMN,
]

# Metadados e tempos nunca entram em X.
NON_FEATURE_EXACT = {
    "stay_id",
    "subject_id",
    "hadm_id",
    "patient_id",
    "icustay_id",
    "row_id",
    "charttime",
    "intime",
    "outtime",
    "rel_charttime",
    "rel_charttime_seconds",
    "feature_time",
    "label_time",
}

IDENTIFIER_PATTERN = re.compile(
    r"(^|_)(subject|patient|hadm|stay|icustay|admission)_?id$",
    flags=re.IGNORECASE,
)


def is_non_feature(column):
    low = column.lower()

    return (
        low in NON_FEATURE_EXACT
        or IDENTIFIER_PATTERN.search(low) is not None
    )


def atomic_write_json(data, destination):
    temp_path = destination.with_suffix(
        destination.suffix + ".tmp"
    )

    temp_path.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    os.replace(temp_path, destination)


def atomic_write_parquet(dataframe, destination):
    temp_path = destination.with_suffix(
        destination.suffix + ".tmp"
    )

    dataframe.to_parquet(
        temp_path,
        index=False,
        compression="zstd",
    )

    os.replace(temp_path, destination)


def build_time_since_last_measurement(dataframe):
    """
    Para cada variável que possui máscara *_imputed:

    - 0 minuto no timestep com medida;
    - 5, 10, 15... após a última medida;
    - NaN antes da primeira medida do stay.

    O NaN é intencional e será tratado nativamente pelo LightGBM.
    """
    result = dataframe.copy()

    mask_columns = sorted(
        column
        for column in result.columns
        if column.endswith("_imputed")
        and column[:-len("_imputed")] in result.columns
    )

    created_columns = []

    for mask_column in mask_columns:
        base_column = mask_column[:-len("_imputed")]

        new_column = (
            f"{base_column}"
            "_time_since_last_measurement_min"
        )

        observed_now = (
            pd.to_numeric(
                result[mask_column],
                errors="coerce",
            )
            .eq(0)
        )

        observed_time = result["charttime"].where(
            observed_now
        )

        last_observed_time = (
            observed_time
            .groupby(
                result["stay_id"],
                sort=False,
            )
            .ffill()
        )

        elapsed_minutes = (
            result["charttime"] - last_observed_time
        ).dt.total_seconds() / 60.0

        if elapsed_minutes.dropna().lt(0).any():
            raise RuntimeError(
                f"Tempo negativo gerado para {base_column}."
            )

        result[new_column] = elapsed_minutes.astype(
            "float32"
        )

        created_columns.append(new_column)

    return result, created_columns


def output_already_approved(clean_file):
    output_file = DATA_DIR / clean_file.name

    summary_file = (
        SUMMARY_DIR
        / f"{clean_file.stem}_summary.json"
    )

    if not output_file.exists():
        return False

    if not summary_file.exists():
        return False

    try:
        summary = json.loads(
            summary_file.read_text(
                encoding="utf-8"
            )
        )

        label_file = LABEL_DIR / clean_file.name
        label_stat = label_file.stat()

        return (
            summary.get("status") == "approved"
            and summary.get("dataset_version") == DATASET_VERSION
            and summary.get("source_step5_version") == STEP5_VERSION
            and summary.get("source_step5_manifest_sha256") == STEP5_MANIFEST_SHA256
            and summary.get("source_label_size_bytes") == label_stat.st_size
            and summary.get("source_label_mtime_ns") == label_stat.st_mtime_ns
            and summary.get("lag_minutes") == STEP_MINUTES
            and summary.get("target") == TARGET_COLUMN
            and summary.get("rows_output") is not None
            and summary.get("features") is not None
        )

    except Exception:
        return False


def process_batch(clean_file, force=False):
    label_file = LABEL_DIR / clean_file.name
    output_file = DATA_DIR / clean_file.name

    summary_file = (
        SUMMARY_DIR
        / f"{clean_file.stem}_summary.json"
    )

    if output_already_approved(clean_file) and not force:
        return json.loads(
            summary_file.read_text(
                encoding="utf-8"
            )
        )

    started = time.perf_counter()

    clean = pd.read_parquet(clean_file)
    labels = pd.read_parquet(label_file)

    missing_clean_key = [
        column
        for column in KEY
        if column not in clean.columns
    ]

    if missing_clean_key:
        raise RuntimeError(
            f"{clean_file.name}: chaves ausentes na ICU-clean: "
            f"{missing_clean_key}"
        )

    missing_label_columns = [
        column
        for column in LABEL_REQUIRED_COLUMNS
        if column not in labels.columns
    ]

    if missing_label_columns:
        raise RuntimeError(
            f"{label_file.name}: colunas ausentes nos labels: "
            f"{missing_label_columns}"
        )

    clean["stay_id"] = pd.to_numeric(
        clean["stay_id"],
        errors="raise",
    ).astype("int64")

    labels["stay_id"] = pd.to_numeric(
        labels["stay_id"],
        errors="raise",
    ).astype("int64")

    clean["charttime"] = pd.to_datetime(
        clean["charttime"],
        errors="raise",
    )

    labels["charttime"] = pd.to_datetime(
        labels["charttime"],
        errors="raise",
    )

    clean = clean.sort_values(KEY).reset_index(drop=True)
    labels = labels.sort_values(KEY).reset_index(drop=True)

    if clean.duplicated(KEY).any():
        raise RuntimeError(
            f"{clean_file.name}: chave duplicada na ICU-clean."
        )

    if labels.duplicated(KEY).any():
        raise RuntimeError(
            f"{label_file.name}: chave duplicada nos labels."
        )

    # Antes do lag, ICU-clean e label devem representar
    # exatamente os mesmos stay_id + charttime.
    if len(clean) != len(labels):
        raise RuntimeError(
            f"{clean_file.name}: ICU-clean possui {len(clean)} "
            f"linhas e label possui {len(labels)}."
        )

    if not clean[KEY].equals(labels[KEY]):
        raise RuntimeError(
            f"{clean_file.name}: chaves ICU-clean/label "
            "não correspondem antes do lag."
        )

    time_differences = (
        clean.groupby(
            "stay_id",
            sort=False,
        )["charttime"]
        .diff()
        .dropna()
        .dt.total_seconds()
        .div(60)
    )

    if not time_differences.eq(STEP_MINUTES).all():
        invalid = time_differences[
            ~time_differences.eq(STEP_MINUTES)
        ].head(10).tolist()

        raise RuntimeError(
            f"{clean_file.name}: grade não contínua de "
            f"{STEP_MINUTES} minutos. Exemplos: {invalid}"
        )

    # Acrescenta tempo desde a última medida em t−5.
    clean_enriched, time_since_columns = (
        build_time_since_last_measurement(clean)
    )

    original_candidates = [
        column
        for column in clean_enriched.columns
        if not is_non_feature(column)
    ]

    # A blacklist é fail-closed: qualquer variável
    # farmacológica, endpoint ou label interrompe o batch.
    feature_guard.assert_no_forbidden_features(
        original_candidates
    )

    feature_columns = original_candidates

    if not feature_columns:
        raise RuntimeError(
            f"{clean_file.name}: nenhuma feature selecionada."
        )

    # feature_time representa t−5.
    features = clean_enriched[
        ["stay_id", "charttime", *feature_columns]
    ].rename(
        columns={"charttime": "feature_time"}
    )

    # Cada feature em t−5 é ligada ao label referenciado em t.
    features["label_time"] = (
        features["feature_time"]
        + pd.Timedelta(minutes=STEP_MINUTES)
    )

    labels_for_merge = labels[
        [
            "stay_id",
            "charttime",
            TARGET_COLUMN,
            ELIGIBILITY_COLUMN,
        ]
    ].rename(
        columns={"charttime": "label_time"}
    )

    aligned = features.merge(
        labels_for_merge,
        on=["stay_id", "label_time"],
        how="inner",
        validate="one_to_one",
        sort=False,
    )

    # Em uma grade contínua, perde-se somente a última linha
    # de features de cada stay, pois ela não possui label em t+5.
    expected_aligned_rows = (
        len(clean_enriched)
        - clean_enriched["stay_id"].nunique()
    )

    if len(aligned) != expected_aligned_rows:
        raise RuntimeError(
            f"{clean_file.name}: esperávamos "
            f"{expected_aligned_rows} linhas após o lag, "
            f"mas obtivemos {len(aligned)}."
        )

    eligibility = (
        aligned[ELIGIBILITY_COLUMN]
        .astype("boolean")
        .fillna(False)
    )

    aligned = aligned.loc[eligibility].copy()

    if aligned[TARGET_COLUMN].isna().any():
        raise RuntimeError(
            f"{clean_file.name}: target ausente em linha elegível."
        )

    aligned[TARGET_COLUMN] = pd.to_numeric(
        aligned[TARGET_COLUMN],
        errors="raise",
    ).astype("int8")

    invalid_targets = (
        set(aligned[TARGET_COLUMN].unique())
        - {0, 1}
    )

    if invalid_targets:
        raise RuntimeError(
            f"{clean_file.name}: valores inválidos no target: "
            f"{invalid_targets}"
        )

    aligned = aligned.drop(
        columns=[ELIGIBILITY_COLUMN]
    )

    output_columns = [
        "stay_id",
        "feature_time",
        "label_time",
        TARGET_COLUMN,
        *feature_columns,
    ]

    aligned = aligned[
        output_columns
    ].sort_values(
        ["stay_id", "feature_time"]
    ).reset_index(drop=True)

    # Confirma materialmente o lag.
    lag_minutes = (
        aligned["label_time"]
        - aligned["feature_time"]
    ).dt.total_seconds().div(60)

    if not lag_minutes.eq(STEP_MINUTES).all():
        raise RuntimeError(
            f"{clean_file.name}: lag final diferente de "
            f"{STEP_MINUTES} minutos."
        )

    # Confirma que X permanece sem leakage.
    feature_guard.assert_no_forbidden_features(
        feature_columns
    )

    if FEATURE_NAMES_PATH.exists():
        registered_features = json.loads(
            FEATURE_NAMES_PATH.read_text(
                encoding="utf-8"
            )
        )["feature_names"]

        if registered_features != feature_columns:
            raise RuntimeError(
                f"{clean_file.name}: schema de X diverge "
                "do batch de referência."
            )

    else:
        atomic_write_json(
            {
                "dataset_version": DATASET_VERSION,
                "source_step5_version": STEP5_VERSION,
                "source_step5_manifest_sha256": STEP5_MANIFEST_SHA256,
                "feature_names": feature_columns,
                "feature_count": len(feature_columns),
                "target": TARGET_COLUMN,
                "metadata_not_in_x": [
                    "stay_id",
                    "feature_time",
                    "label_time",
                ],
                "lag_minutes": STEP_MINUTES,
                "time_since_before_first_measurement": (
                    "NaN"
                ),
            },
            FEATURE_NAMES_PATH,
        )

    atomic_write_parquet(
        aligned,
        output_file,
    )

    # Gate de leitura após gravação.
    written_schema = (
        pq.ParquetFile(output_file)
        .schema_arrow.names
    )

    if written_schema != output_columns:
        raise RuntimeError(
            f"{clean_file.name}: schema gravado incorreto."
        )

    positives = int(
        aligned[TARGET_COLUMN].sum()
    )

    label_stat = label_file.stat()

    summary = {
        "status": "approved",
        "batch": clean_file.name,
        "dataset_version": DATASET_VERSION,
        "source_step5_version": STEP5_VERSION,
        "source_step5_manifest_sha256": STEP5_MANIFEST_SHA256,
        "source_label_size_bytes": int(label_stat.st_size),
        "source_label_mtime_ns": int(label_stat.st_mtime_ns),
        "lag_minutes": STEP_MINUTES,
        "target": TARGET_COLUMN,
        "rows_icu_clean": int(len(clean)),
        "rows_after_lag": int(expected_aligned_rows),
        "rows_output": int(len(aligned)),
        "rows_removed_by_eligibility": int(
            expected_aligned_rows - len(aligned)
        ),
        "positive_rows": positives,
        "negative_rows": int(len(aligned) - positives),
        "stays_output": int(
            aligned["stay_id"].nunique()
        ),
        "features": int(len(feature_columns)),
        "time_since_features": int(
            len(time_since_columns)
        ),
        "output_bytes": int(
            output_file.stat().st_size
        ),
        "duration_seconds": round(
            time.perf_counter() - started,
            3,
        ),
    }

    atomic_write_json(
        summary,
        summary_file,
    )

    return summary

# %% [source cell 8]
smoke_summary = process_batch(
    clean_files[0],
    force=False,
)

print(
    json.dumps(
        smoke_summary,
        indent=2,
        ensure_ascii=False,
    )
)

smoke_output = pd.read_parquet(
    DATA_DIR / "batch_0000.parquet"
)

registered = json.loads(
    FEATURE_NAMES_PATH.read_text(
        encoding="utf-8"
    )
)

feature_names = registered["feature_names"]

print("\nAMOSTRA DO DATASET ALINHADO")
display(
    smoke_output[
        [
            "stay_id",
            "feature_time",
            "label_time",
            TARGET_COLUMN,
        ]
    ].head(10)
)

print("\nNúmero de features em X:", len(feature_names))
print(
    "Features de tempo desde a medida:",
    sum(
        column.endswith(
            "_time_since_last_measurement_min"
        )
        for column in feature_names
    )
)

print(
    "Lag observado:",
    (
        smoke_output["label_time"]
        - smoke_output["feature_time"]
    )
    .dt.total_seconds()
    .div(60)
    .value_counts()
    .to_dict()
)

print(
    "Target:",
    smoke_output[TARGET_COLUMN]
    .value_counts(dropna=False)
    .sort_index()
    .to_dict()
)

# %% [source cell 10]
ORDERED_AUDIT_CSV = (
    OUTPUT_DIR
    / "audit_batch_0000_ordered_timelines_v2.csv"
)

aligned_path = DATA_DIR / "batch_0000.parquet"
label_path = LABEL_DIR / "batch_0000.parquet"

if not aligned_path.exists():
    raise FileNotFoundError(
        "O Parquet alinhado do batch_0000 não existe."
    )

aligned = pd.read_parquet(aligned_path)
source_labels = pd.read_parquet(label_path)

aligned["feature_time"] = pd.to_datetime(
    aligned["feature_time"]
)

aligned["label_time"] = pd.to_datetime(
    aligned["label_time"]
)

source_labels["charttime"] = pd.to_datetime(
    source_labels["charttime"]
)

source_labels["stay_id"] = pd.to_numeric(
    source_labels["stay_id"],
    errors="raise",
).astype("int64")

failure_state = (
    source_labels["failure_state_new_at_t"]
    .astype("boolean")
    .fillna(False)
    .astype("int8")
)

source_labels = source_labels.assign(
    _failure_state_numeric=failure_state
)

# Primeiro instante de falência de cada stay.
first_onsets = (
    source_labels.loc[
        source_labels["_failure_state_numeric"].eq(1),
        ["stay_id", "charttime"],
    ]
    .groupby("stay_id", as_index=False)["charttime"]
    .min()
    .rename(
        columns={
            "charttime": "first_onset_time"
        }
    )
)

onset_map = dict(
    zip(
        first_onsets["stay_id"],
        first_onsets["first_onset_time"],
    )
)

all_stays = sorted(
    aligned["stay_id"].unique().tolist()
)

# Seleção determinística: menores stay_ids do batch.
positive_stays = [
    stay_id
    for stay_id in all_stays
    if stay_id in onset_map
][:3]

negative_stays = [
    stay_id
    for stay_id in all_stays
    if stay_id not in onset_map
][:2]

if len(positive_stays) < 3:
    raise RuntimeError(
        "O batch_0000 não contém três stays com onset."
    )

if len(negative_stays) < 2:
    raise RuntimeError(
        "O batch_0000 não contém dois stays sem onset."
    )

audit_frames = []

# Três trajetórias com onset:
# mostra de 9 horas antes até o último ponto elegível.
for case_number, stay_id in enumerate(
    positive_stays,
    start=1,
):
    onset_time = pd.Timestamp(
        onset_map[stay_id]
    )

    stay_data = (
        aligned.loc[
            aligned["stay_id"].eq(stay_id)
            & aligned["label_time"].ge(
                onset_time - pd.Timedelta(hours=9)
            )
            & aligned["label_time"].lt(onset_time)
        ]
        .sort_values("label_time")
        .copy()
    )

    stay_data.insert(
        0,
        "case",
        f"with_onset_{case_number}",
    )

    stay_data.insert(
        4,
        "first_onset_time",
        onset_time,
    )

    stay_data.insert(
        5,
        "hours_to_first_onset",
        (
            onset_time - stay_data["label_time"]
        ).dt.total_seconds() / 3600,
    )

    audit_frames.append(stay_data)

# Dois pacientes sem onset:
# mostra as primeiras nove horas elegíveis.
for case_number, stay_id in enumerate(
    negative_stays,
    start=1,
):
    stay_data = (
        aligned.loc[
            aligned["stay_id"].eq(stay_id)
        ]
        .sort_values("label_time")
        .head(108)  # 108 × 5 min = 9 horas
        .copy()
    )

    stay_data.insert(
        0,
        "case",
        f"without_onset_{case_number}",
    )

    stay_data.insert(
        4,
        "first_onset_time",
        pd.NaT,
    )

    stay_data.insert(
        5,
        "hours_to_first_onset",
        np.nan,
    )

    audit_frames.append(stay_data)

audit = pd.concat(
    audit_frames,
    ignore_index=True,
)

# Verifica a relação exata entre onset e target.
has_onset = audit[
    "first_onset_time"
].notna()

expected_target = (
    has_onset
    & audit["hours_to_first_onset"].gt(0)
    & audit["hours_to_first_onset"].le(8)
).astype("int8")

actual_target = pd.to_numeric(
    audit[TARGET_COLUMN],
    errors="raise",
).astype("int8")

if not actual_target.equals(expected_target):
    mismatches = audit.loc[
        actual_target.ne(expected_target),
        [
            "case",
            "stay_id",
            "label_time",
            "first_onset_time",
            "hours_to_first_onset",
            TARGET_COLUMN,
        ],
    ].head(20)

    display(mismatches)

    raise RuntimeError(
        "O target não corresponde à janela "
        "(t, t+8h] na amostra."
    )

# CSV compacto e legível.
audit_columns = [
    column
    for column in [
        "case",
        "stay_id",
        "feature_time",
        "label_time",
        "first_onset_time",
        "hours_to_first_onset",
        TARGET_COLUMN,
        "heart_rate",
        "heart_rate_imputed",
        "heart_rate_time_since_last_measurement_min",
        "mbp",
        "mbp_imputed",
        "mbp_time_since_last_measurement_min",
        "lab_50813",
        "lab_50813_imputed",
        "lab_50813_time_since_last_measurement_min",
    ]
    if column in audit.columns
]

audit = (
    audit[audit_columns]
    .sort_values(
        ["case", "label_time"]
    )
    .reset_index(drop=True)
)

audit.to_csv(
    ORDERED_AUDIT_CSV,
    index=False,
    encoding="utf-8",
)

print("CSV criado:", ORDERED_AUDIT_CSV)
print("Linhas:", len(audit))
print("Casos:", audit["case"].unique().tolist())

print("\nResumo:")
display(
    audit.groupby("case").agg(
        stay_id=("stay_id", "first"),
        primeira_linha=("label_time", "min"),
        ultima_linha=("label_time", "max"),
        onset=("first_onset_time", "first"),
        linhas=(TARGET_COLUMN, "size"),
        linhas_target_1=(TARGET_COLUMN, "sum"),
    )
)

display(audit.head(40))

# %% [source cell 12]
run_started = time.perf_counter()
processed_now = 0
already_current = 0

for index, clean_file in enumerate(clean_files, start=1):
    if output_already_approved(clean_file):
        already_current += 1
        print(
            f"[{index:03d}/{EXPECTED_BATCHES}] "
            f"{clean_file.name}: já aprovado em {DATASET_VERSION}"
        )
        continue

    print(
        f"[{index:03d}/{EXPECTED_BATCHES}] "
        f"{clean_file.name}: construindo...",
        flush=True,
    )

    summary = process_batch(clean_file, force=False)
    processed_now += 1

    print(
        f"[{index:03d}/{EXPECTED_BATCHES}] "
        f"{clean_file.name}: APROVADO — "
        f"{summary['rows_output']:,} linhas, "
        f"{summary['positive_rows']:,} positivas, "
        f"{summary['duration_seconds']:.1f}s",
        flush=True,
    )

print("\nCONSTRUÇÃO DOS BATCHES ENCERRADA")
print("Processados agora:", processed_now)
print("Já atuais e pulados:", already_current)
print(
    "Duração:",
    round((time.perf_counter() - run_started) / 60, 2),
    "minutos",
)

# %% [source cell 14]
approved_summaries = []
missing_outputs = []

for clean_file in clean_files:
    if not output_already_approved(clean_file):
        missing_outputs.append(clean_file.name)
        continue

    summary_file = (
        SUMMARY_DIR
        / f"{clean_file.stem}_summary.json"
    )

    approved_summaries.append(
        json.loads(
            summary_file.read_text(
                encoding="utf-8"
            )
        )
    )

if missing_outputs:
    raise RuntimeError(
        f"Há {len(missing_outputs)} batches não aprovados: "
        f"{missing_outputs[:20]}"
    )

output_files = canonical_batches(DATA_DIR)

if len(output_files) != EXPECTED_BATCHES:
    raise RuntimeError(
        f"Esperávamos {EXPECTED_BATCHES} saídas, "
        f"mas encontramos {len(output_files)}."
    )

all_stays = set()
duplicated_between_batches = set()

for output_file in output_files:
    stay_ids = set(
        pq.read_table(
            output_file,
            columns=["stay_id"],
        )["stay_id"].to_pylist()
    )

    duplicated_between_batches.update(
        all_stays.intersection(stay_ids)
    )

    all_stays.update(stay_ids)

if duplicated_between_batches:
    raise RuntimeError(
        "Existem stays presentes em mais de um batch: "
        f"{sorted(duplicated_between_batches)[:20]}"
    )

total_rows = sum(
    summary["rows_output"]
    for summary in approved_summaries
)

total_positives = sum(
    summary["positive_rows"]
    for summary in approved_summaries
)

total_size_gb = sum(
    file.stat().st_size
    for file in output_files
) / 1024**3

registered = json.loads(
    FEATURE_NAMES_PATH.read_text(
        encoding="utf-8"
    )
)

final_manifest = {
    "status": "approved",
    "step": 7,
    "dataset_version": DATASET_VERSION,
    "source_step5_version": STEP5_VERSION,
    "source_step5_manifest": str(STEP5_MANIFEST_PATH),
    "source_step5_manifest_sha256": STEP5_MANIFEST_SHA256,
    "source_step5_eligible_rows": int(step5_manifest["eligible_rows"]),
    "source_step5_positive_rows": int(step5_manifest["target_positive_rows"]),
    "batches": len(output_files),
    "unique_stays_with_eligible_rows": len(all_stays),
    "rows": int(total_rows),
    "positive_rows": int(total_positives),
    "negative_rows": int(total_rows - total_positives),
    "positive_prevalence": (
        float(total_positives / total_rows)
        if total_rows
        else None
    ),
    "feature_count": int(
        registered["feature_count"]
    ),
    "target": TARGET_COLUMN,
    "lag_minutes": STEP_MINUTES,
    "feature_time_definition": "t-5min",
    "label_time_definition": "t",
    "prediction_window": "(t, t+8h]",
    "onset_and_post_onset_excluded": True,
    "vasoactive_features_excluded_from_x": True,
    "raw_identifiers_and_times_excluded_from_x": True,
    "output_size_gb": round(total_size_gb, 4),
    "feature_names_file": str(
        FEATURE_NAMES_PATH
    ),
    "data_directory": str(DATA_DIR),
}

if total_rows > int(step5_manifest["eligible_rows"]):
    raise RuntimeError("Step 7 possui mais linhas que as elegíveis no Step 5.")

if total_positives > int(step5_manifest["target_positive_rows"]):
    raise RuntimeError("Step 7 possui mais positivos que o Step 5.")

atomic_write_json(
    final_manifest,
    MANIFEST_PATH,
)

print(
    json.dumps(
        final_manifest,
        indent=2,
        ensure_ascii=False,
    )
)

print("\nSTEP 7 CONCLUÍDO E APROVADO.")
