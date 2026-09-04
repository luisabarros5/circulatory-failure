"""Source exported from the reviewed development notebook.

This file contains code only; patient-level data and notebook outputs are not distributed.
"""

# %% [source cell 2]
from pathlib import Path
import importlib.util
import json
import os
import re
import time

import pandas as pd
import pyarrow.parquet as pq

STEP_MINUTES = 5
EXPECTED_BATCHES = 285
EXPECTED_SOURCE_STAYS = 71054
STEP5_VERSION = "observed_onset_sufficient_without_full_8h_followup_v2"
DATASET_VERSION = "dataset_aligned_xy_v2"
TARGET_COLUMN = "target_first_onset_0_8h"

PROJECT_DIR = Path("/path/to/circulatory_failure_prediction")
CLEAN_DIR = Path(
    "/path/to/mimic_source_project/"
    "output_icu_clean_v2/raw_grid_all_icu_clean_v2"
)
LABEL_DIR = PROJECT_DIR / "outputs" / "step05_labels_v2"
STEP5_MANIFEST_PATH = LABEL_DIR / "manifest_step5_eligibility_v2.json"

STEP7_DIR = PROJECT_DIR / "outputs" / "step07_aligned_xy_v2"
DATA_DIR = STEP7_DIR / "batches"
SUMMARY_DIR = STEP7_DIR / "summaries"
FEATURE_NAMES_PATH = STEP7_DIR / "feature_names.json"
STEP7_MANIFEST_PATH = STEP7_DIR / "manifest_step7.json"

VALIDATION_DIR = PROJECT_DIR / "outputs" / "step08_validation_v2"
BATCH_REPORT_PATH = VALIDATION_DIR / "batch_validation.csv"
EXCLUSION_REPORT_PATH = VALIDATION_DIR / "stays_without_training_rows.csv"
EXCLUSION_SUMMARY_PATH = VALIDATION_DIR / "exclusion_summary.csv"
STEP8_MANIFEST_PATH = VALIDATION_DIR / "manifest_step8.json"

FEATURE_GUARD_PATH = PROJECT_DIR / "scripts" / "feature_guard.py"
VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

print("Dataset Step 7:", STEP7_DIR)
print("Validação Step 8:", VALIDATION_DIR)

# %% [source cell 4]
def canonical_batches(directory):
    return sorted(
        path
        for path in directory.glob("*.parquet")
        if re.fullmatch(r"batch_\d{4}\.parquet", path.name)
    )


required_files = [
    STEP5_MANIFEST_PATH,
    STEP7_MANIFEST_PATH,
    FEATURE_NAMES_PATH,
    FEATURE_GUARD_PATH,
]

missing_required = [str(path) for path in required_files if not path.exists()]
if missing_required:
    raise FileNotFoundError(f"Arquivos obrigatórios ausentes: {missing_required}")

step5_manifest = json.loads(STEP5_MANIFEST_PATH.read_text(encoding="utf-8"))
step7_manifest = json.loads(STEP7_MANIFEST_PATH.read_text(encoding="utf-8"))
feature_registry = json.loads(FEATURE_NAMES_PATH.read_text(encoding="utf-8"))

if step5_manifest.get("status") != "approved" or step5_manifest.get("version") != STEP5_VERSION:
    raise RuntimeError("O Step 5 aprovado não corresponde à versão exigida.")

if step7_manifest.get("status") != "approved" or step7_manifest.get("dataset_version") != DATASET_VERSION:
    raise RuntimeError("O Step 7 ainda não está aprovado na versão v2.")

if step7_manifest.get("source_step5_version") != STEP5_VERSION:
    raise RuntimeError("O Step 7 não foi construído a partir do Step 5 v2.")

if step7_manifest.get("lag_minutes") != STEP_MINUTES:
    raise RuntimeError("O manifesto do Step 7 não registra lag de cinco minutos.")

clean_files = canonical_batches(CLEAN_DIR)
label_files = canonical_batches(LABEL_DIR)
output_files = canonical_batches(DATA_DIR)
expected_names = [f"batch_{index:04d}.parquet" for index in range(EXPECTED_BATCHES)]

for label, files in [
    ("ICU-clean", clean_files),
    ("labels", label_files),
    ("Step 7", output_files),
]:
    names = [path.name for path in files]
    if names != expected_names:
        raise RuntimeError(f"{label}: sequência de batches diferente de 0000–0284.")

feature_names = feature_registry["feature_names"]
if len(feature_names) != feature_registry["feature_count"]:
    raise RuntimeError("Contagem inconsistente no feature_names.json.")

spec = importlib.util.spec_from_file_location("feature_guard_step8", FEATURE_GUARD_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError("Não foi possível carregar feature_guard.py.")

feature_guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(feature_guard)
feature_guard.assert_no_forbidden_features(feature_names)

print("Batches confirmados:", len(output_files))
print("Features confirmadas:", len(feature_names))
print("Step 5:", step5_manifest["version"])
print("Step 7:", step7_manifest["dataset_version"])

# %% [source cell 6]
batch_rows = []
all_output_stays = set()
seen_output_stays = set()
duplicate_stays_between_batches = set()
expected_output_columns = [
    "stay_id",
    "feature_time",
    "label_time",
    TARGET_COLUMN,
    *feature_names,
]

started = time.perf_counter()

for index, (clean_file, label_file, output_file) in enumerate(
    zip(clean_files, label_files, output_files),
    start=1,
):
    output = pd.read_parquet(output_file)
    labels = pd.read_parquet(
        label_file,
        columns=[
            "stay_id",
            "charttime",
            "failure_state_new_at_t",
            TARGET_COLUMN,
            "eligible_for_training",
        ],
    )
    clean_keys = pq.read_table(
        clean_file,
        columns=["stay_id", "charttime"],
    ).to_pandas()

    if list(output.columns) != expected_output_columns:
        raise RuntimeError(f"{output_file.name}: schema diferente do registro oficial.")

    for frame, time_columns in [
        (output, ["feature_time", "label_time"]),
        (labels, ["charttime"]),
        (clean_keys, ["charttime"]),
    ]:
        frame["stay_id"] = pd.to_numeric(frame["stay_id"], errors="raise").astype("int64")
        for column in time_columns:
            frame[column] = pd.to_datetime(frame[column], errors="raise")

    if output.duplicated(["stay_id", "label_time"]).any():
        raise RuntimeError(f"{output_file.name}: chave de label duplicada.")

    if output.duplicated(["stay_id", "feature_time"]).any():
        raise RuntimeError(f"{output_file.name}: chave de feature duplicada.")

    lag = (output["label_time"] - output["feature_time"]).dt.total_seconds().div(60)
    if not lag.eq(STEP_MINUTES).all():
        raise RuntimeError(f"{output_file.name}: existe lag diferente de cinco minutos.")

    target = pd.to_numeric(output[TARGET_COLUMN], errors="raise").astype("int8")
    if not set(target.unique()).issubset({0, 1}):
        raise RuntimeError(f"{output_file.name}: target fora de 0/1.")

    output_label_view = output[["stay_id", "label_time", TARGET_COLUMN]].copy()
    label_view = labels.rename(columns={"charttime": "label_time"})
    label_check = output_label_view.merge(
        label_view,
        on=["stay_id", "label_time"],
        how="left",
        validate="one_to_one",
        suffixes=("_step7", "_step5"),
        indicator=True,
    )

    if not label_check["_merge"].eq("both").all():
        raise RuntimeError(f"{output_file.name}: linha do Step 7 sem label correspondente.")

    eligible = label_check["eligible_for_training"].astype("boolean").fillna(False)
    if not eligible.all():
        raise RuntimeError(f"{output_file.name}: linha não elegível entrou no Step 7.")

    target_step5 = pd.to_numeric(
        label_check[f"{TARGET_COLUMN}_step5"], errors="raise"
    ).astype("int8")
    target_step7 = pd.to_numeric(
        label_check[f"{TARGET_COLUMN}_step7"], errors="raise"
    ).astype("int8")
    if not target_step5.equals(target_step7):
        raise RuntimeError(f"{output_file.name}: target diverge entre Steps 5 e 7.")

    failure_at_t = (
        label_check["failure_state_new_at_t"]
        .astype("boolean")
        .fillna(False)
    )
    if failure_at_t.any():
        raise RuntimeError(f"{output_file.name}: onset ou pós-onset entrou no treinamento.")

    feature_keys = output[["stay_id", "feature_time"]].rename(
        columns={"feature_time": "charttime"}
    )
    feature_key_check = feature_keys.merge(
        clean_keys,
        on=["stay_id", "charttime"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not feature_key_check["_merge"].eq("both").all():
        raise RuntimeError(f"{output_file.name}: X não corresponde à grade ICU-clean em t−5.")

    batch_stays = set(output["stay_id"].unique().tolist())
    duplicate_stays_between_batches.update(seen_output_stays.intersection(batch_stays))
    seen_output_stays.update(batch_stays)
    all_output_stays.update(batch_stays)

    summary_path = SUMMARY_DIR / f"{output_file.stem}_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    positives = int(target.sum())

    if summary.get("rows_output") != len(output) or summary.get("positive_rows") != positives:
        raise RuntimeError(f"{output_file.name}: resumo não corresponde ao Parquet.")

    batch_rows.append(
        {
            "batch": output_file.name,
            "status": "approved",
            "rows": int(len(output)),
            "stays": int(len(batch_stays)),
            "positive_rows": positives,
            "negative_rows": int(len(output) - positives),
            "lag_minutes": STEP_MINUTES,
            "features": len(feature_names),
        }
    )

    if index == 1 or index % 25 == 0 or index == EXPECTED_BATCHES:
        print(f"[{index:03d}/{EXPECTED_BATCHES}] validação dos batches OK", flush=True)

if duplicate_stays_between_batches:
    raise RuntimeError(
        "Há stays em mais de um batch: "
        f"{sorted(duplicate_stays_between_batches)[:20]}"
    )

batch_validation = pd.DataFrame(batch_rows)
batch_validation.to_csv(BATCH_REPORT_PATH, index=False)

print("Relatório:", BATCH_REPORT_PATH)
print("Duração:", round((time.perf_counter() - started) / 60, 2), "minutos")
print("Linhas validadas:", int(batch_validation["rows"].sum()))
print("Stays validados:", len(all_output_stays))

# %% [source cell 8]
stay_frames = []

for label_file, output_file in zip(label_files, output_files):
    labels = pd.read_parquet(
        label_file,
        columns=[
            "stay_id",
            "charttime",
            "failure_state_new_at_t",
            "eligible_for_training",
        ],
    )
    output_keys = pd.read_parquet(
        output_file,
        columns=["stay_id", "label_time"],
    )

    labels["charttime"] = pd.to_datetime(labels["charttime"], errors="raise")
    labels = labels.sort_values(["stay_id", "charttime"]).reset_index(drop=True)
    labels["grid_index"] = labels.groupby("stay_id", sort=False).cumcount()
    labels["eligible_numeric"] = (
        labels["eligible_for_training"].astype("boolean").fillna(False).astype("int8")
    )
    labels["eligible_after_lag_numeric"] = (
        labels["eligible_numeric"].eq(1) & labels["grid_index"].gt(0)
    ).astype("int8")
    labels["onset_time_aux"] = labels["charttime"].where(
        labels["failure_state_new_at_t"].astype("boolean").fillna(False)
    )
    labels["first_onset_time_group"] = labels.groupby(
        "stay_id", sort=False
    )["onset_time_aux"].transform("min")
    labels["known_pre_onset_within_8h"] = (
        labels["failure_state_new_at_t"].notna()
        & labels["first_onset_time_group"].notna()
        & labels["charttime"].lt(labels["first_onset_time_group"])
        & labels["charttime"].ge(
            labels["first_onset_time_group"] - pd.Timedelta(hours=8)
        )
    ).astype("int8")

    per_stay = labels.groupby("stay_id", as_index=False, sort=False).agg(
        source_rows=("charttime", "size"),
        first_grid_time=("charttime", "min"),
        last_grid_time=("charttime", "max"),
        first_onset_time=("onset_time_aux", "min"),
        eligible_rows_step5=("eligible_numeric", "sum"),
        eligible_rows_after_lag=("eligible_after_lag_numeric", "sum"),
        known_pre_onset_rows_within_8h=("known_pre_onset_within_8h", "sum"),
    )

    output_counts = output_keys.groupby("stay_id", as_index=False).size().rename(
        columns={"size": "output_rows_step7"}
    )
    per_stay = per_stay.merge(output_counts, on="stay_id", how="left", validate="one_to_one")
    per_stay["output_rows_step7"] = per_stay["output_rows_step7"].fillna(0).astype("int64")
    per_stay.insert(0, "batch", label_file.name)
    stay_frames.append(per_stay)

stay_audit = pd.concat(stay_frames, ignore_index=True)

if stay_audit["stay_id"].duplicated().any():
    raise RuntimeError("Há stay_id repetido entre batches na auditoria de exclusões.")

if stay_audit["stay_id"].nunique() != EXPECTED_SOURCE_STAYS:
    raise RuntimeError(
        f"Esperávamos {EXPECTED_SOURCE_STAYS} stays de origem, "
        f"mas encontramos {stay_audit['stay_id'].nunique()}."
    )

def classify_exclusion(row):
    if row["output_rows_step7"] > 0:
        return "com_linhas_de_treinamento"
    if row["eligible_rows_after_lag"] > 0:
        return "ERRO_linha_elegivel_apos_lag_nao_materializada"
    if row["eligible_rows_step5"] > 0:
        return "somente_label_elegivel_sem_x_tmenos5"
    if pd.notna(row["first_onset_time"]):
        if row["first_onset_time"] == row["first_grid_time"]:
            return "onset_no_primeiro_timestep"
        if row["known_pre_onset_rows_within_8h"] > 0:
            return "ERRO_step5_ignorou_estado_conhecido_nas_8h_pre_onset"
        return "onset_sem_estado_conhecido_nas_8h_pre_onset"
    return "sem_onset_sem_horizonte_negativo_completo"

stay_audit["reason"] = stay_audit.apply(classify_exclusion, axis=1)
excluded = stay_audit.loc[stay_audit["output_rows_step7"].eq(0)].copy()

exclusion_summary = (
    excluded.groupby("reason", as_index=False)
    .agg(stays=("stay_id", "nunique"))
    .sort_values(["stays", "reason"], ascending=[False, True])
    .reset_index(drop=True)
)

excluded.to_csv(EXCLUSION_REPORT_PATH, index=False)
exclusion_summary.to_csv(EXCLUSION_SUMMARY_PATH, index=False)

blocking = excluded.loc[excluded["reason"].str.startswith("ERRO_")]
if not blocking.empty:
    display(blocking.head(30))
    raise RuntimeError(
        f"Existem {len(blocking)} stays com exclusão não explicada."
    )

print("Stays ICU-clean:", stay_audit["stay_id"].nunique())
print("Stays com linhas:", stay_audit.loc[stay_audit["output_rows_step7"].gt(0), "stay_id"].nunique())
print("Stays sem linhas:", len(excluded))
display(exclusion_summary)
print("Relatório individual:", EXCLUSION_REPORT_PATH)

# %% [source cell 10]
total_rows = int(batch_validation["rows"].sum())
total_positives = int(batch_validation["positive_rows"].sum())
total_stays = len(all_output_stays)

expected_global = {
    "batches": EXPECTED_BATCHES,
    "rows": total_rows,
    "positive_rows": total_positives,
    "unique_stays_with_eligible_rows": total_stays,
    "feature_count": len(feature_names),
    "lag_minutes": STEP_MINUTES,
}

global_mismatches = {
    key: {"manifest": step7_manifest.get(key), "validated": value}
    for key, value in expected_global.items()
    if step7_manifest.get(key) != value
}

if global_mismatches:
    raise RuntimeError(f"Manifesto Step 7 diverge da validação: {global_mismatches}")

if total_stays + len(excluded) != EXPECTED_SOURCE_STAYS:
    raise RuntimeError("Stays incluídos + excluídos não recompõem a coorte ICU-clean.")

final_manifest = {
    "status": "approved",
    "step": 8,
    "validation_version": "validation_step8_v2",
    "validated_dataset_version": DATASET_VERSION,
    "validated_step5_version": STEP5_VERSION,
    "batches": EXPECTED_BATCHES,
    "source_stays": EXPECTED_SOURCE_STAYS,
    "stays_with_training_rows": total_stays,
    "stays_without_training_rows": int(len(excluded)),
    "rows": total_rows,
    "positive_rows": total_positives,
    "negative_rows": int(total_rows - total_positives),
    "feature_count": len(feature_names),
    "lag_minutes": STEP_MINUTES,
    "alignment": "X(t-5min) -> Y(t,t+8h]",
    "onset_and_post_onset_rows_found": 0,
    "forbidden_features_found": 0,
    "duplicate_keys_found": 0,
    "duplicate_stays_between_batches_found": 0,
    "unexplained_excluded_stays": 0,
    "batch_report": str(BATCH_REPORT_PATH),
    "exclusion_report": str(EXCLUSION_REPORT_PATH),
    "exclusion_summary": str(EXCLUSION_SUMMARY_PATH),
}

temp_manifest = STEP8_MANIFEST_PATH.with_suffix(".json.tmp")
temp_manifest.write_text(
    json.dumps(final_manifest, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
os.replace(temp_manifest, STEP8_MANIFEST_PATH)

print(json.dumps(final_manifest, indent=2, ensure_ascii=False))
print("\nSTEP 8 v2 CONCLUÍDO E APROVADO.")
