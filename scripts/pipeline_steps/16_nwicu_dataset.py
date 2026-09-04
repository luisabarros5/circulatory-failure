"""Source exported from the reviewed development notebook.

This file contains code only; patient-level data and notebook outputs are not distributed.
"""

# %% [source cell 1]
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(os.environ.get(
    "CF_PROJECT", "/path/to/circulatory_failure_prediction"
)).resolve()

MAPPING_FILE = PROJECT / "outputs/step15_nwicu_schema_mapping/nwicu_v010/mapping_approved.json"
FEATURE_FILE = PROJECT / "outputs/step07_aligned_xy_v2/feature_names.json"
OUTPUT_DIR = PROJECT / "outputs/step16_nwicu_aligned_xy/nwicu_v010"
LEGACY_OUTPUT_DIR = PROJECT / "outputs/step16_nwicu_aligned_xy/nwicu_v010_v2"

# Configuração da coorte externa final.
COHORT_SUBJECTS = 10_000
COHORT_SEED = 42
SMOKE_SUBJECTS = 100
RUN_MODE = "full"  # altere somente para "full" após aprovar o smoke

# Use True somente para descartar uma execução canônica incompleta/incompatível.
RESET_STEP16_OUTPUTS = False
REBUILD_RAW_CACHE = False

RAW_BUCKETS = 64
CSV_CHUNK_ROWS = 500_000
TARGET = "target_first_onset_0_8h"
SENSITIVITY_TARGET = "target_first_onset_map_only_0_8h"
CONTRACT_VERSION = 3

if RUN_MODE not in {"smoke", "full"}:
    raise ValueError("RUN_MODE deve ser 'smoke' ou 'full'.")
if LEGACY_OUTPUT_DIR.exists():
    shutil.rmtree(LEGACY_OUTPUT_DIR)
if RESET_STEP16_OUTPUTS and OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)

assert MAPPING_FILE.is_file(), f"Execute primeiro o Step 15 canônico: {MAPPING_FILE}"
assert FEATURE_FILE.is_file(), FEATURE_FILE

MAPPING = json.loads(MAPPING_FILE.read_text())
assert MAPPING["approved"] is True
assert MAPPING["strict_semantic_equivalence"] is False

registry = json.loads(FEATURE_FILE.read_text())
FEATURES = list(registry["feature_names"])
FEATURE_HASH = hashlib.sha256("\n".join(FEATURES).encode()).hexdigest()
assert len(FEATURES) == 204
assert MAPPING["mimic_feature_sha256"] == FEATURE_HASH

RAW_DIR = OUTPUT_DIR / "raw_partitioned"
FULL_BATCH_DIR = OUTPUT_DIR / "batches"
SMOKE_BATCH_DIR = OUTPUT_DIR / "smoke_batches"
FULL_SUMMARY_DIR = OUTPUT_DIR / "batch_summaries"
SMOKE_SUMMARY_DIR = OUTPUT_DIR / "smoke_batch_summaries"
for directory in (OUTPUT_DIR, RAW_DIR, FULL_BATCH_DIR, SMOKE_BATCH_DIR, FULL_SUMMARY_DIR, SMOKE_SUMMARY_DIR):
    directory.mkdir(parents=True, exist_ok=True)

print("Mapeamento:", MAPPING_FILE)
print("Saída canônica:", OUTPUT_DIR)
print("Contrato congelado:", len(FEATURES), "features")
print("Modo:", RUN_MODE)

# %% [source cell 3]
def read_csv_table(name, usecols=None, nrows=None):
    return pd.read_csv(
        MAPPING["tables"][name], usecols=usecols, nrows=nrows, low_memory=False
    )


icustays = read_csv_table(
    "icustays", usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime"]
)
patients = read_csv_table("patients", usecols=["subject_id", "anchor_age"])

candidate = icustays.merge(
    patients, on="subject_id", how="left", validate="many_to_one"
)
candidate["intime"] = pd.to_datetime(candidate["intime"], errors="raise")
candidate["outtime"] = pd.to_datetime(candidate["outtime"], errors="raise")
candidate["stay_hours"] = (
    candidate["outtime"] - candidate["intime"]
).dt.total_seconds() / 3600
candidate = candidate.loc[
    candidate["anchor_age"].ge(MAPPING["cohort"]["adult_age_min"])
    & candidate["stay_hours"].gt(MAPPING["cohort"]["minimum_stay_hours"])
].drop_duplicates("stay_id").copy()

assert not candidate.duplicated("stay_id").any()
assert candidate["intime"].lt(candidate["outtime"]).all()

all_subjects = np.sort(candidate["subject_id"].astype("int64").unique())
rng = np.random.default_rng(COHORT_SEED)
sampled_subjects = rng.permutation(all_subjects)[: min(COHORT_SUBJECTS, len(all_subjects))]
sample_rank = pd.DataFrame({
    "subject_id": sampled_subjects.astype("int64"),
    "sample_rank": np.arange(1, len(sampled_subjects) + 1, dtype="int64"),
})

cohort = (
    candidate.merge(sample_rank, on="subject_id", how="inner", validate="many_to_one")
    .sort_values(["sample_rank", "stay_id"])
    .reset_index(drop=True)
)
cohort["bucket"] = cohort["stay_id"].astype("int64") % RAW_BUCKETS

cohort_signature_frame = cohort[
    ["subject_id", "hadm_id", "stay_id", "intime", "outtime", "sample_rank"]
].copy()
cohort_signature_frame["intime"] = cohort_signature_frame["intime"].astype(str)
cohort_signature_frame["outtime"] = cohort_signature_frame["outtime"].astype(str)
cohort_hash = hashlib.sha256(
    pd.util.hash_pandas_object(cohort_signature_frame, index=False)
    .to_numpy(dtype="uint64")
    .tobytes()
).hexdigest()

run_contract = {
    "contract_version": CONTRACT_VERSION,
    "source": "NWICU v0.1.0",
    "cohort_subject_target": COHORT_SUBJECTS,
    "cohort_seed": COHORT_SEED,
    "selected_subjects": int(cohort["subject_id"].nunique()),
    "selected_stays": int(cohort["stay_id"].nunique()),
    "selected_patient_days": float(cohort["stay_hours"].sum() / 24),
    "cohort_sha256": cohort_hash,
    "feature_count": len(FEATURES),
    "feature_sha256": FEATURE_HASH,
    "target": TARGET,
    "grid_minutes": int(MAPPING["grid"]["minutes"]),
    "prediction_horizon_hours": int(MAPPING["grid"]["prediction_horizon_hours"]),
}

contract_path = OUTPUT_DIR / "run_contract.json"
if contract_path.exists():
    previous_contract = json.loads(contract_path.read_text())
    if previous_contract != run_contract:
        raise RuntimeError(
            "Há outputs canônicos incompatíveis. Confira o caminho e, se forem de uma "
            "tentativa descartada, defina RESET_STEP16_OUTPUTS=True e execute desde o início."
        )
else:
    unexpected = list(RAW_DIR.rglob("*.parquet")) + list(FULL_BATCH_DIR.glob("*.parquet"))
    if unexpected:
        raise RuntimeError(
            "Foram encontrados outputs sem contrato. Use RESET_STEP16_OUTPUTS=True uma única vez."
        )
    contract_path.write_text(json.dumps(run_contract, indent=2) + "\n")

sample_rank.to_csv(OUTPUT_DIR / "external_validation_subjects.csv", index=False)
cohort.to_parquet(OUTPUT_DIR / "external_validation_stays.parquet", index=False)

active_subjects = set(
    sampled_subjects[:SMOKE_SUBJECTS] if RUN_MODE == "smoke" else sampled_subjects
)
active_cohort = cohort.loc[cohort["subject_id"].isin(active_subjects)].copy()

print("Pacientes adultos candidatos:", len(all_subjects))
print("Pacientes selecionados para validação externa:", cohort["subject_id"].nunique())
print("Stays selecionados:", cohort["stay_id"].nunique())
print("Pacientes-dia selecionados:", round(cohort["stay_hours"].sum() / 24, 1))
print("Pacientes ativos nesta execução:", active_cohort["subject_id"].nunique())
display(cohort.head())

# %% [source cell 5]
VALUE_SPECS = [spec for spec in MAPPING["features"] if spec.get("role") == "value"]
RAW_SPECS = [
    spec for spec in VALUE_SPECS
    if spec.get("source_table") in {"chartevents", "labevents"}
]

item_to_features = {}
for spec in RAW_SPECS:
    for itemid in spec.get("itemids", []):
        key = (spec["source_table"], int(itemid))
        item_to_features.setdefault(key, []).append(spec["output_feature"])


def clear_raw_cache():
    if RAW_DIR.exists():
        shutil.rmtree(RAW_DIR)
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def write_bucketed(frame, source, chunk_id):
    if frame.empty:
        return
    frame = frame.copy()
    frame["bucket"] = frame["stay_id"].astype("int64") % RAW_BUCKETS
    for bucket, piece in frame.groupby("bucket", sort=False):
        directory = RAW_DIR / source / f"bucket_{int(bucket):02d}"
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / f"part_{chunk_id:05d}.parquet"
        piece.drop(columns="bucket").to_parquet(destination, index=False)


def attach_labs_to_stays(chunk):
    bounds = cohort[["subject_id", "hadm_id", "stay_id", "intime", "outtime"]]
    merged = chunk.merge(bounds, on=["subject_id", "hadm_id"], how="inner")
    return merged.loc[
        merged["charttime"].between(merged["intime"], merged["outtime"], inclusive="both")
    ].copy()


def build_event_cache():
    marker = RAW_DIR / "CACHE_COMPLETED.json"
    if marker.exists() and not REBUILD_RAW_CACHE:
        recorded = json.loads(marker.read_text())
        if recorded.get("cohort_sha256") != cohort_hash or recorded.get("feature_sha256") != FEATURE_HASH:
            raise RuntimeError(
                "O cache bruto pertence a outro contrato. Defina REBUILD_RAW_CACHE=True e execute novamente."
            )
        print("Cache bruto já concluído:", marker)
        return

    clear_raw_cache()
    unit_counts = defaultdict(Counter)

    for source in ("chartevents", "labevents"):
        wanted = {itemid for table, itemid in item_to_features if table == source}
        columns = ["subject_id", "hadm_id", "itemid", "charttime", "valuenum", "valueuom"]
        if source == "chartevents":
            columns.append("stay_id")

        for chunk_id, chunk in enumerate(pd.read_csv(
            MAPPING["tables"][source],
            usecols=columns,
            chunksize=CSV_CHUNK_ROWS,
            low_memory=False,
        )):
            chunk = chunk.loc[
                chunk["itemid"].isin(wanted) & chunk["valuenum"].notna()
            ].copy()
            if chunk.empty:
                continue
            chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
            chunk = chunk.loc[chunk["charttime"].notna()].copy()
            if source == "chartevents":
                chunk = chunk.loc[chunk["stay_id"].isin(cohort["stay_id"])].copy()
            else:
                chunk = attach_labs_to_stays(chunk)
            if chunk.empty:
                continue

            chunk["output_feature"] = [
                item_to_features.get((source, int(itemid)), [])
                for itemid in chunk["itemid"].to_numpy()
            ]
            chunk = chunk.explode("output_feature")
            chunk = chunk.loc[chunk["output_feature"].notna()].copy()

            chunk["unit_key"] = chunk["valueuom"].fillna("<missing>").astype(str)
            for (feature, unit), count in chunk.groupby(
                ["output_feature", "unit_key"]
            ).size().items():
                unit_counts[str(feature)][str(unit)] += int(count)

            temperature = chunk["output_feature"].eq("temperature")
            chunk.loc[temperature, "valuenum"] = (
                chunk.loc[temperature, "valuenum"] - 32.0
            ) * 5.0 / 9.0

            write_bucketed(
                chunk[[
                    "subject_id", "hadm_id", "stay_id", "charttime",
                    "output_feature", "valuenum",
                ]],
                "features",
                chunk_id + (0 if source == "chartevents" else 10000),
            )
            if chunk_id == 0 or chunk_id % 5 == 0:
                print(source, "chunk", chunk_id)

    vaso = MAPPING["target"]["vasoactive"]
    regex = "|".join(f"(?:{pattern})" for pattern in vaso["name_patterns"])
    columns = ["subject_id", "hadm_id", "starttime", "stoptime", "drug", "route"]
    vaso_rows = 0
    for chunk_id, chunk in enumerate(pd.read_csv(
        MAPPING["tables"]["prescriptions"],
        usecols=columns,
        chunksize=CSV_CHUNK_ROWS,
        low_memory=False,
    )):
        chunk = chunk.loc[
            chunk["route"].isin(vaso["allowed_routes"])
            & chunk["drug"].astype(str).str.contains(regex, case=False, regex=True, na=False)
        ].copy()
        if chunk.empty:
            continue
        chunk["starttime"] = pd.to_datetime(chunk["starttime"], errors="coerce")
        chunk["endtime"] = pd.to_datetime(chunk["stoptime"], errors="coerce")
        chunk = chunk.loc[
            chunk["starttime"].notna()
            & chunk["endtime"].notna()
            & chunk["endtime"].ge(chunk["starttime"])
        ].copy()
        merged = chunk.merge(
            cohort[["subject_id", "hadm_id", "stay_id", "intime", "outtime"]],
            on=["subject_id", "hadm_id"],
            how="inner",
        )
        merged = merged.loc[
            merged["starttime"].le(merged["outtime"])
            & merged["endtime"].ge(merged["intime"])
        ].copy()
        if merged.empty:
            continue
        merged["starttime"] = merged[["starttime", "intime"]].max(axis=1)
        merged["endtime"] = merged[["endtime", "outtime"]].min(axis=1)
        vaso_rows += len(merged)
        write_bucketed(
            merged[["subject_id", "hadm_id", "stay_id", "starttime", "endtime", "drug"]],
            "vasoactive",
            chunk_id,
        )

    unit_rows = []
    for feature, counts in sorted(unit_counts.items()):
        for unit, count in counts.most_common():
            unit_rows.append({"feature": feature, "valueuom": unit, "rows": count})
    unit_audit = pd.DataFrame(unit_rows)
    unit_audit.to_csv(OUTPUT_DIR / "observed_unit_audit.csv", index=False)

    required_observed = {"heart_rate", "sbp", "dbp", "lab_50813"}
    observed_features = set(unit_audit["feature"]) if not unit_audit.empty else set()
    missing_required = sorted(required_observed - observed_features)
    if missing_required:
        raise RuntimeError(f"Variáveis essenciais sem observações na coorte: {missing_required}")

    marker.write_text(json.dumps({
        "status": "completed",
        "cohort_sha256": cohort_hash,
        "feature_sha256": FEATURE_HASH,
        "raw_buckets": RAW_BUCKETS,
        "selected_subjects": int(cohort["subject_id"].nunique()),
        "selected_stays": int(cohort["stay_id"].nunique()),
        "vasoactive_proxy_intervals": int(vaso_rows),
    }, indent=2) + "\n")
    print("Cache bruto concluído:", marker)


build_event_cache()

# %% [source cell 7]
SPEC_BY_FEATURE = {spec["output_feature"]: spec for spec in VALUE_SPECS}
BASE_FEATURES = [
    name for name in FEATURES
    if not name.endswith("_imputed")
    and not name.endswith("_time_since_last_measurement_min")
]
assert len(BASE_FEATURES) == 68


def read_bucket(source, bucket):
    files = sorted((RAW_DIR / source / f"bucket_{bucket:02d}").glob("*.parquet"))
    if not files:
        columns = (
            ["stay_id", "charttime", "output_feature", "valuenum"]
            if source == "features"
            else ["stay_id", "starttime", "endtime", "drug"]
        )
        return pd.DataFrame(columns=columns)
    return pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)


def observed_series(events, feature, grid_time):
    selected = events.loc[events["output_feature"].eq(feature)].copy()
    if selected.empty:
        return pd.Series(np.nan, index=grid_time, dtype="float32")
    selected["bin"] = selected["charttime"].dt.ceil("5min")
    selected = selected.loc[selected["bin"].isin(grid_time)]
    return (
        selected.groupby("bin")["valuenum"].last()
        .reindex(grid_time)
        .astype("float32")
    )


def failure_and_target(grid, vasoactive, include_vaso=True):
    lactate = pd.to_numeric(grid["lab_50813"], errors="coerce").ffill(limit=48)
    mbp = pd.to_numeric(grid["mbp"], errors="coerce")
    vaso = pd.Series(vasoactive, index=grid.index, dtype="boolean")
    if not include_vaso:
        vaso[:] = False

    raw = pd.Series(pd.NA, index=grid.index, dtype="boolean")
    raw.loc[lactate.notna() & lactate.lt(2.0)] = False
    lactate_high = lactate.notna() & lactate.ge(2.0)
    map_low = mbp.notna() & mbp.le(65.0)
    map_normal = mbp.notna() & mbp.gt(65.0)
    vaso_true = vaso.fillna(False)
    vaso_false_known = vaso.notna() & ~vaso.fillna(False)
    raw.loc[lactate_high & (map_low | vaso_true)] = True
    raw.loc[lactate_high & map_normal & vaso_false_known] = False

    known_count = raw.notna().astype("int8").rolling(3, min_periods=3).sum()
    positive_count = raw.fillna(False).astype("int8").rolling(3, min_periods=3).sum()
    complete_window = known_count.eq(3)
    failure = pd.Series(pd.NA, index=grid.index, dtype="boolean")
    failure.loc[complete_window] = positive_count.loc[complete_window].ge(2)

    positive_times = grid.loc[failure.fillna(False), "charttime"]
    onset = positive_times.iloc[0] if len(positive_times) else pd.NaT
    horizon_steps = 96
    state_known = failure.notna()
    future_all_known = (
        state_known.shift(-1)
        .iloc[::-1]
        .rolling(horizon_steps, min_periods=horizon_steps)
        .sum()
        .iloc[::-1]
        .eq(horizon_steps)
    )

    if pd.isna(onset):
        before_onset = pd.Series(True, index=grid.index, dtype=bool)
        target_positive = pd.Series(False, index=grid.index, dtype=bool)
    else:
        before_onset = grid["charttime"].lt(onset)
        target_positive = (
            onset > grid["charttime"]
        ) & (
            onset <= grid["charttime"] + pd.Timedelta(hours=8)
        )

    eligible_positive = state_known & before_onset & target_positive
    eligible_negative = state_known & before_onset & ~target_positive & future_all_known
    eligible = eligible_positive | eligible_negative
    target = pd.Series(pd.NA, index=grid.index, dtype="Int8")
    target.loc[eligible_positive] = 1
    target.loc[eligible_negative] = 0
    return target, eligible.astype(bool), onset


def build_stay(stay, events, intervals):
    grid_time = pd.date_range(
        stay.intime.ceil("5min"), stay.outtime.floor("5min"), freq="5min"
    )
    if len(grid_time) < 4:
        return pd.DataFrame(), None

    grid = pd.DataFrame({"charttime": grid_time})
    grid["stay_id"] = int(stay.stay_id)
    grid["subject_id"] = int(stay.subject_id)

    stay_events = events.loc[events["stay_id"].eq(stay.stay_id)].copy()
    if not stay_events.empty:
        stay_events["charttime"] = pd.to_datetime(stay_events["charttime"], errors="raise")

    observed = {}
    filled = {}
    imputed = {}
    for feature in BASE_FEATURES:
        if feature == "mbp":
            continue
        spec = SPEC_BY_FEATURE[feature]
        raw = (
            observed_series(stay_events, feature, grid_time)
            if spec.get("source_table")
            else pd.Series(np.nan, index=grid_time, dtype="float32")
        )
        observed[feature] = raw
        propagated = raw.ffill()
        default = spec.get("default")
        if default is not None and not pd.isna(default):
            propagated = propagated.fillna(float(default))
        filled[feature] = propagated.astype("float32")
        imputed[feature] = raw.isna().astype("int8")

    # Proxy de MAP, derivada causalmente dos valores de PAS/PAD disponíveis.
    observed["mbp"] = (observed["sbp"] + 2 * observed["dbp"]) / 3
    filled["mbp"] = ((filled["sbp"] + 2 * filled["dbp"]) / 3).astype("float32")
    imputed["mbp"] = (
        imputed["sbp"].astype(bool) | imputed["dbp"].astype(bool)
    ).astype("int8")

    grid_time_series = pd.Series(grid_time, index=grid_time)
    feature_data = {}
    for feature in BASE_FEATURES:
        feature_data[feature] = filled[feature].to_numpy()
        feature_data[f"{feature}_imputed"] = imputed[feature].to_numpy()
        measured_at = pd.Series(pd.NaT, index=grid_time, dtype="datetime64[ns]")
        measured = observed[feature].notna()
        measured_at.loc[measured] = grid_time[measured]
        elapsed = (grid_time_series - measured_at.ffill()).dt.total_seconds() / 60
        feature_data[f"{feature}_time_since_last_measurement_min"] = (
            elapsed.astype("float32").to_numpy()
        )
    grid = pd.concat([grid, pd.DataFrame(feature_data, index=grid.index)], axis=1)

    active = np.zeros(len(grid), dtype=bool)
    stay_intervals = intervals.loc[intervals["stay_id"].eq(stay.stay_id)]
    for interval in stay_intervals.itertuples(index=False):
        active |= (
            (grid_time >= np.datetime64(interval.starttime))
            & (grid_time <= np.datetime64(interval.endtime))
        )

    target, eligible, onset = failure_and_target(grid, active, include_vaso=True)
    sensitivity, sensitivity_eligible, sensitivity_onset = failure_and_target(
        grid, active, include_vaso=False
    )
    grid[TARGET] = target
    grid[SENSITIVITY_TARGET] = sensitivity
    grid["eligible"] = eligible
    grid["sensitivity_eligible"] = sensitivity_eligible

    # Mesmo alinhamento do Step 7 MIMIC: X em t-5 min prediz o label em t.
    aligned = grid.iloc[:-1].copy()
    aligned["feature_time"] = grid["charttime"].iloc[:-1].to_numpy()
    aligned["label_time"] = grid["charttime"].iloc[1:].to_numpy()
    aligned[TARGET] = grid[TARGET].iloc[1:].to_numpy()
    aligned[SENSITIVITY_TARGET] = grid[SENSITIVITY_TARGET].iloc[1:].to_numpy()
    aligned["eligible"] = grid["eligible"].iloc[1:].to_numpy()
    aligned["sensitivity_eligible"] = grid["sensitivity_eligible"].iloc[1:].to_numpy()
    aligned = aligned.loc[aligned["eligible"]].copy()

    if not aligned.empty:
        aligned[TARGET] = aligned[TARGET].astype("int8")
        aligned = aligned[[
            "subject_id", "stay_id", "feature_time", "label_time", TARGET,
            SENSITIVITY_TARGET, "sensitivity_eligible", *FEATURES,
        ]]

    has_vital = bool(
        not stay_events.empty
        and stay_events["output_feature"].isin(
            ["heart_rate", "sbp", "dbp", "resp_rate", "temperature", "spo2"]
        ).any()
    )
    has_lab = bool(
        not stay_events.empty
        and (
            stay_events["output_feature"].astype(str).str.startswith("lab_").any()
            or stay_events["output_feature"].eq("glucose").any()
        )
    )
    summary = {
        "subject_id": int(stay.subject_id),
        "stay_id": int(stay.stay_id),
        "rows": int(len(aligned)),
        "positive_rows": int(aligned[TARGET].sum()) if not aligned.empty else 0,
        "first_onset": onset,
        "map_only_first_onset": sensitivity_onset,
        "has_selected_vital": has_vital,
        "has_selected_lab": has_lab,
    }
    return aligned, summary

# %% [source cell 9]
batch_dir = SMOKE_BATCH_DIR if RUN_MODE == "smoke" else FULL_BATCH_DIR
summary_dir = SMOKE_SUMMARY_DIR if RUN_MODE == "smoke" else FULL_SUMMARY_DIR

for bucket in sorted(active_cohort["bucket"].unique()):
    output_path = batch_dir / f"bucket_{int(bucket):02d}.parquet"
    summary_path = summary_dir / f"bucket_{int(bucket):02d}.parquet"
    if output_path.exists() and summary_path.exists():
        print("Já concluído:", output_path.name)
        continue

    events = read_bucket("features", int(bucket))
    intervals = read_bucket("vasoactive", int(bucket))
    if not intervals.empty:
        intervals["starttime"] = pd.to_datetime(intervals["starttime"], errors="raise")
        intervals["endtime"] = pd.to_datetime(intervals["endtime"], errors="raise")

    frames = []
    summaries = []
    bucket_cohort = active_cohort.loc[active_cohort["bucket"].eq(bucket)]
    for stay in bucket_cohort.itertuples(index=False):
        frame, summary = build_stay(stay, events, intervals)
        if summary is None:
            continue
        summaries.append(summary)
        if (
            summary["has_selected_vital"]
            and summary["has_selected_lab"]
            and not frame.empty
        ):
            frames.append(frame)

    if frames:
        pd.concat(frames, ignore_index=True).to_parquet(
            output_path, index=False, compression="zstd"
        )
    else:
        pd.DataFrame(columns=[
            "subject_id", "stay_id", "feature_time", "label_time", TARGET,
            SENSITIVITY_TARGET, "sensitivity_eligible", *FEATURES,
        ]).to_parquet(output_path, index=False, compression="zstd")

    pd.DataFrame(summaries).to_parquet(summary_path, index=False, compression="zstd")
    print(
        "Bucket", int(bucket),
        "| stays examinados:", len(summaries),
        "| stays produzidos:", len(frames),
    )

# %% [source cell 11]
files = sorted(batch_dir.glob("bucket_*.parquet"))
assert files, "Nenhum bucket foi produzido."

rows = 0
positives = 0
stay_ids = set()
subject_ids = set()
observed_counts = Counter()
mask_columns = [name for name in FEATURES if name.endswith("_imputed")]

for path in files:
    frame = pd.read_parquet(path)
    if frame.empty:
        continue
    assert list(frame.columns[-len(FEATURES):]) == FEATURES
    assert (
        pd.to_datetime(frame["label_time"]) - pd.to_datetime(frame["feature_time"])
    ).eq(pd.Timedelta(minutes=5)).all()
    assert set(frame[TARGET].unique()).issubset({0, 1})
    assert not frame.duplicated(["stay_id", "label_time"]).any()

    rows += len(frame)
    positives += int(frame[TARGET].sum())
    stay_ids.update(frame["stay_id"].astype(int).unique())
    subject_ids.update(frame["subject_id"].astype(int).unique())
    for column, count in frame[mask_columns].eq(0).sum().items():
        observed_counts[column] += int(count)

assert rows > 0
availability = pd.DataFrame({
    "feature": mask_columns,
    "observed_rows": [observed_counts[name] for name in mask_columns],
    "total_rows": rows,
})
availability["observed_fraction"] = availability["observed_rows"] / rows
availability_path = OUTPUT_DIR / (
    "smoke_feature_availability.csv" if RUN_MODE == "smoke" else "feature_availability.csv"
)
availability.to_csv(availability_path, index=False)

all_summaries = sorted(summary_dir.glob("bucket_*.parquet"))
summary = pd.concat([pd.read_parquet(path) for path in all_summaries], ignore_index=True)
summary_path = OUTPUT_DIR / (
    "smoke_stay_summary.parquet" if RUN_MODE == "smoke" else "stay_summary.parquet"
)
summary.to_parquet(summary_path, index=False, compression="zstd")

manifest = {
    "status": "smoke" if RUN_MODE == "smoke" else "completed",
    "source": "NWICU v0.1.0",
    "contract_version": CONTRACT_VERSION,
    "cohort_kind": "fixed_external_validation_cohort",
    "cohort_subject_target": COHORT_SUBJECTS,
    "cohort_seed": COHORT_SEED,
    "cohort_sha256": cohort_hash,
    "selected_subjects": int(cohort["subject_id"].nunique()),
    "selected_stays": int(cohort["stay_id"].nunique()),
    "produced_subjects": int(len(subject_ids)),
    "produced_stays": int(len(stay_ids)),
    "selected_patient_days": float(cohort["stay_hours"].sum() / 24),
    "feature_count": len(FEATURES),
    "feature_sha256": FEATURE_HASH,
    "files": len(files),
    "rows": int(rows),
    "positive_rows": int(positives),
    "prevalence": float(positives / rows),
    "all_eligible_target_rows_retained": True,
    "external_data_used_for_training": False,
    "external_scaler_refit": False,
    "external_threshold_tuning": False,
    "strict_semantic_equivalence": False,
    "target_harmonization": "derived MAP + continuous-prescription vasoactive proxy",
}

if RUN_MODE == "full":
    assert manifest["produced_subjects"] <= manifest["selected_subjects"]
    assert 0 < positives < rows, "A execução completa precisa conter as duas classes."

manifest_path = OUTPUT_DIR / (
    "smoke_manifest.json" if RUN_MODE == "smoke" else "manifest.json"
)
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

display(availability.sort_values("observed_fraction").head(30))
print(json.dumps(manifest, indent=2))
if RUN_MODE == "smoke":
    print("SMOKE concluído. Se as auditorias estiverem corretas, altere somente RUN_MODE para 'full'.")
