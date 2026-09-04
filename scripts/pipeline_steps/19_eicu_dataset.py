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
from collections import Counter
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

PROJECT = Path(os.environ.get(
    "CF_PROJECT", "/path/to/circulatory_failure_prediction"
)).resolve()
MAPPING_FILE = PROJECT / "outputs/step18_eicu_schema_mapping/eicu_v20/mapping_approved.json"
FEATURE_FILE = PROJECT / "outputs/step07_aligned_xy_v2/feature_names.json"
OUTPUT_DIR = PROJECT / "outputs/step19_eicu_aligned_xy/eicu_v20"

COHORT_SUBJECTS = 10_000
COHORT_SEED = 42
SMOKE_SUBJECTS = 100
RUN_MODE = "full"  # após conferir o smoke, altere somente para "full"
RESET_STEP19_OUTPUTS = False
REBUILD_PARQUET_CACHE = False
BATCH_STAYS = 250
TARGET = "target_first_onset_0_8h"
SENSITIVITY_TARGET = "target_first_onset_map_only_0_8h"

if RUN_MODE not in {"smoke", "full"}:
    raise ValueError("RUN_MODE deve ser 'smoke' ou 'full'.")
if RESET_STEP19_OUTPUTS and OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)
assert MAPPING_FILE.is_file(), f"Execute primeiro o Step 18: {MAPPING_FILE}"
assert FEATURE_FILE.is_file(), FEATURE_FILE

MAPPING = json.loads(MAPPING_FILE.read_text())
assert MAPPING["approved"] is True and MAPPING["strict_semantic_equivalence"] is False
assert MAPPING.get("harmonization_revision") == 3, "Execute novamente o Step 18 corrigido."
FEATURES = json.loads(FEATURE_FILE.read_text())["feature_names"]
FEATURE_HASH = hashlib.sha256("\n".join(FEATURES).encode()).hexdigest()
assert len(FEATURES) == 204 and MAPPING["mimic_feature_sha256"] == FEATURE_HASH

CACHE_DIR = OUTPUT_DIR / "source_parquet"
FULL_BATCH_DIR = OUTPUT_DIR / "batches"
SMOKE_BATCH_DIR = OUTPUT_DIR / "smoke_batches"
FULL_SUMMARY_DIR = OUTPUT_DIR / "batch_summaries"
SMOKE_SUMMARY_DIR = OUTPUT_DIR / "smoke_batch_summaries"
for path in (OUTPUT_DIR,CACHE_DIR,FULL_BATCH_DIR,SMOKE_BATCH_DIR,FULL_SUMMARY_DIR,SMOKE_SUMMARY_DIR):
    path.mkdir(parents=True, exist_ok=True)

print("Modo:", RUN_MODE)
print("Mapeamento:", MAPPING_FILE)
print("Saída canônica:", OUTPUT_DIR)

# %% [source cell 3]
con = duckdb.connect()

EXPORTS = {
    "patient": "SELECT patientunitstayid, uniquepid, age, unitdischargeoffset FROM read_csv_auto({path}, ignore_errors=true)",
    "vital": "SELECT patientunitstayid, observationoffset, temperature, sao2, heartrate, respiration, systemicsystolic, systemicdiastolic, systemicmean FROM read_csv_auto({path}, ignore_errors=true)",
    "lab": "SELECT patientunitstayid, labresultoffset, labname, try_cast(labresult AS DOUBLE) AS labresult FROM read_csv_auto({path}, ignore_errors=true)",
    "customLab": "SELECT patientunitstayid, labotheroffset AS labresultoffset, labothername AS labname, try_cast(labotherresult AS DOUBLE) AS labresult FROM read_csv_auto({path}, ignore_errors=true)",
    "infusion": "SELECT patientunitstayid, infusionoffset, drugname FROM read_csv_auto({path}, ignore_errors=true)",
}
SOURCE_KEYS = {"patient":"patient","vital":"vitalPeriodic","lab":"lab","customLab":"customLab","infusion":"infusionDrug"}

if REBUILD_PARQUET_CACHE:
    for path in CACHE_DIR.glob("*.parquet"):
        path.unlink()

for name, sql in EXPORTS.items():
    destination = CACHE_DIR / f"{name}.parquet"
    if destination.is_file():
        print("Cache existente:", destination.name)
        continue
    source = MAPPING["tables"][SOURCE_KEYS[name]].replace("'", "''")
    rendered = sql.format(path=f"'{source}'")
    con.execute(f"COPY ({rendered}) TO '{destination.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    print("Cache criado:", destination.name)

# %% [source cell 5]
patient = con.execute(f"SELECT * FROM read_parquet('{(CACHE_DIR/'patient.parquet').as_posix()}')").fetch_df()

def parse_age(value):
    text = str(value).strip()
    if text.startswith(">"):
        return 90.0
    return pd.to_numeric(text, errors="coerce")

patient["age_numeric"] = patient["age"].map(parse_age)
patient["unitdischargeoffset"] = pd.to_numeric(patient["unitdischargeoffset"], errors="coerce")
candidate = patient.loc[
    patient["age_numeric"].ge(MAPPING["cohort"]["adult_age_min"])
    & patient["unitdischargeoffset"].gt(MAPPING["cohort"]["minimum_stay_minutes"])
    & patient["uniquepid"].notna()
].drop_duplicates("patientunitstayid").copy()
candidate["subject_id"] = candidate["uniquepid"].astype(str)
candidate["stay_id"] = candidate["patientunitstayid"].astype("int64")
candidate["stay_hours"] = candidate["unitdischargeoffset"] / 60.0

all_subjects = np.sort(candidate["subject_id"].unique())
rng = np.random.default_rng(COHORT_SEED)
sampled_subjects = rng.permutation(all_subjects)[:min(COHORT_SUBJECTS,len(all_subjects))]
sample_rank = pd.DataFrame({"subject_id":sampled_subjects,
                            "sample_rank":np.arange(1,len(sampled_subjects)+1,dtype="int64")})
cohort = (candidate.merge(sample_rank,on="subject_id",how="inner",validate="many_to_one")
          .sort_values(["sample_rank","stay_id"]).reset_index(drop=True))
signature = cohort[["subject_id","stay_id","unitdischargeoffset","sample_rank"]].copy()
cohort_hash = hashlib.sha256(pd.util.hash_pandas_object(signature,index=False).to_numpy(dtype="uint64").tobytes()).hexdigest()

contract = {"source":"eICU-CRD v2.0","cohort_subject_target":COHORT_SUBJECTS,
            "cohort_seed":COHORT_SEED,"selected_subjects":int(cohort.subject_id.nunique()),
            "selected_stays":int(cohort.stay_id.nunique()),"cohort_sha256":cohort_hash,
            "feature_count":len(FEATURES),"feature_sha256":FEATURE_HASH,"target":TARGET,
            "harmonization_revision":MAPPING["harmonization_revision"]}
contract_path = OUTPUT_DIR / "run_contract.json"
if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
    previous = json.loads(contract_path.read_text())
    same_cohort = previous.get("cohort_sha256") == cohort_hash
    same_features = previous.get("feature_sha256") == FEATURE_HASH
    old_revision = int(previous.get("harmonization_revision", 1))
    if same_cohort and same_features and old_revision < MAPPING["harmonization_revision"]:
        for directory in (FULL_BATCH_DIR,SMOKE_BATCH_DIR,FULL_SUMMARY_DIR,SMOKE_SUMMARY_DIR):
            if directory.exists(): shutil.rmtree(directory)
            directory.mkdir(parents=True,exist_ok=True)
        for name in ("manifest.json","smoke_manifest.json","feature_availability.csv",
                     "smoke_feature_availability.csv","stay_summary.parquet",
                     "smoke_stay_summary.parquet"):
            path = OUTPUT_DIR / name
            if path.exists(): path.unlink()
        print("Harmonização atualizada: derivados antigos removidos; cache Parquet preservado.")
    else:
        raise RuntimeError("Outputs incompatíveis. Confira coorte, features e caminhos antes de continuar.")
contract_path.write_text(json.dumps(contract,indent=2)+"\n")
sample_rank.to_csv(OUTPUT_DIR/"external_validation_subjects.csv",index=False)
cohort.to_parquet(OUTPUT_DIR/"external_validation_stays.parquet",index=False)

active_subjects = set(sampled_subjects[:SMOKE_SUBJECTS] if RUN_MODE=="smoke" else sampled_subjects)
active_cohort = cohort.loc[cohort.subject_id.isin(active_subjects)].copy()
print("Pacientes adultos candidatos:",len(all_subjects))
print("Pacientes selecionados:",cohort.subject_id.nunique())
print("Stays selecionados:",cohort.stay_id.nunique())
print("Pacientes ativos nesta execução:",active_cohort.subject_id.nunique())
display(cohort.head())

# %% [source cell 7]
BASE_FEATURES = [x for x in FEATURES if not x.endswith("_imputed") and not x.endswith("_time_since_last_measurement_min")]
assert len(BASE_FEATURES) == 68
SPEC = {x["output_feature"]:x for x in MAPPING["features"]}

alias_lookup = {}
for canonical, variants in MAPPING["lab_aliases"].items():
    for variant in variants:
        alias_lookup[variant.strip().lower()] = canonical

def normalize_labname(value):
    if pd.isna(value): return None
    text = re.sub(r"\s+"," ",str(value).strip().lower())
    if any(token in text for token in ("urine","urinary","urinario","urinária")) and "phos" in text:
        return None
    if text in alias_lookup: return alias_lookup[text]
    for alias, canonical in sorted(alias_lookup.items(),key=lambda x:len(x[0]),reverse=True):
        if len(alias)>=4 and alias in text: return canonical
    return None

def load_events(stay_ids):
    ids = pd.DataFrame({"stay_id":np.asarray(stay_ids,dtype="int64")})
    con.register("active_ids",ids)
    vital = con.execute(f"""SELECT v.* FROM read_parquet('{(CACHE_DIR/'vital.parquet').as_posix()}') v
                            JOIN active_ids i ON v.patientunitstayid=i.stay_id""").fetch_df()
    lab = con.execute(f"""SELECT l.* FROM read_parquet('{(CACHE_DIR/'lab.parquet').as_posix()}') l
                          JOIN active_ids i ON l.patientunitstayid=i.stay_id""").fetch_df()
    custom = con.execute(f"""SELECT l.* FROM read_parquet('{(CACHE_DIR/'customLab.parquet').as_posix()}') l
                             JOIN active_ids i ON l.patientunitstayid=i.stay_id""").fetch_df()
    infusion = con.execute(f"""SELECT d.* FROM read_parquet('{(CACHE_DIR/'infusion.parquet').as_posix()}') d
                               JOIN active_ids i ON d.patientunitstayid=i.stay_id""").fetch_df()
    con.unregister("active_ids")
    lab = pd.concat([lab,custom],ignore_index=True)
    if not lab.empty:
        lab["canonical"] = lab["labname"].map(normalize_labname)
        lab["labresult"] = pd.to_numeric(lab["labresult"],errors="coerce")
        lab["labresultoffset"] = pd.to_numeric(lab["labresultoffset"],errors="coerce")
        lab = lab.dropna(subset=["canonical","labresult","labresultoffset"])
    return vital,lab,infusion

def binned_series(frame, offset_col, value_col, grid, transform=None):
    if frame.empty: return pd.Series(np.nan,index=grid,dtype="float32")
    work = frame[[offset_col,value_col]].copy()
    work[offset_col] = pd.to_numeric(work[offset_col],errors="coerce")
    work[value_col] = pd.to_numeric(work[value_col],errors="coerce")
    work = work.dropna()
    work["bin"] = np.ceil(work[offset_col]/5.0).astype(int)*5
    work = work.loc[work["bin"].isin(grid)]
    values = work.groupby("bin")[value_col].last().reindex(grid).astype("float32")
    if transform == "ionized_calcium_auto_to_mmol_l":
        observed = values.dropna()
        if len(observed) and observed.median() > 2.5:
            values = values * 0.2495
    elif transform == "mixed_temperature_to_celsius":
        fahrenheit = values.gt(50)
        values.loc[fahrenheit] = (values.loc[fahrenheit] - 32.0) * 5.0 / 9.0
        values = values.where(values.between(25.0,45.0))
    elif transform == "serum_phosphate_range_filter":
        values = values.where(values.between(0.1,30.0))
    return values

def failure_and_target(grid, vasoactive, include_vaso=True):
    lactate = pd.to_numeric(grid["lab_50813"],errors="coerce").ffill(limit=48)
    mbp = pd.to_numeric(grid["mbp"],errors="coerce")
    vaso = pd.Series(vasoactive,index=grid.index,dtype="boolean")
    if not include_vaso: vaso[:] = False
    raw = pd.Series(pd.NA,index=grid.index,dtype="boolean")
    raw.loc[lactate.notna() & lactate.lt(2.0)] = False
    high = lactate.notna() & lactate.ge(2.0)
    low = mbp.notna() & mbp.le(65.0)
    normal = mbp.notna() & mbp.gt(65.0)
    raw.loc[high & (low | vaso.fillna(False))] = True
    raw.loc[high & normal & vaso.notna() & ~vaso.fillna(False)] = False
    known = raw.notna().astype("int8").rolling(3,min_periods=3).sum()
    positive = raw.fillna(False).astype("int8").rolling(3,min_periods=3).sum()
    failure = pd.Series(pd.NA,index=grid.index,dtype="boolean")
    failure.loc[known.eq(3)] = positive.loc[known.eq(3)].ge(2)
    onset_offsets = grid.loc[failure.fillna(False),"offset_min"]
    onset = float(onset_offsets.iloc[0]) if len(onset_offsets) else np.nan
    state_known = failure.notna()
    horizon_steps = 96
    future_all_known = (state_known.shift(-1).iloc[::-1].rolling(horizon_steps,min_periods=horizon_steps)
                        .sum().iloc[::-1].eq(horizon_steps))
    before = pd.Series(True,index=grid.index) if np.isnan(onset) else grid.offset_min.lt(onset)
    target_positive = pd.Series(False,index=grid.index) if np.isnan(onset) else (
        grid.offset_min.lt(onset) & (onset <= grid.offset_min + 480))
    eligible_pos = state_known & before & target_positive
    eligible_neg = state_known & before & ~target_positive & future_all_known
    eligible = eligible_pos | eligible_neg
    target = pd.Series(pd.NA,index=grid.index,dtype="Int8")
    target.loc[eligible_pos]=1; target.loc[eligible_neg]=0
    return target,eligible.astype(bool),onset

def build_stay(stay, vital, lab, infusion):
    end = int(np.floor(float(stay.unitdischargeoffset)/5.0)*5)
    grid_index = np.arange(0,end+1,5,dtype=int)
    if len(grid_index)<4: return pd.DataFrame(),None
    grid = pd.DataFrame({"offset_min":grid_index})
    grid["subject_id"] = str(stay.subject_id); grid["stay_id"] = int(stay.stay_id)
    sv = vital.loc[vital.patientunitstayid.eq(stay.stay_id)].copy()
    sl = lab.loc[lab.patientunitstayid.eq(stay.stay_id)].copy()
    observed={}; filled={}; imputed={}
    for feature in BASE_FEATURES:
        spec=SPEC[feature]
        if feature=="mbp": continue
        if spec.get("source_table")=="vitalPeriodic":
            raw=binned_series(sv,"observationoffset",spec["source_name"],grid_index,spec.get("transform"))
        elif spec.get("source_table")=="lab":
            raw=binned_series(sl.loc[sl.canonical.eq(spec["source_name"])],"labresultoffset","labresult",grid_index,spec.get("transform"))
        else:
            raw=pd.Series(np.nan,index=grid_index,dtype="float32")
        observed[feature]=raw
        propagated=raw.ffill()
        if spec.get("default") is not None: propagated=propagated.fillna(float(spec["default"]))
        filled[feature]=propagated.astype("float32")
        imputed[feature]=raw.isna().astype("int8")
    direct=binned_series(sv,"observationoffset","systemicmean",grid_index)
    derived=(observed["sbp"]+2*observed["dbp"])/3
    observed["mbp"]=direct.combine_first(derived)
    filled["mbp"]=(observed["mbp"].ffill().fillna(float(SPEC["mbp"]["default"]))).astype("float32")
    imputed["mbp"]=observed["mbp"].isna().astype("int8")
    feature_data={}
    for feature in BASE_FEATURES:
        feature_data[feature]=filled[feature].to_numpy()
        feature_data[f"{feature}_imputed"]=imputed[feature].to_numpy()
        measured_at=pd.Series(np.nan,index=grid_index,dtype="float64")
        measured=observed[feature].notna(); measured_at.loc[measured]=grid_index[measured]
        feature_data[f"{feature}_time_since_last_measurement_min"]=(pd.Series(grid_index,index=grid_index)-measured_at.ffill()).astype("float32").to_numpy()
    grid=pd.concat([grid,pd.DataFrame(feature_data)],axis=1)

    active=np.zeros(len(grid),dtype=bool)
    si=infusion.loc[infusion.patientunitstayid.eq(stay.stay_id)].copy()
    if not si.empty:
        names=si.drugname.astype(str).str.lower()
        include=names.str.contains("|".join(MAPPING["target"]["vasoactive_patterns"]),regex=True,na=False)
        exclude=names.str.contains("|".join(MAPPING["target"]["vasoactive_exclusions"]),regex=True,na=False)
        offsets=pd.to_numeric(si.loc[include & ~exclude,"infusionoffset"],errors="coerce").dropna()
        carry=int(MAPPING["target"]["vasoactive_proxy_minutes_after_record"])
        for offset in offsets:
            start=int(np.ceil(offset/5.0)*5); active |= (grid_index>=start)&(grid_index<=start+carry)

    target,eligible,onset=failure_and_target(grid,active,True)
    sensitivity,sensitivity_eligible,sensitivity_onset=failure_and_target(grid,active,False)
    grid[TARGET]=target; grid[SENSITIVITY_TARGET]=sensitivity
    grid["eligible"]=eligible; grid["sensitivity_eligible"]=sensitivity_eligible
    aligned=grid.iloc[:-1].copy()
    aligned["feature_time"]=pd.to_timedelta(grid.offset_min.iloc[:-1].to_numpy(),unit="m")
    aligned["label_time"]=pd.to_timedelta(grid.offset_min.iloc[1:].to_numpy(),unit="m")
    aligned[TARGET]=grid[TARGET].iloc[1:].to_numpy()
    aligned[SENSITIVITY_TARGET]=grid[SENSITIVITY_TARGET].iloc[1:].to_numpy()
    aligned["sensitivity_eligible"]=grid.sensitivity_eligible.iloc[1:].to_numpy()
    aligned["eligible"]=grid.eligible.iloc[1:].to_numpy()
    aligned=aligned.loc[aligned.eligible].copy()
    if not aligned.empty:
        aligned[TARGET]=aligned[TARGET].astype("int8")
        aligned=aligned[["subject_id","stay_id","feature_time","label_time",TARGET,SENSITIVITY_TARGET,"sensitivity_eligible",*FEATURES]]
    summary={"subject_id":str(stay.subject_id),"stay_id":int(stay.stay_id),"rows":len(aligned),
             "positive_rows":int(aligned[TARGET].sum()) if len(aligned) else 0,
             "first_onset_offset_min":onset,"map_only_first_onset_offset_min":sensitivity_onset,
             "has_selected_vital":bool(len(sv)),"has_selected_lab":bool(len(sl))}
    return aligned,summary

# %% [source cell 9]
batch_dir = SMOKE_BATCH_DIR if RUN_MODE=="smoke" else FULL_BATCH_DIR
summary_dir = SMOKE_SUMMARY_DIR if RUN_MODE=="smoke" else FULL_SUMMARY_DIR
ordered = active_cohort.sort_values(["sample_rank","stay_id"]).reset_index(drop=True)

for batch_no,start in enumerate(range(0,len(ordered),BATCH_STAYS)):
    output_path=batch_dir/f"batch_{batch_no:04d}.parquet"
    summary_path=summary_dir/f"batch_{batch_no:04d}.parquet"
    if output_path.exists() and summary_path.exists():
        print("Já concluído:",output_path.name); continue
    part=ordered.iloc[start:start+BATCH_STAYS]
    vital,lab,infusion=load_events(part.stay_id.to_numpy())
    frames=[]; summaries=[]
    for stay in part.itertuples(index=False):
        frame,summary=build_stay(stay,vital,lab,infusion)
        summaries.append(summary)
        if summary["has_selected_vital"] and summary["has_selected_lab"] and len(frame): frames.append(frame)
    if frames:
        pd.concat(frames,ignore_index=True).to_parquet(output_path,index=False,compression="zstd")
    else:
        pd.DataFrame(columns=["subject_id","stay_id","feature_time","label_time",TARGET,SENSITIVITY_TARGET,"sensitivity_eligible",*FEATURES]).to_parquet(output_path,index=False)
    pd.DataFrame(summaries).to_parquet(summary_path,index=False,compression="zstd")
    print("Batch",batch_no,"| stays examinados:",len(summaries),"| stays produzidos:",len(frames))

# %% [source cell 11]
files=sorted(batch_dir.glob("batch_*.parquet")); assert files
rows=positives=0; stay_ids=set(); subject_ids=set(); observed_counts=Counter()
mask_columns=[x for x in FEATURES if x.endswith("_imputed")]
for path in files:
    frame=pd.read_parquet(path)
    if frame.empty: continue
    assert list(frame.columns[-len(FEATURES):])==FEATURES
    assert (frame.label_time-frame.feature_time).eq(pd.Timedelta(minutes=5)).all()
    assert not frame.duplicated(["stay_id","label_time"]).any()
    rows+=len(frame); positives+=int(frame[TARGET].sum())
    stay_ids.update(frame.stay_id.astype(int).unique()); subject_ids.update(frame.subject_id.astype(str).unique())
    for column,count in frame[mask_columns].eq(0).sum().items(): observed_counts[column]+=int(count)
assert rows>0
availability=pd.DataFrame({"feature":mask_columns,"observed_rows":[observed_counts[x] for x in mask_columns],"total_rows":rows})
availability["observed_fraction"]=availability.observed_rows/rows
availability.to_csv(OUTPUT_DIR/("smoke_feature_availability.csv" if RUN_MODE=="smoke" else "feature_availability.csv"),index=False)
summaries=pd.concat([pd.read_parquet(x) for x in sorted(summary_dir.glob("batch_*.parquet"))],ignore_index=True)
summaries.to_parquet(OUTPUT_DIR/("smoke_stay_summary.parquet" if RUN_MODE=="smoke" else "stay_summary.parquet"),index=False)

manifest={"status":"smoke" if RUN_MODE=="smoke" else "completed","source":"eICU-CRD v2.0",
          "harmonization_revision":MAPPING["harmonization_revision"],
          "cohort_kind":"fixed_external_validation_cohort","cohort_subject_target":COHORT_SUBJECTS,
          "cohort_seed":COHORT_SEED,"cohort_sha256":cohort_hash,
          "selected_subjects":int(cohort.subject_id.nunique()),"selected_stays":int(cohort.stay_id.nunique()),
          "produced_subjects":len(subject_ids),"produced_stays":len(stay_ids),
          "selected_patient_days":float(cohort.stay_hours.sum()/24),"feature_count":len(FEATURES),
          "feature_sha256":FEATURE_HASH,"files":len(files),"rows":rows,"positive_rows":positives,
          "prevalence":float(positives/rows),"all_eligible_target_rows_retained":True,
          "external_data_used_for_training":False,"external_scaler_refit":False,
          "external_threshold_tuning":False,"strict_semantic_equivalence":False,
          "target_harmonization":"direct MAP with SBP/DBP fallback + 60-minute infusion-record proxy",
          "input_harmonization":"mixed temperature F-to-C conversion + serum phosphate urine/range exclusion + semantic separation of glucose, troponin and assay-specific laboratory features"}
if RUN_MODE=="full": assert 0<positives<rows
manifest_path=OUTPUT_DIR/("smoke_manifest.json" if RUN_MODE=="smoke" else "manifest.json")
manifest_path.write_text(json.dumps(manifest,indent=2)+"\n")
display(availability.sort_values("observed_fraction").head(30)); print(json.dumps(manifest,indent=2))
if RUN_MODE=="smoke": print("SMOKE concluído. Confira a auditoria e altere somente RUN_MODE para 'full'.")
