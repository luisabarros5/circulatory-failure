"""Source exported from the reviewed development notebook.

This file contains code only; patient-level data and notebook outputs are not distributed.
"""
from __future__ import annotations

# %% [source cell 1]
import hashlib
import json
import os
import re
import shutil
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(os.environ.get(
    "CF_PROJECT",
    "/path/to/circulatory_failure_prediction",
)).resolve()
NWICU_ROOT = Path(os.environ.get(
    "NWICU_ROOT",
    "/path/to/nwicu/data",
)).resolve()
STEP15_ROOT = PROJECT / "outputs/step15_nwicu_schema_mapping"
OUTPUT_DIR = STEP15_ROOT / "nwicu_v010"
LEGACY_V2_DIR = STEP15_ROOT / "nwicu_v010_v2"

# Limpeza estritamente limitada aos artefatos antigos deste Step 15.
if LEGACY_V2_DIR.exists():
    shutil.rmtree(LEGACY_V2_DIR)
for stale_name in (
    "schema_inventory.csv",
    "candidate_concepts.csv",
    "nwicu_mapping_template.json",
    "file_inventory.csv",
):
    stale_path = OUTPUT_DIR / stale_name
    if stale_path.exists():
        stale_path.unlink()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_FILE = PROJECT / "outputs/step07_aligned_xy_v2/feature_names.json"
NW_D_LABS = NWICU_ROOT / "nw_hosp/d_labitems.csv"
NW_D_ITEMS = NWICU_ROOT / "nw_icu/d_items.csv"

VITAL_DEFAULTS = {'heart_rate': 70.0, 'sbp': 125.0, 'dbp': 75.0, 'mbp': 90.0, 'resp_rate': 12.0, 'temperature': 37.0, 'spo2': 98.0, 'glucose': 5.0}
LAB_DEFAULTS = {'lab_50803': 24.0, 'lab_50805': 1.5, 'lab_50808': 1.1, 'lab_50809': 90.0, 'lab_50813': 1.0, 'lab_50852': 5.5, 'lab_50855': 14.0, 'lab_50856': 10.0, 'lab_50861': 30.0, 'lab_50862': 4.3, 'lab_50878': 40.0, 'lab_50883': 0.3, 'lab_50884': 0.6, 'lab_50885': 1.0, 'lab_50889': 5.0, 'lab_50908': 0.8, 'lab_50910': 150.0, 'lab_50912': 1.0, 'lab_50915': 0.5, 'lab_50924': 120.0, 'lab_50931': 90.0, 'lab_50935': 100.0, 'lab_50960': 2.0, 'lab_50963': 100.0, 'lab_50966': 15.0, 'lab_50967': 15.0, 'lab_50968': 10.0, 'lab_50969': 80.0, 'lab_50970': 3.5, 'lab_50971': 4.5, 'lab_50983': 140.0, 'lab_51002': 0.01, 'lab_51003': 0.01, 'lab_51006': 15.0, 'lab_51099': 0.2, 'lab_51196': 0.5, 'lab_51214': 300.0, 'lab_51222': 14.0, 'lab_51223': 3.0, 'lab_51265': 200.0, 'lab_51266': 1.0, 'lab_51274': 12.0, 'lab_51275': 35.0, 'lab_51464': 0.0, 'lab_51580': 0.8, 'lab_51623': 300.0, 'lab_51631': 6.0, 'lab_51638': 45.0, 'lab_51640': 14.0, 'lab_51966': 0.0, 'lab_52116': 300.0, 'lab_52117': 300.0, 'lab_52142': 10.0, 'lab_52144': 1.0, 'lab_52546': 1.0, 'lab_52551': 0.5, 'lab_52642': 0.01, 'lab_53085': 4.3, 'lab_53161': None, 'lab_53180': None}

assert NWICU_ROOT.exists(), NWICU_ROOT
assert NW_D_LABS.exists(), NW_D_LABS
assert NW_D_ITEMS.exists(), NW_D_ITEMS
assert FEATURE_FILE.exists(), FEATURE_FILE

registry = json.loads(FEATURE_FILE.read_text())
FEATURES = list(registry["feature_names"])
FEATURE_HASH = hashlib.sha256("\n".join(FEATURES).encode()).hexdigest()
assert len(FEATURES) == 204, len(FEATURES)

print("NWICU:", NWICU_ROOT)
print("Saída:", OUTPUT_DIR)
print("Features congeladas:", len(FEATURES))

# %% [source cell 3]
nw_labs = pd.read_csv(NW_D_LABS)
nw_items = pd.read_csv(NW_D_ITEMS)
nw_chart = nw_items.loc[nw_items.linksto.eq("chartevents")].copy()

display(nw_chart)
display(nw_labs.head(30))

assert 100031 in set(nw_labs.itemid), "Lactato NWICU 100031 não encontrado."
assert nw_labs.loc[nw_labs.itemid.eq(100031), "label"].str.fullmatch("Lactate", case=False).all()

print("d_labitems:", len(nw_labs), "itens")
print("d_items/chartevents:", len(nw_chart), "itens")
print("Lactato laboratorial NWICU: itemid 100031")

# %% [source cell 5]
import psycopg2

source_lab_ids = sorted(
    int(name.removeprefix("lab_"))
    for name in LAB_DEFAULTS
)

mimic_database_url = os.environ.get("MIMIC_DATABASE_URL")
if not mimic_database_url:
    raise RuntimeError(
        "Configure MIMIC_DATABASE_URL securely outside the repository."
    )
connection = psycopg2.connect(mimic_database_url)
try:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT itemid, label, fluid, category
            FROM mimiciv_hosp.d_labitems
            WHERE itemid = ANY(%s)
            ORDER BY itemid
            """,
            (source_lab_ids,),
        )
        mimic_labs = pd.DataFrame(
            cursor.fetchall(),
            columns=[column[0] for column in cursor.description],
        )
finally:
    connection.close()

mimic_labs["dictionary_available"] = True
returned_ids = set(mimic_labs.itemid.astype(int))
missing_source_ids = sorted(set(source_lab_ids) - returned_ids)

if missing_source_ids:
    missing_rows = pd.DataFrame({
        "itemid": missing_source_ids,
        "label": [f"Unresolved MIMIC itemid {itemid}" for itemid in missing_source_ids],
        "fluid": [None] * len(missing_source_ids),
        "category": [None] * len(missing_source_ids),
        "dictionary_available": [False] * len(missing_source_ids),
    })
    mimic_labs = (
        pd.concat([mimic_labs, missing_rows], ignore_index=True)
        .sort_values("itemid")
        .reset_index(drop=True)
    )
    print(
        "Itemids ausentes no dicionário MIMIC; serão marcados como indisponíveis: ",
        missing_source_ids,
    )
else:
    print("Todos os itemids esperados existem no dicionário MIMIC.")

assert set(mimic_labs.itemid.astype(int)) == set(source_lab_ids)
display(mimic_labs)

# %% [source cell 7]
def normalize_text(value):
    value = "" if pd.isna(value) else str(value)
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


nw_labs = nw_labs.assign(
    norm_label=nw_labs.label.map(normalize_text),
    norm_fluid=nw_labs.fluid.map(normalize_text),
    norm_category=nw_labs.category.map(normalize_text),
)

crosswalk_rows = []
for row in mimic_labs.itertuples(index=False):
    if not row.dictionary_available:
        candidates = nw_labs.head(0)
        status = "SOURCE_DICTIONARY_MISSING"
        score = np.nan
    else:
        exact = nw_labs.loc[nw_labs.norm_label.eq(normalize_text(row.label))].copy()
        same_fluid = exact.loc[exact.norm_fluid.eq(normalize_text(row.fluid))]
        if not same_fluid.empty:
            exact = same_fluid
        same_category = exact.loc[exact.norm_category.eq(normalize_text(row.category))]
        if not same_category.empty:
            exact = same_category

    if row.dictionary_available and not exact.empty:
        status = "DIRECT"
        candidates = exact
        score = 1.0
    elif row.dictionary_available:
        scores = nw_labs.norm_label.map(
            lambda x: SequenceMatcher(None, normalize_text(row.label), x).ratio()
        )
        best = float(scores.max())
        candidates = nw_labs.loc[scores.eq(best)].head(3)
        status = "UNAVAILABLE_DEFAULT_ONLY"
        score = best

    crosswalk_rows.append({
        "mimic_itemid": int(row.itemid),
        "mimic_label": row.label,
        "mimic_fluid": row.fluid,
        "mimic_category": row.category,
        "status": status,
        "nwicu_itemids": [int(x) for x in candidates.itemid] if status == "DIRECT" else [],
        "nwicu_labels": " | ".join(candidates.label.astype(str)),
        "source_dictionary_available": bool(row.dictionary_available),
        "best_label_similarity": score,
    })

crosswalk = pd.DataFrame(crosswalk_rows)
crosswalk.to_csv(OUTPUT_DIR / "lab_crosswalk_mimic_to_nwicu.csv", index=False)
display(crosswalk)
print("Diretos:", int(crosswalk.status.eq("DIRECT").sum()))
print("Default-only:", int(crosswalk.status.eq("UNAVAILABLE_DEFAULT_ONLY").sum()))

# %% [source cell 9]
VITAL_SPECS = {
    "heart_rate": dict(source_table="chartevents", itemids=[320045], unit_rule="identity", default=70.0),
    "sbp": dict(source_table="chartevents", itemids=[320050, 320179], unit_rule="identity", default=125.0),
    "dbp": dict(source_table="chartevents", itemids=[320051, 320180], unit_rule="identity", default=75.0),
    "mbp": dict(source_table=None, itemids=[], unit_rule="derived_sbp_2dbp_over_3", default=90.0),
    "resp_rate": dict(source_table="chartevents", itemids=[320210], unit_rule="identity", default=12.0),
    "temperature": dict(source_table="chartevents", itemids=[323761], unit_rule="fahrenheit_to_celsius", default=37.0),
    "spo2": dict(source_table="chartevents", itemids=[320277], unit_rule="identity", default=98.0),
    "glucose": dict(source_table="labevents", itemids=[100001, 100045], unit_rule="identity", default=5.0),
}

crosswalk_by_id = crosswalk.set_index("mimic_itemid")
feature_specs = []
for feature in FEATURES:
    if feature.endswith("_time_since_last_measurement_min"):
        base = feature[:-len("_time_since_last_measurement_min")]
        feature_specs.append({"output_feature": feature, "base_feature": base, "role": "time_since", "status": "DERIVED"})
        continue
    if feature.endswith("_imputed"):
        base = feature[:-len("_imputed")]
        feature_specs.append({"output_feature": feature, "base_feature": base, "role": "imputed", "status": "DERIVED"})
        continue

    if feature in VITAL_SPECS:
        spec = VITAL_SPECS[feature]
        status = "DERIVED_PROXY" if feature == "mbp" else "DIRECT_PROXY_SOURCE" if feature == "glucose" else "DIRECT"
        feature_specs.append({
            "output_feature": feature, "base_feature": feature, "role": "value", "status": status,
            "source_table": spec["source_table"], "itemids": spec["itemids"],
            "itemid_column": "itemid", "time_column": "charttime", "value_column": "valuenum",
            "unit_column": "valueuom", "unit_rule": spec["unit_rule"], "default": spec["default"],
        })
        continue

    mimic_id = int(feature.removeprefix("lab_"))
    row = crosswalk_by_id.loc[mimic_id]
    feature_specs.append({
        "output_feature": feature, "base_feature": feature, "role": "value", "status": row.status,
        "source_table": "labevents" if row.status == "DIRECT" else None,
        "itemids": list(row.nwicu_itemids), "itemid_column": "itemid",
        "time_column": "charttime", "value_column": "valuenum", "unit_column": "valueuom",
        "unit_rule": "identity", "default": LAB_DEFAULTS.get(feature),
        "mimic_label": row.mimic_label, "nwicu_labels": row.nwicu_labels,
    })

assert [x["output_feature"] for x in feature_specs] == FEATURES
feature_audit = pd.DataFrame(feature_specs)
feature_audit.to_csv(OUTPUT_DIR / "feature_mapping_audit.csv", index=False)
display(feature_audit.loc[feature_audit.role.eq("value")])

# %% [source cell 11]
VASOACTIVE_INFUSION_PATTERNS = [
    r"^NOREPINEPHRINE .* IV SOLN$",
    r"^PHENYLEPHRINE HCL IN 0\.9% NACL 40 MG/250 ML .* IV SOLN$",
    r"^VASOPRESSIN IN 0\.9 % SOD CHLOR 100 UNIT/100 ML .* IV SOLN$",
    r"^DOPAMINE IN 5 % DEXTROSE .* IV SOLN$",
    r"^DOBUTAMINE IN D5W .* IV SOLP$",
    r"^MILRINONE IN 5 % DEXTROSE .* IV PGBK$",
    r"^EPINEPHRINE HCL IN 5% DEXTROSE .* IV SOLN$",
    r"^EPINEPHRINE HCL 1 MG/ NS 500 ML BAG BUNDLE.*$",
]

lactate_feature = next(x for x in feature_specs if x["output_feature"] == "lab_50813")
assert lactate_feature["status"] == "DIRECT"
assert 100031 in lactate_feature["itemids"]

mapping = {
    "approved": True,
    "approval_scope": "HARMONIZED_EXTERNAL_VALIDATION_WITH_DOCUMENTED_TARGET_PROXIES",
    "strict_semantic_equivalence": False,
    "source": "NWICU v0.1.0",
    "nwicu_root": str(NWICU_ROOT),
    "mimic_feature_count": len(FEATURES),
    "mimic_feature_sha256": FEATURE_HASH,
    "tables": {
        "patients": str(NWICU_ROOT / "nw_hosp/patients.csv"),
        "admissions": str(NWICU_ROOT / "nw_hosp/admissions.csv"),
        "icustays": str(NWICU_ROOT / "nw_icu/icustays.csv"),
        "d_items": str(NW_D_ITEMS),
        "chartevents": str(NWICU_ROOT / "nw_icu/chartevents.csv"),
        "d_labitems": str(NW_D_LABS),
        "labevents": str(NWICU_ROOT / "nw_hosp/labevents.csv"),
        "prescriptions": str(NWICU_ROOT / "nw_hosp/prescriptions.csv"),
        "emar": str(NWICU_ROOT / "nw_hosp/emar.csv"),
    },
    "cohort": {
        "adult_age_min": 18, "minimum_stay_hours": 1,
        "subject_column": "subject_id", "admission_column": "hadm_id", "stay_column": "stay_id",
        "icu_intime_column": "intime", "icu_outtime_column": "outtime",
    },
    "grid": {"minutes": 5, "history_rows": 96, "prediction_horizon_hours": 8},
    "target": {
        "definition": "lactate>=2 AND (derived_MAP<=65 OR vasoactive_prescription_proxy)",
        "persistence_rule": "2_of_3_complete_instantaneous_criteria",
        "lactate_forward_fill_hours": 4,
        "lactate": {**lactate_feature, "status": "DIRECT"},
        "map": {"status": "DERIVED_PROXY", "derived_feature": "mbp", "formula": "(sbp + 2*dbp)/3"},
        "vasoactive": {
            "status": "PRESCRIPTION_INTERVAL_PROXY", "source_table": "prescriptions",
            "name_column": "drug", "start_time_column": "starttime", "end_time_column": "stoptime",
            "route_column": "route", "allowed_routes": ["Intravenous"],
            "name_patterns": VASOACTIVE_INFUSION_PATTERNS,
            "exclude_invalid_intervals": True,
            "limitation": "Prescription interval is not identical to measured positive infusion rate.",
        },
    },
    "features": feature_specs,
}

mapping_path = OUTPUT_DIR / "mapping_approved.json"
mapping_path.write_text(json.dumps(mapping, indent=2, ensure_ascii=False))
print("Mapeamento harmonizado salvo:", mapping_path)
print("ATENÇÃO: MAP e vasoativo são proxies documentados.")

# %% [source cell 13]
counts = feature_audit.loc[feature_audit.role.eq("value"), "status"].value_counts().to_dict()
report = f"""# NWICU mapping report

- Features congeladas: {len(FEATURES)}
- Hash: `{FEATURE_HASH}`
- Mapeamento por status: {counts}
- Lactato: NWICU itemid 100031, direto.
- MAP: derivada por `(PAS + 2×PAD)/3`; proxy, pois não há MAP direta em `d_items`.
- Vasoativo: intervalos de prescrições de formulações de infusão contínua; proxy de atividade.
- Bolus, anestésicos locais com epinefrina e vias não intravenosas: excluídos.
- Features laboratoriais ausentes: default original do treino + `_imputed=1`.
- Classificação: validação externa harmonizada com proxies documentados, não equivalência semântica estrita.
"""
(OUTPUT_DIR / "REPORT.md").write_text(report)
print(report)
