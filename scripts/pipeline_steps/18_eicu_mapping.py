"""Source exported from the reviewed development notebook.

This file contains code only; patient-level data and notebook outputs are not distributed.
"""

# %% [source cell 1]
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(os.environ.get(
    "CF_PROJECT", "/path/to/circulatory_failure_prediction"
)).resolve()
EICU_ROOT = Path(os.environ.get(
    "EICU_ROOT", "/path/to/eicu/data"
)).resolve()
OUTPUT_DIR = PROJECT / "outputs/step18_eicu_schema_mapping/eicu_v20"
FEATURE_FILE = PROJECT / "outputs/step07_aligned_xy_v2/feature_names.json"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TABLES = {
    "patient": EICU_ROOT / "patient.csv",
    "vitalPeriodic": EICU_ROOT / "vitalPeriodic.csv",
    "lab": EICU_ROOT / "lab.csv",
    "customLab": EICU_ROOT / "customLab.csv",
    "infusionDrug": EICU_ROOT / "infusionDrug.csv",
}
for name, path in TABLES.items():
    assert path.is_file(), f"Tabela eICU ausente ({name}): {path}"
assert FEATURE_FILE.is_file(), FEATURE_FILE

registry = json.loads(FEATURE_FILE.read_text())
FEATURES = list(registry["feature_names"])
FEATURE_HASH = hashlib.sha256("\n".join(FEATURES).encode()).hexdigest()
assert len(FEATURES) == 204

print("eICU:", EICU_ROOT)
print("Contrato MIMIC congelado:", len(FEATURES), "features")
print("Saída:", OUTPUT_DIR)

# %% [source cell 3]
EXPECTED_COLUMNS = {
    "patient": {"patientunitstayid", "uniquepid", "age", "unitdischargeoffset"},
    "vitalPeriodic": {
        "patientunitstayid", "observationoffset", "temperature", "sao2",
        "heartrate", "respiration", "systemicsystolic", "systemicdiastolic",
        "systemicmean",
    },
    "lab": {"patientunitstayid", "labresultoffset", "labname", "labresult"},
    "customLab": {
        "patientunitstayid", "labotheroffset", "labothername", "labotherresult"
    },
    "infusionDrug": {"patientunitstayid", "infusionoffset", "drugname"},
}

schema_rows = []
for table, path in TABLES.items():
    columns = list(pd.read_csv(path, nrows=0).columns)
    missing = sorted(EXPECTED_COLUMNS[table] - set(columns))
    assert not missing, f"Colunas ausentes em {table}: {missing}"
    schema_rows.append({
        "table": table,
        "path": str(path),
        "columns": len(columns),
        "required_columns_present": True,
    })

schema_inventory = pd.DataFrame(schema_rows)
schema_inventory.to_csv(OUTPUT_DIR / "schema_inventory.csv", index=False)
display(schema_inventory)

# %% [source cell 5]
# Defaults abaixo são os defaults do treinamento MIMIC, não os defaults do notebook eICU antigo.
VITAL_SPECS = {
    "heart_rate": ("vitalPeriodic", "heartrate", 70.0, "direct"),
    "sbp": ("vitalPeriodic", "systemicsystolic", 125.0, "direct"),
    "dbp": ("vitalPeriodic", "systemicdiastolic", 75.0, "direct"),
    "mbp": ("vitalPeriodic", "systemicmean", 90.0, "direct_with_sbp_dbp_fallback"),
    "resp_rate": ("vitalPeriodic", "respiration", 12.0, "direct"),
    "temperature": ("vitalPeriodic", "temperature", 37.0, "mixed_celsius_fahrenheit_harmonized"),
    "spo2": ("vitalPeriodic", "sao2", 98.0, "direct"),
    "glucose": ("lab", "bedside glucose", 5.0, "exact_semantic_match"),
}
VITAL_TRANSFORMS = {"temperature": "mixed_temperature_to_celsius"}

LAB_DEFAULTS = {
    50803:24.0,50805:1.5,50808:1.1,50809:90.0,50813:1.0,50852:5.5,
    50855:14.0,50856:10.0,50861:30.0,50862:4.3,50878:40.0,50883:0.3,
    50884:0.6,50885:1.0,50889:5.0,50908:0.8,50910:150.0,50912:1.0,
    50915:0.5,50924:120.0,50931:90.0,50935:100.0,50960:2.0,50963:100.0,
    50966:15.0,50967:15.0,50968:10.0,50969:80.0,50970:3.5,50971:4.5,
    50983:140.0,51002:0.01,51003:0.01,51006:15.0,51099:0.2,51196:0.5,
    51214:300.0,51222:14.0,51223:3.0,51265:200.0,51266:1.0,51274:12.0,
    51275:35.0,51464:0.0,51580:0.8,51623:300.0,51631:6.0,51638:45.0,
    51640:14.0,51966:0.0,52116:300.0,52117:300.0,52142:10.0,52144:1.0,
    52546:1.0,52551:0.5,52642:0.01,53085:4.3,53161:None,53180:None,
}

# Mapeamento baseado nos nomes/aliases já listados no eicu_preprocess.ipynb.
LAB_SOURCE = {
    50803:"bicarbonate",50805:"carboxyhemoglobin",50808:"ionized calcium",
    50809:"blood gas glucose",50813:"lactate",50861:"alt (sgpt)",
    50862:"albumin",50878:"ast (sgot)",50883:"direct bilirubin",
    50885:"total bilirubin",50908:"cpk-mb index",50910:"cpk",50912:"creatinine",
    50924:"ferritin",50931:"chemistry glucose",50960:"magnesium",50970:"phosphate",
    50971:"potassium",50983:"sodium",51002:"troponin i",51003:"troponin t",
    51006:"bun",51214:"fibrinogen",51222:"hgb",51265:"platelets x 1000",
    51274:"pt",51275:"ptt",51580:"cpk-mb",51638:"hct",51640:"hgb",
    52142:"mpv",52144:"methemoglobin",52546:"creatinine",52642:"troponin i",
    53085:"albumin",
}

LAB_ALIASES = {
    "albumin":["albumin","pre-albumin","prealbumin"],
    "alt (sgpt)":["alt (sgpt)","alanine aminotransferase","alt"],
    "ast (sgot)":["ast (sgot)","aspartate aminotransferase","ast"],
    "bicarbonate":["bicarbonate","co2","total co2","tco2","hco3"],
    "bun":["bun","urea nitrogen","blood urea nitrogen"],
    "carboxyhemoglobin":["carboxyhemoglobin","carboxy hb","cohb"],
    "cpk":["cpk","creatine kinase","ck"],
    "cpk-mb":["cpk-mb","ck-mb","cpk mb","ck mb"],
    "cpk-mb index":["cpk-mb index","ck-mb index","cpk mb index","ck mb index"],
    "creatinine":["creatinine","creatinine w gfr","creatinine w est gfr","creatinine w/ est gfr","creatinine w/ gfr"],
    "direct bilirubin":["direct bilirubin"],"ferritin":["ferritin"],
    "fibrinogen":["fibrinogen","fibrinogen, functional","functional fibrinogen"],
    "bedside glucose":["bedside glucose","point of care glucose","point-of-care glucose","poc glucose"],
    "chemistry glucose":["glucose","blood glucose"],
    "blood gas glucose":["blood gas glucose","glucose, blood gas","glucose - arterial","glucose, arterial","arterial glucose"],
    "hct":["hct","hematocrit"],"hgb":["hgb","hemoglobin"],
    "ionized calcium":["ionized calcium","free calcium","ica","i-ca"],
    "lactate":["lactate","lactic acid"],"magnesium":["magnesium"],
    "methemoglobin":["methemoglobin","met-hb","met hb"],
    "mpv":["mpv","mean platelet volume"],"phosphate":["phosphate","phosphorus","phos"],
    "platelets x 1000":["platelets x 1000","platelets","platelet"],
    "potassium":["potassium","k"],"pt":["pt","prothrombin time"],
    "ptt":["ptt","ptt ratio","partial thromboplastin time","aptt"],
    "sodium":["sodium","na"],"total bilirubin":["total bilirubin","bilirubin"],
    "troponin i":["troponin i","troponin - i","trop i"],
    "troponin t":["troponin t","troponin - t","trop t"],
}

feature_specs = []
for feature in FEATURES:
    if feature.endswith("_imputed") or feature.endswith("_time_since_last_measurement_min"):
        continue
    if feature in VITAL_SPECS:
        table, source, default, status = VITAL_SPECS[feature]
        feature_specs.append({"output_feature":feature,"role":"value","source_table":table,
                              "source_name":source,"default":default,"status":status,
                              "transform":VITAL_TRANSFORMS.get(feature)})
    else:
        itemid = int(feature.removeprefix("lab_"))
        source = LAB_SOURCE.get(itemid)
        status = "name_harmonized" if source else "unavailable_default_only"
        transform = {50808:"ionized_calcium_auto_to_mmol_l",
                     50970:"serum_phosphate_range_filter"}.get(itemid)
        feature_specs.append({"output_feature":feature,"role":"value",
                              "source_table":"lab" if source else None,
                              "source_name":source,"mimic_itemid":itemid,
                              "default":LAB_DEFAULTS[itemid],"status":status,
                              "transform":transform})

assert len(feature_specs) == 68
assert {x["output_feature"] for x in feature_specs} == {
    x for x in FEATURES if not x.endswith("_imputed") and not x.endswith("_time_since_last_measurement_min")
}
audit = pd.DataFrame(feature_specs)

# Uma fonte externa só pode preencher mais de uma feature quando os códigos MIMIC
# representam exatamente o mesmo analito. Exames de espécimes ou métodos diferentes
# permanecem separados; quando não há fonte equivalente, usa-se o default congelado.
allowed_shared_lab_sources = {
    "albumin": {"lab_50862", "lab_53085"},
    "creatinine": {"lab_50912", "lab_52546"},
    "hgb": {"lab_51222", "lab_51640"},
    "troponin i": {"lab_51002", "lab_52642"},
}
shared = (
    audit.loc[audit.source_table.eq("lab") & audit.source_name.notna()]
    .groupby("source_name").output_feature.agg(lambda x: set(x))
)
unexpected_shared = {
    source: outputs
    for source, outputs in shared.items()
    if len(outputs) > 1 and allowed_shared_lab_sources.get(source) != outputs
}
assert not unexpected_shared, f"Fontes laboratoriais indevidamente duplicadas: {unexpected_shared}"

audit.to_csv(OUTPUT_DIR / "feature_mapping_audit.csv", index=False)
display(audit.groupby(["source_table","status"], dropna=False).size().rename("n").reset_index())
display(pd.DataFrame([
    {"source_name": source, "equivalent_outputs": sorted(outputs)}
    for source, outputs in allowed_shared_lab_sources.items()
]))

# %% [source cell 7]
VASOACTIVE_PATTERNS = [
    "dopamine","epinephrine","adrenalin","norepinephrine","norepi","levophed",
    "phenylephrine","neosynephrine","vasopressin","dobutamine","milrinone",
    "isoproterenol","isuprel",
]
VASOACTIVE_EXCLUSIONS = [
    "not given","cancel","discontinued","flush","ophthalm","nasal","oral",
    "lidocaine.*epinephrine","bupivacaine.*epinephrine",
]

mapping = {
    "approved": True,
    "source": "eICU-CRD v2.0",
    "strict_semantic_equivalence": False,
    "harmonization_revision": 3,
    "mimic_feature_count": len(FEATURES),
    "mimic_feature_sha256": FEATURE_HASH,
    "tables": {name: str(path) for name, path in TABLES.items()},
    "cohort": {"patient_id":"uniquepid","stay_id":"patientunitstayid",
               "adult_age_min":18,"minimum_stay_minutes":60},
    "grid": {"minutes":5,"prediction_horizon_hours":8,"history_rows":96,
             "feature_to_label_offset_minutes":5},
    "features": feature_specs,
    "lab_aliases": LAB_ALIASES,
    "target": {
        "definition":"lactate >= 2 AND (MAP <= 65 OR vasoactive proxy)",
        "persistence":"at least 2 of 3 consecutive 5-minute states",
        "lactate_source_names":LAB_ALIASES["lactate"],
        "map_source":"vitalPeriodic.systemicmean; causal SBP/DBP fallback",
        "vasoactive_source":"infusionDrug.drugname",
        "vasoactive_patterns":VASOACTIVE_PATTERNS,
        "vasoactive_exclusions":VASOACTIVE_EXCLUSIONS,
        "vasoactive_proxy_minutes_after_record":60,
        "sensitivity_target":"MAP-only without vasoactive proxy",
    },
    "external_data_used_for_training": False,
    "external_scaler_refit": False,
    "external_threshold_tuning": False,
}
mapping_path = OUTPUT_DIR / "mapping_approved.json"
mapping_path.write_text(json.dumps(mapping, indent=2) + "\n")

report = f"""# Relatório Step 18 — eICU

- Caminho eICU: `{EICU_ROOT}`
- Tabelas obrigatórias: {len(TABLES)} conferidas.
- Features do contrato: {len(FEATURES)}.
- Features-base mapeadas por nome/sinal: {(audit.status != "unavailable_default_only").sum()} de 68.
- Target principal: lactato ≥ 2 E (MAP ≤ 65 OU proxy vasoativo), persistência 2/3.
- Proxy vasoativo: 60 minutos após cada registro em `infusionDrug`; não equivale a intervalo de infusão.
- Temperatura: valores >50 são convertidos de °F para °C; valores finais fora de 25–45 °C são descartados.
- Fosfato: exames urinários são excluídos e valores séricos fora de 0,1–30 mg/dL são descartados.
- Revisão 3: glicose de leito, química e gasometria foram separadas; troponina I/T foram separadas; códigos sem equivalência de método/espécime permanecem indisponíveis e usam o default congelado.
- Sensibilidade obrigatória: target MAP-only.
- Nenhum modelo, scaler ou threshold é treinado/ajustado na eICU.
"""
(OUTPUT_DIR / "REPORT.md").write_text(report)
print("APROVADO:", mapping_path)
print(report)
