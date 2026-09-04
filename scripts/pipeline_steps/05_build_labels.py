"""Source exported from the reviewed development notebook.

This file contains code only; patient-level data and notebook outputs are not distributed.
"""
from __future__ import annotations

# %% [source cell 1]
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
import importlib.util
import json
import os
import re
import shutil
import time
import warnings

import pyarrow.parquet as pq

PROJECT_DIR = Path("/path/to/circulatory_failure_prediction")
CLEAN_DIR = Path(
    "/path/to/mimic_source_project/"
    "output_icu_clean_v2/raw_grid_all_icu_clean_v2"
)
LABEL_DIR = PROJECT_DIR / "outputs" / "step05_labels_v2"
SCRIPT_PATH = PROJECT_DIR / "scripts/step5_build_clean_labels.py"
LOG_DIR = PROJECT_DIR / "logs"
LOG_PATH = LOG_DIR / "step5_labels_eligibility_v2.log"
MANIFEST_PATH = LABEL_DIR / "manifest_step5_eligibility_v2.json"
EXPECTED_BATCHES = 285
EXPECTED_STAYS = 71054
RULE_VERSION = "observed_onset_sufficient_without_full_8h_followup_v2"

LABEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

if not CLEAN_DIR.exists():
    raise FileNotFoundError(CLEAN_DIR)
if not SCRIPT_PATH.exists():
    raise FileNotFoundError(SCRIPT_PATH)

print("ICU-clean:", CLEAN_DIR)
print("Labels:", LABEL_DIR)
print("Script produtivo:", SCRIPT_PATH)
print("Versão da regra:", RULE_VERSION)

# %% [source cell 3]
source = SCRIPT_PATH.read_text(encoding="utf-8")
original_source = source
backup_path = SCRIPT_PATH.with_name(
    "step5_build_clean_labels_BACKUP_before_eligibility_v2.py"
)

negative_old = '''    doses = intervals[VASOACTIVE_COLUMNS].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if doses.lt(0).any().any():
        raise RuntimeError(
            "Há dose/taxa farmacológica negativa."
        )

    intervals["interval_active"] = doses.gt(0).any(axis=1)
'''

negative_new = '''    doses = intervals[VASOACTIVE_COLUMNS].apply(
        pd.to_numeric,
        errors="coerce",
    )

    # Taxas negativas são registros inválidos e não constituem
    # evidência de suporte vasoativo ativo.
    negative_mask = doses.lt(0)
    negative_count = int(negative_mask.sum().sum())

    if negative_count > 0:
        negative_rows = int(negative_mask.any(axis=1).sum())
        print(
            "AVISO FARMACOLÓGICO: "
            f"{negative_count} valor(es) negativo(s) em "
            f"{negative_rows} intervalo(s); tratados como ausentes."
        )
        doses = doses.mask(negative_mask)

    intervals["interval_active"] = doses.gt(0).any(axis=1)
'''

if negative_old in source:
    source = source.replace(negative_old, negative_new, 1)
elif "negative_mask = doses.lt(0)" not in source:
    raise RuntimeError("Não reconheci a regra atual para taxas negativas.")

positive_old = '''    eligible_positive = (
        state_known
        & before_onset
        & future_has_full_steps
        & target_positive
    )
'''

positive_new = '''    # Para um target positivo, o onset observado dentro de
    # (t, t+8h] é suficiente; não se exige seguimento após o evento.
    eligible_positive = (
        state_known
        & before_onset
        & target_positive
    )
'''

if positive_old in source:
    source = source.replace(positive_old, positive_new, 1)
elif "não se exige seguimento após o evento" not in source:
    raise RuntimeError("Não reconheci a regra atual de elegibilidade positiva.")

summary_anchor = '''        "persistence_rule": "2_of_3_complete_instantaneous_criteria",
'''
summary_version = f'''        "persistence_rule": "2_of_3_complete_instantaneous_criteria",
        "target_eligibility_rule": "{RULE_VERSION}",
'''

if RULE_VERSION not in source:
    if summary_anchor not in source:
        raise RuntimeError("Não encontrei o ponto de versão do resumo.")
    source = source.replace(summary_anchor, summary_version, 1)

compile(source, str(SCRIPT_PATH), "exec")

if source != original_source:
    if not backup_path.exists():
        shutil.copy2(SCRIPT_PATH, backup_path)
        print("Backup criado:", backup_path)
    else:
        print("Backup existente preservado:", backup_path)
    SCRIPT_PATH.write_text(source, encoding="utf-8")
    print("Script produtivo atualizado.")
else:
    print("Script já continha todas as correções; nenhuma escrita necessária.")

final_source = SCRIPT_PATH.read_text(encoding="utf-8")
if RULE_VERSION not in final_source:
    raise RuntimeError("Versão v2 não registrada no script.")
if "& future_has_full_steps\n        & target_positive" in final_source:
    raise RuntimeError("A exigência incorreta de seguimento positivo permanece.")
if "negative_mask = doses.lt(0)" not in final_source:
    raise RuntimeError("Tratamento de taxas negativas ausente.")

print("STEP 5 v2 CORRIGIDO E COMPILADO.")

# %% [source cell 5]
def canonical_batches(directory):
    return sorted(
        path
        for path in directory.glob("*.parquet")
        if re.fullmatch(r"batch_\d{4}\.parquet", path.name)
    )

clean_files = canonical_batches(CLEAN_DIR)
expected_names = [f"batch_{index:04d}.parquet" for index in range(EXPECTED_BATCHES)]

if [path.name for path in clean_files] != expected_names:
    raise RuntimeError(
        f"Esperávamos batch_0000–batch_0284; encontramos {len(clean_files)} arquivos."
    )

def label_is_current(clean_file):
    output_file = LABEL_DIR / clean_file.name
    summary_file = LABEL_DIR / f"{clean_file.stem}_summary.json"
    if not output_file.exists() or not summary_file.exists():
        return False
    try:
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        summary.get("target_eligibility_rule") == RULE_VERSION
        and summary.get("vaso_method") == "direct_closed_start_end_intervals"
        and summary.get("persistence_rule") == "2_of_3_complete_instantaneous_criteria"
        and summary.get("vasopressor_saved_in_output") is False
        and summary.get("rows") is not None
        and output_file.stat().st_size > 0
    )

already_current = sum(label_is_current(path) for path in clean_files)
print("Batches ICU-clean:", len(clean_files))
print("Labels já produzidos pela regra v2:", already_current)
print("Labels que serão reconstruídos:", EXPECTED_BATCHES - already_current)

# %% [source cell 7]
mimic_database_url = os.environ.get("MIMIC_DATABASE_URL")
if not mimic_database_url:
    raise RuntimeError(
        "Configure MIMIC_DATABASE_URL securely outside the repository."
    )
processed_now = 0
skipped = 0
started_all = time.perf_counter()

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="DataFrameGroupBy.apply operated on the grouping columns",
)

try:
    spec = importlib.util.spec_from_file_location("step5_labels_v2", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Não foi possível carregar {SCRIPT_PATH}")
    step5 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(step5)

    for number, clean_file in enumerate(clean_files, start=1):
        if label_is_current(clean_file):
            skipped += 1
            print(f"[{number:03d}/285] {clean_file.name}: v2 já aprovado, pulando", flush=True)
            continue

        started_batch = time.perf_counter()
        print(f"[{number:03d}/285] {clean_file.name}: reconstruindo...", flush=True)
        try:
            with LOG_PATH.open("a", encoding="utf-8") as log:
                print(f"\n{'=' * 80}\nINÍCIO v2: {clean_file.name}\n{'=' * 80}", file=log)
                with redirect_stdout(log), redirect_stderr(log):
                    step5.process_batch(clean_file)

            if not label_is_current(clean_file):
                raise RuntimeError("Saída não foi reconhecida como regra v2.")

            processed_now += 1
            duration = time.perf_counter() - started_batch
            print(f"[{number:03d}/285] {clean_file.name}: APROVADO v2 em {duration:.1f}s", flush=True)

        except Exception as error:
            print(f"\nEXECUÇÃO INTERROMPIDA EM {clean_file.name}")
            print("Erro:", repr(error))
            print("Log:", LOG_PATH)
            raise
finally:
    pass

duration_minutes = (time.perf_counter() - started_all) / 60
print("\nRECONSTRUÇÃO STEP 5 v2 ENCERRADA")
print("Processados agora:", processed_now)
print("Já v2 e pulados:", skipped)
print("Duração:", round(duration_minutes, 2), "minutos")

# %% [source cell 9]
summaries = []
stays_seen = set()
expected_schema = None

for clean_file in clean_files:
    if not label_is_current(clean_file):
        raise RuntimeError(f"Label v2 não aprovado: {clean_file.name}")

    output_file = LABEL_DIR / clean_file.name
    summary_file = LABEL_DIR / f"{clean_file.stem}_summary.json"
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    parquet = pq.ParquetFile(output_file)
    schema = parquet.schema_arrow.names

    if expected_schema is None:
        expected_schema = schema
    elif schema != expected_schema:
        raise RuntimeError(f"Schema divergente: {clean_file.name}")

    if parquet.metadata.num_rows != int(summary["rows"]):
        raise RuntimeError(f"Linhas divergentes: {clean_file.name}")

    stay_ids = set(
        pq.read_table(output_file, columns=["stay_id"])["stay_id"].to_pylist()
    )
    overlap = stays_seen.intersection(stay_ids)
    if overlap:
        raise RuntimeError(f"Stays repetidos entre batches: {sorted(overlap)[:10]}")
    stays_seen.update(stay_ids)
    summaries.append(summary)

if len(stays_seen) != EXPECTED_STAYS:
    raise RuntimeError(f"Esperávamos {EXPECTED_STAYS} stays; encontramos {len(stays_seen)}.")

manifest = {
    "status": "approved",
    "step": 5,
    "version": RULE_VERSION,
    "batches": len(summaries),
    "stays": len(stays_seen),
    "rows": int(sum(item["rows"] for item in summaries)),
    "stays_with_first_onset_sum": int(sum(item["stays_with_first_onset"] for item in summaries)),
    "eligible_rows": int(sum(item["eligible_rows"] for item in summaries)),
    "target_positive_rows": int(sum(item["target_positive_rows"] for item in summaries)),
    "output_columns": expected_schema,
    "negative_requires_complete_known_8h_horizon": True,
    "positive_requires_complete_8h_followup_after_t": False,
    "log": str(LOG_PATH),
}

temporary = MANIFEST_PATH.with_suffix(".json.tmp")
temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
os.replace(temporary, MANIFEST_PATH)

print(json.dumps(manifest, indent=2, ensure_ascii=False))
print("\nSTEP 5 v2 CONCLUÍDO E APROVADO.")
