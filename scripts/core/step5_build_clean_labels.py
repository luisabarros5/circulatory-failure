#!/usr/bin/env python3

from pathlib import Path
import argparse
import json
import os
import sys

import psycopg2

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


CLEAN_DIR = Path(
    "/path/to/mimic_source_project/"
    "output_icu_clean_v2/raw_grid_all_icu_clean_v2"
)

LEGACY_DIR = Path(
    "/path/to/mimic_source_project/"
    "output/features_filtradas"
)

OUTPUT_DIR = Path(
    "/path/to/circulatory_failure_prediction/"
    "labels_icu_clean_v2"
)

KEY = ["stay_id", "charttime"]

OUTPUT_COLUMNS = [
    "stay_id",
    "charttime",
    "failure_state_new_at_t",
    "target_first_onset_0_8h",
    "eligible_for_training",
]

GRID_MINUTES = 5
LACTATE_FORWARD_FILL_HOURS = 4
HORIZON_HOURS = 8
PERSISTENCE_WINDOW = 3
PERSISTENCE_MIN_POSITIVE = 2

VASOACTIVE_COLUMNS = [
    "dopamine",
    "epinephrine",
    "norepinephrine",
    "phenylephrine",
    "vasopressin",
    "dobutamine",
    "milrinone",
]

MIMIC_DATABASE_URL = os.getenv("MIMIC_DATABASE_URL")



def read_vasoactive_intervals(stay_ids):
    """
    Consulta os intervalos farmacológicos originais.

    Um intervalo é considerado ativo somente quando pelo menos uma das
    sete taxas/doses é estritamente maior que zero.
    """
    query = """
        SELECT
            stay_id,
            starttime,
            endtime,
            dopamine,
            epinephrine,
            norepinephrine,
            phenylephrine,
            vasopressin,
            dobutamine,
            milrinone
        FROM mimiciv_derived.vasoactive_agent
        WHERE stay_id = ANY(%s)
        ORDER BY stay_id, starttime, endtime
    """

    if not MIMIC_DATABASE_URL:
        raise RuntimeError(
            "Configure MIMIC_DATABASE_URL securely outside the repository."
        )
    connection = psycopg2.connect(MIMIC_DATABASE_URL)

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                query,
                ([int(stay_id) for stay_id in stay_ids],),
            )
            rows = cursor.fetchall()
            columns = [
                description.name
                for description in cursor.description
            ]
    finally:
        connection.close()

    intervals = pd.DataFrame(rows, columns=columns)

    if intervals.empty:
        intervals = pd.DataFrame(
            columns=[
                "stay_id",
                "starttime",
                "endtime",
                *VASOACTIVE_COLUMNS,
                "interval_active",
            ]
        )
        return intervals

    intervals["stay_id"] = pd.to_numeric(
        intervals["stay_id"],
        errors="raise",
    ).astype("int64")

    intervals["starttime"] = pd.to_datetime(
        intervals["starttime"],
        errors="raise",
    )
    intervals["endtime"] = pd.to_datetime(
        intervals["endtime"],
        errors="raise",
    )

    if intervals[["starttime", "endtime"]].isna().any().any():
        raise RuntimeError(
            "Há intervalo farmacológico sem starttime ou endtime."
        )

    invalid_interval = (
        intervals["endtime"] < intervals["starttime"]
    )

    if invalid_interval.any():
        raise RuntimeError(
            f"Há {int(invalid_interval.sum())} intervalos com "
            "endtime anterior ao starttime."
        )

    doses = intervals[VASOACTIVE_COLUMNS].apply(
        pd.to_numeric,
        errors="coerce",
    )

    # Taxas negativas são registros farmacológicos inválidos:
    # não constituem evidência de suporte vasoativo ativo.
    negative_mask = doses.lt(0)
    negative_count = int(negative_mask.sum().sum())

    if negative_count > 0:
        negative_rows = int(negative_mask.any(axis=1).sum())

        print(
            "AVISO FARMACOLÓGICO: "
            f"{negative_count} valor(es) negativo(s) em "
            f"{negative_rows} intervalo(s); "
            "tratados como ausentes para definir atividade."
        )

        doses = doses.mask(negative_mask)

    # Um intervalo é ativo se pelo menos uma droga possuir
    # dose/taxa estritamente positiva.
    intervals["interval_active"] = doses.gt(0).any(axis=1)

    return intervals


def mark_vasoactive_directly(clean, intervals):
    """
    Marca suporte vasoativo/inotrópico diretamente na grade ICU-clean.

    Regra inclusiva:
        starttime <= charttime <= endtime

    Não utiliza merge_asof, forward fill, backward fill,
    arredondamento ou valores futuros.
    """
    data = clean.sort_values(KEY).copy()

    data["vasopressor_ativo_aux"] = pd.Series(
        False,
        index=data.index,
        dtype="boolean",
    )

    active_intervals = intervals.loc[
        intervals["interval_active"]
    ].copy()

    intervals_by_stay = {
        int(stay_id): stay_intervals
        for stay_id, stay_intervals
        in active_intervals.groupby("stay_id", sort=False)
    }

    for stay_id, indices in data.groupby(
        "stay_id",
        sort=False,
    ).groups.items():

        stay_intervals = intervals_by_stay.get(int(stay_id))

        if stay_intervals is None:
            continue

        times = data.loc[indices, "charttime"].to_numpy()
        active = np.zeros(len(indices), dtype=bool)

        for interval in stay_intervals.itertuples(index=False):
            active |= (
                (times >= np.datetime64(interval.starttime))
                & (times <= np.datetime64(interval.endtime))
            )

        data.loc[indices, "vasopressor_ativo_aux"] = active

    return data




def calculate_raw_failure(group):
    group = group.sort_values("charttime").copy()

    # Grade regular de 5 min: 4 horas correspondem a 48 timesteps.
    ffill_limit = int(
        LACTATE_FORWARD_FILL_HOURS * 60 / GRID_MINUTES
    )

    lactate = pd.to_numeric(group["lab_50813"], errors="coerce")
    lactate = lactate.ffill(limit=ffill_limit)

    mbp = pd.to_numeric(group["mbp"], errors="coerce")
    vaso = group["vasopressor_ativo_aux"].astype("boolean")

    raw = pd.Series(pd.NA, index=group.index, dtype="boolean")

    # Lactato conhecido abaixo de 2 torna a regra negativa,
    # independentemente de MAP/vasoativo.
    raw.loc[lactate.notna() & (lactate < 2)] = False

    lactate_high = lactate.notna() & (lactate >= 2)
    map_low = mbp.notna() & (mbp <= 65)
    map_normal = mbp.notna() & (mbp > 65)
    vaso_true = vaso.fillna(False)
    vaso_false_known = vaso.notna() & (~vaso.fillna(False))

    raw.loc[lactate_high & (map_low | vaso_true)] = True
    raw.loc[lactate_high & map_normal & vaso_false_known] = False

    group["_lactate_for_label"] = lactate
    group["_failure_raw"] = raw

    return group



def apply_persistence(group):
    """
    Confirma falência quando o critério instantâneo completo for
    positivo em pelo menos 2 de 3 timesteps consecutivos.

    A janela contém somente o instante atual e os dois anteriores.
    Nenhuma informação futura é utilizada.
    """
    group = group.sort_values("charttime").copy()

    raw = group["_failure_raw"].astype("boolean")

    known_count = (
        raw.notna()
        .astype("int8")
        .rolling(
            PERSISTENCE_WINDOW,
            min_periods=PERSISTENCE_WINDOW,
        )
        .sum()
    )

    positive_count = (
        raw.fillna(False)
        .astype("int8")
        .rolling(
            PERSISTENCE_WINDOW,
            min_periods=PERSISTENCE_WINDOW,
        )
        .sum()
    )

    complete_window = (
        known_count == PERSISTENCE_WINDOW
    )

    failure = pd.Series(
        pd.NA,
        index=group.index,
        dtype="boolean",
    )

    failure.loc[complete_window] = (
        positive_count.loc[complete_window]
        >= PERSISTENCE_MIN_POSITIVE
    )

    group["failure_state_new_at_t"] = failure

    return group




def build_target(group):
    group = group.sort_values("charttime").copy()

    horizon_steps = int(
        HORIZON_HOURS * 60 / GRID_MINUTES
    )

    if horizon_steps <= 0:
        raise RuntimeError(
            f"Horizonte inválido em timesteps: {horizon_steps}"
        )

    positive_times = group.loc[
        group["failure_state_new_at_t"].fillna(False),
        "charttime",
    ]

    first_onset = (
        positive_times.iloc[0]
        if len(positive_times)
        else pd.NaT
    )

    horizon_end = group["charttime"] + pd.Timedelta(
        hours=HORIZON_HOURS
    )

    state_known = group["failure_state_new_at_t"].notna()

    # Qualquer linha elegível precisa ter 96 timesteps futuros na grade.
    # Para positivos, esses estados podem conter indeterminações desde que
    # o primeiro onset conhecido esteja dentro do horizonte.
    future_has_full_steps = pd.Series(
        np.arange(len(group)) + horizon_steps < len(group),
        index=group.index,
        dtype=bool,
    )

    # Para um target negativo, exige que os próximos 96 timesteps
    # (t, t+8h] sejam todos conhecidos. O estado atual é excluído.
    # A inversão permite uma janela prospectiva sem incluir qualquer
    # informação posterior ao horizonte de oito horas.
    future_all_known = (
        state_known.shift(-1)
        .iloc[::-1]
        .rolling(
            horizon_steps,
            min_periods=horizon_steps,
        )
        .sum()
        .iloc[::-1]
        .eq(horizon_steps)
    )

    if pd.isna(first_onset):
        before_onset = pd.Series(
            True, index=group.index, dtype=bool
        )
        target_positive = pd.Series(
            False, index=group.index, dtype=bool
        )
    else:
        before_onset = group["charttime"] < first_onset
        target_positive = (
            (first_onset > group["charttime"])
            & (first_onset <= horizon_end)
        )

    # Um onset observado no horizonte mantém o target positivo mesmo
    # quando o restante das oito horas não está completamente observado.
    # Um target negativo só é válido se todo o horizonte futuro é conhecido.
    # Para um target positivo, o onset observado dentro de
    # (t, t+8h] é suficiente; não se exige seguimento após o evento.
    eligible_positive = (
        state_known
        & before_onset
        & target_positive
    )
    eligible_negative = (
        state_known
        & before_onset
        & (~target_positive)
        & future_all_known
    )
    eligible = eligible_positive | eligible_negative

    target = pd.Series(
        pd.NA, index=group.index, dtype="Int8"
    )
    target.loc[eligible_positive] = 1
    target.loc[eligible_negative] = 0

    group["target_first_onset_0_8h"] = target
    group["eligible_for_training"] = eligible.astype(bool)
    group["_first_onset"] = first_onset

    return group


def process_batch(clean_file):
    print(f"\nProcessando: {clean_file}")

    required_clean = {
        "stay_id",
        "charttime",
        "mbp",
        "lab_50813",
    }
    schema = set(pq.ParquetFile(clean_file).schema.names)
    missing = sorted(required_clean - schema)

    if missing:
        raise RuntimeError(
            f"Colunas ausentes na grade limpa: {missing}"
        )

    clean = pq.read_table(
        clean_file,
        columns=["stay_id", "charttime", "mbp", "lab_50813"],
    ).to_pandas()

    clean["charttime"] = pd.to_datetime(
        clean["charttime"], errors="coerce"
    )

    if clean[KEY].isna().any().any():
        raise RuntimeError("Há stay_id/charttime nulo na grade limpa.")

    duplicate_clean = int(clean.duplicated(KEY).sum())
    if duplicate_clean:
        raise RuntimeError(
            f"Há {duplicate_clean} chaves duplicadas na grade limpa."
        )

    clean = clean.sort_values(KEY).reset_index(drop=True)
    stay_ids = clean["stay_id"].drop_duplicates().tolist()

    vaso_intervals = read_vasoactive_intervals(stay_ids)
    data = mark_vasoactive_directly(clean, vaso_intervals)

    data = (
        data.groupby("stay_id", group_keys=False, sort=False)
        .apply(calculate_raw_failure)
        .reset_index(drop=True)
    )

    data = (
        data.groupby("stay_id", group_keys=False, sort=False)
        .apply(apply_persistence)
        .reset_index(drop=True)
    )

    data = (
        data.groupby("stay_id", group_keys=False, sort=False)
        .apply(build_target)
        .reset_index(drop=True)
    )

    output = data[OUTPUT_COLUMNS].copy()

    # Travas contra leakage e contra colunas inesperadas.
    forbidden = [
        c for c in output.columns
        if any(
            term in c.lower()
            for term in [
                "vaso",
                "norepine",
                "epinephrine",
                "phenylephrine",
                "dopamine",
                "dobutamine",
                "milrinone",
            ]
        )
    ]

    if forbidden:
        raise RuntimeError(
            f"Coluna proibida apareceu na saída: {forbidden}"
        )

    if list(output.columns) != OUTPUT_COLUMNS:
        raise RuntimeError(
            f"Schema final inesperado: {list(output.columns)}"
        )

    if len(output) != len(clean):
        raise RuntimeError(
            f"Número de linhas mudou: {len(clean)} -> {len(output)}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / clean_file.name

    table = pa.Table.from_pandas(
        output,
        preserve_index=False,
    )
    pq.write_table(
        table,
        output_file,
        compression="zstd",
    )

    active_intervals = vaso_intervals.loc[
        vaso_intervals["interval_active"]
    ]
    active_grid_rows = int(
        data["vasopressor_ativo_aux"].fillna(False).sum()
    )

    onset_stays = int(
        data.loc[
            data["failure_state_new_at_t"].fillna(False)
        ]["stay_id"].nunique()
    )

    eligible = data["eligible_for_training"]
    target = data["target_first_onset_0_8h"]

    summary = {
        "batch": clean_file.name,
        "rows": int(len(data)),
        "stays": int(data["stay_id"].nunique()),
        "vaso_method": "direct_closed_start_end_intervals",
        "vaso_positive_rule": "any_of_seven_doses_strictly_gt_zero",
        "vaso_interval_rows_total": int(len(vaso_intervals)),
        "vaso_interval_rows_active": int(len(active_intervals)),
        "stays_with_active_vaso_intervals": int(
            active_intervals["stay_id"].nunique()
        ),
        "vaso_active_grid_rows": active_grid_rows,
        "vaso_future_information_used": False,
        "persistence_rule": "2_of_3_complete_instantaneous_criteria",
        "target_eligibility_rule": "observed_onset_sufficient_without_full_8h_followup_v2",
        "raw_failure_positive_rows": int(
            data["_failure_raw"].fillna(False).sum()
        ),
        "persistent_failure_positive_rows": int(
            data["failure_state_new_at_t"].fillna(False).sum()
        ),
        "stays_with_first_onset": onset_stays,
        "eligible_rows": int(eligible.sum()),
        "target_positive_rows": int(
            target.loc[eligible].fillna(0).sum()
        ),
        "output_file": str(output_file),
        "output_columns": OUTPUT_COLUMNS,
        "vasopressor_saved_in_output": False,
    }

    summary_file = OUTPUT_DIR / (
        clean_file.stem + "_summary.json"
    )
    summary_file.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nLabels gravados em: {output_file}")
    print(f"Resumo gravado em: {summary_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch",
        default="batch_0000.parquet",
        help="Nome do batch limpo a processar.",
    )
    args = parser.parse_args()

    clean_file = CLEAN_DIR / args.batch

    if not CLEAN_DIR.exists():
        raise RuntimeError(
            f"Pasta da grade limpa não encontrada: {CLEAN_DIR}"
        )

    if not LEGACY_DIR.exists():
        raise RuntimeError(
            f"Pasta preservada não encontrada: {LEGACY_DIR}"
        )

    if not clean_file.exists():
        raise RuntimeError(
            f"Batch limpo não encontrado: {clean_file}"
        )

    process_batch(clean_file)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERRO: {exc}", file=sys.stderr)
        raise
