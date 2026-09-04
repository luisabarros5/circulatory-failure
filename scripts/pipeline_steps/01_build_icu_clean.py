"""Source exported from the reviewed development notebook.

This file contains code only; patient-level data and notebook outputs are not distributed.
"""
from __future__ import annotations

# %% [source cell 0]
import os
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from tqdm.auto import tqdm

mimic_database_url = os.environ.get("MIMIC_DATABASE_URL")
if not mimic_database_url:
    raise RuntimeError(
        "Configure MIMIC_DATABASE_URL securely outside the repository."
    )

conn = psycopg2.connect(mimic_database_url)

with conn.cursor() as cursor:
    cursor.execute("SELECT current_user, current_database();")
    print(cursor.fetchone())

print("Conexão com MIMIC-IV criada com sucesso.")

# %% [source cell 1]
conn.rollback()

with conn.cursor() as cursor:
    cursor.execute("""
        CREATE TEMP TABLE todas_utis AS
        SELECT
            i.subject_id,
            i.hadm_id,
            i.stay_id,
            i.intime,
            i.outtime,
            EXTRACT(EPOCH FROM (i.outtime - i.intime)) / 60.0
                AS duracao_minutos
        FROM mimiciv_icu.icustays AS i
        WHERE EXTRACT(EPOCH FROM (i.outtime - i.intime)) > 3600;
    """)

conn.commit()

df_total = pd.read_sql(
    "SELECT COUNT(*) AS total_admissoes FROM todas_utis;",
    conn,
)

print(
    "UTI stays com duração acima de 1 hora:",
    int(df_total.loc[0, "total_admissoes"]),
)

# %% [source cell 2]
VITAL_DEFAULTS = {
    "heart_rate": 70.0,
    "sbp": 125.0,
    "dbp": 75.0,
    "mbp": 90.0,
    "resp_rate": 12.0,
    "temperature": 37.0,
    "spo2": 98.0,
    "glucose": 5.0,
}

LAB_DEFAULTS = {
    "lab_50803": 24.0,
    "lab_50805": 1.5,
    "lab_50808": 1.1,
    "lab_50809": 90.0,
    "lab_50813": 1.0,
    "lab_50852": 5.5,
    "lab_50855": 14.0,
    "lab_50856": 10.0,
    "lab_50861": 30.0,
    "lab_50862": 4.3,
    "lab_50878": 40.0,
    "lab_50883": 0.3,
    "lab_50884": 0.6,
    "lab_50885": 1.0,
    "lab_50889": 5.0,
    "lab_50908": 0.8,
    "lab_50910": 150.0,
    "lab_50912": 1.0,
    "lab_50915": 0.5,
    "lab_50924": 120.0,
    "lab_50931": 90.0,
    "lab_50935": 100.0,
    "lab_50960": 2.0,
    "lab_50963": 100.0,
    "lab_50966": 15.0,
    "lab_50967": 15.0,
    "lab_50968": 10.0,
    "lab_50969": 80.0,
    "lab_50970": 3.5,
    "lab_50971": 4.5,
    "lab_50983": 140.0,
    "lab_51002": 0.01,
    "lab_51003": 0.01,
    "lab_51006": 15.0,
    "lab_51099": 0.2,
    "lab_51196": 0.5,
    "lab_51214": 300.0,
    "lab_51222": 14.0,
    "lab_51223": 3.0,
    "lab_51265": 200.0,
    "lab_51266": 1.0,
    "lab_51274": 12.0,
    "lab_51275": 35.0,
    "lab_51464": 0.0,
    "lab_51580": 0.8,
    "lab_51623": 300.0,
    "lab_51631": 6.0,
    "lab_51638": 45.0,
    "lab_51640": 14.0,
    "lab_51966": 0.0,
    "lab_52116": 300.0,
    "lab_52117": 300.0,
    "lab_52142": 10.0,
    "lab_52144": 1.0,
    "lab_52546": 1.0,
    "lab_52551": 0.5,
    "lab_52642": 0.01,
    "lab_53085": 4.3,
}

# Itens sem default anterior, preservados nesta extração.
LAB_ITEMIDS_WITHOUT_DEFAULT = [53161, 53180]

# Blacklist: teofilina + exames raros decididos na auditoria.
EXCLUDED_LAB_ITEMIDS = {
    50928,
    50975,
    50990,
    51290,
    51291,
    51292,
    51568,
    51569,
    51570,
    51643,
    51647,
}

LAB_ITEMIDS = sorted(
    {
        int(name.removeprefix("lab_"))
        for name in LAB_DEFAULTS
    }
    | set(LAB_ITEMIDS_WITHOUT_DEFAULT)
)

LAB_ITEMIDS = [
    itemid
    for itemid in LAB_ITEMIDS
    if itemid not in EXCLUDED_LAB_ITEMIDS
]

VITAL_COLUMNS = list(VITAL_DEFAULTS)
LAB_COLUMNS = [f"lab_{itemid}" for itemid in LAB_ITEMIDS]

assert 50990 not in LAB_ITEMIDS
assert not (set(EXCLUDED_LAB_ITEMIDS) & set(LAB_ITEMIDS))

print(f"Vitais: {len(VITAL_COLUMNS)}")
print(f"Laboratórios permitidos: {len(LAB_ITEMIDS)}")
print("Blacklist sem teofilina e exames raros: OK.")

# %% [source cell 3]
def gerar_timegrid_icu(intime, outtime):
    """
    Grade de cinco minutos definida somente pela permanência real na UTI.
    """
    start = pd.Timestamp(intime).ceil("5min")
    end = pd.Timestamp(outtime).floor("5min")

    if end < start:
        return pd.DatetimeIndex([])

    return pd.date_range(
        start=start,
        end=end,
        freq="5min",
    )


def imputar_causalmente(
    events_df,
    stay_id,
    timegrid,
    variables,
    defaults,
):
    """
    Usa somente eventos com timestamp <= instante da grade.
    Forward-fill causal; antes da primeira medida usa o default.
    *_imputed = 0 quando houve medida naquele intervalo de 5 min;
    *_imputed = 1 quando o valor foi propagado/default.
    """
    output = pd.DataFrame({
        "stay_id": np.full(len(timegrid), stay_id, dtype=np.int64),
        "charttime": timegrid,
    })

    if events_df.empty:
        for variable in variables:
            output[variable] = np.float32(defaults.get(variable, np.nan))
            output[f"{variable}_imputed"] = np.int8(1)
        return output

    work = events_df.copy()
    work["charttime"] = pd.to_datetime(work["charttime"])
    work = work.sort_values("charttime")

    for variable in variables:
        default = defaults.get(variable, np.nan)

        if variable not in work.columns:
            output[variable] = np.float32(default)
            output[f"{variable}_imputed"] = np.int8(1)
            continue

        observed = work[
            ["charttime", variable]
        ].dropna(subset=[variable]).copy()

        if observed.empty:
            output[variable] = np.float32(default)
            output[f"{variable}_imputed"] = np.int8(1)
            continue

        observed = (
            observed.groupby("charttime", as_index=False)[variable]
            .last()
            .sort_values("charttime")
        )

        series = (
            observed.set_index("charttime")[variable]
            .reindex(timegrid, method="ffill")
            .fillna(default)
            .astype(np.float32)
        )

        # A medição é atribuída ao primeiro ponto da grade após sua coleta.
        event_bins = observed["charttime"].dt.ceil("5min")
        measured_bins = pd.Index(event_bins).intersection(pd.Index(timegrid))

        is_imputed = np.ones(len(timegrid), dtype=np.int8)
        if len(measured_bins) > 0:
            positions = pd.Index(timegrid).get_indexer(measured_bins)
            is_imputed[positions[positions >= 0]] = 0

        output[variable] = series.to_numpy()
        output[f"{variable}_imputed"] = is_imputed

    return output


def processar_batch_vitals(batch_stay_ids):
    query = """
        SELECT
            vs.stay_id,
            vs.charttime,
            vs.heart_rate,
            vs.sbp,
            vs.dbp,
            vs.mbp,
            vs.resp_rate,
            vs.temperature,
            vs.spo2,
            vs.glucose
        FROM mimiciv_derived.vitalsign AS vs
        INNER JOIN todas_utis AS icu
            ON vs.stay_id = icu.stay_id
        WHERE vs.stay_id = ANY(%s)
          AND vs.charttime >= icu.intime
          AND vs.charttime <= icu.outtime
        ORDER BY vs.stay_id, vs.charttime;
    """

    df = pd.read_sql(
        query,
        conn,
        params=(list(map(int, batch_stay_ids)),),
    )

    df["charttime"] = pd.to_datetime(df["charttime"])
    return df


def processar_batch_labs(batch_stay_ids):
    query = """
        SELECT
            icu.stay_id,
            le.charttime,
            le.itemid,
            le.valuenum
        FROM mimiciv_hosp.labevents AS le
        INNER JOIN todas_utis AS icu
            ON le.subject_id = icu.subject_id
           AND le.hadm_id = icu.hadm_id
           AND le.charttime >= icu.intime
           AND le.charttime <= icu.outtime
        WHERE le.itemid = ANY(%s)
          AND le.valuenum IS NOT NULL
          AND icu.stay_id = ANY(%s)
        ORDER BY icu.stay_id, le.charttime;
    """

    long_df = pd.read_sql(
        query,
        conn,
        params=(
            list(map(int, LAB_ITEMIDS)),
            list(map(int, batch_stay_ids)),
        ),
    )

    if long_df.empty:
        return pd.DataFrame(columns=["stay_id", "charttime"])

    long_df["charttime"] = pd.to_datetime(long_df["charttime"])

    wide_df = long_df.pivot_table(
        index=["stay_id", "charttime"],
        columns="itemid",
        values="valuenum",
        aggfunc="last",
    )

    wide_df.columns = [
        f"lab_{int(itemid)}"
        for itemid in wide_df.columns
    ]

    return (
        wide_df.reset_index()
        .sort_values(["stay_id", "charttime"])
        .reset_index(drop=True)
    )

print("Funções causais e queries ICU-bounded prontas.")

# %% [source cell 4]
cohort_query = """
WITH vital_stays AS (
    SELECT DISTINCT vs.stay_id
    FROM mimiciv_derived.vitalsign AS vs
    INNER JOIN todas_utis AS icu
        ON vs.stay_id = icu.stay_id
    WHERE vs.charttime >= icu.intime
      AND vs.charttime <= icu.outtime
),
lab_stays AS (
    SELECT DISTINCT icu.stay_id
    FROM mimiciv_hosp.labevents AS le
    INNER JOIN todas_utis AS icu
        ON le.subject_id = icu.subject_id
       AND le.hadm_id = icu.hadm_id
       AND le.charttime >= icu.intime
       AND le.charttime <= icu.outtime
    WHERE le.itemid = ANY(%s)
      AND le.valuenum IS NOT NULL
)
SELECT
    icu.stay_id,
    icu.intime,
    icu.outtime
FROM todas_utis AS icu
INNER JOIN vital_stays AS v
    ON icu.stay_id = v.stay_id
INNER JOIN lab_stays AS l
    ON icu.stay_id = l.stay_id
ORDER BY icu.stay_id;
"""

clean_cohort = pd.read_sql(
    cohort_query,
    conn,
    params=(list(map(int, LAB_ITEMIDS)),),
)

clean_cohort["stay_id"] = clean_cohort["stay_id"].astype(int)
clean_cohort["intime"] = pd.to_datetime(clean_cohort["intime"])
clean_cohort["outtime"] = pd.to_datetime(clean_cohort["outtime"])

stay_bounds_clean = (
    clean_cohort
    .set_index("stay_id")[["intime", "outtime"]]
)

print(
    f"Coorte ICU-clean: {clean_cohort['stay_id'].nunique():,} stays."
)

# %% [source cell 5]
SMOKE_STAYS = 100

OUTPUT_DIR = Path(
    "/path/to/mimic_source_project/"
    "output_icu_clean_v2/smoke_icu_bounds"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / (
    f"icu_clean_smoke_{SMOKE_STAYS}_stays.parquet"
)
TEMP_FILE = OUTPUT_FILE.with_suffix(".parquet.tmp")

if OUTPUT_FILE.exists():
    raise FileExistsError(
        f"Arquivo já existe: {OUTPUT_FILE}\n"
        "Não vou sobrescrever um smoke anterior."
    )

smoke_stays = (
    clean_cohort
    .sort_values("stay_id")
    .head(SMOKE_STAYS)
    .copy()
)

stay_ids = smoke_stays["stay_id"].astype(int).tolist()

print(f"Consultando dados de {len(stay_ids)} stays...")

df_vitals = processar_batch_vitals(stay_ids)
df_labs = processar_batch_labs(stay_ids)

if df_vitals.empty:
    raise RuntimeError("A consulta de vitais retornou zero linhas.")

if df_labs.empty:
    raise RuntimeError("A consulta de laboratórios retornou zero linhas.")

frames = []

for stay in tqdm(
    smoke_stays.itertuples(index=False),
    total=len(smoke_stays),
    desc="Construindo grade ICU-clean",
):
    stay_id = int(stay.stay_id)
    intime = pd.Timestamp(stay.intime)
    outtime = pd.Timestamp(stay.outtime)

    timegrid = gerar_timegrid_icu(intime, outtime)

    if timegrid.empty:
        raise RuntimeError(f"stay_id={stay_id}: grade temporal vazia.")

    vital_stay = df_vitals.loc[
        df_vitals["stay_id"] == stay_id
    ].copy()

    lab_stay = df_labs.loc[
        df_labs["stay_id"] == stay_id
    ].copy()

    vital_grid = imputar_causalmente(
        vital_stay,
        stay_id,
        timegrid,
        VITAL_COLUMNS,
        VITAL_DEFAULTS,
    )

    lab_grid = imputar_causalmente(
        lab_stay,
        stay_id,
        timegrid,
        LAB_COLUMNS,
        LAB_DEFAULTS,
    )

    merged = vital_grid.merge(
        lab_grid,
        on=["stay_id", "charttime"],
        how="inner",
        validate="one_to_one",
    )

    assert not merged.empty
    assert (merged["charttime"] >= intime).all()
    assert (merged["charttime"] <= outtime).all()

    frames.append(merged)

result = (
    pd.concat(frames, ignore_index=True)
    .sort_values(["stay_id", "charttime"])
    .reset_index(drop=True)
)

assert not result.duplicated(["stay_id", "charttime"]).any()

result.to_parquet(
    TEMP_FILE,
    index=False,
    compression="zstd",
)

os.replace(TEMP_FILE, OUTPUT_FILE)

print("\nSMOKE CONCLUÍDO")
print(f"Arquivo: {OUTPUT_FILE}")
print(f"Stays: {result['stay_id'].nunique():,}")
print(f"Linhas: {len(result):,}")
print(f"Colunas: {len(result.columns):,}")

del result, frames, df_vitals, df_labs
gc.collect()

# %% [source cell 6]
check = pd.read_parquet(OUTPUT_FILE)

assert check["stay_id"].nunique() == SMOKE_STAYS
assert not check.duplicated(["stay_id", "charttime"]).any()

bounds = stay_bounds_clean.reset_index()

audit = check[
    ["stay_id", "charttime"]
].merge(
    bounds,
    on="stay_id",
    how="left",
    validate="many_to_one",
)

assert audit["intime"].notna().all()
assert (audit["charttime"] >= audit["intime"]).all()
assert (audit["charttime"] <= audit["outtime"]).all()

deltas = (
    check
    .sort_values(["stay_id", "charttime"])
    .groupby("stay_id")["charttime"]
    .diff()
    .dropna()
    .dt.total_seconds()
)

assert (deltas == 300).all()

forbidden = [
    column
    for column in check.columns
    if any(
        f"lab_{itemid}" in column
        for itemid in EXCLUDED_LAB_ITEMIDS
    )
]

assert not forbidden, f"Colunas proibidas encontradas: {forbidden}"

print("VALIDAÇÃO APROVADA")
print(f"Stays: {check['stay_id'].nunique():,}")
print(f"Linhas: {len(check):,}")
print(f"Colunas: {len(check.columns):,}")
print("Todos os dados estão dentro de intime/outtime.")
print("Grade temporal: 5 minutos.")
print("Teofilina e exames removidos: ausentes.")

# %% [source cell 7]
import json
import time
import traceback

BATCH_SIZE = 250
START_BATCH = 0
END_BATCH = None  # None = processa até o último batch

FULL_OUTPUT_DIR = Path(
    "/path/to/mimic_source_project/"
    "output_icu_clean_v2/raw_grid_all_icu_clean_v2"
)
FULL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROGRESS_PATH = FULL_OUTPUT_DIR / "progress.jsonl"

all_stay_ids = (
    clean_cohort["stay_id"]
    .astype(int)
    .sort_values()
    .tolist()
)

batches = [
    all_stay_ids[start:start + BATCH_SIZE]
    for start in range(0, len(all_stay_ids), BATCH_SIZE)
]

# Ordem canônica e fixa para todos os Parquets.
# O merge gera a mesma quantidade de colunas, mas em ordem diferente.
expected_columns = (
    ["stay_id", "charttime"]
    + VITAL_COLUMNS
    + [f"{column}_imputed" for column in VITAL_COLUMNS]
    + LAB_COLUMNS
    + [f"{column}_imputed" for column in LAB_COLUMNS]
)

if END_BATCH is None:
    END_BATCH = len(batches)

if not (0 <= START_BATCH < len(batches)):
    raise ValueError(
        f"START_BATCH inválido: {START_BATCH}. "
        f"Total de batches: {len(batches)}."
    )

if not (START_BATCH < END_BATCH <= len(batches)):
    raise ValueError(
        f"Intervalo inválido: START_BATCH={START_BATCH}, "
        f"END_BATCH={END_BATCH}, total={len(batches)}."
    )

print(f"Coorte total: {len(all_stay_ids):,} stays")
print(f"Batches totais: {len(batches):,}")
print(f"Batches desta execução: {START_BATCH} até {END_BATCH - 1}")
print(f"Stays por batch: {BATCH_SIZE}")
print(f"Saída: {FULL_OUTPUT_DIR}")

for batch_idx in range(START_BATCH, END_BATCH):
    batch_stay_ids = batches[batch_idx]

    parquet_path = FULL_OUTPUT_DIR / f"batch_{batch_idx:04d}.parquet"
    temp_path = FULL_OUTPUT_DIR / f"batch_{batch_idx:04d}.parquet.tmp"
    done_path = FULL_OUTPUT_DIR / f"batch_{batch_idx:04d}.parquet.done"

    if (
        done_path.exists()
        and parquet_path.exists()
        and parquet_path.stat().st_size > 0
    ):
        print(
            f"[{batch_idx + 1}/{len(batches)}] "
            "já concluído; pulando."
        )
        continue

    if done_path.exists() and (
        not parquet_path.exists()
        or parquet_path.stat().st_size == 0
    ):
        raise RuntimeError(
            f"Estado inconsistente: existe {done_path}, "
            "mas o Parquet correspondente não é válido."
        )

    if temp_path.exists():
        raise RuntimeError(
            "Existe arquivo temporário de execução interrompida:\n"
            f"{temp_path}\n"
            "Não vou sobrescrevê-lo automaticamente."
        )

    started = time.time()

    try:
        print(
            f"\n[{batch_idx + 1}/{len(batches)}] "
            f"Processando {len(batch_stay_ids)} stays..."
        )

        df_vitals = processar_batch_vitals(batch_stay_ids)
        df_labs = processar_batch_labs(batch_stay_ids)

        if df_vitals.empty:
            raise RuntimeError(
                f"batch {batch_idx}: consulta de vitais retornou zero linhas."
            )

        if df_labs.empty:
            raise RuntimeError(
                f"batch {batch_idx}: consulta de labs retornou zero linhas."
            )

        frames = []

        for stay_id in tqdm(
            batch_stay_ids,
            desc=f"Batch {batch_idx:04d}",
            leave=False,
        ):
            bounds = stay_bounds_clean.loc[int(stay_id)]
            intime = pd.Timestamp(bounds["intime"])
            outtime = pd.Timestamp(bounds["outtime"])

            timegrid = gerar_timegrid_icu(intime, outtime)

            if timegrid.empty:
                raise RuntimeError(
                    f"batch {batch_idx}, stay_id={stay_id}: "
                    "grade temporal vazia."
                )

            vital_stay = df_vitals.loc[
                df_vitals["stay_id"] == stay_id
            ].copy()

            lab_stay = df_labs.loc[
                df_labs["stay_id"] == stay_id
            ].copy()

            vital_grid = imputar_causalmente(
                vital_stay,
                stay_id,
                timegrid,
                VITAL_COLUMNS,
                VITAL_DEFAULTS,
            )

            lab_grid = imputar_causalmente(
                lab_stay,
                stay_id,
                timegrid,
                LAB_COLUMNS,
                LAB_DEFAULTS,
            )

            merged = vital_grid.merge(
                lab_grid,
                on=["stay_id", "charttime"],
                how="inner",
                validate="one_to_one",
            )

            if merged.empty:
                raise RuntimeError(
                    f"batch {batch_idx}, stay_id={stay_id}: "
                    "merge vazio."
                )

            if not (
                (merged["charttime"] >= intime).all()
                and (merged["charttime"] <= outtime).all()
            ):
                raise RuntimeError(
                    f"batch {batch_idx}, stay_id={stay_id}: "
                    "dados fora de intime/outtime."
                )

            frames.append(merged)

        if not frames:
            raise RuntimeError(
                f"batch {batch_idx}: nenhum stay gerou linhas."
            )

        batch_result = (
            pd.concat(frames, ignore_index=True)
            .sort_values(["stay_id", "charttime"])
            .reset_index(drop=True)
        )

        if batch_result.duplicated(["stay_id", "charttime"]).any():
            raise RuntimeError(
                f"batch {batch_idx}: duplicatas stay_id + charttime."
            )

        # Confere presença das colunas, independentemente da ordem
        # gerada pelo merge; depois padroniza a ordem canônica.
        missing_columns = sorted(
            set(expected_columns) - set(batch_result.columns)
        )

        extra_columns = sorted(
            set(batch_result.columns) - set(expected_columns)
        )

        if missing_columns or extra_columns:
            raise RuntimeError(
                f"batch {batch_idx}: schema incompatível.\n"
                f"Colunas faltantes: {missing_columns}\n"
                f"Colunas extras: {extra_columns}"
            )

        batch_result = batch_result.reindex(columns=expected_columns)

        if list(batch_result.columns) != expected_columns:
            raise RuntimeError(
                f"batch {batch_idx}: não foi possível padronizar "
                "a ordem das colunas."
            )

        batch_result.to_parquet(
            temp_path,
            index=False,
            compression="zstd",
        )

        if not temp_path.exists() or temp_path.stat().st_size == 0:
            raise RuntimeError(
                f"batch {batch_idx}: falha ao gravar Parquet temporário."
            )

        os.replace(temp_path, parquet_path)

        done_path.write_text(
            json.dumps(
                {
                    "batch_idx": batch_idx,
                    "stays": int(batch_result["stay_id"].nunique()),
                    "rows": int(len(batch_result)),
                    "columns": int(len(batch_result.columns)),
                    "seconds": round(time.time() - started, 2),
                    "status": "completed",
                },
                ensure_ascii=False,
            )
            + "\n"
        )

        with PROGRESS_PATH.open("a", encoding="utf-8") as progress_file:
            progress_file.write(
                json.dumps(
                    {
                        "batch_idx": batch_idx,
                        "parquet": str(parquet_path),
                        "stays": int(batch_result["stay_id"].nunique()),
                        "rows": int(len(batch_result)),
                        "columns": int(len(batch_result.columns)),
                        "seconds": round(time.time() - started, 2),
                        "status": "completed",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

        print(
            f"[{batch_idx + 1}/{len(batches)}] concluído: "
            f"{len(batch_result):,} linhas, "
            f"{batch_result['stay_id'].nunique():,} stays, "
            f"{time.time() - started:.1f}s"
        )

        del batch_result, frames, df_vitals, df_labs
        gc.collect()

    except Exception:
        print(f"\nERRO no batch {batch_idx}. Nenhum .done foi criado.")
        traceback.print_exc()
        raise

print("\nEXTRAÇÃO COMPLETA FINALIZADA")
print(f"Saída: {FULL_OUTPUT_DIR}")

# %% [source cell 8]
from pathlib import Path
import pyarrow.parquet as pq

GRID_DIR = Path(
    "/path/to/mimic_source_project/"
    "output_icu_clean_v2/raw_grid_all_icu_clean_v2"
)

files = sorted(GRID_DIR.glob("*.parquet"))

if len(files) != 285:
    raise RuntimeError(f"Esperava 285 arquivos, encontrei {len(files)}.")

schema = pq.ParquetFile(files[0]).schema_arrow
columns = schema.names

print(f"Total de colunas: {len(columns)}\n")

for i, col in enumerate(columns, start=1):
    print(f"{i:03d}. {col}")

# %% [source cell 9]
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm.auto import tqdm

GRID_DIR = Path(
    "/path/to/mimic_source_project/"
    "output_icu_clean_v2/raw_grid_all_icu_clean_v2"
)
files = sorted(GRID_DIR.glob("*.parquet"))

# A coorte usada na extração deve continuar disponível no notebook.
cohort_bounds = (
    clean_cohort[["stay_id", "intime", "outtime"]]
    .drop_duplicates("stay_id")
    .assign(
        stay_id=lambda x: x["stay_id"].astype("int64"),
        intime=lambda x: pd.to_datetime(x["intime"]),
        outtime=lambda x: pd.to_datetime(x["outtime"]),
    )
    .set_index("stay_id")
)

if len(files) != 285:
    raise RuntimeError(f"Esperava 285 batches; encontrei {len(files)}.")

expected_stays = set(cohort_bounds.index.tolist())
seen_stays = set()

total_rows = 0
duplicate_keys = 0
outside_icu = 0
bad_intervals = 0
problems = []

for path in tqdm(files, desc="Validando 285 batches"):
    try:
        pf = pq.ParquetFile(path)

        # Cada batch foi construído por stay; ainda assim, verificamos tudo.
        for record_batch in pf.iter_batches(
            batch_size=500_000,
            columns=["stay_id", "charttime"],
        ):
            df = record_batch.to_pandas()
            df["stay_id"] = df["stay_id"].astype("int64")
            df["charttime"] = pd.to_datetime(df["charttime"])

            total_rows += len(df)

            if df.duplicated(["stay_id", "charttime"]).any():
                duplicate_keys += int(df.duplicated(["stay_id", "charttime"]).sum())

            unknown = set(df["stay_id"].unique()) - expected_stays
            if unknown:
                raise RuntimeError(
                    f"{path.name}: stay_ids fora da coorte, exemplo: "
                    f"{list(unknown)[:5]}"
                )

            bounds = cohort_bounds.loc[df["stay_id"]]
            invalid_bounds = (
                (df["charttime"].to_numpy() < bounds["intime"].to_numpy()) |
                (df["charttime"].to_numpy() > bounds["outtime"].to_numpy())
            )
            outside_icu += int(invalid_bounds.sum())

            # Checa grade de 5 min dentro de cada stay neste trecho.
            df = df.sort_values(["stay_id", "charttime"])
            diffs = df.groupby("stay_id")["charttime"].diff().dropna()
            bad_intervals += int((diffs != pd.Timedelta(minutes=5)).sum())

        batch_stays = set(
            pq.read_table(path, columns=["stay_id"])
            .column("stay_id")
            .to_pylist()
        )

        overlap = seen_stays & batch_stays
        if overlap:
            raise RuntimeError(
                f"{path.name}: stays repetidos entre batches; exemplo: "
                f"{list(overlap)[:5]}"
            )
        seen_stays.update(batch_stays)

    except Exception as exc:
        problems.append(f"{path.name}: {exc}")

missing_stays = expected_stays - seen_stays
extra_stays = seen_stays - expected_stays

print("\nRESULTADO DA VALIDAÇÃO INTEGRAL")
print(f"Linhas lidas: {total_rows:,}")
print(f"Stays encontrados: {len(seen_stays):,}")
print(f"Stays esperados: {len(expected_stays):,}")
print(f"Chaves duplicadas: {duplicate_keys:,}")
print(f"Timestamps fora de intime/outtime: {outside_icu:,}")
print(f"Intervalos diferentes de 5 min: {bad_intervals:,}")
print(f"Stays ausentes: {len(missing_stays):,}")
print(f"Stays extras: {len(extra_stays):,}")
print(f"Erros de leitura/validação: {len(problems):,}")

if problems:
    raise RuntimeError("\n".join(problems[:10]))
if duplicate_keys or outside_icu or bad_intervals or missing_stays or extra_stays:
    raise RuntimeError("VALIDAÇÃO REPROVADA — veja as contagens acima.")
print("\nVALIDAÇÃO INTEGRAL APROVADA.")
