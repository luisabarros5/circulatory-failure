#!/usr/bin/env python3
"""
01_build_positive_stays_v2.py

Constrói a lista de ICU stays positivos usando os labels NOVOS de falência.

- Lê Parquets de labels em streaming (arquivo por arquivo).
- Detecta ou recebe explicitamente a coluna target.
- Mantém apenas stays com pelo menos um target == 1.
- Salva uma linha por stay_id, com contagem de positivos e primeiro ponto positivo.
- NÃO recalcula a regra clínica: usa o target já materializado.

Example:
python 01_build_positive_stays.py \
  --labels-dir /path/to/labels \
  --out /path/to/outputs/positive_stays.parquet \
  --target-col cf_first_onset_primary
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


TARGET_CANDIDATES = (
    "falencia",
    "target",
    "target_onset_0_8h",
    "y",
    "label",
)

TIME_CANDIDATES = (
    "rel_charttime",
    "charttime",
    "feature_time",
    "label_time",
)


def detect_target(schema_names: list[str], explicit: str | None) -> str:
    if explicit:
        if explicit not in schema_names:
            raise SystemExit(
                f"Target solicitado '{explicit}' não existe. "
                f"Colunas disponíveis: {schema_names}"
            )
        return explicit

    found = [c for c in TARGET_CANDIDATES if c in schema_names]
    if not found:
        raise SystemExit(
            "Não consegui detectar a coluna target. "
            f"Use --target-col. Colunas: {schema_names}"
        )
    if len(found) > 1:
        print(
            "[ATENÇÃO] Mais de um candidato a target encontrado: "
            f"{found}. Usando '{found[0]}'. "
            "Se não for o desejado, rode novamente com --target-col."
        )
    return found[0]


def detect_time(schema_names: list[str]) -> str | None:
    for c in TIME_CANDIDATES:
        if c in schema_names:
            return c
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--target-col", default=None)
    ap.add_argument("--pattern", default="*.parquet")
    ap.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Caminho opcional do JSON de auditoria.",
    )
    args = ap.parse_args()

    paths = sorted(args.labels_dir.glob(args.pattern))
    paths = [p for p in paths if p.is_file() and "summary" not in p.name.lower()]
    if not paths:
        raise SystemExit(f"Nenhum parquet encontrado em {args.labels_dir}")

    first_schema = pq.read_schema(paths[0]).names
    if "stay_id" not in first_schema:
        raise SystemExit(f"'stay_id' ausente em {paths[0]}")

    target_col = detect_target(first_schema, args.target_col)
    time_col = detect_time(first_schema)

    print(f"[INFO] Arquivos: {len(paths)}")
    print(f"[INFO] Target selecionado: {target_col}")
    print(f"[INFO] Coluna temporal: {time_col or 'não encontrada'}")

    summaries: list[pd.DataFrame] = []
    total_rows = 0
    total_positive_rows = 0

    for i, path in enumerate(paths, 1):
        schema = pq.read_schema(path).names
        required = ["stay_id", target_col]
        if time_col and time_col in schema:
            required.append(time_col)

        missing = [c for c in required if c not in schema]
        if missing:
            raise RuntimeError(f"{path.name}: faltam colunas {missing}")

        df = pd.read_parquet(path, columns=required)
        total_rows += len(df)

        target = pd.to_numeric(df[target_col], errors="coerce").fillna(0)
        pos = df.loc[target == 1].copy()
        total_positive_rows += len(pos)

        if pos.empty:
            continue

        pos["stay_id"] = pd.to_numeric(pos["stay_id"], errors="coerce")
        pos = pos.dropna(subset=["stay_id"])
        pos["stay_id"] = pos["stay_id"].astype("int64")

        if time_col and time_col in pos.columns:
            g = (
                pos.groupby("stay_id", as_index=False)
                .agg(
                    n_positive_rows=(target_col, "size"),
                    first_positive_time=(time_col, "min"),
                    last_positive_time=(time_col, "max"),
                )
            )
        else:
            g = (
                pos.groupby("stay_id", as_index=False)
                .agg(n_positive_rows=(target_col, "size"))
            )
            g["first_positive_time"] = pd.NA
            g["last_positive_time"] = pd.NA

        summaries.append(g)

        if i % 25 == 0 or i == len(paths):
            print(
                f"[INFO] {i}/{len(paths)} | "
                f"rows={total_rows:,} | positive_rows={total_positive_rows:,}"
            )

    if not summaries:
        raise SystemExit(
            f"Nenhuma linha com {target_col} == 1 foi encontrada. "
            "Confira se escolhemos o target correto."
        )

    all_pos = pd.concat(summaries, ignore_index=True)

    # Stays não deveriam atravessar arquivos, mas agregamos de novo por segurança.
    positive_stays = (
        all_pos.groupby("stay_id", as_index=False)
        .agg(
            n_positive_rows=("n_positive_rows", "sum"),
            first_positive_time=("first_positive_time", "min"),
            last_positive_time=("last_positive_time", "max"),
        )
        .sort_values("stay_id")
        .reset_index(drop=True)
    )

    positive_stays["falencia_any"] = 1
    positive_stays["source_target_col"] = target_col

    args.out.parent.mkdir(parents=True, exist_ok=True)
    positive_stays.to_parquet(args.out, index=False)

    summary = {
        "labels_dir": str(args.labels_dir),
        "n_label_files": len(paths),
        "target_col": target_col,
        "time_col": time_col,
        "total_rows_scanned": int(total_rows),
        "total_positive_rows": int(total_positive_rows),
        "n_positive_stays": int(positive_stays["stay_id"].nunique()),
        "output": str(args.out),
    }

    summary_path = args.summary_json or args.out.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== RESULTADO ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print("\nPrimeiros stays:")
    print(positive_stays.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
