#!/usr/bin/env python3
"""
03_audit_temporal_availability.py

Auditoria ÚNICA da disponibilidade temporal pré-onset.

Entrada:
- positive_stays_v2.parquet já validado, com:
    stay_id
    first_positive_time  (segundos desde o início da série)
    source_target_col    (cf_first_onset_primary)

Objetivo:
- informar quantos e qual % dos 15.739 stays possuem
  1h, 2h, 4h, 6h e 8h COMPLETAS antes do primeiro onset.

Não:
- reconstrói labels;
- lê os 214M timesteps;
- gera séries;
- roda MiniRocket;
- roda K-means.

Saída:
- temporal_availability_audit.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


WINDOWS_HOURS = [1, 2, 4, 6, 8]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positive-stays", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    df = pd.read_parquet(args.positive_stays)

    required = {"stay_id", "first_positive_time"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Faltam colunas obrigatórias: {sorted(missing)}")

    df = df[["stay_id", "first_positive_time"]].copy()
    df["first_positive_time"] = pd.to_numeric(
        df["first_positive_time"], errors="coerce"
    )
    df = df.dropna(subset=["stay_id", "first_positive_time"])
    df = df.drop_duplicates("stay_id")

    n_total = len(df)
    if n_total == 0:
        raise SystemExit("Nenhum stay válido encontrado.")

    df["onset_hours"] = df["first_positive_time"] / 3600.0

    rows = []
    for h in WINDOWS_HOURS:
        eligible = df["first_positive_time"] >= h * 3600
        n = int(eligible.sum())
        rows.append({
            "window_hours": h,
            "required_pre_onset_timesteps_5min": h * 12,
            "n_stays_with_complete_window": n,
            "pct_of_positive_stays": 100.0 * n / n_total,
            "n_stays_excluded": n_total - n,
            "pct_excluded": 100.0 * (n_total - n) / n_total,
        })

    audit = pd.DataFrame(rows)

    q = df["onset_hours"].quantile(
        [0.00, 0.10, 0.25, 0.50, 0.75, 0.90, 1.00]
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.out, index=False)

    print("=== AUDITORIA ÚNICA DE DISPONIBILIDADE PRÉ-ONSET ===")
    print(f"Stays positivos avaliados: {n_total:,}")
    print()
    print(
        audit.to_string(
            index=False,
            formatters={
                "pct_of_positive_stays": lambda x: f"{x:.1f}%",
                "pct_excluded": lambda x: f"{x:.1f}%",
            },
        )
    )

    print("\n=== TEMPO ATÉ O PRIMEIRO ONSET (HORAS) ===")
    labels = {
        0.00: "mínimo",
        0.10: "P10",
        0.25: "P25",
        0.50: "mediana",
        0.75: "P75",
        0.90: "P90",
        1.00: "máximo",
    }
    for p, value in q.items():
        print(f"{labels[p]:>7}: {value:.2f} h")

    print(f"\nCSV salvo em: {args.out}")


if __name__ == "__main__":
    main()
