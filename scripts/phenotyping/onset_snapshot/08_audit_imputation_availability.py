#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path("outputs/mimic_cluster_circews_v2/onset_snapshot")
RAW = BASE / "onset_snapshot_raw.parquet"
OUT = BASE / "validation_minimal"
OUT.mkdir(parents=True, exist_ok=True)

print("=== 08 | AUDITORIA DE DISPONIBILIDADE / IMPUTAÇÃO ===")

if not RAW.exists():
    raise FileNotFoundError(f"Não encontrei: {RAW}")

df = pd.read_parquet(RAW)
print(f"Snapshot: {len(df):,} stays")
print(f"Colunas: {len(df.columns)}")

flag_cols = [
    c for c in df.columns
    if c.endswith("_imputed")
    or c.endswith("__imputed")
    or c.startswith("imputed_")
]

rows = []

if flag_cols:
    print(f"Flags de imputação encontradas: {len(flag_cols)}")

    for flag in flag_cols:
        if flag.endswith("__imputed"):
            feature = flag[:-10]
        elif flag.endswith("_imputed"):
            feature = flag[:-8]
        else:
            feature = flag[len("imputed_"):]

        x = pd.to_numeric(df[flag], errors="coerce")
        x = x[x.notna()]

        if len(x) == 0:
            continue

        vals = set(np.unique(x))
        if not vals.issubset({0, 1, 0.0, 1.0}):
            continue

        imp = 100.0 * float(x.mean())
        rows.append({
            "feature": feature,
            "n": len(x),
            "observed_pct": 100.0 - imp,
            "imputed_pct": imp,
            "method": "imputed_flag",
        })

else:
    suffix = "_time_since_last_measurement_min"
    ts_cols = [c for c in df.columns if c.endswith(suffix)]

    if ts_cols:
        print(f"Sem flags *_imputed. Usando {len(ts_cols)} colunas time-since.")
        for col in ts_cols:
            feature = col[:-len(suffix)]
            x = pd.to_numeric(df[col], errors="coerce")
            valid = x.notna()

            if valid.sum() == 0:
                continue

            observed_pct = 100.0 * float(x[valid].eq(0).mean())
            rows.append({
                "feature": feature,
                "n": int(valid.sum()),
                "observed_pct": observed_pct,
                "imputed_pct": 100.0 - observed_pct,
                "method": "time_since_eq_0",
            })

if rows:
    out = pd.DataFrame(rows).sort_values("observed_pct")
    out.to_csv(OUT / "imputation_availability_audit.csv", index=False)

    print("\n=== MENOR DISPONIBILIDADE NO ONSET ===")
    for _, r in out.head(30).iterrows():
        print(
            f"{r['feature']:<38} "
            f"observado={r['observed_pct']:6.2f}% | "
            f"imputado={r['imputed_pct']:6.2f}%"
        )

    print("\n[OK] Auditoria concluída.")
    print(OUT / "imputation_availability_audit.csv")
else:
    print("\n[ATENÇÃO] O snapshot raw não contém flags *_imputed nem time-since.")
    print("Não vou inferir disponibilidade a partir dos valores já imputados.")
    print("As demais validações podem ser executadas normalmente.")
