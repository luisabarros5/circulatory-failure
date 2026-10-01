#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path("outputs/mimic_cluster_circews_v2/onset_snapshot")
RAW_PATH = BASE / "onset_snapshot_raw.parquet"
SUB_ROOT = BASE / "log_curated" / "subclusters"
OUT = BASE / "validation_minimal"
OUT.mkdir(parents=True, exist_ok=True)

print("=== 10 | C0-A vs C0-B EM VALORES CLÍNICOS ORIGINAIS ===")

def read_table(path):
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)

def find_c0_k2_assignments():
    if not SUB_ROOT.exists():
        raise FileNotFoundError(f"Não encontrei: {SUB_ROOT}")

    candidates = list(SUB_ROOT.rglob("*.parquet")) + list(SUB_ROOT.rglob("*.csv"))
    found = []

    for p in candidates:
        try:
            d = read_table(p)
        except Exception:
            continue

        if "stay_id" not in d.columns:
            continue

        cluster_cols = [
            c for c in d.columns
            if c.lower() in {"cluster", "label", "labels", "subcluster"}
            or "cluster" in c.lower()
        ]

        for cc in cluster_cols:
            vals = d[cc].dropna()
            if vals.nunique() != 2:
                continue

            name = str(p).lower()
            score = 0

            if 4000 <= len(d) <= 4600:
                score += 20
            elif 3500 <= len(d) <= 5000:
                score += 10

            if "c0" in name:
                score += 10
            if "k2" in name or "k_2" in name:
                score += 5

            found.append((score, p, cc, d))

    if not found:
        raise RuntimeError(
            "Não achei automaticamente um arquivo de assignments "
            "C0-A/C0-B em log_curated/subclusters."
        )

    found.sort(key=lambda x: x[0], reverse=True)
    _, path, col, d = found[0]
    return path, col, d[["stay_id", col]].copy()

def summarize(x):
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    return float(x.median()), float(x.quantile(0.25)), float(x.quantile(0.75))

def smd(a, b):
    a = pd.to_numeric(a, errors="coerce").dropna().to_numpy(float)
    b = pd.to_numeric(b, errors="coerce").dropna().to_numpy(float)

    if len(a) < 2 or len(b) < 2:
        return np.nan

    pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    if pooled == 0 or not np.isfinite(pooled):
        return np.nan

    return float((np.mean(b) - np.mean(a)) / pooled)

if not RAW_PATH.exists():
    raise FileNotFoundError(f"Não encontrei: {RAW_PATH}")

raw = pd.read_parquet(RAW_PATH)
assign_path, cluster_col, assign = find_c0_k2_assignments()

assign = assign.rename(columns={cluster_col: "subcluster"})

print(f"Assignments: {assign_path}")
print(f"N: {len(assign):,}")
print(assign["subcluster"].value_counts().sort_index())

df = assign.merge(raw, on="stay_id", how="left", validate="one_to_one")

groups = sorted(df["subcluster"].dropna().unique().tolist())
if len(groups) != 2:
    raise RuntimeError(f"Esperava 2 subclusters; encontrei {groups}")

g0, g1 = groups

preferred = [
    "heart_rate", "mbp", "resp_rate", "temperature", "spo2", "glucose",
    "lactate", "sodium", "potassium", "creatinine", "bun", "ast", "alt",
    "total_bilirubin", "albumin", "hemoglobin", "platelets", "pt", "ptt",
    "vasopressor_ativo",
]

features = [c for c in preferred if c in df.columns]

if len(features) < 10:
    exclude = {
        "stay_id", "subject_id", "hadm_id",
        "cluster", "subcluster",
        "cf_state_primary_2h", "cf_first_onset_primary",
    }
    features = [
        c for c in raw.select_dtypes(include=np.number).columns
        if c not in exclude
        and "imputed" not in c.lower()
        and "time_since" not in c.lower()
        and "charttime" not in c.lower()
    ]

rows = []

for feature in features:
    a = df.loc[df["subcluster"] == g0, feature]
    b = df.loc[df["subcluster"] == g1, feature]

    m0, q00, q01 = summarize(a)
    m1, q10, q11 = summarize(b)
    effect = smd(a, b)

    rows.append({
        "feature": feature,
        "cluster_a": str(g0),
        "a_median": m0,
        "a_q25": q00,
        "a_q75": q01,
        "cluster_b": str(g1),
        "b_median": m1,
        "b_q25": q10,
        "b_q75": q11,
        "smd_b_minus_a": effect,
        "abs_smd": abs(effect) if np.isfinite(effect) else np.nan,
    })

out = pd.DataFrame(rows).sort_values("abs_smd", ascending=False)
out.to_csv(OUT / "c0_subclusters_original_values.csv", index=False)

print("\n=== MAIORES DIFERENÇAS ===")
for _, r in out.head(25).iterrows():
    print(
        f"{r['feature']:<32} "
        f"{g0}={r['a_median']:.2f} | "
        f"{g1}={r['b_median']:.2f} | "
        f"SMD={r['smd_b_minus_a']:+.3f}"
    )

print("\n[OK] Caracterização concluída.")
print(OUT / "c0_subclusters_original_values.csv")
