#!/usr/bin/env python3
"""
24_characterize_C_k3_for_clinical_review.py

Caracterização clínica do Experimento C, k=3.

Entradas:
- assignments do C k=3
- cubo temporal normalizado [N,96,20]
- meta do C
- onset_snapshot_raw.parquet (valores em escala clínica/original)

Saídas:
- cluster_sizes.csv
- onset_clinical_table.csv
- onset_axis_ranking.csv
- temporal_axis_ranking.csv
- TOP_AXES.txt

Objetivo:
1) mostrar como os 3 clusters se dividem;
2) descrever os grupos em unidades clínicas no onset;
3) identificar quais eixos temporais mais diferenciam os clusters.

IMPORTANTE:
O ranking temporal usa o dataset pré-processado/normalizado usado no clustering.
Serve para identificar EIXOS discriminativos, não para relatar valores clínicos absolutos.
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = [
    "heart_rate",
    "mbp",
    "resp_rate",
    "temperature",
    "spo2",
    "glucose",
    "lab_50813",  # lactate
    "lab_50983",  # sodium
    "lab_50971",  # potassium
    "lab_50912",  # creatinine
    "lab_51006",  # BUN
    "lab_50878",  # AST
    "lab_50861",  # ALT
    "lab_50885",  # total bilirubin
    "lab_50862",  # albumin
    "lab_51222",  # hemoglobin
    "lab_51265",  # platelets
    "lab_51274",  # PT
    "lab_51275",  # PTT
    "vasopressor_ativo",
]

PRETTY = {
    "heart_rate": "Heart rate",
    "sbp": "SBP",
    "dbp": "DBP",
    "mbp": "MAP",
    "resp_rate": "Respiratory rate",
    "temperature": "Temperature",
    "spo2": "SpO2",
    "glucose": "Glucose",
    "lab_50813": "Lactate",
    "lab_50983": "Sodium",
    "lab_50971": "Potassium",
    "lab_50912": "Creatinine",
    "lab_51006": "BUN",
    "lab_50878": "AST",
    "lab_50861": "ALT",
    "lab_50885": "Total bilirubin",
    "lab_50862": "Albumin",
    "lab_51222": "Hemoglobin",
    "lab_51265": "Platelets",
    "lab_51274": "PT",
    "lab_51275": "PTT",
    "vasopressor_ativo": "Active vasopressor",
}

CLINICAL_ORDER = [
    "vasopressor_ativo",
    "heart_rate", "sbp", "dbp", "mbp", "resp_rate", "temperature", "spo2",
    "lab_50813", "glucose", "lab_50983", "lab_50971",
    "lab_50912", "lab_51006",
    "lab_50878", "lab_50861", "lab_50885", "lab_50862",
    "lab_51222", "lab_51265", "lab_51274", "lab_51275",
]


def eta_squared(values, labels):
    v = np.asarray(values, dtype=float)
    y = np.asarray(labels)
    ok = np.isfinite(v)
    v, y = v[ok], y[ok]
    if len(v) < 2:
        return np.nan
    grand = v.mean()
    sst = ((v - grand) ** 2).sum()
    if sst <= 0:
        return 0.0
    ssb = 0.0
    for c in np.unique(y):
        x = v[y == c]
        ssb += len(x) * (x.mean() - grand) ** 2
    return float(ssb / sst)


def fmt_med_iqr(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0:
        return ""
    q1 = s.quantile(.25)
    med = s.median()
    q3 = s.quantile(.75)
    return f"{med:.3g} [{q1:.3g}–{q3:.3g}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assignments", required=True, type=Path)
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--meta", required=True, type=Path)
    ap.add_argument("--onset-raw", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    assign = pd.read_parquet(args.assignments)
    meta = pd.read_parquet(args.meta)
    raw = pd.read_parquet(args.onset_raw)
    X = np.load(args.dataset, mmap_mode="r")

    if "cluster" not in assign:
        raise SystemExit("assignments sem coluna cluster")
    if "stay_id" not in assign or "stay_id" not in meta or "stay_id" not in raw:
        raise SystemExit("stay_id ausente")
    if len(X) != len(meta):
        raise SystemExit(f"Dataset N={len(X)} != meta N={len(meta)}")
    if X.shape[2] != len(FEATURES):
        raise SystemExit(
            f"Esperava {len(FEATURES)} canais, dataset tem {X.shape[2]}"
        )

    # Labels na mesma ordem do meta/dataset.
    labels_df = meta[["stay_id"]].merge(
        assign[["stay_id", "cluster"]].drop_duplicates("stay_id"),
        on="stay_id",
        how="left",
        validate="one_to_one",
    )
    if labels_df["cluster"].isna().any():
        raise SystemExit("Há stays do meta sem cluster")
    y = labels_df["cluster"].to_numpy(int)

    if len(np.unique(y)) != 3:
        print(f"[AVISO] clusters encontrados: {np.unique(y)}")

    # ------------------------------------------------------------------
    # 1. Cluster sizes + onset timing
    # ------------------------------------------------------------------
    sizes = (
        meta[["stay_id", "onset_hours", "source_preonset_rows"]]
        .merge(assign[["stay_id", "cluster"]], on="stay_id", validate="one_to_one")
        .groupby("cluster")
        .agg(
            n=("stay_id", "size"),
            median_onset_hours=("onset_hours", "median"),
            q1_onset_hours=("onset_hours", lambda s: s.quantile(.25)),
            q3_onset_hours=("onset_hours", lambda s: s.quantile(.75)),
            median_source_rows=("source_preonset_rows", "median"),
        )
        .reset_index()
    )
    sizes["pct"] = 100 * sizes["n"] / sizes["n"].sum()
    sizes.to_csv(args.out_dir / "cluster_sizes.csv", index=False)

    # ------------------------------------------------------------------
    # 2. Clinical onset table in original clinical units
    # ------------------------------------------------------------------
    clinical = raw.merge(
        assign[["stay_id", "cluster"]].drop_duplicates("stay_id"),
        on="stay_id",
        how="inner",
        validate="one_to_one",
    )

    onset_rows = []
    onset_rank = []

    for var in CLINICAL_ORDER:
        if var not in clinical.columns:
            continue

        numeric = pd.to_numeric(clinical[var], errors="coerce")
        if numeric.notna().sum() == 0:
            continue

        row = {
            "variable": var,
            "label": PRETTY.get(var, var),
            "overall_n_available": int(numeric.notna().sum()),
            "overall_available_pct": float(100 * numeric.notna().mean()),
        }

        if var == "vasopressor_ativo":
            for c in sorted(clinical["cluster"].unique()):
                s = pd.to_numeric(
                    clinical.loc[clinical["cluster"] == c, var],
                    errors="coerce",
                )
                row[f"cluster_{c}"] = (
                    f"{100*s.mean():.1f}%"
                    if s.notna().any() else ""
                )
        else:
            for c in sorted(clinical["cluster"].unique()):
                row[f"cluster_{c}"] = fmt_med_iqr(
                    clinical.loc[clinical["cluster"] == c, var]
                )

        onset_rows.append(row)

        vals = pd.to_numeric(clinical[var], errors="coerce").to_numpy(float)
        eta = eta_squared(vals, clinical["cluster"].to_numpy(int))

        means = {}
        medians = {}
        for c in sorted(clinical["cluster"].unique()):
            s = pd.to_numeric(
                clinical.loc[clinical["cluster"] == c, var],
                errors="coerce",
            )
            means[f"mean_c{c}"] = float(s.mean()) if s.notna().any() else np.nan
            medians[f"median_c{c}"] = float(s.median()) if s.notna().any() else np.nan

        onset_rank.append({
            "variable": var,
            "label": PRETTY.get(var, var),
            "eta2_cluster": eta,
            "available_pct": float(100 * numeric.notna().mean()),
            **means,
            **medians,
        })

    onset_table = pd.DataFrame(onset_rows)
    onset_table.to_csv(args.out_dir / "onset_clinical_table.csv", index=False)

    onset_rank = pd.DataFrame(onset_rank).sort_values(
        "eta2_cluster", ascending=False
    )
    onset_rank.to_csv(args.out_dir / "onset_axis_ranking.csv", index=False)

    # ------------------------------------------------------------------
    # 3. Temporal axes on the actual clustering representation
    # ------------------------------------------------------------------
    # Dataset C is already preprocessed/scaled.
    # We summarize trajectory by:
    # early = first quartile, late = last quartile,
    # delta = late - early, mean, variability.
    Xf = np.asarray(X, dtype=np.float32)

    early = Xf[:, :24, :].mean(axis=1)
    late = Xf[:, -24:, :].mean(axis=1)
    delta = late - early
    mean = Xf.mean(axis=1)
    variability = Xf.std(axis=1)

    summaries = {
        "early_mean": early,
        "late_mean": late,
        "delta_late_minus_early": delta,
        "trajectory_mean": mean,
        "trajectory_sd": variability,
    }

    temporal_rows = []

    for measure, arr in summaries.items():
        for j, feature in enumerate(FEATURES):
            vals = arr[:, j]
            eta = eta_squared(vals, y)
            row = {
                "measure": measure,
                "feature": feature,
                "label": PRETTY.get(feature, feature),
                "eta2_cluster": eta,
            }
            for c in sorted(np.unique(y)):
                s = vals[y == c]
                row[f"median_c{c}"] = float(np.median(s))
                row[f"mean_c{c}"] = float(np.mean(s))
            temporal_rows.append(row)

    temporal = pd.DataFrame(temporal_rows).sort_values(
        "eta2_cluster", ascending=False
    )
    temporal.to_csv(
        args.out_dir / "temporal_axis_ranking.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # 4. Concise terminal/text summary
    # ------------------------------------------------------------------
    lines = []
    lines.append("EXPERIMENTO C — K=3 — CARACTERIZAÇÃO\n")
    lines.append("=== CLUSTER SIZES ===")
    lines.append(sizes.to_string(index=False))

    lines.append("\n\n=== TOP ONSET CLINICAL AXES BY ETA² ===")
    show_onset = onset_rank.head(12)
    lines.append(
        show_onset[
            ["label", "eta2_cluster", "available_pct"]
            + [c for c in show_onset.columns if c.startswith("median_c")]
        ].to_string(index=False)
    )

    lines.append("\n\n=== TOP TEMPORAL AXES BY ETA² ===")
    show_temp = temporal.head(20)
    lines.append(
        show_temp[
            ["measure", "label", "eta2_cluster"]
            + [c for c in show_temp.columns if c.startswith("median_c")]
        ].to_string(index=False)
    )

    text = "\n".join(lines)
    (args.out_dir / "TOP_AXES.txt").write_text(text, encoding="utf-8")

    print(text)
    print("\n\n[OK] Outputs:", args.out_dir)


if __name__ == "__main__":
    main()
