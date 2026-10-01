#!/usr/bin/env python3
"""
06_characterize_pca80_k2.py

Caracterização clínica do PCA80 + K-means k=2 no snapshot do primeiro onset.

Outputs:
- cluster_summary_k2.csv
- clinical_table_k2.csv
- clinical_stats_long_k2.csv
- top_feature_differences_k2.csv

A caracterização clínica usa valores na escala original do onset_snapshot_raw.
As diferenças padronizadas usam o dataset log-curated de 20 features.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


VARIABLES = [
    ("Vital signs", "heart_rate", "Heart rate", "bpm", 1),
    ("Vital signs", "sbp", "Systolic blood pressure", "mmHg", 1),
    ("Vital signs", "dbp", "Diastolic blood pressure", "mmHg", 1),
    ("Vital signs", "mbp", "Mean blood pressure", "mmHg", 1),
    ("Vital signs", "resp_rate", "Respiratory rate", "insp/min", 1),
    ("Vital signs", "temperature", "Temperature", "°C", 2),
    ("Vital signs", "spo2", "SpO2", "%", 1),

    ("Metabolic / perfusion", "lab_50813", "Lactate", "mmol/L", 2),
    ("Metabolic / perfusion", "glucose", "Glucose", "mg/dL", 1),
    ("Metabolic / perfusion", "lab_50983", "Sodium", "mmol/L", 1),
    ("Metabolic / perfusion", "lab_50971", "Potassium", "mmol/L", 2),

    ("Renal / hepatic", "lab_50912", "Creatinine", "mg/dL", 2),
    ("Renal / hepatic", "lab_51006", "BUN", "mg/dL", 1),
    ("Renal / hepatic", "lab_50878", "AST", "U/L", 1),
    ("Renal / hepatic", "lab_50861", "ALT", "U/L", 1),
    ("Renal / hepatic", "lab_50885", "Total bilirubin", "mg/dL", 2),
    ("Renal / hepatic", "lab_50862", "Albumin", "g/dL", 2),

    ("Hematologic / coagulation", "lab_51222", "Hemoglobin", "g/dL", 1),
    ("Hematologic / coagulation", "lab_51265", "Platelets", "10^3/µL", 1),
    ("Hematologic / coagulation", "lab_51274", "Prothrombin time", "s", 1),
    ("Hematologic / coagulation", "lab_51275", "Partial thromboplastin time", "s", 1),
]

SPARSE_WARNING_THRESHOLD_PCT = 5.0


def fmt(x: float, decimals: int) -> str:
    if pd.isna(x):
        return "NA"
    return f"{x:.{decimals}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-snapshot", required=True, type=Path)
    ap.add_argument("--assignments", required=True, type=Path)
    ap.add_argument("--coverage-file", required=True, type=Path)
    ap.add_argument("--scaled-dataset", required=True, type=Path)
    ap.add_argument("--scaled-meta", required=True, type=Path)
    ap.add_argument("--features-csv", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_parquet(args.raw_snapshot)
    ass = pd.read_parquet(args.assignments)
    meta = pd.read_parquet(args.scaled_meta)
    X = np.load(args.scaled_dataset)

    if raw["stay_id"].duplicated().any():
        raise SystemExit("raw snapshot contém stay_id duplicado.")
    if ass["stay_id"].duplicated().any():
        raise SystemExit("assignments contém stay_id duplicado.")
    if meta["stay_id"].duplicated().any():
        raise SystemExit("scaled-meta contém stay_id duplicado.")
    if len(meta) != X.shape[0]:
        raise SystemExit("scaled-meta e scaled-dataset têm N diferente.")

    d = raw.merge(
        ass[["stay_id", "cluster"]],
        on="stay_id",
        how="inner",
        validate="one_to_one",
    )

    if len(d) != len(raw):
        raise RuntimeError(
            f"Merge perdeu pacientes: raw={len(raw)} merged={len(d)}"
        )

    clusters = sorted(d["cluster"].unique().tolist())
    if clusters != [0, 1]:
        raise RuntimeError(f"Esperava clusters [0,1], encontrei {clusters}")

    # Cobertura observada no onset.
    cov = pd.read_csv(args.coverage_file)
    coverage_map = {}
    if {"feature", "observed_at_onset_pct"}.issubset(cov.columns):
        coverage_map = dict(zip(cov["feature"], cov["observed_at_onset_pct"]))

    # Resumo geral.
    summary_rows = []
    for c in clusters:
        g = d[d["cluster"] == c]
        onset_h = pd.to_numeric(g["first_positive_time"], errors="coerce") / 3600.0
        vaso = pd.to_numeric(g["vasopressor_ativo"], errors="coerce")

        summary_rows.append({
            "cluster": c,
            "n_stays": len(g),
            "pct_stays": 100.0 * len(g) / len(d),
            "onset_hours_median": onset_h.median(),
            "onset_hours_q25": onset_h.quantile(0.25),
            "onset_hours_q75": onset_h.quantile(0.75),
            "active_vasopressor_at_onset_pct": 100.0 * (vaso > 0).mean(),
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "cluster_summary_k2.csv", index=False)

    # Tabela clínica legível.
    wide_rows = [{
        "section": "Cohort",
        "variable": "N (ICU stays)",
        "unit": "",
        "overall_observed_pct": 100.0,
        "interpretation_warning": "",
        "C0": str(int((d["cluster"] == 0).sum())),
        "C1": str(int((d["cluster"] == 1).sum())),
    }]

    wide_rows.append({
        "section": "Cohort",
        "variable": "Active vasopressor at onset",
        "unit": "%",
        "overall_observed_pct": 100.0,
        "interpretation_warning": "",
        "C0": f"{100.0 * (pd.to_numeric(d.loc[d.cluster == 0, 'vasopressor_ativo'], errors='coerce') > 0).mean():.1f}",
        "C1": f"{100.0 * (pd.to_numeric(d.loc[d.cluster == 1, 'vasopressor_ativo'], errors='coerce') > 0).mean():.1f}",
    })

    long_rows = []

    for section, feature, display, unit, decimals in VARIABLES:
        if feature not in d.columns:
            continue

        overall_cov = coverage_map.get(feature, np.nan)
        warning = ""
        if pd.notna(overall_cov) and overall_cov < SPARSE_WARNING_THRESHOLD_PCT:
            warning = (
                f"very sparse observed-at-onset coverage "
                f"({overall_cov:.2f}%); interpret cautiously"
            )

        row = {
            "section": section,
            "variable": display,
            "unit": unit,
            "overall_observed_pct": overall_cov,
            "interpretation_warning": warning,
        }

        for c in clusters:
            s = pd.to_numeric(
                d.loc[d["cluster"] == c, feature],
                errors="coerce",
            ).dropna()

            med = float(s.median()) if len(s) else np.nan
            q1 = float(s.quantile(0.25)) if len(s) else np.nan
            q3 = float(s.quantile(0.75)) if len(s) else np.nan

            row[f"C{c}"] = (
                "NA"
                if pd.isna(med)
                else f"{fmt(med, decimals)} [{fmt(q1, decimals)}–{fmt(q3, decimals)}]"
            )

            long_rows.append({
                "cluster": c,
                "feature": feature,
                "variable": display,
                "unit": unit,
                "median": med,
                "q1": q1,
                "q3": q3,
                "overall_observed_at_onset_pct": overall_cov,
                "warning": warning,
            })

        wide_rows.append(row)

    pd.DataFrame(wide_rows).to_csv(
        args.output_dir / "clinical_table_k2.csv",
        index=False,
    )
    pd.DataFrame(long_rows).to_csv(
        args.output_dir / "clinical_stats_long_k2.csv",
        index=False,
    )

    # Diferenças nas 20 features preprocessadas usadas no clustering.
    fdf = pd.read_csv(args.features_csv)
    if "feature" not in fdf.columns:
        raise SystemExit("features-csv precisa conter coluna 'feature'.")

    feature_names = fdf["feature"].tolist()
    if len(feature_names) != X.shape[1]:
        raise RuntimeError(
            f"N features no CSV={len(feature_names)} != X.shape[1]={X.shape[1]}"
        )

    xmeta = meta[["stay_id"]].copy()
    xmeta["row_idx"] = np.arange(len(xmeta))
    aligned = xmeta.merge(
        ass[["stay_id", "cluster"]],
        on="stay_id",
        how="inner",
        validate="one_to_one",
    )

    if len(aligned) != len(meta):
        raise RuntimeError("Assignments e scaled-meta não cobrem a mesma coorte.")

    idx0 = aligned.loc[aligned["cluster"] == 0, "row_idx"].to_numpy()
    idx1 = aligned.loc[aligned["cluster"] == 1, "row_idx"].to_numpy()

    rows = []
    for j, feature in enumerate(feature_names):
        a = X[idx0, j]
        b = X[idx1, j]

        m0 = float(np.mean(a))
        m1 = float(np.mean(b))
        v0 = float(np.var(a, ddof=1))
        v1 = float(np.var(b, ddof=1))
        pooled = np.sqrt((v0 + v1) / 2.0)

        smd = np.nan if pooled == 0 else (m1 - m0) / pooled

        rows.append({
            "feature": feature,
            "mean_scaled_C0": m0,
            "mean_scaled_C1": m1,
            "smd_C1_minus_C0": smd,
            "abs_smd": abs(smd) if pd.notna(smd) else np.nan,
        })

    diffs = (
        pd.DataFrame(rows)
        .sort_values("abs_smd", ascending=False)
        .reset_index(drop=True)
    )
    diffs.to_csv(
        args.output_dir / "top_feature_differences_k2.csv",
        index=False,
    )

    print("=== CLUSTER SUMMARY ===")
    print(summary.to_string(index=False))

    print("\n=== TOP 15 FEATURES ===")
    print(diffs.head(15).to_string(index=False))

    print("\n[OK] Outputs em:", args.output_dir)


if __name__ == "__main__":
    main()
