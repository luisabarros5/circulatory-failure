#!/usr/bin/env python3
"""
08_characterize_k2_k3_phenotypes.py

Caracterização clínica dos candidatos k=2 e k=3 para a fenotipagem
de falência circulatória.

Usa:
- os 15.739 stays no primeiro onset;
- os valores clínicos ORIGINAIS do onset_snapshot_raw.parquet;
- os assignments produzidos pelo K-means log-curated.

Importante:
- os valores apresentados NÃO são log-transformados;
- NÃO são winsorizados;
- NÃO são padronizados;
- são apresentados na escala clínica original;
- a organização segue os eixos fisiológicos da Table 3 do trabalho anterior.

Outputs para k=2 e k=3:
- clinical_table_k{K}.csv        -> tabela pronta para leitura
- clinical_stats_long_k{K}.csv   -> estatísticas numéricas completas
- cluster_sizes_k{K}.csv
- k2_vs_k3_crosstab_counts.csv
- k2_vs_k3_crosstab_rowpct.csv

Também incorpora a cobertura observada global do arquivo
onset_snapshot_features.csv, quando disponível, para sinalizar biomarcadores
extremamente esparsos.

A cobertura é INFORMATIVA: o valor no onset_snapshot_raw corresponde ao estado
da feature usado pelo pipeline no onset, que pode incluir preenchimento causal
do pipeline temporal. Portanto, a tabela não deve ser descrita como
"observed measurements only" sem uma extração específica dos valores crus.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


VARIABLES = [
    # section, feature, display, unit, decimals
    ("Vital signs", "heart_rate", "Heart rate", "bpm", 1),
    ("Vital signs", "sbp", "Systolic blood pressure", "mmHg", 1),
    ("Vital signs", "dbp", "Diastolic blood pressure", "mmHg", 1),
    ("Vital signs", "mbp", "Mean blood pressure", "mmHg", 1),
    ("Vital signs", "resp_rate", "Respiratory rate", "insp/min", 1),
    ("Vital signs", "temperature", "Temperature", "°C", 2),
    ("Vital signs", "spo2", "SpO2", "%", 1),

    ("Metabolic / perfusion axis", "lab_50813", "Lactate", "mmol/L", 2),
    ("Metabolic / perfusion axis", "glucose", "Glucose", "mg/dL", 1),
    ("Metabolic / perfusion axis", "lab_50983", "Sodium", "mmol/L", 1),
    ("Metabolic / perfusion axis", "lab_50971", "Potassium", "mmol/L", 2),

    ("Cardiac / inflammatory axis", "lab_50963", "NT-proBNP", "pg/mL", 1),
    ("Cardiac / inflammatory axis", "lab_51003", "Troponin T", "ng/mL", 3),
    ("Cardiac / inflammatory axis", "lab_50910", "Creatine kinase", "U/L", 1),
    ("Cardiac / inflammatory axis", "lab_50924", "Ferritin", "ng/mL", 1),

    ("Renal / hepatic function", "lab_50912", "Creatinine", "mg/dL", 2),
    ("Renal / hepatic function", "lab_51006", "BUN", "mg/dL", 1),
    ("Renal / hepatic function", "lab_50878", "AST", "U/L", 1),
    ("Renal / hepatic function", "lab_50861", "ALT", "U/L", 1),
    ("Renal / hepatic function", "lab_50885", "Total bilirubin", "mg/dL", 2),
    ("Renal / hepatic function", "lab_50862", "Albumin", "g/dL", 2),

    ("Hematologic / coagulation", "lab_51222", "Hemoglobin", "g/dL", 1),
    ("Hematologic / coagulation", "lab_51265", "Platelets", "10^3/µL", 1),
    ("Hematologic / coagulation", "lab_51274", "Prothrombin time", "s", 1),
    ("Hematologic / coagulation", "lab_51275",
     "Partial thromboplastin time", "s", 1),
]

SPARSE_WARNING_THRESHOLD_PCT = 5.0


def fmt(x: float, decimals: int) -> str:
    if pd.isna(x):
        return "NA"
    return f"{x:.{decimals}f}"


def build_for_k(
    k: int,
    raw: pd.DataFrame,
    assignments: pd.DataFrame,
    coverage_map: dict[str, float],
    out_dir: Path,
):
    d = raw.merge(
        assignments[["stay_id", "cluster"]],
        on="stay_id",
        how="inner",
        validate="one_to_one",
    )

    if len(d) != len(raw):
        raise RuntimeError(
            f"k={k}: merge perdeu pacientes: raw={len(raw)} merged={len(d)}"
        )

    clusters = sorted(d["cluster"].unique().tolist())
    if clusters != list(range(k)):
        raise RuntimeError(f"k={k}: clusters inesperados: {clusters}")

    counts = (
        d["cluster"]
        .value_counts()
        .sort_index()
        .rename_axis("cluster")
        .reset_index(name="n")
    )
    counts["pct"] = 100.0 * counts["n"] / len(d)
    counts.to_csv(out_dir / f"cluster_sizes_k{k}.csv", index=False)

    # Wide human-readable table.
    wide_rows = []

    row = {
        "section": "Cohort",
        "variable": "N (ICU stays)",
        "unit": "",
        "overall_observed_pct": 100.0,
        "interpretation_warning": "",
    }
    for c in clusters:
        row[f"C{c+1}"] = str(int((d["cluster"] == c).sum()))
    wide_rows.append(row)

    # Vasopressor at onset.
    if "vasopressor_ativo" not in d.columns:
        raise RuntimeError("vasopressor_ativo ausente do snapshot.")

    row = {
        "section": "Cohort",
        "variable": "Active vasopressor at onset",
        "unit": "%",
        "overall_observed_pct": 100.0,
        "interpretation_warning": "",
    }
    for c in clusters:
        s = pd.to_numeric(
            d.loc[d["cluster"] == c, "vasopressor_ativo"],
            errors="coerce",
        )
        pct = 100.0 * (s > 0).mean()
        row[f"C{c+1}"] = f"{pct:.1f}"
    wide_rows.append(row)

    long_rows = []

    for section, feature, display, unit, decimals in VARIABLES:
        if feature not in d.columns:
            # Keep explicit NA row if a legacy-table variable is unavailable.
            row = {
                "section": section,
                "variable": display,
                "unit": unit,
                "overall_observed_pct": coverage_map.get(feature, np.nan),
                "interpretation_warning": "feature unavailable in onset snapshot",
            }
            for c in clusters:
                row[f"C{c+1}"] = "NA"
            wide_rows.append(row)
            continue

        overall_cov = coverage_map.get(feature, np.nan)
        warning = ""
        if pd.notna(overall_cov) and overall_cov < SPARSE_WARNING_THRESHOLD_PCT:
            warning = (
                f"very sparse observed-at-onset coverage "
                f"({overall_cov:.2f}%); do not use alone for phenotype naming"
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

            if len(s) == 0:
                med = q1 = q3 = np.nan
            else:
                med = float(s.median())
                q1 = float(s.quantile(0.25))
                q3 = float(s.quantile(0.75))

            row[f"C{c+1}"] = (
                "NA"
                if pd.isna(med)
                else f"{fmt(med, decimals)} [{fmt(q1, decimals)}–{fmt(q3, decimals)}]"
            )

            long_rows.append({
                "k": k,
                "cluster": c,
                "cluster_label": f"C{c+1}",
                "section": section,
                "feature": feature,
                "variable": display,
                "unit": unit,
                "n_nonmissing_feature_state": int(len(s)),
                "cluster_n": int((d["cluster"] == c).sum()),
                "median": med,
                "q1": q1,
                "q3": q3,
                "overall_observed_at_onset_pct": overall_cov,
                "sparse_warning": warning,
            })

        wide_rows.append(row)

    wide = pd.DataFrame(wide_rows)
    long = pd.DataFrame(long_rows)

    wide.to_csv(out_dir / f"clinical_table_k{k}.csv", index=False)
    long.to_csv(out_dir / f"clinical_stats_long_k{k}.csv", index=False)

    print(f"\n=================== K={k} ===================")
    print(counts.to_string(index=False))
    print()
    print(wide.to_string(index=False))

    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-snapshot", required=True, type=Path)
    ap.add_argument("--coverage-file", required=True, type=Path)
    ap.add_argument("--kmeans-dir", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_parquet(args.raw_snapshot)
    if raw["stay_id"].duplicated().any():
        raise SystemExit("raw snapshot contém stay_id duplicado.")

    if len(raw) != 15739:
        print(
            f"[AVISO] N do snapshot={len(raw):,}; esperado no projeto=15.739."
        )

    cov = pd.read_csv(args.coverage_file)
    coverage_map = {}
    if {"feature", "observed_at_onset_pct"}.issubset(cov.columns):
        coverage_map = dict(
            zip(cov["feature"], cov["observed_at_onset_pct"])
        )

    datasets = {}

    for k in (2, 3):
        apath = args.kmeans_dir / f"cluster_assignments_k{k}.parquet"
        if not apath.exists():
            raise FileNotFoundError(apath)

        assignments = pd.read_parquet(apath)

        if assignments["stay_id"].duplicated().any():
            raise RuntimeError(f"k={k}: assignments duplicados.")

        datasets[k] = build_for_k(
            k,
            raw,
            assignments,
            coverage_map,
            args.output_dir,
        )

    # Crosswalk k=2 -> k=3.
    a2 = pd.read_parquet(
        args.kmeans_dir / "cluster_assignments_k2.parquet"
    )[["stay_id", "cluster"]].rename(columns={"cluster": "cluster_k2"})

    a3 = pd.read_parquet(
        args.kmeans_dir / "cluster_assignments_k3.parquet"
    )[["stay_id", "cluster"]].rename(columns={"cluster": "cluster_k3"})

    cross = a2.merge(a3, on="stay_id", validate="one_to_one")

    counts = pd.crosstab(
        cross["cluster_k2"],
        cross["cluster_k3"],
    )
    rowpct = pd.crosstab(
        cross["cluster_k2"],
        cross["cluster_k3"],
        normalize="index",
    ) * 100.0

    counts.to_csv(args.output_dir / "k2_vs_k3_crosstab_counts.csv")
    rowpct.to_csv(args.output_dir / "k2_vs_k3_crosstab_rowpct.csv")

    print("\n================ K=2 vs K=3: COUNTS ================")
    print(counts.to_string())

    print("\n================ K=2 vs K=3: ROW % =================")
    print(rowpct.round(1).to_string())

    print("\n[OK] Caracterização concluída em:")
    print(args.output_dir)


if __name__ == "__main__":
    main()
