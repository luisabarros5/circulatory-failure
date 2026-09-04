#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

VARIABLES = [
    ("Vital signs", "Heart rate", "bpm", "heart_rate"),
    ("Vital signs", "Systolic blood pressure", "mmHg", "sbp"),
    ("Vital signs", "Diastolic blood pressure", "mmHg", "dbp"),
    ("Vital signs", "Mean blood pressure", "mmHg", "mbp"),
    ("Vital signs", "Respiratory rate", "insp/min", "resp_rate"),
    ("Vital signs", "Temperature", "°C", "temperature"),
    ("Vital signs", "SpO2", "%", "spo2"),
    ("Metabolic / perfusion markers", "Lactate", "mmol/L", "lab_50813"),
    ("Metabolic / perfusion markers", "Glucose", "mg/dL", "glucose"),
    ("Metabolic / perfusion markers", "Sodium", "mmol/L", "lab_50983"),
    ("Metabolic / perfusion markers", "Potassium", "mmol/L", "lab_50971"),
    ("Cardiac / inflammatory markers", "NT-proBNP", "pg/mL", "lab_50963"),
    ("Cardiac / inflammatory markers", "Troponin I", "ng/mL", "lab_51002"),
    ("Cardiac / inflammatory markers", "Troponin T", "ng/mL", "lab_51003"),
    ("Cardiac / inflammatory markers", "Creatine kinase", "U/L", "lab_50910"),
    ("Cardiac / inflammatory markers", "Ferritin", "ng/mL", "lab_50924"),
    ("Renal / hepatic function", "Creatinine", "mg/dL", "lab_50912"),
    ("Renal / hepatic function", "BUN", "mg/dL", "lab_51006"),
    ("Renal / hepatic function", "AST", "U/L", "lab_50878"),
    ("Renal / hepatic function", "ALT", "U/L", "lab_50861"),
    ("Renal / hepatic function", "Total bilirubin", "mg/dL", "lab_50885"),
    ("Renal / hepatic function", "Albumin", "g/dL", "lab_50862"),
    ("Hematologic / coagulation", "Hemoglobin", "g/dL", "lab_51222"),
    ("Hematologic / coagulation", "Platelets", "10^3/uL", "lab_51265"),
    ("Hematologic / coagulation", "Prothrombin time", "s", "lab_51274"),
    ("Hematologic / coagulation", "Partial thromboplastin time", "s", "lab_51275"),
]

def paired_label_path(labels_dir: Path, feature_path: Path) -> Path:
    return labels_dir / f"{feature_path.stem}_labels.parquet"

def fmt_value(median, q1, q3):
    if pd.isna(median):
        return ""
    if abs(median) >= 1000:
        return f"{median:,.1f} [{q1:,.1f}–{q3:,.1f}]"
    if abs(median) >= 10:
        return f"{median:.1f} [{q1:.1f}–{q3:.1f}]"
    return f"{median:.2f} [{q1:.2f}–{q3:.2f}]"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-dir", required=True, type=Path)
    ap.add_argument("--labels-dir", required=True, type=Path)
    ap.add_argument("--clusters", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--k", required=True, type=int)
    ap.add_argument("--timesteps", type=int, default=96)
    ap.add_argument("--step-seconds", type=int, default=300)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    clusters = pd.read_parquet(args.clusters)
    required_cluster_cols = {"stay_id", "first_positive_time", "cluster"}
    missing = required_cluster_cols - set(clusters.columns)
    if missing:
        raise SystemExit(f"Cluster parquet sem colunas: {sorted(missing)}")

    clusters = clusters[["stay_id", "first_positive_time", "cluster"]].drop_duplicates("stay_id").copy()
    clusters["stay_id"] = clusters["stay_id"].astype("int64")
    clusters["cluster"] = clusters["cluster"].astype(int)

    if clusters["cluster"].nunique() != args.k:
        raise SystemExit(
            f"--k={args.k}, mas arquivo contém {clusters['cluster'].nunique()} clusters."
        )

    onset_map = dict(
        zip(
            clusters["stay_id"].tolist(),
            clusters["first_positive_time"].astype(float).tolist(),
        )
    )

    feature_paths = sorted(
        p for p in args.feature_dir.glob("batch_*.parquet")
        if not p.name.endswith("_labels.parquet")
    )
    if not feature_paths:
        raise SystemExit(f"Nenhum feature parquet em {args.feature_dir}")

    first_schema = set(pq.read_schema(feature_paths[0]).names)
    available_vars = [v for v in VARIABLES if v[3] in first_schema]
    missing_vars = [v[3] for v in VARIABLES if v[3] not in first_schema]

    print(f"[INFO] K={args.k}")
    print(f"[INFO] stays={len(clusters):,}")
    print(f"[INFO] variáveis disponíveis={len(available_vars)}/{len(VARIABLES)}")
    if missing_vars:
        print("[AVISO] variáveis ausentes:", ", ".join(missing_vars))

    sums = {}
    counts = {}
    vaso_any = {}

    selected_feature_cols = ["stay_id", "vasopressor_ativo"]
    for _, _, _, col in available_vars:
        selected_feature_cols.append(col)
        m = f"{col}_imputed"
        if m in first_schema:
            selected_feature_cols.append(m)
    selected_feature_cols = list(dict.fromkeys(selected_feature_cols))

    window_seconds = (args.timesteps - 1) * args.step_seconds
    total_rows_window = 0

    for idx, fp in enumerate(feature_paths, 1):
        lp = paired_label_path(args.labels_dir, fp)
        if not lp.exists():
            raise FileNotFoundError(f"Label pareado ausente: {lp}")

        fpf = pq.ParquetFile(fp)
        ldf = pd.read_parquet(lp, columns=["stay_id", "rel_charttime"])

        if len(ldf) != fpf.metadata.num_rows:
            raise RuntimeError(f"Row count incompatível: {fp.name}")

        sid_all = ldf["stay_id"].to_numpy(np.int64)
        rel_all = pd.to_numeric(ldf["rel_charttime"], errors="coerce").to_numpy(float)

        offset = 0
        for rg in range(fpf.num_row_groups):
            nrg = fpf.metadata.row_group(rg).num_rows
            sl = slice(offset, offset + nrg)
            sid_rg = sid_all[sl]
            rel_rg = rel_all[sl]

            onset_rg = np.fromiter(
                (onset_map.get(int(s), np.nan) for s in sid_rg),
                dtype=float,
                count=len(sid_rg),
            )
            delta = onset_rg - rel_rg
            mask_window = (
                np.isfinite(onset_rg)
                & np.isfinite(rel_rg)
                & (delta >= 0)
                & (delta <= window_seconds)
            )

            if mask_window.any():
                cols_here = [c for c in selected_feature_cols if c in fpf.schema_arrow.names]
                table = fpf.read_row_group(rg, columns=cols_here)
                take_idx = np.flatnonzero(mask_window)
                df = table.take(take_idx).to_pandas()

                sid_sel = sid_rg[take_idx]
                if not np.array_equal(df["stay_id"].to_numpy(np.int64), sid_sel):
                    raise RuntimeError(f"Desalinhamento features/labels: {fp.name} RG {rg}")

                df["stay_id"] = sid_sel
                total_rows_window += len(df)

                vaso = pd.to_numeric(df["vasopressor_ativo"], errors="coerce").fillna(0)
                temp_vaso = (
                    pd.DataFrame({"stay_id": df["stay_id"].to_numpy(), "vaso": vaso.to_numpy()})
                    .groupby("stay_id")["vaso"].max()
                )
                for sid, val in temp_vaso.items():
                    sid = int(sid)
                    vaso_any[sid] = max(vaso_any.get(sid, 0.0), float(val > 0))

                for _, _, _, col in available_vars:
                    vals = pd.to_numeric(df[col], errors="coerce")
                    mcol = f"{col}_imputed"
                    if mcol in df.columns:
                        imp = pd.to_numeric(df[mcol], errors="coerce").fillna(1)
                        valid = vals.notna() & (imp == 0)
                    else:
                        valid = vals.notna()

                    if not valid.any():
                        continue

                    tmp = pd.DataFrame({
                        "stay_id": df.loc[valid, "stay_id"].to_numpy(),
                        "value": vals.loc[valid].to_numpy(float),
                    })
                    agg = tmp.groupby("stay_id")["value"].agg(["sum", "count"])

                    for sid, row in agg.iterrows():
                        key = (int(sid), col)
                        sums[key] = sums.get(key, 0.0) + float(row["sum"])
                        counts[key] = counts.get(key, 0) + int(row["count"])

            offset += nrg

        if idx % 10 == 0 or idx == len(feature_paths):
            print(f"[INFO] {idx}/{len(feature_paths)} arquivos | linhas na janela={total_rows_window:,}")

    cluster_lookup = clusters.set_index("stay_id")["cluster"].to_dict()
    rows = []
    for sid in clusters["stay_id"]:
        row = {
            "stay_id": int(sid),
            "cluster": int(cluster_lookup[int(sid)]),
            "any_vasopressor": float(vaso_any.get(int(sid), 0.0)),
        }
        for _, _, _, col in available_vars:
            key = (int(sid), col)
            n = counts.get(key, 0)
            row[col] = sums[key] / n if n > 0 else np.nan
            row[f"{col}__n_observed"] = n
        rows.append(row)

    per_stay = pd.DataFrame(rows)
    per_stay_path = args.output_dir / f"per_stay_clinical_k{args.k}.parquet"
    per_stay.to_parquet(per_stay_path, index=False)

    cluster_counts = per_stay.groupby("cluster")["stay_id"].nunique().sort_index()
    vaso_pct = per_stay.groupby("cluster")["any_vasopressor"].mean().mul(100).sort_index()

    long_rows = []
    coverage_rows = []

    for section, name, unit, col in available_vars:
        for cluster, g in per_stay.groupby("cluster"):
            vals = pd.to_numeric(g[col], errors="coerce").dropna()
            n_total = len(g)
            n_available = len(vals)
            med = vals.median() if n_available else np.nan
            q1 = vals.quantile(0.25) if n_available else np.nan
            q3 = vals.quantile(0.75) if n_available else np.nan

            long_rows.append({
                "section": section,
                "variable": name,
                "unit": unit,
                "feature": col,
                "cluster": int(cluster),
                "median": med,
                "q1": q1,
                "q3": q3,
                "n_stays_observed": n_available,
                "coverage_pct": (100*n_available/n_total) if n_total else np.nan,
            })
            coverage_rows.append({
                "feature": col,
                "variable": name,
                "cluster": int(cluster),
                "n_total": n_total,
                "n_stays_observed": n_available,
                "coverage_pct": (100*n_available/n_total) if n_total else np.nan,
            })

    long_df = pd.DataFrame(long_rows)
    coverage_df = pd.DataFrame(coverage_rows)
    long_df.to_csv(args.output_dir / f"clinical_table_long_k{args.k}.csv", index=False)
    coverage_df.to_csv(args.output_dir / f"coverage_k{args.k}.csv", index=False)

    clusters_order = sorted(per_stay["cluster"].unique())
    wide_rows = [
        {
            "section": "Cohort", "variable": "N (ICU stays)", "unit": "",
            **{f"cluster_{c}": str(int(cluster_counts[c])) for c in clusters_order},
        },
        {
            "section": "Cohort", "variable": "Any vasopressor use", "unit": "%",
            **{f"cluster_{c}": f"{vaso_pct[c]:.1f}" for c in clusters_order},
        },
    ]

    for section, name, unit, col in available_vars:
        row = {"section": section, "variable": name, "unit": unit}
        sub = long_df[long_df["feature"] == col].set_index("cluster")
        for c in clusters_order:
            if c in sub.index:
                r = sub.loc[c]
                row[f"cluster_{c}"] = fmt_value(r["median"], r["q1"], r["q3"])
            else:
                row[f"cluster_{c}"] = ""
        wide_rows.append(row)

    wide = pd.DataFrame(wide_rows)
    wide_path = args.output_dir / f"clinical_table_k{args.k}.csv"
    wide.to_csv(wide_path, index=False)

    print("\n=== CLUSTER COUNTS ===")
    print(cluster_counts.to_string())
    print("\n=== ANY VASOPRESSOR (%) ===")
    print(vaso_pct.round(1).to_string())
    print("\n[OK] Gerados:")
    print(wide_path)
    print(args.output_dir / f"clinical_table_long_k{args.k}.csv")
    print(args.output_dir / f"coverage_k{args.k}.csv")
    print(per_stay_path)

if __name__ == "__main__":
    main()
