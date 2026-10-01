#!/usr/bin/env python3
"""
07_subcluster_macrophenotypes.py

Subclusterização não supervisionada dentro dos dois macrophenotypes
obtidos por PCA80 + K-means k=2 no snapshot do primeiro onset.

Fluxo:
    15.739 pacientes
        -> parent cluster C0 / C1
        -> subset de cada parent
        -> re-padronização local das mesmas 20 features
        -> K-means k=2 e k=3 dentro de cada parent

Objetivo:
identificar subfenótipos locais sem usar rótulos clínicos/outcomes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.preprocessing import StandardScaler


def safe_silhouette(X, labels, sample_size, seed):
    n = len(X)
    size = min(sample_size, n)
    return float(
        silhouette_score(
            X,
            labels,
            sample_size=size if size < n else None,
            random_state=seed,
        )
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--meta", required=True, type=Path)
    ap.add_argument("--parent-assignments", required=True, type=Path)
    ap.add_argument("--features-csv", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--k-values", nargs="+", type=int, default=[2, 3])
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--n-init", type=int, default=50)
    ap.add_argument("--max-iter", type=int, default=500)
    ap.add_argument("--silhouette-sample", type=int, default=15000)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(args.dataset)
    meta = pd.read_parquet(args.meta)
    parent = pd.read_parquet(args.parent_assignments)
    features = pd.read_csv(args.features_csv)

    if X.ndim != 2:
        raise SystemExit(f"Dataset precisa ser 2D; shape={X.shape}")
    if len(meta) != X.shape[0]:
        raise SystemExit("Meta e dataset têm N diferente.")
    if "stay_id" not in meta.columns:
        raise SystemExit("Meta sem stay_id.")
    if "stay_id" not in parent.columns or "cluster" not in parent.columns:
        raise SystemExit("Parent assignments precisa ter stay_id e cluster.")
    if meta["stay_id"].duplicated().any():
        raise SystemExit("Meta contém stay_id duplicado.")
    if parent["stay_id"].duplicated().any():
        raise SystemExit("Parent assignments contém stay_id duplicado.")
    if np.isnan(X).any() or np.isinf(X).any():
        raise SystemExit("Dataset contém NaN/Inf.")
    if "feature" not in features.columns:
        raise SystemExit("features-csv precisa conter coluna 'feature'.")

    feature_names = features["feature"].tolist()
    if len(feature_names) != X.shape[1]:
        raise SystemExit(
            f"N features ({len(feature_names)}) != X.shape[1] ({X.shape[1]})."
        )

    aligned = meta[["stay_id"]].merge(
        parent[["stay_id", "cluster"]],
        on="stay_id",
        how="left",
        validate="one_to_one",
    )
    if aligned["cluster"].isna().any():
        raise SystemExit("Há pacientes sem parent cluster.")
    aligned["cluster"] = aligned["cluster"].astype(int)

    parent_values = sorted(aligned["cluster"].unique().tolist())
    if parent_values != [0, 1]:
        raise SystemExit(f"Esperava parent clusters [0,1], encontrei {parent_values}")

    metrics_rows = []
    size_rows = []

    print("=== SUBCLUSTERIZAÇÃO DOS MACROPHENOTYPES ===")
    print(f"N total: {len(aligned):,}")
    print(f"Features: {X.shape[1]}")
    print(f"k testados dentro de cada parent: {args.k_values}")
    print("Supervisão clínica/outcome: NÃO")
    print()

    for parent_cluster in parent_values:
        mask = aligned["cluster"].to_numpy() == parent_cluster
        idx = np.flatnonzero(mask)

        X_parent_global = X[idx].astype(np.float64)
        meta_parent = meta.iloc[idx].reset_index(drop=True)

        print(f"===== PARENT C{parent_cluster} =====")
        print(f"N={len(idx):,}")

        # Re-padronização LOCAL para focar nas diferenças internas ao parent.
        scaler = StandardScaler()
        X_local = scaler.fit_transform(X_parent_global).astype(np.float32)

        np.savez(
            args.output_dir / f"local_scaling_parentC{parent_cluster}.npz",
            mean=scaler.mean_.astype(np.float32),
            scale=scaler.scale_.astype(np.float32),
            feature_names=np.array(feature_names, dtype=object),
        )

        for k in args.k_values:
            km = KMeans(
                n_clusters=k,
                init="k-means++",
                n_init=args.n_init,
                max_iter=args.max_iter,
                random_state=args.seed,
                algorithm="lloyd",
            )
            labels = km.fit_predict(X_local)

            sil = safe_silhouette(
                X_local, labels, args.silhouette_sample, args.seed
            )
            ch = float(calinski_harabasz_score(X_local, labels))
            db = float(davies_bouldin_score(X_local, labels))

            counts = pd.Series(labels).value_counts().sort_index()
            min_pct = float(100.0 * counts.min() / len(labels))

            metrics_rows.append({
                "parent_cluster": parent_cluster,
                "parent_n": len(idx),
                "sub_k": k,
                "silhouette": sil,
                "calinski_harabasz": ch,
                "davies_bouldin": db,
                "inertia": float(km.inertia_),
                "min_subcluster_pct_within_parent": min_pct,
            })

            for subcluster, n in counts.items():
                size_rows.append({
                    "parent_cluster": parent_cluster,
                    "sub_k": k,
                    "subcluster": int(subcluster),
                    "n": int(n),
                    "pct_within_parent": float(100.0 * n / len(labels)),
                    "pct_total_cohort": float(100.0 * n / len(aligned)),
                })

            out = meta_parent.copy()
            out["parent_cluster"] = parent_cluster
            out["subcluster"] = labels.astype(int)
            out["subgroup"] = [
                f"C{parent_cluster}-{chr(65 + int(x))}" for x in labels
            ]
            out.to_parquet(
                args.output_dir
                / f"assignments_parentC{parent_cluster}_k{k}.parquet",
                index=False,
            )

            print(
                f"k={k} | silhouette={sil:.6f} | "
                f"CH={ch:.1f} | DB={db:.4f} | "
                f"menor subgrupo={min_pct:.2f}%"
            )

            if k == 2:
                rows = []
                for j, feature in enumerate(feature_names):
                    a = X_local[labels == 0, j]
                    b = X_local[labels == 1, j]
                    m0 = float(np.mean(a))
                    m1 = float(np.mean(b))
                    v0 = float(np.var(a, ddof=1))
                    v1 = float(np.var(b, ddof=1))
                    pooled = np.sqrt((v0 + v1) / 2.0)
                    smd = np.nan if pooled == 0 else (m1 - m0) / pooled

                    rows.append({
                        "feature": feature,
                        "mean_local_scaled_A": m0,
                        "mean_local_scaled_B": m1,
                        "smd_B_minus_A": smd,
                        "abs_smd": abs(smd) if pd.notna(smd) else np.nan,
                    })

                (
                    pd.DataFrame(rows)
                    .sort_values("abs_smd", ascending=False)
                    .to_csv(
                        args.output_dir
                        / f"top_feature_differences_parentC{parent_cluster}_k2.csv",
                        index=False,
                    )
                )

        print()

    metrics = pd.DataFrame(metrics_rows)
    sizes = pd.DataFrame(size_rows)

    metrics.to_csv(args.output_dir / "subcluster_metrics.csv", index=False)
    sizes.to_csv(args.output_dir / "subcluster_sizes.csv", index=False)

    print("=== MÉTRICAS ===")
    print(metrics.to_string(index=False))
    print("\n[OK] Outputs em:", args.output_dir)


if __name__ == "__main__":
    main()
