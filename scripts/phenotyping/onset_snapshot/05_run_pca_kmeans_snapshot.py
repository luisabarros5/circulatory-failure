#!/usr/bin/env python3
"""
05_run_pca_kmeans_snapshot.py

PCA + K-means sobre o snapshot clínico log-curated no primeiro onset.

Input esperado:
- onset_snapshot_log_curated_scaled.npy
- onset_snapshot_meta.parquet

Fluxo:
20 features clínicas já preprocessadas
    -> PCA preservando 80%, 90% e 95% da variância
    -> K-means k=2..6
    -> métricas no espaço PCA
    -> silhouette secundária no espaço original de 20 features
    -> assignments por stay

Não usa MiniRocket.

Outputs:
- pca_kmeans_metrics.csv
- cluster_sizes_all.csv
- cluster_assignments_PCA{80,90,95}_k{2..6}.parquet
- pca_PCA{80,90,95}.npz
- metric_winners.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)


DEFAULT_VARIANCE_TARGETS = [0.80, 0.90, 0.95]


def safe_silhouette(
    X: np.ndarray,
    labels: np.ndarray,
    sample_size: int,
    seed: int,
) -> float:
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
    ap.add_argument("--output-dir", required=True, type=Path)

    ap.add_argument(
        "--variance-targets",
        nargs="+",
        type=float,
        default=DEFAULT_VARIANCE_TARGETS,
        help="Frações de variância acumulada para PCA. Default: 0.80 0.90 0.95",
    )
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=6)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--n-init", type=int, default=50)
    ap.add_argument("--max-iter", type=int, default=500)
    ap.add_argument(
        "--silhouette-sample",
        type=int,
        default=15000,
        help="Amostra máxima usada no silhouette. Default: 15000",
    )
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(args.dataset)
    meta = pd.read_parquet(args.meta)

    if X.ndim != 2:
        raise SystemExit(f"Esperava matriz 2D; shape={X.shape}")
    if len(meta) != X.shape[0]:
        raise SystemExit(
            f"Meta ({len(meta)}) e dataset ({X.shape[0]}) têm N diferente."
        )
    if np.isnan(X).any() or np.isinf(X).any():
        raise SystemExit("Dataset contém NaN/Inf.")
    if "stay_id" not in meta.columns:
        raise SystemExit("Meta sem stay_id.")

    targets = sorted(set(args.variance_targets))
    for target in targets:
        if not (0.0 < target < 1.0):
            raise SystemExit(
                f"variance target inválido: {target}. Use valores entre 0 e 1."
            )

    print("=== SNAPSHOT ONSET: PCA + K-MEANS ===")
    print(f"N pacientes:      {X.shape[0]:,}")
    print(f"N features input: {X.shape[1]}")
    print(f"PCA targets:      {targets}")
    print(f"k:                {args.k_min}..{args.k_max}")
    print("MiniRocket:       NÃO")
    print()

    metrics_rows = []
    size_rows = []

    for target in targets:
        pct = int(round(target * 100))
        variant = f"PCA{pct}"

        print(f"===== {variant} =====")

        # n_components como fração exige svd_solver='full'.
        pca = PCA(
            n_components=target,
            svd_solver="full",
        )
        Z = pca.fit_transform(X).astype(np.float32)

        n_components = int(Z.shape[1])
        cumulative_variance = float(
            np.sum(pca.explained_variance_ratio_)
        )

        print(
            f"Componentes: {n_components} | "
            f"variância acumulada: {cumulative_variance:.6f}"
        )

        np.savez(
            args.output_dir / f"pca_{variant}.npz",
            components=pca.components_.astype(np.float32),
            explained_variance=pca.explained_variance_.astype(np.float32),
            explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
            mean=pca.mean_.astype(np.float32),
            n_components=np.array([n_components], dtype=np.int32),
            cumulative_variance=np.array([cumulative_variance], dtype=np.float64),
            variance_target=np.array([target], dtype=np.float64),
        )

        for k in range(args.k_min, args.k_max + 1):
            model = KMeans(
                n_clusters=k,
                init="k-means++",
                n_init=args.n_init,
                max_iter=args.max_iter,
                random_state=args.seed,
                algorithm="lloyd",
            )
            labels = model.fit_predict(Z)

            sil_pca = safe_silhouette(
                Z,
                labels,
                args.silhouette_sample,
                args.seed,
            )

            # Métrica secundária:
            # mesmos labels, mas avaliados no espaço original das 20 features.
            sil_original = safe_silhouette(
                X,
                labels,
                args.silhouette_sample,
                args.seed,
            )

            ch = float(calinski_harabasz_score(Z, labels))
            db = float(davies_bouldin_score(Z, labels))

            counts = pd.Series(labels).value_counts().sort_index()
            min_cluster_pct = float(100.0 * counts.min() / len(labels))

            metrics_rows.append(
                {
                    "variant": variant,
                    "variance_target": target,
                    "pca_components": n_components,
                    "pca_cumulative_variance": cumulative_variance,
                    "k": k,
                    "silhouette_pca": sil_pca,
                    "silhouette_original_20d": sil_original,
                    "calinski_harabasz": ch,
                    "davies_bouldin": db,
                    "inertia": float(model.inertia_),
                    "n_iter": int(model.n_iter_),
                    "min_cluster_pct": min_cluster_pct,
                }
            )

            for cluster, n in counts.items():
                size_rows.append(
                    {
                        "variant": variant,
                        "k": k,
                        "cluster": int(cluster),
                        "n": int(n),
                        "pct": float(100.0 * n / len(labels)),
                    }
                )

            out = meta.copy()
            out["cluster"] = labels.astype(int)
            out.to_parquet(
                args.output_dir
                / f"cluster_assignments_{variant}_k{k}.parquet",
                index=False,
            )

            np.save(
                args.output_dir
                / f"cluster_centers_{variant}_k{k}.npy",
                model.cluster_centers_.astype(np.float32),
            )

            print(
                f"k={k} | "
                f"sil_PCA={sil_pca:.6f} | "
                f"sil_20D={sil_original:.6f} | "
                f"CH={ch:.1f} | "
                f"DB={db:.4f} | "
                f"min_cluster={min_cluster_pct:.2f}%"
            )

        print()

    metrics = pd.DataFrame(metrics_rows)
    sizes = pd.DataFrame(size_rows)

    metrics_path = args.output_dir / "pca_kmeans_metrics.csv"
    sizes_path = args.output_dir / "cluster_sizes_all.csv"

    metrics.to_csv(metrics_path, index=False)
    sizes.to_csv(sizes_path, index=False)

    best_sil = metrics.loc[metrics["silhouette_pca"].idxmax()]
    best_ch = metrics.loc[metrics["calinski_harabasz"].idxmax()]
    best_db = metrics.loc[metrics["davies_bouldin"].idxmin()]

    winners = {
        "best_silhouette": {
            "variant": str(best_sil["variant"]),
            "k": int(best_sil["k"]),
            "score": float(best_sil["silhouette_pca"]),
        },
        "best_calinski_harabasz": {
            "variant": str(best_ch["variant"]),
            "k": int(best_ch["k"]),
            "score": float(best_ch["calinski_harabasz"]),
        },
        "best_davies_bouldin": {
            "variant": str(best_db["variant"]),
            "k": int(best_db["k"]),
            "score": float(best_db["davies_bouldin"]),
        },
    }

    (args.output_dir / "metric_winners.json").write_text(
        json.dumps(winners, indent=2),
        encoding="utf-8",
    )

    print("=== RESULTADOS COMPLETOS ===")
    print(
        metrics[
            [
                "variant",
                "k",
                "pca_components",
                "pca_cumulative_variance",
                "silhouette_pca",
                "silhouette_original_20d",
                "calinski_harabasz",
                "davies_bouldin",
                "min_cluster_pct",
            ]
        ].to_string(index=False)
    )

    print()
    print("[OK] Outputs:")
    print(metrics_path)
    print(sizes_path)
    print(args.output_dir / "metric_winners.json")


if __name__ == "__main__":
    main()
