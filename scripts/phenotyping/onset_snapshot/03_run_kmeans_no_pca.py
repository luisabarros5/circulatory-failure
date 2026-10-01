#!/usr/bin/env python3
"""
05_run_kmeans_snapshot.py

K-means sobre o snapshot clínico padronizado no primeiro onset.

Fluxo:
snapshot no onset -> StandardScaler (já feito no step 04) -> K-means

Testa k=2..6 e salva:
- métricas;
- tamanho dos clusters;
- labels por stay para cada k;
- centroides padronizados.

Não usa MiniRocket.
Não usa PCA.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--meta", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=6)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--n-init", type=int, default=50)
    ap.add_argument("--max-iter", type=int, default=500)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(args.dataset, mmap_mode="r")
    meta = pd.read_parquet(args.meta)

    if len(meta) != X.shape[0]:
        raise SystemExit(
            f"Meta ({len(meta)}) e matriz ({X.shape[0]}) têm N diferente."
        )
    if np.isnan(X).any():
        raise SystemExit("Dataset contém NaNs.")
    if X.ndim != 2:
        raise SystemExit(f"Esperava matriz 2D; shape={X.shape}")

    print("=== K-MEANS NO SNAPSHOT DO PRIMEIRO ONSET ===")
    print(f"N pacientes: {X.shape[0]:,}")
    print(f"N features:  {X.shape[1]}")
    print(f"k:           {args.k_min}..{args.k_max}")
    print("MiniRocket:  NÃO")
    print("PCA:         NÃO")
    print("Dados:       já padronizados")

    metrics_rows = []
    summary_rows = []

    for k in range(args.k_min, args.k_max + 1):
        print(f"\n[INFO] Rodando K-means k={k}...")

        model = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=args.n_init,
            max_iter=args.max_iter,
            random_state=args.seed,
            algorithm="lloyd",
        )
        labels = model.fit_predict(X)

        sil = silhouette_score(
            X,
            labels,
            sample_size=min(15000, X.shape[0]),
            random_state=args.seed,
        )
        ch = calinski_harabasz_score(X, labels)
        db = davies_bouldin_score(X, labels)

        metrics_rows.append({
            "k": k,
            "silhouette": sil,
            "calinski_harabasz": ch,
            "davies_bouldin": db,
            "inertia": float(model.inertia_),
            "n_iter": int(model.n_iter_),
        })

        out = meta.copy()
        out["cluster"] = labels.astype(int)
        out.to_parquet(
            args.output_dir / f"cluster_assignments_k{k}.parquet",
            index=False,
        )

        counts = pd.Series(labels).value_counts().sort_index()
        for c, n in counts.items():
            summary_rows.append({
                "k": k,
                "cluster": int(c),
                "n": int(n),
                "pct": 100.0 * int(n) / len(labels),
            })

        np.save(
            args.output_dir / f"cluster_centers_scaled_k{k}.npy",
            model.cluster_centers_.astype("float32"),
        )

        print(
            f"k={k} | silhouette={sil:.4f} | "
            f"CH={ch:.1f} | DB={db:.4f}"
        )

    metrics = pd.DataFrame(metrics_rows)
    summary = pd.DataFrame(summary_rows)

    metrics.to_csv(
        args.output_dir / "k_sweep_metrics.csv",
        index=False,
    )
    summary.to_csv(
        args.output_dir / "cluster_summary_all.csv",
        index=False,
    )

    best_sil = metrics.loc[metrics["silhouette"].idxmax()]
    best_ch = metrics.loc[metrics["calinski_harabasz"].idxmax()]
    best_db = metrics.loc[metrics["davies_bouldin"].idxmin()]

    decision = {
        "best_silhouette_k": int(best_sil["k"]),
        "best_silhouette": float(best_sil["silhouette"]),
        "best_calinski_harabasz_k": int(best_ch["k"]),
        "best_calinski_harabasz": float(best_ch["calinski_harabasz"]),
        "best_davies_bouldin_k": int(best_db["k"]),
        "best_davies_bouldin": float(best_db["davies_bouldin"]),
    }
    (args.output_dir / "metric_winners.json").write_text(
        json.dumps(decision, indent=2),
        encoding="utf-8",
    )

    print("\n=== MÉTRICAS ===")
    print(metrics.to_string(index=False))
    print("\n=== TAMANHOS ===")
    print(summary.to_string(index=False))
    print("\n[OK] Outputs em:", args.output_dir)


if __name__ == "__main__":
    main()
