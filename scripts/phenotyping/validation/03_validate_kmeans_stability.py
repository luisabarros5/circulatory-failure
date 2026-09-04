#!/usr/bin/env python3
"""
30_validate_kmeans_stability.py

Seed stability validation for the two current candidate solutions:
- Bonito (onset8h)
- C (normalized pre-onset)

Uses the already-created MiniRocket embeddings.

For each approach:
embedding -> StandardScaler -> PCA50 (fixed, random_state=43)
-> MiniBatchKMeans k=3 with several random seeds

Outputs:
- metrics per seed
- assignments per seed
- pairwise Adjusted Rand Index (ARI)
- summary with mean/min ARI and silhouette mean/SD
- optional ARI versus the saved reference k=3 solution

The point is to test whether the 3-group solution depends strongly on
K-means initialization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.preprocessing import StandardScaler


def prepare_space(embedding_path: Path):
    emb = np.load(embedding_path)
    X = emb.astype(np.float32)
    X = StandardScaler().fit_transform(X)

    # Match the original cluster_minirocket.py behavior.
    pca = PCA(n_components=50, random_state=43)
    X = pca.fit_transform(X)
    return X.astype(np.float32)


def fit_one(X, seed, epochs=10, batch=4096, n_init=5):
    model = MiniBatchKMeans(
        n_clusters=3,
        batch_size=batch,
        n_init=n_init,
        random_state=seed,
    )
    n = len(X)
    for _ in range(epochs):
        for start in range(0, n, batch):
            model.partial_fit(X[start:min(start + batch, n)])
    labels = model.predict(X)
    return labels


def run_approach(name, embedding, meta_path, out_dir, seeds, reference=None):
    out = out_dir / name
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {name.upper()} ===")
    X = prepare_space(embedding)
    meta = pd.read_parquet(meta_path).reset_index(drop=True)
    if len(meta) != len(X):
        raise RuntimeError(f"{name}: N meta != N embedding")
    if "stay_id" not in meta.columns:
        raise RuntimeError(f"{name}: meta sem stay_id")

    labels_by_seed = {}
    metric_rows = []

    for seed in seeds:
        print(f"[{name}] seed={seed}", flush=True)
        labels = fit_one(X, seed)
        labels_by_seed[seed] = labels

        sil = silhouette_score(X, labels)
        ch = calinski_harabasz_score(X, labels)
        db = davies_bouldin_score(X, labels)
        uniq, cnt = np.unique(labels, return_counts=True)
        sizes = {int(c): int(n) for c, n in zip(uniq, cnt)}

        metric_rows.append({
            "approach": name,
            "seed": seed,
            "silhouette": float(sil),
            "calinski_harabasz": float(ch),
            "davies_bouldin": float(db),
            "min_cluster_n": int(cnt.min()),
            "max_cluster_n": int(cnt.max()),
            "cluster_sizes": json.dumps(sizes, sort_keys=True),
        })

        a = meta[["stay_id"]].copy()
        a["cluster"] = labels
        a.to_parquet(out / f"assignments_seed_{seed}.parquet", index=False)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out / "metrics_by_seed.csv", index=False)

    # Pairwise ARI.
    ari_rows = []
    for i, s1 in enumerate(seeds):
        for s2 in seeds[i + 1:]:
            ari_rows.append({
                "approach": name,
                "seed_1": s1,
                "seed_2": s2,
                "ari": float(adjusted_rand_score(labels_by_seed[s1], labels_by_seed[s2])),
            })
    aris = pd.DataFrame(ari_rows)
    aris.to_csv(out / "pairwise_ari.csv", index=False)

    summary = {
        "approach": name,
        "n": int(len(X)),
        "seeds": list(map(int, seeds)),
        "silhouette_mean": float(metrics["silhouette"].mean()),
        "silhouette_sd": float(metrics["silhouette"].std(ddof=1)),
        "silhouette_min": float(metrics["silhouette"].min()),
        "silhouette_max": float(metrics["silhouette"].max()),
        "pairwise_ari_mean": float(aris["ari"].mean()),
        "pairwise_ari_min": float(aris["ari"].min()),
        "pairwise_ari_max": float(aris["ari"].max()),
    }

    if reference and reference.exists():
        ref = pd.read_parquet(reference)[["stay_id", "cluster"]].rename(
            columns={"cluster": "reference_cluster"}
        )
        for seed in seeds:
            a = meta[["stay_id"]].copy()
            a["seed_cluster"] = labels_by_seed[seed]
            d = a.merge(ref, on="stay_id", how="inner", validate="one_to_one")
            summary[f"ari_seed_{seed}_vs_reference"] = float(
                adjusted_rand_score(d["seed_cluster"], d["reference_cluster"])
            )

    (out / "SUMMARY.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\nMETRICS")
    print(metrics.to_string(index=False))
    print("\nPAIRWISE ARI")
    print(aris.to_string(index=False))
    print("\nSUMMARY")
    print(json.dumps(summary, indent=2))

    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bonito-embedding", required=True, type=Path)
    ap.add_argument("--bonito-meta", required=True, type=Path)
    ap.add_argument("--bonito-reference", type=Path, default=None)
    ap.add_argument("--c-embedding", required=True, type=Path)
    ap.add_argument("--c-meta", required=True, type=Path)
    ap.add_argument("--c-reference", type=Path, default=None)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 43, 71, 101])
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    bonito = run_approach(
        "bonito",
        args.bonito_embedding,
        args.bonito_meta,
        args.out_dir,
        args.seeds,
        reference=args.bonito_reference,
    )
    c = run_approach(
        "C",
        args.c_embedding,
        args.c_meta,
        args.out_dir,
        args.seeds,
        reference=args.c_reference,
    )

    combined = pd.DataFrame([bonito, c])
    combined.to_csv(args.out_dir / "STABILITY_SUMMARY.csv", index=False)

    print("\n=== FINAL STABILITY COMPARISON ===")
    print(combined[[
        "approach", "n",
        "silhouette_mean", "silhouette_sd",
        "pairwise_ari_mean", "pairwise_ari_min"
    ]].to_string(index=False))
    print("\n[OK]", args.out_dir / "STABILITY_SUMMARY.csv")


if __name__ == "__main__":
    main()
