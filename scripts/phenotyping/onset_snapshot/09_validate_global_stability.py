#!/usr/bin/env python3
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

BASE = Path("outputs/mimic_cluster_circews_v2/onset_snapshot")
X_PATH = BASE / "log_curated/onset_snapshot_log_curated_scaled.npy"
META_PATH = BASE / "onset_snapshot_meta.parquet"
OUT = BASE / "validation_minimal"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [1, 7, 13, 23, 31, 43, 57, 71, 89, 101]

print("=== 09 | ESTABILIDADE GLOBAL PCA80 + K=2 ===")

if not X_PATH.exists():
    raise FileNotFoundError(f"Não encontrei: {X_PATH}")
if not META_PATH.exists():
    raise FileNotFoundError(f"Não encontrei: {META_PATH}")

X = np.load(X_PATH)
meta = pd.read_parquet(META_PATH)

if len(X) != len(meta):
    raise RuntimeError(f"X/meta desalinhados: {len(X)} vs {len(meta)}")
if not np.isfinite(X).all():
    raise RuntimeError("X contém NaN/Inf.")

print(f"X: {X.shape}")
print(f"Seeds: {SEEDS}")

pca = PCA(n_components=0.80, svd_solver="full")
Z = pca.fit_transform(X)

print(
    f"PCA80: {Z.shape[1]} componentes | "
    f"variância={pca.explained_variance_ratio_.sum():.4f}"
)

labels_by_seed = {}
metrics = []

for seed in SEEDS:
    km = KMeans(
        n_clusters=2,
        init="k-means++",
        n_init=50,
        max_iter=500,
        random_state=seed,
    )
    labels = km.fit_predict(Z)
    labels_by_seed[seed] = labels

    sil = silhouette_score(
        Z,
        labels,
        sample_size=min(5000, len(Z)),
        random_state=43,
    )

    counts = np.bincount(labels, minlength=2)

    metrics.append({
        "seed": seed,
        "silhouette": float(sil),
        "inertia": float(km.inertia_),
        "cluster0_n": int(counts[0]),
        "cluster1_n": int(counts[1]),
        "min_cluster_pct": float(100 * counts.min() / len(labels)),
    })

    print(f"seed={seed:3d} | sil={sil:.4f} | sizes={counts.tolist()}")

pairwise = []
for a, b in combinations(SEEDS, 2):
    la = labels_by_seed[a]
    lb = labels_by_seed[b]
    pairwise.append({
        "seed_a": a,
        "seed_b": b,
        "ARI": adjusted_rand_score(la, lb),
        "NMI": normalized_mutual_info_score(la, lb),
    })

metrics = pd.DataFrame(metrics)
pairwise = pd.DataFrame(pairwise)

metrics.to_csv(OUT / "global_stability_by_seed.csv", index=False)
pairwise.to_csv(OUT / "global_assignment_concordance.csv", index=False)

assign = pd.DataFrame({"stay_id": meta["stay_id"].to_numpy()})
for seed in SEEDS:
    assign[f"cluster_seed_{seed}"] = labels_by_seed[seed]

assign.to_parquet(OUT / "global_stability_assignments.parquet", index=False)

print("\n=== CONCORDÂNCIA ENTRE SEEDS ===")
print(f"ARI médio:   {pairwise['ARI'].mean():.4f}")
print(f"ARI mínimo:  {pairwise['ARI'].min():.4f}")
print(f"NMI médio:   {pairwise['NMI'].mean():.4f}")
print(f"NMI mínimo:  {pairwise['NMI'].min():.4f}")
print(f"Silhouette:  {metrics['silhouette'].mean():.4f} ± {metrics['silhouette'].std():.4f}")

print("\n[OK] Estabilidade global concluída.")
