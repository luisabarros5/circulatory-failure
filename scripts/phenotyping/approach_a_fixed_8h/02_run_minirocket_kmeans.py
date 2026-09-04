#!/usr/bin/env python3
"""
04_run_minirocket_kmeans_onset.py

Executa MiniRocket + K-means sobre o tensor já alinhado ao onset.
Não reconstrói a janela temporal.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def run(cmd, dry):
    print("$", " ".join(shlex.quote(str(x)) for x in cmd), flush=True)
    if not dry:
        subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--meta", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--n-kernels", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=6)
    args = ap.parse_args()

    bundle = Path(__file__).resolve().parents[1] / "core"
    embed_py = bundle / "embed_minirocket.py"
    cluster_py = bundle / "cluster_minirocket.py"

    for p in [embed_py, cluster_py, args.dataset, args.meta]:
        if not p.exists():
            raise SystemExit(f"Não encontrado: {p}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    emb = args.out_dir / "mimic_onset8h_minirocket.npy"
    emb_meta = args.out_dir / "mimic_onset8h_minirocket_meta.parquet"
    cluster_dir = args.out_dir / "clusters_kmeans"

    embed_cmd = [
        sys.executable, str(embed_py),
        "--dataset", str(args.dataset),
        "--meta_in", str(args.meta),
        "--emb_out", str(emb),
        "--meta_out", str(emb_meta),
        "--n_kernels", str(args.n_kernels),
        "--fit_stays", "50000",
        "--batch_size", "128",
        "--seed", str(args.seed),
        "--log_path", str(args.out_dir / "04_minirocket.log"),
    ]

    cluster_cmd = [
        sys.executable, str(cluster_py),
        "--embeddings", str(emb),
        "--meta", str(emb_meta),
        "--output_dir", str(cluster_dir),
        "--clusterer", "kmeans",
        "--k_min", str(args.k_min),
        "--k_max", str(args.k_max),
        "--kmeans_epochs", "10",
        "--kmeans_batch", "4096",
        "--kmeans_n_init", "5",
        "--pca_dim", "50",
        "--pca_sample", "100000",
        "--silhouette_sample", "15739",
        "--save_all_summaries",
        "--log_path", str(args.out_dir / "05_kmeans.log"),
    ]

    run(embed_cmd, args.dry_run)
    run(cluster_cmd, args.dry_run)

    if not args.dry_run:
        print("\n[OK] MiniRocket + K-means concluídos.")
        print("Métricas:", cluster_dir / "k_sweep_metrics.csv")


if __name__ == "__main__":
    main()
