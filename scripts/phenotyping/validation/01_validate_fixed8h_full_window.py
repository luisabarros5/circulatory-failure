#!/usr/bin/env python3
"""
28_validate_bonito_full8h.py

Sensitivity analysis for the "Bonito" onset8h solution.

Question:
Does the k=3 structure remain when we keep ONLY patients with all 96 real
time positions available (i.e. no structural left-padding caused by onset
before 8h)?

Important:
- This is NOT a replacement cohort.
- The primary cohort remains all 15,739 positive stays.
- This is only a sensitivity analysis to isolate the structural padding issue.
- It re-runs MiniRocket on the full-window subset, rather than merely subsetting
  the old embedding, so MiniRocket fitting is also restricted to that subset.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score


def run(cmd):
    print("$", " ".join(shlex.quote(str(x)) for x in cmd), flush=True)
    subprocess.run(cmd, check=True)


def copy_subset_memmap(src_path: Path, dst_path: Path, idx: np.ndarray, chunk: int = 128):
    src = np.load(src_path, mmap_mode="r")
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst = np.lib.format.open_memmap(
        dst_path, mode="w+", dtype=src.dtype,
        shape=(len(idx), src.shape[1], src.shape[2]),
    )
    for start in range(0, len(idx), chunk):
        end = min(start + chunk, len(idx))
        dst[start:end] = src[idx[start:end]]
        if start % (chunk * 10) == 0:
            print(f"[COPY] {end:,}/{len(idx):,}", flush=True)
    dst.flush()
    del dst
    return src.shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=Path,
                    help="Bonito full 15,739 onset8h tensor")
    ap.add_argument("--meta", required=True, type=Path,
                    help="Bonito meta containing stay_id and length")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--original-k3", type=Path, default=None,
                    help="Optional original all-cohort k3 assignments for ARI on the same full-window stays")
    ap.add_argument("--original-metrics", type=Path, default=None,
                    help="Optional original k_sweep_metrics.csv for side-by-side comparison")
    ap.add_argument("--n-kernels", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--copy-chunk", type=int, default=128)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_parquet(args.meta).reset_index(drop=True)
    if "length" not in meta.columns:
        raise SystemExit("Meta precisa conter coluna 'length'.")
    if "stay_id" not in meta.columns:
        raise SystemExit("Meta precisa conter stay_id.")

    X = np.load(args.dataset, mmap_mode="r")
    if len(meta) != len(X):
        raise SystemExit(f"N meta={len(meta)} != N dataset={len(X)}")

    full_mask = pd.to_numeric(meta["length"], errors="coerce").eq(X.shape[1]).to_numpy()
    idx = np.flatnonzero(full_mask)
    full_meta = meta.loc[full_mask].reset_index(drop=True)

    print("=== BONITO FULL-8H SENSITIVITY ===")
    print(f"Total cohort: {len(meta):,}")
    print(f"Full 96-position window: {len(idx):,} ({100*len(idx)/len(meta):.2f}%)")
    print(f"Excluded only for sensitivity: {len(meta)-len(idx):,}")

    subset_ts = args.out_dir / "bonito_full8h_ts.npy"
    subset_meta = args.out_dir / "bonito_full8h_meta.parquet"

    if not subset_ts.exists():
        shape = copy_subset_memmap(args.dataset, subset_ts, idx, chunk=args.copy_chunk)
        print("[OK] subset tensor:", subset_ts)
        print("[INFO] source shape:", shape)
    else:
        existing = np.load(subset_ts, mmap_mode="r")
        if len(existing) != len(idx):
            raise RuntimeError("Subset tensor já existe com N incompatível; remova-o e rerode.")
        print("[SKIP] subset tensor já existe:", subset_ts)

    full_meta.to_parquet(subset_meta, index=False)

    bundle = Path(__file__).resolve().parents[1] / "core"
    embed_py = bundle / "embed_minirocket.py"
    cluster_py = bundle / "cluster_minirocket.py"
    for p in [embed_py, cluster_py]:
        if not p.exists():
            raise SystemExit(f"Não encontrado: {p}")

    emb = args.out_dir / "bonito_full8h_minirocket.npy"
    emb_meta = args.out_dir / "bonito_full8h_minirocket_meta.parquet"
    cluster_dir = args.out_dir / "clusters_kmeans"

    run([
        sys.executable, str(embed_py),
        "--dataset", str(subset_ts),
        "--meta_in", str(subset_meta),
        "--emb_out", str(emb),
        "--meta_out", str(emb_meta),
        "--n_kernels", str(args.n_kernels),
        "--fit_stays", "50000",
        "--batch_size", "128",
        "--seed", str(args.seed),
        "--log_path", str(args.out_dir / "01_minirocket.log"),
    ])

    run([
        sys.executable, str(cluster_py),
        "--embeddings", str(emb),
        "--meta", str(emb_meta),
        "--output_dir", str(cluster_dir),
        "--clusterer", "kmeans",
        "--k_min", "2",
        "--k_max", "6",
        "--kmeans_epochs", "10",
        "--kmeans_batch", "4096",
        "--kmeans_n_init", "5",
        "--pca_dim", "50",
        "--pca_sample", "100000",
        "--silhouette_sample", str(len(idx)),
        "--save_all_summaries",
        "--log_path", str(args.out_dir / "02_kmeans.log"),
    ])

    new_metrics = pd.read_csv(cluster_dir / "k_sweep_metrics.csv")
    new_metrics.to_csv(args.out_dir / "full8h_k_sweep_metrics.csv", index=False)

    summary = {
        "n_total_primary_cohort": int(len(meta)),
        "n_full_8h_sensitivity": int(len(idx)),
        "pct_full_8h": float(100 * len(idx) / len(meta)),
    }

    k3 = new_metrics.loc[new_metrics["k"] == 3]
    if len(k3):
        r = k3.iloc[0]
        summary.update({
            "full8h_k3_silhouette": float(r["silhouette"]),
            "full8h_k3_calinski_harabasz": float(r["calinski_harabasz"]),
            "full8h_k3_davies_bouldin": float(r["davies_bouldin"]),
        })

    # Compare cluster structure with original Bonito k3, restricted to same stays.
    if args.original_k3 and args.original_k3.exists():
        orig = pd.read_parquet(args.original_k3)[["stay_id", "cluster"]].rename(
            columns={"cluster": "cluster_original"}
        )
        new = pd.read_parquet(
            cluster_dir / "cluster_embeddings_kmeans_k3.parquet"
        )[["stay_id", "cluster"]].rename(columns={"cluster": "cluster_full8h"})
        d = new.merge(orig, on="stay_id", how="inner", validate="one_to_one")
        ari = adjusted_rand_score(d["cluster_original"], d["cluster_full8h"])
        summary["ari_original_vs_full8h_k3_on_same_stays"] = float(ari)
        pd.crosstab(
            d["cluster_original"], d["cluster_full8h"]
        ).to_csv(args.out_dir / "original_vs_full8h_k3_crosstab.csv")

    if args.original_metrics and args.original_metrics.exists():
        old = pd.read_csv(args.original_metrics)
        old3 = old.loc[old["k"] == 3]
        if len(old3):
            r = old3.iloc[0]
            summary["original_all15739_k3_silhouette"] = float(r["silhouette"])
            summary["silhouette_delta_full8h_minus_original"] = (
                summary.get("full8h_k3_silhouette", np.nan) - float(r["silhouette"])
            )

    (args.out_dir / "SUMMARY.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\n=== RESULTADO PRINCIPAL ===")
    print(new_metrics.to_string(index=False))
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print("\n[OK]", args.out_dir)


if __name__ == "__main__":
    main()
