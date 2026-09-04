from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans
from sklearn.decomposition import PCA

from common import setup_logger, load_engineered_features


def _safe_silhouette(X: np.ndarray, labels: np.ndarray, sample: int | None):
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return float("nan")
    if sample is not None and len(X) > sample:
        rng = np.random.default_rng(43)
        idx = rng.choice(len(X), size=sample, replace=False)
        return float(silhouette_score(X[idx], labels[idx]))
    return float(silhouette_score(X, labels))


def _cluster_metrics(X: np.ndarray, labels: np.ndarray, sample: int | None):
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return {
            "silhouette": float("nan"),
            "calinski_harabasz": float("nan"),
            "davies_bouldin": float("nan"),
            "n_clusters": int(len(uniq)),
        }
    return {
        "silhouette": _safe_silhouette(X, labels, sample),
        "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
        "n_clusters": int(len(uniq)),
    }


def _build_summary(df: pd.DataFrame, cluster_col: str, label_cols: list[str]) -> pd.DataFrame:
    agg = {"cluster": "size"}
    for col in label_cols:
        if col in df.columns:
            agg[col] = "mean"
    summary = df.groupby(cluster_col).agg(agg).rename(columns={"cluster": "n"})
    summary = summary.reset_index()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster TS2Vec embeddings for eICU.")
    parser.add_argument("--embeddings", type=str, required=True)
    parser.add_argument("--meta", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--clusterer", type=str, default="auto", choices=["auto", "hdbscan", "agglomerative", "kmeans"])
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--k_min", type=int, default=2)
    parser.add_argument("--k_max", type=int, default=6)
    parser.add_argument("--silhouette_sample", type=int, default=20000)
    parser.add_argument("--min_cluster_size", type=int, default=100)
    parser.add_argument("--min_samples", type=int, default=10)
    parser.add_argument("--use_engineered", action="store_true", default=False)
    parser.add_argument("--engineered_dir", type=str, default=None)
    parser.add_argument("--pca_dim", type=int, default=0, help="Reduce dim before clustering (0=off).")
    parser.add_argument("--pca_sample", type=int, default=50000, help="Sample size to fit PCA (0=full).")
    parser.add_argument("--kmeans_batch", type=int, default=4096)
    parser.add_argument("--kmeans_epochs", type=int, default=5)
    parser.add_argument("--kmeans_n_init", type=int, default=5)
    parser.add_argument("--save_all_summaries", action="store_true", default=False,
                        help="Save cluster_summary_kmeans_k{K}.csv for every K and a combined CSV.")
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = setup_logger(Path(args.log_path) if args.log_path else None)

    emb = np.load(args.embeddings)
    meta_path = Path(args.meta)
    if meta_path.suffix.lower() == ".parquet":
        meta = pd.read_parquet(meta_path)
    else:
        meta = pd.read_csv(meta_path)

    if len(meta) != emb.shape[0]:
        logger.warning("Meta rows (%d) != embeddings (%d)", len(meta), emb.shape[0])

    X = emb.astype(np.float32)

    if args.use_engineered:
        if not args.engineered_dir:
            raise SystemExit("--engineered_dir is required when --use_engineered is set")
        eng = load_engineered_features(Path(args.engineered_dir), meta.columns[0], logger)
        meta_id = meta.columns[0]
        eng = eng.merge(meta[[meta_id]], on=meta_id, how="right")
        eng = eng.drop(columns=[meta_id])
        eng = eng.fillna(0.0)
        eng_scaled = StandardScaler().fit_transform(eng.to_numpy(dtype=np.float32))
        X = np.concatenate([X, eng_scaled], axis=1)
        logger.info("Appended engineered features: new dim=%d", X.shape[1])

    X = StandardScaler().fit_transform(X)

    if args.pca_dim and args.pca_dim < X.shape[1]:
        rng = np.random.default_rng(43)
        if args.pca_sample and args.pca_sample < len(X):
            idx = rng.choice(len(X), size=args.pca_sample, replace=False)
            X_fit = X[idx]
            logger.info("Fitting PCA on %d samples (dim=%d -> %d)", len(X_fit), X.shape[1], args.pca_dim)
        else:
            X_fit = X
            logger.info("Fitting PCA on full data (dim=%d -> %d)", X.shape[1], args.pca_dim)
        pca = PCA(n_components=args.pca_dim, random_state=43)
        pca.fit(X_fit)
        X = pca.transform(X)
        logger.info("PCA transform done. New dim=%d", X.shape[1])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_cols = [c for c in ["falencia_point", "falencia", "falencia_any", "mortality"] if c in meta.columns]
    results = []
    all_summaries = []

    def save_outputs(labels: np.ndarray, tag: str, metrics: dict):
        out_df = meta.copy()
        out_df["cluster"] = labels
        out_path = output_dir / f"cluster_embeddings_{tag}.parquet"
        out_df.to_parquet(out_path, index=False)

        summary = _build_summary(out_df, "cluster", label_cols)
        summary_path = output_dir / f"cluster_summary_{tag}.csv"
        summary.to_csv(summary_path, index=False)

        metrics_path = output_dir / f"metrics_{tag}.json"
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Saved %s outputs to %s", tag, output_dir)

    if args.clusterer == "auto":
        try:
            import hdbscan  # type: ignore

            clusterer = "hdbscan"
        except Exception:
            clusterer = "agglomerative"
    else:
        clusterer = args.clusterer

    if clusterer == "hdbscan":
        import hdbscan  # type: ignore

        logger.info("Clustering with HDBSCAN (min_cluster_size=%d)", args.min_cluster_size)
        model = hdbscan.HDBSCAN(min_cluster_size=args.min_cluster_size, min_samples=args.min_samples)

        stop_event = threading.Event()

        def _heartbeat():
            start = time.time()
            while not stop_event.wait(60):
                elapsed = time.time() - start
                logger.info("HDBSCAN running... elapsed %.1f min", elapsed / 60.0)

        hb = threading.Thread(target=_heartbeat, daemon=True)
        hb.start()
        labels = model.fit_predict(X)
        stop_event.set()
        hb.join(timeout=1)
        metrics = _cluster_metrics(X, labels, args.silhouette_sample)
        metrics["clusterer"] = "hdbscan"
        save_outputs(labels, "hdbscan", metrics)
    elif clusterer == "agglomerative":
        logger.info("Clustering with Agglomerative: k=%d..%d", args.k_min, args.k_max)
        best = None
        for k in range(args.k_min, args.k_max + 1):
            model = AgglomerativeClustering(n_clusters=k)
            labels = model.fit_predict(X)
            metrics = _cluster_metrics(X, labels, args.silhouette_sample)
            metrics["clusterer"] = "agglomerative"
            metrics["k"] = k
            results.append(metrics)
            if best is None or (metrics["silhouette"] > best["silhouette"]):
                best = {"labels": labels, "metrics": metrics}

        if results:
            pd.DataFrame(results).to_csv(output_dir / "k_sweep_metrics.csv", index=False)
        if best is not None:
            save_outputs(best["labels"], f"agglo_k{best['metrics']['k']}", best["metrics"])
    else:
        logger.info("Clustering with MiniBatchKMeans: k=%d..%d", args.k_min, args.k_max)
        best = None
        n_samples = len(X)
        for k in range(args.k_min, args.k_max + 1):
            model = MiniBatchKMeans(
                n_clusters=k,
                batch_size=args.kmeans_batch,
                n_init=args.kmeans_n_init,
                random_state=43,
            )
            from tqdm import tqdm

            for epoch in range(args.kmeans_epochs):
                pbar = tqdm(range(0, n_samples, args.kmeans_batch), desc=f"kmeans k={k} epoch {epoch+1}", unit="batch")
                for start in pbar:
                    end = min(start + args.kmeans_batch, n_samples)
                    model.partial_fit(X[start:end])
                pbar.close()

            labels = model.predict(X)
            metrics = _cluster_metrics(X, labels, args.silhouette_sample)
            metrics["clusterer"] = "kmeans"
            metrics["k"] = k
            results.append(metrics)

            if args.save_all_summaries:
                out_df = meta.copy()
                out_df["cluster"] = labels
                emb_path = output_dir / f"cluster_embeddings_kmeans_k{k}.parquet"
                out_df.to_parquet(emb_path, index=False)
                summary = _build_summary(out_df, "cluster", label_cols)
                summary.insert(0, "k", k)
                summary_path = output_dir / f"cluster_summary_kmeans_k{k}.csv"
                summary.to_csv(summary_path, index=False)
                all_summaries.append(summary)

            score = metrics.get("silhouette", float("nan"))
            if not np.isfinite(score):
                score = -float("inf")
            if best is None:
                best = {"labels": labels, "metrics": metrics, "score": score}
            else:
                if score > best.get("score", -float("inf")):
                    best = {"labels": labels, "metrics": metrics, "score": score}

        if results:
            pd.DataFrame(results).to_csv(output_dir / "k_sweep_metrics.csv", index=False)
        if args.save_all_summaries and all_summaries:
            combined = pd.concat(all_summaries, ignore_index=True)
            combined.to_csv(output_dir / "cluster_summary_kmeans_all.csv", index=False)
        if best is not None:
            save_outputs(best["labels"], f"kmeans_k{best['metrics']['k']}", best["metrics"])


if __name__ == "__main__":
    main()
