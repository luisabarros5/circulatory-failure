from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from common import detect_id_col, detect_time_col, select_feature_columns, setup_logger


def _read_schema_columns(path: Path) -> list[str]:
    try:
        import pyarrow.parquet as pq

        return pq.read_schema(path).names
    except Exception:
        df = pd.read_parquet(path)
        return list(df.columns)


def _quantile_names(qs: list[float]) -> list[str]:
    return [f"p{int(q*100)}" if q * 100 == int(q * 100) else f"p{q}".replace(".", "_") for q in qs]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize per-cluster feature means from MIMIC parquets.")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--cluster_embeddings", type=str, required=True)
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--time_col", type=str, default=None)
    parser.add_argument("--keep_imputed", action="store_true", default=False)
    parser.add_argument("--quantiles", type=str, default="0.25,0.5,0.75,0.9,0.99",
                        help="Comma-separated quantiles to compute (e.g., 0.25,0.5,0.75,0.9,0.99).")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Optional cache directory for per-stay batch summaries.")
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = setup_logger(Path(args.log_path) if args.log_path else None)

    input_dir = Path(args.input_dir)
    paths = sorted(
        p for p in input_dir.glob("batch_*.parquet") if not p.name.endswith("_labels.parquet")
    )
    if not paths:
        raise SystemExit(f"No batch_*.parquet found in {input_dir}")

    # cluster assignments
    emb_path = Path(args.cluster_embeddings)
    if emb_path.suffix.lower() == ".parquet":
        emb = pd.read_parquet(emb_path)
    else:
        emb = pd.read_csv(emb_path)

    id_col = detect_id_col(emb.columns)
    if "cluster" not in emb.columns:
        raise SystemExit("cluster column not found in cluster_embeddings.")

    cluster_map = emb.set_index(id_col)["cluster"].to_dict()
    clusters = sorted(emb["cluster"].dropna().unique().tolist())

    # label rates
    label_cols = [c for c in ["falencia_point", "falencia", "falencia_any", "mortality"] if c in emb.columns]
    label_rates = None
    if label_cols:
        label_rates = emb.groupby("cluster")[label_cols].mean().reset_index()

    # detect features
    columns = _read_schema_columns(paths[0])
    if args.time_col:
        time_col = args.time_col
        if time_col not in columns:
            raise SystemExit(f"time_col '{time_col}' not found in input files.")
    else:
        time_col = detect_time_col(columns)
    feat_cols = select_feature_columns(columns, drop_imputed=not args.keep_imputed)
    feat_cols = [c for c in feat_cols if c not in {id_col, time_col}]

    per_stay_frames: list[pd.DataFrame] = []
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    for i, path in enumerate(tqdm(paths, desc="parquets", unit="file"), start=1):
        logger.info("Processing %s (%d/%d)", path.name, i, len(paths))
        cache_path = None
        if cache_dir:
            cache_path = cache_dir / f"{path.stem}_perstay.parquet"
            if cache_path.exists():
                per_stay = pd.read_parquet(cache_path)
                per_stay["cluster"] = per_stay[id_col].map(cluster_map)
                per_stay = per_stay.dropna(subset=["cluster"])
                per_stay_frames.append(per_stay)
                continue

        df = pd.read_parquet(path, columns=[id_col, time_col] + feat_cols)
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()]
        df = df[df[id_col].isin(cluster_map.keys())]
        if df.empty:
            continue
        df = df.dropna(subset=[id_col])
        df[id_col] = df[id_col].astype(int)

        # per-stay mean across time
        per_stay = df.groupby(id_col)[feat_cols].mean().reset_index()
        if cache_path:
            per_stay.to_parquet(cache_path, index=False)
        per_stay["cluster"] = per_stay[id_col].map(cluster_map)
        per_stay = per_stay.dropna(subset=["cluster"])
        per_stay_frames.append(per_stay)

    if not per_stay_frames:
        raise SystemExit("No stays found to summarize.")

    per_stay_all = pd.concat(per_stay_frames, ignore_index=True)

    qs = [float(x) for x in args.quantiles.split(",") if x.strip()]
    q_names = _quantile_names(qs)

    # mean
    grouped = per_stay_all.groupby("cluster")[feat_cols].mean().reset_index()
    grouped.insert(1, "n", per_stay_all.groupby("cluster").size().values)

    # quantiles
    if qs:
        qdf = per_stay_all.groupby("cluster")[feat_cols].quantile(qs)
        qdf.index = qdf.index.set_names(["cluster", "q"])
        qdf = qdf.reset_index()
        for q, qn in zip(qs, q_names):
            q_slice = qdf[qdf["q"] == q].drop(columns=["q"])
            q_slice = q_slice.rename(columns={c: f"{c}_{qn}" for c in feat_cols})
            grouped = grouped.merge(q_slice, on="cluster", how="left")

    summary = grouped
    if label_rates is not None:
        summary = summary.merge(label_rates, on="cluster", how="left")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)
    logger.info("Saved full summary to %s", out_csv)


if __name__ == "__main__":
    main()
