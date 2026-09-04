from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from common import setup_logger
from minirocket_lite import MiniRocketLite


def _sample_indices(n: int, max_stays: int | None, seed: int) -> np.ndarray:
    if max_stays is None or max_stays >= n:
        return np.arange(n, dtype=int)
    rng = np.random.default_rng(seed)
    return rng.choice(n, size=max_stays, replace=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="MiniRocket embeddings for eICU (from npy dataset).")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--meta_in", type=str, required=True)
    parser.add_argument("--emb_out", type=str, required=True)
    parser.add_argument("--meta_out", type=str, required=True)
    parser.add_argument("--n_kernels", type=int, default=256)
    parser.add_argument("--fit_stays", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = setup_logger(Path(args.log_path) if args.log_path else None)

    data = np.load(args.dataset, mmap_mode="r")
    n, t, d = data.shape
    logger.info("Dataset loaded: %s (N=%d, T=%d, D=%d)", args.dataset, n, t, d)

    fit_idx = _sample_indices(n, args.fit_stays, args.seed)
    X_fit = data[fit_idx]
    nan_rate = np.isnan(X_fit).mean()
    logger.info("NaN rate in fit sample: %.6f", nan_rate)
    X_fit = np.nan_to_num(X_fit, nan=0.0, posinf=0.0, neginf=0.0)
    # MiniRocket expects (N, channels, time); we have (N, time, features)
    X_fit = np.transpose(X_fit, (0, 2, 1))
    embedder = MiniRocketLite(n_kernels=args.n_kernels, random_state=args.seed)
    logger.info("Fitting MiniRocketLite on %d stays", len(X_fit))
    embedder.fit(X_fit)

    batch = max(1, int(args.batch_size))
    embs = []
    for i in tqdm(range(0, n, batch), desc="minirocket", unit="batch"):
        X = data[i : i + batch]
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X = np.transpose(X, (0, 2, 1))
        embs.append(embedder.transform(X))
    emb = np.vstack(embs).astype(np.float32)
    emb_var = float(np.nanvar(emb))
    row_var_zero = int((np.nanvar(emb, axis=1) == 0).sum())
    logger.info("Embedding variance: %.6f | rows var=0: %d/%d", emb_var, row_var_zero, emb.shape[0])
    if emb_var == 0.0:
        logger.warning("All embeddings are constant. Check input data or MiniRocket config.")

    emb_out = Path(args.emb_out)
    emb_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(emb_out, emb)
    logger.info("Embeddings saved to %s", emb_out)

    meta_in = Path(args.meta_in)
    if meta_in.suffix.lower() == ".parquet":
        meta = pd.read_parquet(meta_in)
    else:
        meta = pd.read_csv(meta_in)
    meta_out = Path(args.meta_out)
    meta_out.parent.mkdir(parents=True, exist_ok=True)
    if meta_out.suffix.lower() == ".parquet":
        meta.to_parquet(meta_out, index=False)
    else:
        meta.to_csv(meta_out, index=False)
    logger.info("Meta saved to %s", meta_out)


if __name__ == "__main__":
    main()
