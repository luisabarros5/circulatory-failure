#!/usr/bin/env python3
"""
03_build_onset_aligned_dataset.py

Constrói o tensor temporal para fenotipagem alinhado ao PRIMEIRO onset da nova
definição de falência circulatória.

Janela principal:
- grid: 5 min
- 96 posições
- última posição = onset (cf_first_onset_primary)
- 95 posições anteriores + onset
- stays com onset precoce ficam com padding à ESQUERDA (NaN)

Importante:
- charttime NÃO entra como feature.
- *_imputed ficam fora (são flags/máscaras no dataset atual).
- vasopressor_ativo permanece.
- target/mortalidade/IDs não entram como features.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def compute_mean_std(memmap: np.ndarray, chunk: int = 128):
    n, _, d = memmap.shape
    sum_v = np.zeros(d, dtype=np.float64)
    cnt_v = np.zeros(d, dtype=np.float64)

    for i in range(0, n, chunk):
        block = memmap[i:i + chunk]
        mask = ~np.isnan(block)
        sum_v += np.nansum(block, axis=(0, 1))
        cnt_v += mask.sum(axis=(0, 1))

    mean = np.divide(sum_v, cnt_v, out=np.zeros_like(sum_v), where=cnt_v > 0)

    var_sum = np.zeros(d, dtype=np.float64)
    for i in range(0, n, chunk):
        block = memmap[i:i + chunk]
        diff = block - mean
        diff[np.isnan(diff)] = 0.0
        var_sum += np.sum(diff * diff, axis=(0, 1))

    var = np.divide(var_sum, cnt_v, out=np.zeros_like(var_sum), where=cnt_v > 0)
    std = np.sqrt(var)
    std[std == 0] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def standardize_inplace(memmap: np.ndarray, mean: np.ndarray, std: np.ndarray, chunk: int = 128):
    for i in range(0, memmap.shape[0], chunk):
        block = memmap[i:i + chunk]
        memmap[i:i + chunk] = (block - mean) / std


def label_name_for(feature_path: Path) -> str:
    return f"{feature_path.stem}_labels.parquet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-dir", required=True, type=Path)
    ap.add_argument("--labels-dir", required=True, type=Path)
    ap.add_argument("--positive-stays", required=True, type=Path)
    ap.add_argument("--out-npy", required=True, type=Path)
    ap.add_argument("--out-meta", required=True, type=Path)
    ap.add_argument("--scaler-out", required=True, type=Path)
    ap.add_argument("--timesteps", type=int, default=96)
    ap.add_argument("--step-seconds", type=int, default=300)
    args = ap.parse_args()

    bundle = Path(__file__).resolve().parents[1] / "core"
    sys.path.insert(0, str(bundle))
    from common import select_feature_columns  # noqa

    pos = pd.read_parquet(args.positive_stays)
    required = {"stay_id", "first_positive_time"}
    if not required.issubset(pos.columns):
        raise SystemExit(f"positive_stays precisa ter {sorted(required)}")

    pos = pos[["stay_id", "first_positive_time"]].dropna().copy()
    pos["stay_id"] = pos["stay_id"].astype("int64")
    pos["first_positive_time"] = pos["first_positive_time"].astype(float)
    pos = pos.drop_duplicates("stay_id").sort_values("stay_id").reset_index(drop=True)

    stays = pos["stay_id"].to_numpy(np.int64)
    onset = pos["first_positive_time"].to_numpy(float)
    onset_map = dict(zip(stays.tolist(), onset.tolist()))
    id_to_idx = {int(s): i for i, s in enumerate(stays)}

    feature_paths = sorted(
        p for p in args.feature_dir.glob("batch_*.parquet")
        if not p.name.endswith("_labels.parquet")
    )
    if not feature_paths:
        raise SystemExit(f"Nenhum batch_*.parquet em {args.feature_dir}")

    schema = pq.read_schema(feature_paths[0]).names
    feat_cols = select_feature_columns(schema, drop_imputed=True)

    # Fail-safe explícito: tempo/ID nunca entram no tensor.
    forbidden = {
        "stay_id", "patientunitstayid", "patientunitstay_id",
        "charttime", "rel_charttime", "time", "timestamp",
        "falencia", "falencia_point", "falencia_any", "falencia_onset",
        "mortality",
    }
    feat_cols = [c for c in feat_cols if c not in forbidden]

    if "vasopressor_ativo" not in feat_cols:
        raise SystemExit("vasopressor_ativo não está entre as features; abortando.")

    n, t, d = len(stays), args.timesteps, len(feat_cols)
    print(f"[INFO] stays={n:,} | timesteps={t} | features={d}")
    print("[INFO] alinhamento: último timestep = primeiro onset")
    print("[INFO] charttime: somente chave temporal; NÃO é feature")
    print("[INFO] vasopressor_ativo: MANTIDO")

    args.out_npy.parent.mkdir(parents=True, exist_ok=True)
    data = np.lib.format.open_memmap(
        args.out_npy, mode="w+", dtype="float32", shape=(n, t, d)
    )
    data[:] = np.nan

    filled_positions = np.zeros(n, dtype=np.int16)
    onset_seen = np.zeros(n, dtype=bool)

    total_selected_rows = 0
    paired_files = 0

    for file_i, fp in enumerate(feature_paths, 1):
        lp = args.labels_dir / label_name_for(fp)
        if not lp.exists():
            raise FileNotFoundError(f"Label pareado não encontrado para {fp.name}: {lp}")

        fpf = pq.ParquetFile(fp)
        ldf = pd.read_parquet(
            lp,
            columns=["stay_id", "rel_charttime", "cf_first_onset_primary"],
        )

        if fpf.metadata.num_rows != len(ldf):
            raise RuntimeError(
                f"Row count difere em {fp.name}: features={fpf.metadata.num_rows}, labels={len(ldf)}"
            )

        lbl_sid = ldf["stay_id"].to_numpy(np.int64)
        lbl_rel = pd.to_numeric(ldf["rel_charttime"], errors="coerce").to_numpy(float)

        offset = 0
        for rg in range(fpf.num_row_groups):
            nrg = fpf.metadata.row_group(rg).num_rows
            sl = slice(offset, offset + nrg)
            sid_rg = lbl_sid[sl]
            rel_rg = lbl_rel[sl]

            unique_sid = np.unique(sid_rg)
            relevant = [int(s) for s in unique_sid if int(s) in onset_map]

            if relevant:
                onset_rg = np.fromiter(
                    (onset_map.get(int(s), np.nan) for s in sid_rg),
                    dtype=float,
                    count=len(sid_rg),
                )
                delta = onset_rg - rel_rg
                mask = (
                    np.isfinite(onset_rg)
                    & np.isfinite(rel_rg)
                    & (delta >= 0)
                    & (delta <= (args.timesteps - 1) * args.step_seconds)
                )

                if mask.any():
                    table = fpf.read_row_group(rg, columns=["stay_id", *feat_cols])
                    f_sid = table["stay_id"].to_numpy(zero_copy_only=False).astype(np.int64)

                    if not np.array_equal(f_sid, sid_rg):
                        raise RuntimeError(
                            f"Desalinhamento stay_id entre feature/label em {fp.name}, row-group {rg}"
                        )

                    take_idx = np.flatnonzero(mask)
                    sub = table.take(take_idx)
                    vals = sub.select(feat_cols).to_pandas().to_numpy(dtype=np.float32)
                    sid_sel = sid_rg[take_idx]
                    delta_sel = delta[take_idx]

                    # onset = posição T-1; 5 min antes = T-2 etc.
                    pos_idx = (args.timesteps - 1) - np.rint(
                        delta_sel / args.step_seconds
                    ).astype(int)

                    good = (pos_idx >= 0) & (pos_idx < args.timesteps)
                    vals = vals[good]
                    sid_sel = sid_sel[good]
                    pos_idx = pos_idx[good]

                    for row, sid, pidx in zip(vals, sid_sel, pos_idx):
                        dst = id_to_idx[int(sid)]
                        data[dst, int(pidx), :] = row
                        if not np.isnan(row).all():
                            filled_positions[dst] += 1
                        if int(pidx) == args.timesteps - 1:
                            onset_seen[dst] = True

                    total_selected_rows += len(vals)

            offset += nrg

        paired_files += 1
        if file_i % 10 == 0 or file_i == len(feature_paths):
            print(
                f"[INFO] {file_i}/{len(feature_paths)} arquivos | "
                f"linhas da janela={total_selected_rows:,}"
            )

    if paired_files != len(feature_paths):
        raise RuntimeError("Nem todos os arquivos foram pareados.")

    n_seen = int((filled_positions > 0).sum())
    n_onset = int(onset_seen.sum())

    meta = pos.copy()
    meta["length"] = filled_positions
    meta["onset_at_last_timestep"] = onset_seen

    args.out_meta.parent.mkdir(parents=True, exist_ok=True)
    meta.to_parquet(args.out_meta, index=False)

    print("\n=== AUDITORIA ANTES DA PADRONIZAÇÃO ===")
    print(f"stays esperados: {n:,}")
    print(f"stays com >=1 timestep: {n_seen:,}")
    print(f"stays com onset na posição final: {n_onset:,}")
    print(
        "length min/mediana/max:",
        int(filled_positions.min()),
        float(np.median(filled_positions)),
        int(filled_positions.max()),
    )
    print(f"stays com 96 posições: {int((filled_positions == args.timesteps).sum()):,}")

    if n_seen != n or n_onset != n:
        raise RuntimeError(
            "Nem todos os stays foram alinhados ao onset. "
            "Não prossiga para MiniRocket."
        )

    print("\n[INFO] Calculando mean/std...")
    mean, std = compute_mean_std(data)
    standardize_inplace(data, mean, std)
    args.scaler_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.scaler_out, mean=mean, std=std)

    print("\n[OK] Tensor alinhado criado.")
    print(f"NPY:    {args.out_npy}")
    print(f"META:   {args.out_meta}")
    print(f"SCALER: {args.scaler_out}")


if __name__ == "__main__":
    main()
