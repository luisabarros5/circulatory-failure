#!/usr/bin/env python3
"""
29_audit_C_observation_density.py

Audits whether Experiment C clusters are associated with the amount of REAL
clinical information available before onset.

For each of the 20 C variables and each stay, within [0, first onset]:
- number of feature-store rows with a finite value;
- number marked observed (_imputed == 0), when that flag exists;
- number marked imputed (_imputed != 0), when that flag exists.

Then compares those quantities across C k=3 clusters.

This does NOT alter the clustering and does NOT define phenotypes using
missingness. It is only an artefact audit.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr


FEATURES = [
    "heart_rate", "mbp", "resp_rate", "temperature", "spo2", "glucose",
    "lab_50813", "lab_50983", "lab_50971", "lab_50912", "lab_51006",
    "lab_50878", "lab_50861", "lab_50885", "lab_50862", "lab_51222",
    "lab_51265", "lab_51274", "lab_51275", "vasopressor_ativo",
]


def paired_label(labels_dir: Path, fp: Path) -> Path:
    return labels_dir / f"{fp.stem}_labels.parquet"


def safe_spearman(x, y):
    x = pd.to_numeric(x, errors="coerce").to_numpy(float)
    y = pd.to_numeric(y, errors="coerce").to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return np.nan, np.nan
    r, p = spearmanr(x[ok], y[ok])
    return float(r), float(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-dir", required=True, type=Path)
    ap.add_argument("--labels-dir", required=True, type=Path)
    ap.add_argument("--positive-stays", required=True, type=Path)
    ap.add_argument("--c-k3", required=True, type=Path,
                    help="C k=3 assignments parquet")
    ap.add_argument("--c-meta", required=True, type=Path,
                    help="normalized_preonset96_meta.parquet")
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pos = pd.read_parquet(args.positive_stays)[["stay_id", "first_positive_time"]].copy()
    pos = pos.dropna().drop_duplicates("stay_id").sort_values("stay_id").reset_index(drop=True)
    pos["stay_id"] = pos["stay_id"].astype("int64")
    pos["first_positive_time"] = pd.to_numeric(pos["first_positive_time"], errors="coerce")

    stays = pos["stay_id"].to_numpy(np.int64)
    onset = pos["first_positive_time"].to_numpy(float)
    onset_map = dict(zip(stays.tolist(), onset.tolist()))
    n = len(stays)
    d = len(FEATURES)

    # Efficient lookup because stays is sorted.
    def map_sid_to_idx(sid):
        ix = np.searchsorted(stays, sid)
        good = (ix >= 0) & (ix < n)
        safe_ix = np.clip(ix, 0, n - 1)
        good &= stays[safe_ix] == sid
        return safe_ix, good

    fps = sorted(
        p for p in args.feature_dir.glob("batch_*.parquet")
        if not p.name.endswith("_labels.parquet")
    )
    if not fps:
        raise SystemExit(f"Nenhum batch_*.parquet em {args.feature_dir}")

    schema = set(pq.read_schema(fps[0]).names)
    missing = [c for c in FEATURES if c not in schema]
    if missing:
        raise SystemExit(f"Features do C ausentes no feature store: {missing}")

    has_flag = np.array([f"{c}_imputed" in schema for c in FEATURES], dtype=bool)
    print("=== FLAGS DE IMPUTAÇÃO ===")
    for c, flag in zip(FEATURES, has_flag):
        print(f"{c:24s} {'SIM' if flag else 'NÃO'}")

    # Counts per stay x feature.
    present = np.zeros((n, d), dtype=np.int32)
    observed = np.zeros((n, d), dtype=np.int32)
    imputed = np.zeros((n, d), dtype=np.int32)
    source_rows = np.zeros(n, dtype=np.int32)

    total_selected = 0

    for fi, fp in enumerate(fps, 1):
        lp = paired_label(args.labels_dir, fp)
        if not lp.exists():
            raise FileNotFoundError(f"Label pareado ausente: {lp}")

        fpf = pq.ParquetFile(fp)
        ldf = pd.read_parquet(lp, columns=["stay_id", "rel_charttime"])
        if fpf.metadata.num_rows != len(ldf):
            raise RuntimeError(f"Row count incompatível: {fp.name}")

        sid_all = ldf["stay_id"].to_numpy(np.int64)
        rel_all = pd.to_numeric(ldf["rel_charttime"], errors="coerce").to_numpy(float)

        offset = 0
        for rg in range(fpf.num_row_groups):
            nrg = fpf.metadata.row_group(rg).num_rows
            sl = slice(offset, offset + nrg)
            sid_rg = sid_all[sl]
            rel_rg = rel_all[sl]

            onset_rg = np.fromiter(
                (onset_map.get(int(s), np.nan) for s in sid_rg),
                dtype=float, count=len(sid_rg),
            )
            mask = (
                np.isfinite(onset_rg) & np.isfinite(rel_rg)
                & (rel_rg >= 0) & (rel_rg <= onset_rg)
            )

            if mask.any():
                cols = ["stay_id", *FEATURES]
                cols += [
                    f"{c}_imputed" for c in FEATURES
                    if f"{c}_imputed" in fpf.schema_arrow.names
                ]
                cols = list(dict.fromkeys(cols))
                table = fpf.read_row_group(rg, columns=cols)
                f_sid = table["stay_id"].to_numpy(zero_copy_only=False).astype(np.int64)
                if not np.array_equal(f_sid, sid_rg):
                    raise RuntimeError(f"Desalinhamento feature/label: {fp.name}, RG {rg}")

                take = np.flatnonzero(mask)
                df = table.take(take).to_pandas()
                sid_sel = sid_rg[take]
                idx_sel, good = map_sid_to_idx(sid_sel)

                if not good.all():
                    df = df.loc[good].reset_index(drop=True)
                    idx_sel = idx_sel[good]

                np.add.at(source_rows, idx_sel, 1)
                total_selected += len(idx_sel)

                for j, c in enumerate(FEATURES):
                    vals = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
                    finite = np.isfinite(vals)
                    np.add.at(present[:, j], idx_sel, finite.astype(np.int8))

                    mcol = f"{c}_imputed"
                    if mcol in df.columns:
                        flag = pd.to_numeric(df[mcol], errors="coerce").fillna(1).to_numpy(float)
                        obs = finite & (flag == 0)
                        imp = finite & (flag != 0)
                        np.add.at(observed[:, j], idx_sel, obs.astype(np.int8))
                        np.add.at(imputed[:, j], idx_sel, imp.astype(np.int8))

            offset += nrg

        if fi % 10 == 0 or fi == len(fps):
            print(f"[INFO] {fi}/{len(fps)} | linhas pré-onset selecionadas={total_selected:,}")

    # Per-stay output.
    per = pos.copy()
    per["onset_hours"] = per["first_positive_time"] / 3600.0
    per["source_rows_recounted"] = source_rows
    per["features_with_any_value"] = (present > 0).sum(axis=1)
    flagged_idx = np.flatnonzero(has_flag)
    if len(flagged_idx):
        per["features_with_any_real_observation"] = (observed[:, flagged_idx] > 0).sum(axis=1)
        obs_total = observed[:, flagged_idx].sum(axis=1)
        imp_total = imputed[:, flagged_idx].sum(axis=1)
        denom = obs_total + imp_total
        per["real_observed_entries_flagged"] = obs_total
        per["imputed_entries_flagged"] = imp_total
        per["real_observed_fraction_flagged"] = np.divide(
            obs_total, denom,
            out=np.full(n, np.nan, dtype=float),
            where=denom > 0,
        )
    else:
        per["features_with_any_real_observation"] = np.nan
        per["real_observed_entries_flagged"] = np.nan
        per["imputed_entries_flagged"] = np.nan
        per["real_observed_fraction_flagged"] = np.nan

    for j, c in enumerate(FEATURES):
        per[f"{c}__n_value_rows"] = present[:, j]
        if has_flag[j]:
            per[f"{c}__n_real_observed"] = observed[:, j]
            per[f"{c}__n_imputed"] = imputed[:, j]

    c_assign = pd.read_parquet(args.c_k3)[["stay_id", "cluster"]].copy()
    c_assign["stay_id"] = c_assign["stay_id"].astype("int64")
    per = per.merge(c_assign, on="stay_id", how="left", validate="one_to_one")
    if per["cluster"].isna().any():
        raise RuntimeError("Alguns stays positivos não têm cluster C k=3.")

    cmeta = pd.read_parquet(args.c_meta)
    keep_meta = [c for c in ["stay_id", "source_preonset_rows", "onset_hours"] if c in cmeta.columns]
    cmeta = cmeta[keep_meta].drop_duplicates("stay_id")
    cmeta = cmeta.rename(columns={
        "source_preonset_rows": "source_preonset_rows_C",
        "onset_hours": "onset_hours_C",
    })
    per = per.merge(cmeta, on="stay_id", how="left", validate="one_to_one")

    per.to_parquet(args.out_dir / "per_stay_C_observation_density.parquet", index=False)

    # Cluster-level summary.
    rows = []
    for cl, g in per.groupby("cluster"):
        rows.append({
            "cluster": int(cl),
            "n": len(g),
            "onset_median_h": g["onset_hours"].median(),
            "onset_q1_h": g["onset_hours"].quantile(.25),
            "onset_q3_h": g["onset_hours"].quantile(.75),
            "source_rows_median": g["source_rows_recounted"].median(),
            "source_rows_q1": g["source_rows_recounted"].quantile(.25),
            "source_rows_q3": g["source_rows_recounted"].quantile(.75),
            "features_any_value_median": g["features_with_any_value"].median(),
            "features_real_observed_median": g["features_with_any_real_observation"].median(),
            "real_observed_fraction_median": g["real_observed_fraction_flagged"].median(),
            "real_observed_fraction_q1": g["real_observed_fraction_flagged"].quantile(.25),
            "real_observed_fraction_q3": g["real_observed_fraction_flagged"].quantile(.75),
        })
    cluster_summary = pd.DataFrame(rows)
    cluster_summary.to_csv(args.out_dir / "cluster_observation_density_summary.csv", index=False)

    # Per-feature, per-cluster audit.
    frows = []
    for j, c in enumerate(FEATURES):
        for cl, gidx in per.groupby("cluster").groups.items():
            ids = np.asarray(list(gidx), dtype=int)
            ncl = len(ids)
            value_counts = present[ids, j]
            row = {
                "feature": c,
                "has_imputed_flag": bool(has_flag[j]),
                "cluster": int(cl),
                "n": ncl,
                "pct_stays_with_any_value": 100 * np.mean(value_counts > 0),
                "median_value_rows_per_stay": float(np.median(value_counts)),
            }
            if has_flag[j]:
                o = observed[ids, j]
                im = imputed[ids, j]
                denom = o.sum() + im.sum()
                row.update({
                    "pct_stays_with_any_real_observation": 100 * np.mean(o > 0),
                    "median_real_observations_per_stay": float(np.median(o)),
                    "median_imputed_rows_per_stay": float(np.median(im)),
                    "real_fraction_across_all_present_entries": (
                        float(o.sum() / denom) if denom else np.nan
                    ),
                })
            frows.append(row)
    fdf = pd.DataFrame(frows)
    fdf.to_csv(args.out_dir / "feature_observation_density_by_cluster.csv", index=False)

    # Correlations with onset: if very strong, availability is tracking time.
    corr_rows = []
    for col in [
        "source_rows_recounted",
        "features_with_any_value",
        "features_with_any_real_observation",
        "real_observed_fraction_flagged",
    ]:
        r, p = safe_spearman(per["onset_hours"], per[col])
        corr_rows.append({
            "availability_measure": col,
            "spearman_r_with_onset_hours": r,
            "p_value": p,
        })
    corr = pd.DataFrame(corr_rows)
    corr.to_csv(args.out_dir / "availability_vs_onset_spearman.csv", index=False)

    # Sanity check: our source count should reproduce C meta.
    if "source_preonset_rows_C" in per.columns:
        delta = (
            pd.to_numeric(per["source_rows_recounted"], errors="coerce")
            - pd.to_numeric(per["source_preonset_rows_C"], errors="coerce")
        )
        pd.DataFrame({
            "n": [len(delta)],
            "exact_match_pct": [100 * np.mean(delta == 0)],
            "median_abs_difference": [np.nanmedian(np.abs(delta))],
            "max_abs_difference": [np.nanmax(np.abs(delta))],
        }).to_csv(args.out_dir / "source_rows_reproduction_check.csv", index=False)

    print("\n=== CLUSTER OBSERVATION DENSITY ===")
    print(cluster_summary.to_string(index=False))
    print("\n=== AVAILABILITY vs ONSET ===")
    print(corr.to_string(index=False))
    print("\n[OK]", args.out_dir)


if __name__ == "__main__":
    main()
