#!/usr/bin/env python3
"""
04_build_onset_snapshot.py

Dataset estático para fenotipagem da falência circulatória.

População:
- TODOS os stays positivos pela nova definição (cf_first_onset_primary == 1).

Representação:
- exatamente 1 linha por stay;
- estado clínico no PRIMEIRO onset;
- sem janela pré-onset;
- sem MiniRocket;
- sem perda de pacientes por tempo até onset.

Features:
- vitais básicos: heart_rate, sbp, dbp, mbp, resp_rate, temperature, spo2;
- glucose;
- labs brutos lab_<itemid>;
- vasopressor_ativo.

Não entram:
- IDs como feature;
- charttime/rel_charttime;
- target/falência;
- mortalidade/outcomes;
- *_imputed como feature;
- n_meas_*, min_*, max_*, mean_*, *_cumul, *_intens, *_instab.

Tratamento:
- usa o VALOR clínico disponível no feature table no onset;
- *_imputed são usados apenas para registrar cobertura observada;
- NaNs residuais: mediana da coorte;
- features totalmente ausentes ou sem variância são removidas;
- StandardScaler após imputação.

Outputs:
- onset_snapshot_raw.parquet
- onset_snapshot_scaled.npy
- onset_snapshot_meta.parquet
- onset_snapshot_features.csv
- onset_snapshot_preprocessing.npz
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


BASE_FEATURES = [
    "heart_rate",
    "sbp",
    "dbp",
    "mbp",
    "resp_rate",
    "temperature",
    "spo2",
    "glucose",
]
LAB_RE = re.compile(r"^lab_\d+$")


def select_features(schema_names: list[str]) -> list[str]:
    labs = sorted(c for c in schema_names if LAB_RE.fullmatch(c))
    out = [c for c in BASE_FEATURES if c in schema_names]
    out.extend(labs)
    if "vasopressor_ativo" in schema_names:
        out.append("vasopressor_ativo")
    return out


def paired_label_path(labels_dir: Path, feature_path: Path) -> Path:
    return labels_dir / f"{feature_path.stem}_labels.parquet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-dir", required=True, type=Path)
    ap.add_argument("--labels-dir", required=True, type=Path)
    ap.add_argument("--positive-stays", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    pos = pd.read_parquet(args.positive_stays)
    required_pos = {"stay_id", "first_positive_time"}
    missing = required_pos - set(pos.columns)
    if missing:
        raise SystemExit(f"positive_stays sem colunas: {sorted(missing)}")

    pos = pos[["stay_id", "first_positive_time"]].copy()
    pos["stay_id"] = pd.to_numeric(pos["stay_id"], errors="coerce")
    pos["first_positive_time"] = pd.to_numeric(
        pos["first_positive_time"], errors="coerce"
    )
    pos = pos.dropna().drop_duplicates("stay_id")
    pos["stay_id"] = pos["stay_id"].astype("int64")
    pos = pos.sort_values("stay_id").reset_index(drop=True)

    expected_stays = set(pos["stay_id"].tolist())
    expected_n = len(expected_stays)

    feature_paths = sorted(
        p for p in args.feature_dir.glob("batch_*.parquet")
        if not p.name.endswith("_labels.parquet")
    )
    if not feature_paths:
        raise SystemExit(f"Nenhum batch_*.parquet em {args.feature_dir}")

    schema_names = pq.read_schema(feature_paths[0]).names
    feature_cols = select_features(schema_names)

    if "mbp" not in feature_cols:
        raise SystemExit("mbp não encontrado.")
    if "lab_50813" not in feature_cols:
        raise SystemExit("lab_50813 (lactato) não encontrado.")
    if "vasopressor_ativo" not in feature_cols:
        raise SystemExit("vasopressor_ativo não encontrado.")

    suspicious = [
        c for c in feature_cols
        if c.endswith("_imputed")
        or c.startswith(("n_meas_", "min_", "max_", "mean_"))
        or c.endswith(("_cumul", "_intens", "_instab"))
        or c in {"stay_id", "charttime", "rel_charttime"}
    ]
    if suspicious:
        raise SystemExit(f"Features proibidas detectadas: {suspicious}")

    mask_cols = [
        f"{c}_imputed"
        for c in feature_cols
        if f"{c}_imputed" in schema_names
    ]

    print("=== DATASET ESTÁTICO NO PRIMEIRO ONSET ===")
    print(f"Stays positivos esperados: {expected_n:,}")
    print(f"Features clínicas candidatas: {len(feature_cols)}")
    print("Todos os pacientes positivos serão mantidos.")
    print("MiniRocket: NÃO usado nesta análise.")
    print("K-means receberá um vetor clínico por stay.")

    rows = []
    observed_counts = {c: 0 for c in feature_cols}

    for file_i, fp in enumerate(feature_paths, 1):
        lp = paired_label_path(args.labels_dir, fp)
        if not lp.exists():
            raise FileNotFoundError(f"Label pareado ausente: {lp}")

        fpf = pq.ParquetFile(fp)
        ldf = pd.read_parquet(
            lp,
            columns=["stay_id", "rel_charttime", "cf_first_onset_primary"],
        )

        if fpf.metadata.num_rows != len(ldf):
            raise RuntimeError(
                f"Row count incompatível: {fp.name} "
                f"features={fpf.metadata.num_rows} labels={len(ldf)}"
            )

        sid_all = ldf["stay_id"].to_numpy(np.int64)
        rel_all = pd.to_numeric(
            ldf["rel_charttime"], errors="coerce"
        ).to_numpy(float)
        onset_flag_all = pd.to_numeric(
            ldf["cf_first_onset_primary"], errors="coerce"
        ).fillna(0).to_numpy()

        offset = 0
        for rg in range(fpf.num_row_groups):
            nrg = fpf.metadata.row_group(rg).num_rows
            sl = slice(offset, offset + nrg)

            sid_rg = sid_all[sl]
            rel_rg = rel_all[sl]
            flag_rg = onset_flag_all[sl]

            local_mask = (flag_rg == 1)
            if local_mask.any():
                local_idx = np.flatnonzero(local_mask)

                cols_here = ["stay_id", *feature_cols, *mask_cols]
                cols_here = list(dict.fromkeys(
                    c for c in cols_here if c in fpf.schema_arrow.names
                ))

                table = fpf.read_row_group(rg, columns=cols_here).take(local_idx)
                df = table.to_pandas()

                sid_sel = sid_rg[local_idx]
                rel_sel = rel_rg[local_idx]

                if not np.array_equal(
                    df["stay_id"].to_numpy(np.int64), sid_sel
                ):
                    raise RuntimeError(
                        f"Desalinhamento feature/label em {fp.name}, RG {rg}"
                    )

                df["stay_id"] = sid_sel
                df["first_positive_time"] = rel_sel

                # Cobertura: valor realmente observado no onset.
                for c in feature_cols:
                    m = f"{c}_imputed"
                    vals = pd.to_numeric(df[c], errors="coerce")
                    if m in df.columns:
                        imp = pd.to_numeric(
                            df[m], errors="coerce"
                        ).fillna(1)
                        observed_counts[c] += int(
                            (vals.notna() & (imp == 0)).sum()
                        )
                    else:
                        observed_counts[c] += int(vals.notna().sum())

                rows.append(
                    df[["stay_id", "first_positive_time", *feature_cols]].copy()
                )

            offset += nrg

        if file_i % 10 == 0 or file_i == len(feature_paths):
            current = sum(len(x) for x in rows)
            print(
                f"[INFO] {file_i}/{len(feature_paths)} arquivos | "
                f"onsets extraídos={current:,}"
            )

    if not rows:
        raise SystemExit("Nenhum onset extraído.")

    snapshot = pd.concat(rows, ignore_index=True)
    snapshot["stay_id"] = snapshot["stay_id"].astype("int64")

    # Validação de unicidade e correspondência com a coorte oficial.
    dup = snapshot["stay_id"].duplicated().sum()
    found_stays = set(snapshot["stay_id"].tolist())
    missing_stays = expected_stays - found_stays
    unexpected_stays = found_stays - expected_stays

    print("\n=== VALIDAÇÃO DA COORTE ===")
    print(f"Linhas extraídas:          {len(snapshot):,}")
    print(f"Stays únicos:              {snapshot['stay_id'].nunique():,}")
    print(f"Duplicados de stay_id:     {dup:,}")
    print(f"Stays positivos esperados: {expected_n:,}")
    print(f"Stays faltantes:           {len(missing_stays):,}")
    print(f"Stays inesperados:         {len(unexpected_stays):,}")

    if (
        len(snapshot) != expected_n
        or snapshot["stay_id"].nunique() != expected_n
        or dup != 0
        or missing_stays
        or unexpected_stays
    ):
        raise RuntimeError(
            "Snapshot não corresponde exatamente aos stays positivos. "
            "K-means NÃO deve ser executado."
        )

    # Confere que o tempo de onset extraído bate com positive_stays.
    check = snapshot[["stay_id", "first_positive_time"]].merge(
        pos,
        on="stay_id",
        suffixes=("_labels", "_positive_stays"),
        validate="one_to_one",
    )
    time_diff = np.abs(
        check["first_positive_time_labels"].to_numpy(float)
        - check["first_positive_time_positive_stays"].to_numpy(float)
    )
    max_diff = float(np.nanmax(time_diff))
    print(f"Maior diferença de onset:  {max_diff:.6f} s")
    if max_diff > 1e-6:
        raise RuntimeError("Tempo de onset não bate com positive_stays.")

    # Converte features para numérico.
    Xdf = snapshot[feature_cols].apply(pd.to_numeric, errors="coerce")

    # Remove apenas features impossíveis de usar:
    # 1) totalmente ausentes;
    all_nan = [c for c in feature_cols if Xdf[c].notna().sum() == 0]
    usable = [c for c in feature_cols if c not in all_nan]

    # Imputação mediana dos NaNs residuais.
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(Xdf[usable])

    # Remove variância zero após imputação.
    variances = np.nanvar(X_imp, axis=0)
    nonconstant_mask = np.isfinite(variances) & (variances > 0)
    constant = [
        c for c, keep in zip(usable, nonconstant_mask)
        if not keep
    ]
    final_features = [
        c for c, keep in zip(usable, nonconstant_mask)
        if keep
    ]
    X_imp = X_imp[:, nonconstant_mask]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp).astype("float32")

    raw_path = args.output_dir / "onset_snapshot_raw.parquet"
    scaled_path = args.output_dir / "onset_snapshot_scaled.npy"
    meta_path = args.output_dir / "onset_snapshot_meta.parquet"
    features_path = args.output_dir / "onset_snapshot_features.csv"
    prep_path = args.output_dir / "onset_snapshot_preprocessing.npz"

    snapshot[["stay_id", "first_positive_time", *final_features]].to_parquet(
        raw_path, index=False
    )
    np.save(scaled_path, X_scaled)

    snapshot[["stay_id", "first_positive_time"]].to_parquet(
        meta_path, index=False
    )

    coverage_rows = []
    for c in feature_cols:
        n_obs = int(observed_counts[c])
        coverage_rows.append({
            "feature": c,
            "included_final": c in final_features,
            "observed_at_onset_n": n_obs,
            "observed_at_onset_pct": 100.0 * n_obs / expected_n,
            "all_nan": c in all_nan,
            "zero_variance_after_imputation": c in constant,
        })
    pd.DataFrame(coverage_rows).to_csv(features_path, index=False)

    np.savez(
        prep_path,
        features=np.array(final_features, dtype=object),
        imputer_statistics=imputer.statistics_[
            [i for i, c in enumerate(usable) if c in final_features]
        ],
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
    )

    print("\n=== MATRIZ FINAL ===")
    print(f"Pacientes:                 {X_scaled.shape[0]:,}")
    print(f"Features candidatas:       {len(feature_cols)}")
    print(f"Features totalmente NaN:   {len(all_nan)}")
    print(f"Features variância zero:   {len(constant)}")
    print(f"Features finais:           {X_scaled.shape[1]}")
    print(f"NaNs após processamento:   {int(np.isnan(X_scaled).sum()):,}")

    if X_scaled.shape[0] != expected_n or np.isnan(X_scaled).any():
        raise RuntimeError("Matriz final inválida.")

    print("\n[OK] SNAPSHOT NO ONSET CONSTRUÍDO")
    print(f"Raw:        {raw_path}")
    print(f"Scaled:     {scaled_path}")
    print(f"Meta:       {meta_path}")
    print(f"Features:   {features_path}")
    print(f"Preprocess: {prep_path}")


if __name__ == "__main__":
    main()
