#!/usr/bin/env python3
"""
07_prepare_log_curated_onset_snapshot.py

Versão final de pré-processamento para o snapshot clínico no primeiro onset.

Mantém:
- TODOS os 15.739 stays positivos;
- as MESMAS 20 variáveis clínicas da versão curated;
- o mesmo K-means no step seguinte.

Pipeline:
raw snapshot
    -> seleção das mesmas 20 features
    -> imputação mediana
    -> log1p em variáveis pré-definidas de forte assimetria
    -> winsorização 0.5%-99.5%
    -> StandardScaler
    -> K-means (step 05)

Nenhuma feature ou paciente é removido com base no resultado do clustering.

Variáveis transformadas por log1p:
- glucose
- lactate
- creatinine
- BUN
- AST
- ALT
- total bilirubin
- prothrombin time
- partial thromboplastin time

Observação:
- log1p só é aplicado após validar que os valores são >= 0.
- vasopressor_ativo permanece binário e não é winsorizado.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


FEATURES = [
    ("heart_rate", "Heart rate"),
    ("mbp", "Mean blood pressure"),
    ("resp_rate", "Respiratory rate"),
    ("temperature", "Temperature"),
    ("spo2", "SpO2"),
    ("glucose", "Glucose"),
    ("lab_50813", "Lactate"),
    ("lab_50983", "Sodium"),
    ("lab_50971", "Potassium"),
    ("lab_50912", "Creatinine"),
    ("lab_51006", "BUN"),
    ("lab_50878", "AST"),
    ("lab_50861", "ALT"),
    ("lab_50885", "Total bilirubin"),
    ("lab_50862", "Albumin"),
    ("lab_51222", "Hemoglobin"),
    ("lab_51265", "Platelets"),
    ("lab_51274", "Prothrombin time"),
    ("lab_51275", "Partial thromboplastin time"),
    ("vasopressor_ativo", "Active vasopressor"),
]

LOG_FEATURES = {
    "glucose",
    "lab_50813",  # lactate
    "lab_50912",  # creatinine
    "lab_51006",  # BUN
    "lab_50878",  # AST
    "lab_50861",  # ALT
    "lab_50885",  # total bilirubin
    "lab_51274",  # PT
    "lab_51275",  # PTT
}

BINARY_FEATURES = {"vasopressor_ativo"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-snapshot", required=True, type=Path)
    ap.add_argument("--meta", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--lower-q", type=float, default=0.005)
    ap.add_argument("--upper-q", type=float, default=0.995)
    args = ap.parse_args()

    if not (0 <= args.lower_q < args.upper_q <= 1):
        raise SystemExit("Quantis inválidos.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_parquet(args.raw_snapshot)
    meta = pd.read_parquet(args.meta)

    if "stay_id" not in raw.columns or "stay_id" not in meta.columns:
        raise SystemExit("stay_id ausente.")
    if raw["stay_id"].duplicated().any():
        raise SystemExit("Snapshot raw contém stay_id duplicado.")
    if meta["stay_id"].duplicated().any():
        raise SystemExit("Meta contém stay_id duplicado.")

    requested = [f for f, _ in FEATURES]
    missing = [f for f in requested if f not in raw.columns]
    if missing:
        raise SystemExit(
            "Features do painel ausentes no snapshot raw: "
            + ", ".join(missing)
        )

    # Garante exatamente a mesma população e ordem do meta.
    raw_ids = raw["stay_id"].astype("int64").to_numpy()
    meta_ids = meta["stay_id"].astype("int64").to_numpy()

    if len(raw_ids) != len(meta_ids):
        raise SystemExit(
            f"N diferente: raw={len(raw_ids)} meta={len(meta_ids)}"
        )

    if not np.array_equal(raw_ids, meta_ids):
        raw = meta[["stay_id"]].merge(
            raw, on="stay_id", how="left", validate="one_to_one"
        )

    if raw["stay_id"].isna().any():
        raise SystemExit("Falha ao alinhar raw e meta.")

    if len(raw) != 15739:
        print(
            f"[AVISO] N={len(raw):,}; o script preservará todos os pacientes, "
            "mas o esperado neste projeto é 15.739."
        )

    Xdf = raw[requested].apply(pd.to_numeric, errors="coerce")

    # Validação: nenhuma das 20 pode estar completamente ausente.
    all_nan = [c for c in requested if Xdf[c].notna().sum() == 0]
    if all_nan:
        raise SystemExit(
            "Features totalmente ausentes: " + ", ".join(all_nan)
        )

    # Guarda estatísticas antes do processamento.
    pre_stats = {}
    for c in requested:
        s = Xdf[c].dropna()
        pre_stats[c] = {
            "n_nonmissing": int(s.shape[0]),
            "pct_nonmissing": float(100.0 * s.shape[0] / len(Xdf)),
            "min_raw": float(s.min()) if len(s) else np.nan,
            "median_raw": float(s.median()) if len(s) else np.nan,
            "max_raw": float(s.max()) if len(s) else np.nan,
            "skew_raw": float(s.skew()) if len(s) > 2 else np.nan,
        }

    # Imputação mediana.
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(Xdf[requested]).astype(np.float64)

    # log1p nas variáveis pré-definidas.
    log_applied = []
    for j, c in enumerate(requested):
        if c not in LOG_FEATURES:
            continue

        min_val = float(np.nanmin(X[:, j]))
        if min_val < 0:
            raise RuntimeError(
                f"{c} tem valor negativo ({min_val}); log1p não será aplicado "
                "silenciosamente. Revise a variável."
            )

        X[:, j] = np.log1p(X[:, j])
        log_applied.append(c)

    # Winsorização após log, exceto binária.
    lower_bounds = np.empty(len(requested), dtype=np.float64)
    upper_bounds = np.empty(len(requested), dtype=np.float64)

    for j, c in enumerate(requested):
        col = X[:, j]

        if c in BINARY_FEATURES:
            lo = float(np.min(col))
            hi = float(np.max(col))
        else:
            lo = float(np.quantile(col, args.lower_q))
            hi = float(np.quantile(col, args.upper_q))
            X[:, j] = np.clip(col, lo, hi)

        lower_bounds[j] = lo
        upper_bounds[j] = hi

    # Nenhuma das 20 deve virar constante.
    variances = np.var(X, axis=0)
    zero_var = [
        c for c, v in zip(requested, variances)
        if (not np.isfinite(v)) or v <= 0
    ]
    if zero_var:
        raise RuntimeError(
            "Features com variância zero após processamento: "
            + ", ".join(zero_var)
        )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype("float32")

    if np.isnan(X_scaled).any() or np.isinf(X_scaled).any():
        raise RuntimeError("NaN/Inf após processamento.")

    if X_scaled.shape != (len(raw), len(requested)):
        raise RuntimeError(
            f"Shape inesperado: {X_scaled.shape}; "
            f"esperado ({len(raw)}, {len(requested)})"
        )

    label_map = dict(FEATURES)

    # Outputs.
    raw_out = args.output_dir / "onset_snapshot_log_curated_raw.parquet"
    scaled_out = args.output_dir / "onset_snapshot_log_curated_scaled.npy"
    features_out = args.output_dir / "onset_snapshot_log_curated_features.csv"
    prep_out = args.output_dir / "onset_snapshot_log_curated_preprocessing.npz"

    raw[["stay_id", "first_positive_time", *requested]].to_parquet(
        raw_out, index=False
    )
    np.save(scaled_out, X_scaled)

    manifest = []
    for j, c in enumerate(requested):
        st = pre_stats[c]
        manifest.append({
            "feature": c,
            "clinical_name": label_map[c],
            "log1p_applied": c in LOG_FEATURES,
            "n_nonmissing_before_imputation": st["n_nonmissing"],
            "pct_nonmissing_before_imputation": st["pct_nonmissing"],
            "min_raw": st["min_raw"],
            "median_raw": st["median_raw"],
            "max_raw": st["max_raw"],
            "skew_raw": st["skew_raw"],
            "imputation_median_raw": float(imputer.statistics_[j]),
            "winsor_lower_transformed_scale": float(lower_bounds[j]),
            "winsor_upper_transformed_scale": float(upper_bounds[j]),
            "scaler_mean_transformed": float(scaler.mean_[j]),
            "scaler_scale_transformed": float(scaler.scale_[j]),
        })

    pd.DataFrame(manifest).to_csv(features_out, index=False)

    np.savez(
        prep_out,
        features=np.array(requested, dtype=object),
        log_features=np.array(sorted(LOG_FEATURES), dtype=object),
        imputer_statistics_raw=imputer.statistics_,
        winsor_lower_transformed=lower_bounds,
        winsor_upper_transformed=upper_bounds,
        scaler_mean_transformed=scaler.mean_,
        scaler_scale_transformed=scaler.scale_,
        lower_q=args.lower_q,
        upper_q=args.upper_q,
    )

    print("=== SNAPSHOT LOG-CURATED ===")
    print(f"Pacientes mantidos:       {X_scaled.shape[0]:,}")
    print(f"Features mantidas:        {X_scaled.shape[1]}")
    print(f"Features com log1p:       {len(log_applied)}")
    print(f"Winsorização:             {args.lower_q*100:.1f}%–{args.upper_q*100:.1f}%")
    print(f"NaNs finais:              {int(np.isnan(X_scaled).sum()):,}")
    print(f"Infs finais:              {int(np.isinf(X_scaled).sum()):,}")
    print()
    print("log1p aplicado em:")
    for c in requested:
        if c in LOG_FEATURES:
            print(f"  - {c}: {label_map[c]}")
    print()
    print("[OK] REPRESENTAÇÃO FINAL CONSTRUÍDA")
    print(f"Raw:      {raw_out}")
    print(f"Scaled:   {scaled_out}")
    print(f"Features: {features_out}")
    print(f"Prep:     {prep_out}")


if __name__ == "__main__":
    main()
