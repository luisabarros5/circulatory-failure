#!/usr/bin/env python3
"""Confirmatory multi-outcome and incremental-value inference for frozen phenotypes."""
from __future__ import annotations

import argparse
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


def load_module(path: Path, name: str):
    spec = spec_from_file_location(name, path); module = module_from_spec(spec)
    assert spec.loader is not None; spec.loader.exec_module(module); return module


def bh(p: pd.Series) -> np.ndarray:
    values = p.to_numpy(float); order = np.argsort(values); ranked = values[order]
    adjusted = np.minimum.accumulate((ranked * len(ranked) / np.arange(1, len(ranked)+1))[::-1])[::-1]
    out = np.empty(len(values)); out[order] = np.clip(adjusted, 0, 1); return out


def bootstrap_metric_delta(y: np.ndarray, p0: np.ndarray, p1: np.ndarray, metric, seed: int, n_boot: int = 2000):
    rng = np.random.default_rng(seed); deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2: continue
        deltas.append(metric(y[idx], p1[idx]) - metric(y[idx], p0[idx]))
    return float(np.quantile(deltas, .025)), float(np.quantile(deltas, .975))


def incremental_value(data: pd.DataFrame, helper, seed: int) -> pd.DataFrame:
    deriv = data[data.split.eq("derivation")].copy(); valid = data[data.split.eq("validation")].copy()
    x0d = helper.prognostic_matrix(deriv); x0v = helper.prognostic_matrix(valid).reindex(columns=x0d.columns, fill_value=0)
    x1d = x0d.copy(); x1d["phenotype_C1"] = deriv.cluster_original.to_numpy()
    x1v = x0v.copy(); x1v["phenotype_C1"] = valid.cluster_original.to_numpy()
    rows = []
    for outcome in ["mortality_28d", "hospital_expire_flag", "icu_mortality"]:
        train_ok = deriv[outcome].notna(); val_ok = valid[outcome].notna()
        ytr = deriv.loc[train_ok, outcome].astype(int).to_numpy(); y = valid.loc[val_ok, outcome].astype(int).to_numpy()
        base = helper.model().fit(x0d.loc[train_ok], ytr)
        plus = helper.model().fit(x1d.loc[train_ok], ytr)
        p0 = base.predict_proba(x0v.loc[val_ok])[:, 1]; p1 = plus.predict_proba(x1v.loc[val_ok])[:, 1]
        metrics = {"auroc": roc_auc_score, "auprc": average_precision_score,
                   "brier": lambda yy, pp: brier_score_loss(yy, pp),
                   "log_loss": lambda yy, pp: log_loss(yy, pp, labels=[0, 1])}
        for metric_name, metric in metrics.items():
            v0, v1 = float(metric(y, p0)), float(metric(y, p1)); delta = v1-v0
            lo, hi = bootstrap_metric_delta(y, p0, p1, metric, seed + len(rows))
            rows.append({"outcome": outcome, "metric": metric_name, "n_validation": len(y),
                         "events_validation": int(y.sum()), "baseline_model": v0,
                         "baseline_plus_cluster": v1, "delta_plus_cluster": delta,
                         "delta_ci95_low": lo, "delta_ci95_high": hi})
    return pd.DataFrame(rows)


def outcome_frame(data: pd.DataFrame, outcome: str) -> pd.DataFrame:
    d = data.copy()
    if outcome == "ventilation_any":
        d[outcome] = (pd.to_numeric(d.ventilation_hours, errors="coerce") > 0).where(d.ventilation_hours.notna())
    if outcome == "rrt_any": d[outcome] = pd.to_numeric(d.rrt_on, errors="coerce").clip(0, 1)
    if outcome == "high_dose_nee_post6h": d = d[d.observed_through_6h]
    if outcome == "high_dose_nee_post24h": d = d[d.observed_through_24h]
    return d[d[outcome].notna()].copy()


def multi_outcome(data: pd.DataFrame, helper) -> tuple[pd.DataFrame, pd.DataFrame]:
    outcomes = ["mortality_28d", "hospital_expire_flag", "icu_mortality", "rrt_any",
                "ventilation_any", "high_dose_nee_post6h", "high_dose_nee_post24h"]
    rows, balances = [], []
    for scope, source in [("all", data), ("derivation", data[data.split.eq("derivation")]),
                          ("validation", data[data.split.eq("validation")])]:
        for oi, outcome in enumerate(outcomes):
            d = outcome_frame(source, outcome); d["treatment"] = d.cluster_original.astype(int)
            d["mortality_28d_original"] = d.mortality_28d
            d["mortality_28d"] = pd.to_numeric(d[outcome], errors="coerce").astype(int)
            x = helper.prognostic_matrix(d); pred = helper.crossfit(d.reset_index(drop=True), x.reset_index(drop=True), 500 + oi)
            for summary in [helper.aipw_summary(pred, scope, "C1_vs_C0"),
                            helper.overlap_summary(pred, scope, "C1_vs_C0"),
                            helper.psm_summary(pred, scope, "C1_vs_C0")]:
                summary["outcome"] = outcome; rows.append(summary)
            balances.append({"scope": scope, "outcome": outcome,
                             "max_smd_unweighted": helper.max_smd(x, d.treatment.to_numpy()),
                             "max_smd_overlap": helper.max_smd(x, d.treatment.to_numpy(), pred.overlap_weight.to_numpy())})
    result = pd.DataFrame(rows)
    result = result.rename(columns={"risk_no_early_vaso": "risk_C0", "risk_early_vaso": "risk_C1"})
    result["q_value_bh"] = np.nan
    for (_, estimator), idx in result.groupby(["scope", "estimator"]).groups.items():
        if result.loc[idx, "p_value"].notna().all(): result.loc[idx, "q_value_bh"] = bh(result.loc[idx, "p_value"])
    return result, pd.DataFrame(balances)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--enrichment", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(); args.out_dir.mkdir(parents=True, exist_ok=True)
    helper = load_module(Path(__file__).with_name("phenotype_inference_utils.py"), "phenotype_utils")
    data = pd.read_parquet(args.enrichment)
    results, balance = multi_outcome(data, helper)
    incremental = incremental_value(data, helper, 800)
    results.to_csv(args.out_dir / "multi_outcome_balanced_associations.csv", index=False)
    balance.to_csv(args.out_dir / "multi_outcome_balance.csv", index=False)
    incremental.to_csv(args.out_dir / "incremental_prognostic_value.csv", index=False)
    manifest = {"cluster_refit": False, "primary_scope": "frozen validation",
                "primary_outcome": "28-day mortality", "multiplicity": "BH within scope and estimator",
                "covariates": "age, sex, race, ICU, time to failure, 17 Charlson conditions",
                "cluster_variables_excluded_from_balancing": True,
                "interpretation": "case-mix-balanced prognostic association, not causal effect of cluster membership"}
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(results[(results.scope.eq("validation")) & results.estimator.eq("cross_fitted_AIPW")].to_string(index=False))
    print(incremental.to_string(index=False))


if __name__ == "__main__": main()
