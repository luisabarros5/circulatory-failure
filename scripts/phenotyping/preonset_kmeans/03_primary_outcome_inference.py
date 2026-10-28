#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from inference_utils import modified_poisson
from common import write_json


def design_matrix(data: pd.DataFrame, adjusted: bool) -> tuple[np.ndarray, list[str]]:
    frame = pd.DataFrame({"cluster_1": data.cluster_original.astype(float)}, index=data.index)
    if adjusted:
        age = pd.to_numeric(data.age, errors="coerce")
        onset = np.log1p(pd.to_numeric(data.hours_icu_to_onset, errors="coerce").clip(lower=0))
        frame["age_per_10y"] = (age.fillna(age.median()) - age.median()) / 10
        frame["log1p_hours_to_onset"] = onset.fillna(onset.median())
        frame["male"] = data.gender.eq("M").astype(float)
        race = data.race.fillna("UNKNOWN").astype(str).str.upper().map(
            lambda x: next((key for key in ["WHITE", "BLACK", "HISPANIC", "ASIAN"] if key in x), "OTHER_UNKNOWN")
        )
        frame = pd.concat([frame, pd.get_dummies(race, prefix="race", drop_first=True, dtype=float),
                           pd.get_dummies(data.first_careunit.fillna("UNKNOWN"), prefix="unit", drop_first=True, dtype=float)], axis=1)
    frame.insert(0, "intercept", 1.0)
    return frame.to_numpy(float), frame.columns.tolist()


def fit(data: pd.DataFrame, outcome: str, adjusted: bool) -> dict:
    d = data[data[outcome].notna()].copy(); y = pd.to_numeric(d[outcome], errors="coerce").to_numpy(float)
    x, names = design_matrix(d, adjusted); beta, covariance, converged = modified_poisson(y, x)
    j = names.index("cluster_1"); se = float(np.sqrt(max(covariance[j, j], 0))); rr = float(np.exp(beta[j]))
    lo, hi = np.exp(beta[j] + np.array([-1, 1]) * 1.96 * se); z = beta[j] / se
    return {"outcome": outcome, "model": "adjusted" if adjusted else "unadjusted", "n": len(d), "events": int(y.sum()),
            "risk_ratio_c1_vs_c0": rr, "ci95_low": float(lo), "ci95_high": float(hi),
            "p_value": float(2 * norm.sf(abs(z))), "converged": converged}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--assignments", type=Path, required=True); parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(); args.out_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_parquet(args.assignments)
    data["rrt_any"] = pd.to_numeric(data.rrt_on, errors="coerce").clip(0, 1)
    data["ventilation_any"] = (pd.to_numeric(data.ventilation_hours, errors="coerce") > 0).where(data.ventilation_hours.notna())
    outcomes = ["hospital_expire_flag", "mortality_28d", "icu_mortality", "rrt_any", "ventilation_any"]
    models = []; risks = []
    for scope, subset in [("validation_primary", data[data.split.eq("validation")]), ("all_secondary", data)]:
        for outcome in outcomes:
            for adjusted in [False, True]: models.append({**fit(subset, outcome, adjusted), "scope": scope})
            for cluster in [0, 1]:
                values = pd.to_numeric(subset.loc[subset.cluster_original.eq(cluster), outcome], errors="coerce").dropna()
                risks.append({"scope": scope, "outcome": outcome, "cluster": cluster, "n": len(values), "events": int(values.sum()), "risk": float(values.mean())})
    model_df = pd.DataFrame(models); risk_df = pd.DataFrame(risks)
    model_df.to_csv(args.out_dir / "modified_poisson.csv", index=False); risk_df.to_csv(args.out_dir / "absolute_risks.csv", index=False)
    write_json(args.out_dir / "manifest.json", {"primary_scope": "frozen validation", "causal_claim": False,
               "adjustment": "age, sex, race, first ICU, log ICU-to-onset time", "all_converged": bool(model_df.converged.all())})
    print(risk_df[risk_df.scope.eq("validation_primary")].to_string(index=False))
    print(model_df[(model_df.scope.eq("validation_primary")) & model_df.model.eq("adjusted")].to_string(index=False))


if __name__ == "__main__":
    main()
