#!/usr/bin/env python3
"""Shared estimators for case-mix-balanced phenotype comparisons.

These functions compare the frozen phenotypes as prognostic strata. They do
not estimate a treatment effect and deliberately exclude the eight variables
used to define the phenotypes from the balancing model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import norm
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


COMORBIDITIES = [
    "myocardial_infarct", "congestive_heart_failure",
    "peripheral_vascular_disease", "cerebrovascular_disease", "dementia",
    "chronic_pulmonary_disease", "rheumatic_disease",
    "peptic_ulcer_disease", "mild_liver_disease", "diabetes_without_cc",
    "diabetes_with_cc", "paraplegia", "renal_disease", "malignant_cancer",
    "severe_liver_disease", "metastatic_solid_tumor", "aids",
]


def prognostic_matrix(data: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame(index=data.index)
    x["age"] = pd.to_numeric(data.age, errors="coerce")
    x["log_hours_to_onset"] = np.log1p(
        pd.to_numeric(data.hours_icu_to_onset, errors="coerce").clip(lower=0)
    )
    x["male"] = data.gender.eq("M").astype(int)
    for feature in COMORBIDITIES:
        x[feature] = pd.to_numeric(data[feature], errors="coerce")
    race = data.race.fillna("UNKNOWN").astype(str).str.upper().map(
        lambda z: next(
            (key for key in ["WHITE", "BLACK", "HISPANIC", "ASIAN"] if key in z),
            "OTHER_UNKNOWN",
        )
    )
    categorical = pd.DataFrame(
        {"race": race, "unit": data.first_careunit.fillna("UNKNOWN")},
        index=data.index,
    )
    return pd.concat(
        [x, pd.get_dummies(categorical, drop_first=True, dtype=float)], axis=1
    ).astype(float)


def model() -> object:
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        LogisticRegression(C=0.5, max_iter=3000, solver="lbfgs"),
    )


def crossfit(data: pd.DataFrame, x: pd.DataFrame, seed: int = 43) -> pd.DataFrame:
    a = data.treatment.to_numpy(int)
    y = data.mortality_28d.to_numpy(int)
    n_splits = min(5, int(np.bincount(a).min()))
    if n_splits < 2:
        raise RuntimeError("Insufficient phenotype variation for cross-fitting")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    e = np.full(len(data), np.nan)
    mu0 = np.full(len(data), np.nan)
    mu1 = np.full(len(data), np.nan)
    for train, test in splitter.split(x, a):
        e[test] = model().fit(x.iloc[train], a[train]).predict_proba(x.iloc[test])[:, 1]
        for arm, output in [(0, mu0), (1, mu1)]:
            arm_train = train[a[train] == arm]
            if len(np.unique(y[arm_train])) < 2:
                output[test] = y[arm_train].mean()
            else:
                output[test] = model().fit(
                    x.iloc[arm_train], y[arm_train]
                ).predict_proba(x.iloc[test])[:, 1]
    e = np.clip(e, 0.025, 0.975)
    psi1 = mu1 + a / e * (y - mu1)
    psi0 = mu0 + (1 - a) / (1 - e) * (y - mu0)
    out = data[["stay_id", "cluster_original", "split", "treatment", "mortality_28d"]].copy()
    out["propensity"] = e
    out["mu0"] = mu0
    out["mu1"] = mu1
    out["psi0"] = psi0
    out["psi1"] = psi1
    out["effect_score"] = psi1 - psi0
    out["overlap_weight"] = np.where(a == 1, 1 - e, e)
    return out


def ci_mean(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, float)
    mean = float(np.mean(values))
    se = float(np.std(values, ddof=1) / np.sqrt(len(values)))
    return mean, mean - 1.96 * se, mean + 1.96 * se


def aipw_summary(pred: pd.DataFrame, scope: str, phenotype: str) -> dict:
    r0, _, _ = ci_mean(pred.psi0)
    r1, _, _ = ci_mean(pred.psi1)
    rd, lo, hi = ci_mean(pred.effect_score)
    se = (hi - lo) / (2 * 1.96)
    return {
        "scope": scope, "phenotype": phenotype, "estimator": "cross_fitted_AIPW",
        "n": len(pred), "treated": int(pred.treatment.sum()),
        "deaths": int(pred.mortality_28d.sum()), "risk_no_early_vaso": r0,
        "risk_early_vaso": r1, "risk_difference": rd,
        "risk_ratio": float(r1 / r0) if r0 > 0 else np.nan,
        "ci95_low": lo, "ci95_high": hi,
        "p_value": float(2 * norm.sf(abs(rd / se))) if se > 0 else np.nan,
        "propensity_min": float(pred.propensity.min()),
        "propensity_max": float(pred.propensity.max()),
        "effective_sample_size_overlap": float(
            pred.overlap_weight.sum() ** 2 / np.square(pred.overlap_weight).sum()
        ),
    }


def overlap_summary(pred: pd.DataFrame, scope: str, phenotype: str) -> dict:
    risks = []
    for arm in [0, 1]:
        z = pred[pred.treatment.eq(arm)]
        risks.append(float(np.average(z.mortality_28d, weights=z.overlap_weight)))
    return {
        "scope": scope, "phenotype": phenotype, "estimator": "overlap_weighting",
        "n": len(pred), "treated": int(pred.treatment.sum()),
        "deaths": int(pred.mortality_28d.sum()), "risk_no_early_vaso": risks[0],
        "risk_early_vaso": risks[1], "risk_difference": risks[1] - risks[0],
        "risk_ratio": float(risks[1] / risks[0]) if risks[0] > 0 else np.nan,
        "ci95_low": np.nan, "ci95_high": np.nan, "p_value": np.nan,
        "propensity_min": float(pred.propensity.min()),
        "propensity_max": float(pred.propensity.max()),
        "effective_sample_size_overlap": float(
            pred.overlap_weight.sum() ** 2 / np.square(pred.overlap_weight).sum()
        ),
    }


def greedy_match(pred: pd.DataFrame, caliper_sd: float = 0.2) -> pd.DataFrame:
    treated = pred[pred.treatment.eq(1)].copy()
    control = pred[pred.treatment.eq(0)].copy()
    if treated.empty or control.empty:
        return pd.DataFrame()
    score = logit(np.clip(pred.propensity.to_numpy(), 1e-5, 1 - 1e-5))
    caliper = caliper_sd * np.std(score)
    treated["logit_ps"] = logit(treated.propensity)
    control["logit_ps"] = logit(control.propensity)
    candidates = []
    for ti, row in treated.iterrows():
        candidates.append((float((control.logit_ps - row.logit_ps).abs().min()), ti))
    used, pairs = set(), []
    for _, ti in sorted(candidates):
        # Stable sort makes exact propensity-score ties reproducible across runs.
        distances = (control.logit_ps - treated.loc[ti, "logit_ps"]).abs().sort_values(kind="mergesort")
        available = [ci for ci in distances.index if ci not in used and distances.loc[ci] <= caliper]
        if not available:
            continue
        ci = available[0]
        used.add(ci)
        pairs.append({
            "pair": len(pairs), "treated_index": ti, "control_index": ci,
            "difference": float(
                treated.loc[ti, "mortality_28d"] - control.loc[ci, "mortality_28d"]
            ),
        })
    return pd.DataFrame(pairs)


def psm_summary(pred: pd.DataFrame, scope: str, phenotype: str) -> dict:
    pairs = greedy_match(pred)
    if pairs.empty:
        return {"scope": scope, "phenotype": phenotype, "estimator": "PSM_1to1", "n": len(pred)}
    difference = pairs.difference.to_numpy()
    rd, lo, hi = ci_mean(difference)
    se = (hi - lo) / (2 * 1.96)
    risk1 = float(pred.loc[pairs.treated_index.astype(int), "mortality_28d"].mean())
    risk0 = float(pred.loc[pairs.control_index.astype(int), "mortality_28d"].mean())
    return {
        "scope": scope, "phenotype": phenotype, "estimator": "PSM_1to1",
        "n": len(pred), "treated": int(pred.treatment.sum()),
        "deaths": int(pred.mortality_28d.sum()), "matched_pairs": len(pairs),
        "risk_no_early_vaso": risk0, "risk_early_vaso": risk1,
        "risk_difference": rd, "ci95_low": lo, "ci95_high": hi,
        "risk_ratio": float(risk1 / risk0) if risk0 > 0 else np.nan,
        "p_value": float(2 * norm.sf(abs(rd / se))) if se > 0 else np.nan,
        "propensity_min": float(pred.propensity.min()),
        "propensity_max": float(pred.propensity.max()),
        "effective_sample_size_overlap": np.nan,
    }


def max_smd(x: pd.DataFrame, a: np.ndarray, weights: np.ndarray | None = None) -> float:
    xx = x.copy().replace([np.inf, -np.inf], np.nan).fillna(x.median(numeric_only=True)).fillna(0)
    values = []
    for col in xx:
        z = xx[col].to_numpy(float)
        z0, z1 = z[a == 0], z[a == 1]
        if weights is None:
            m0, m1, v0, v1 = z0.mean(), z1.mean(), z0.var(), z1.var()
        else:
            w0, w1 = weights[a == 0], weights[a == 1]
            m0, m1 = np.average(z0, weights=w0), np.average(z1, weights=w1)
            v0 = np.average((z0 - m0) ** 2, weights=w0)
            v1 = np.average((z1 - m1) ** 2, weights=w1)
        denominator = np.sqrt((v0 + v1) / 2)
        if denominator > 1e-12:
            values.append(abs((m1 - m0) / denominator))
    return float(max(values)) if values else 0.0
