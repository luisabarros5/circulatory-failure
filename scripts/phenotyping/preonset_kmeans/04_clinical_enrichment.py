#!/usr/bin/env python3
"""Clinical enrichment of the frozen original-space k=2 solution.

This module never refits or changes the clustering model.  It links the frozen
assignments to admission-coded Charlson comorbidities and to a 5-minute
vasopressor time grid.  The validation split is the primary inferential scope.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu, norm

from inference_utils import modified_poisson


CHARLSON = {
    "myocardial_infarct": ("Myocardial infarction", 1),
    "congestive_heart_failure": ("Congestive heart failure", 1),
    "peripheral_vascular_disease": ("Peripheral vascular disease", 1),
    "cerebrovascular_disease": ("Cerebrovascular disease", 1),
    "dementia": ("Dementia", 1),
    "chronic_pulmonary_disease": ("Chronic pulmonary disease", 1),
    "rheumatic_disease": ("Rheumatic disease", 1),
    "peptic_ulcer_disease": ("Peptic ulcer disease", 1),
    "mild_liver_disease": ("Mild liver disease", 1),
    "diabetes_without_cc": ("Diabetes without complications", 1),
    "diabetes_with_cc": ("Diabetes with complications", 2),
    "paraplegia": ("Paraplegia", 2),
    "renal_disease": ("Moderate/severe renal disease", 2),
    "malignant_cancer": ("Malignant cancer", 2),
    "severe_liver_disease": ("Severe liver disease", 3),
    "metastatic_solid_tumor": ("Metastatic solid tumour", 6),
    "aids": ("AIDS", 6),
}


def bh_adjust(p_values: pd.Series) -> np.ndarray:
    p = np.asarray(p_values, float)
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    values = p[ok]
    if not len(values):
        return out
    order = np.argsort(values)
    ranked = values[order]
    adjusted = np.minimum.accumulate((ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1])[::-1]
    restored = np.empty(len(values)); restored[order] = np.clip(adjusted, 0, 1)
    out[ok] = restored
    return out


def load_official_vasopressor(assignments: pd.DataFrame, mimic_root: Path, sql_dir: Path) -> pd.DataFrame:
    """Build official MIMIC-IV vasoactive intervals locally from inputevents."""
    con = duckdb.connect(":memory:")
    con.execute("CREATE SCHEMA mimiciv_icu; CREATE SCHEMA mimiciv_derived")
    wanted = assignments[["stay_id"]].drop_duplicates().copy()
    wanted["stay_id"] = wanted.stay_id.astype("int64")
    con.register("wanted_stays", wanted)
    inputevents = str(mimic_root / "icu" / "icu" / "inputevents.csv").replace("'", "''")
    itemids = "221662,221289,221906,221749,222315,221653,221986"
    con.execute(f"""
        CREATE TABLE mimiciv_icu.inputevents AS
        SELECT stay_id, linkorderid, rate, rateuom, patientweight, amount, starttime, endtime, itemid
        FROM read_csv_auto('{inputevents}', header=true)
        WHERE itemid IN ({itemids}) AND stay_id IN (SELECT stay_id FROM wanted_stays)
    """)
    for name in ["dopamine", "epinephrine", "norepinephrine", "phenylephrine",
                 "vasopressin", "dobutamine", "milrinone", "vasoactive_agent"]:
        con.execute((sql_dir / f"{name}.sql").read_text())
    drugs = ["dopamine", "epinephrine", "norepinephrine", "phenylephrine",
             "vasopressin", "dobutamine", "milrinone"]
    positive = ", ".join(f"CASE WHEN {x} > 0 THEN {x} ELSE 0 END AS {x}" for x in drugs)
    intervals = con.execute(f"""
        SELECT stay_id, starttime, endtime, {positive}
        FROM mimiciv_derived.vasoactive_agent
    """).df()
    con.close()
    if intervals.empty:
        return pd.DataFrame(columns=["stay_id", "starttime", "endtime", *drugs, "active", "nee"])
    intervals["starttime"] = pd.to_datetime(intervals.starttime)
    intervals["endtime"] = pd.to_datetime(intervals.endtime)
    intervals["active"] = intervals[drugs].gt(0).any(axis=1)
    intervals["nee"] = (intervals.norepinephrine + intervals.epinephrine
                         + intervals.phenylephrine / 10 + intervals.dopamine / 100
                         + intervals.vasopressin * 2.5 / 60)
    return intervals


def load_official_charlson(assignments: pd.DataFrame, mimic_root: Path, sql_path: Path) -> pd.DataFrame:
    """Execute the official MIMIC-IV DuckDB Charlson concept for all admissions."""
    con = duckdb.connect(":memory:")
    con.execute("CREATE SCHEMA mimiciv_hosp; CREATE SCHEMA mimiciv_derived")
    diagnoses = str(mimic_root / "hosp" / "diagnoses_icd.csv.gz").replace("'", "''")
    admissions = str(mimic_root / "hosp" / "admissions.csv.gz").replace("'", "''")
    con.execute(f"CREATE VIEW mimiciv_hosp.diagnoses_icd AS SELECT * FROM read_csv_auto('{diagnoses}', header=true)")
    con.execute(f"CREATE VIEW mimiciv_hosp.admissions AS SELECT * FROM read_csv_auto('{admissions}', header=true)")
    ages = assignments[["hadm_id", "age"]].dropna(subset=["hadm_id"]).copy()
    ages["hadm_id"] = ages.hadm_id.astype("int64")
    ages = ages.groupby("hadm_id", as_index=False).age.first()
    con.register("target_ages", ages)
    con.execute("CREATE TABLE mimiciv_derived.age AS SELECT * FROM target_ages")
    con.execute(sql_path.read_text())
    wanted = assignments[["hadm_id"]].dropna().copy(); wanted["hadm_id"] = wanted.hadm_id.astype("int64")
    con.register("wanted_hadm", wanted.drop_duplicates())
    selected = con.execute(
        "SELECT c.* FROM mimiciv_derived.charlson c INNER JOIN wanted_hadm w USING (hadm_id)"
    ).df()
    con.close()
    return selected[["hadm_id", *CHARLSON]].copy()


def charlson_score(frame: pd.DataFrame) -> pd.Series:
    # Mutually exclusive severity strata avoid double counting diabetes, liver,
    # and cancer categories in the original Charlson weighting scheme.
    score = pd.Series(0.0, index=frame.index)
    simple = ["myocardial_infarct", "congestive_heart_failure", "peripheral_vascular_disease",
              "cerebrovascular_disease", "dementia", "chronic_pulmonary_disease",
              "rheumatic_disease", "peptic_ulcer_disease", "paraplegia", "renal_disease", "aids"]
    for variable in simple:
        score += frame[variable] * CHARLSON[variable][1]
    score += np.where(frame.diabetes_with_cc.eq(1), 2, frame.diabetes_without_cc)
    score += np.where(frame.severe_liver_disease.eq(1), 3, frame.mild_liver_disease)
    score += np.where(frame.metastatic_solid_tumor.eq(1), 6, 2 * frame.malignant_cancer)
    return score


def aggregate(assignments: pd.DataFrame, intervals: pd.DataFrame, charlson: pd.DataFrame) -> pd.DataFrame:
    flags = charlson.copy()
    for variable in CHARLSON:
        flags[variable] = pd.to_numeric(flags[variable], errors="coerce").fillna(0).gt(0).astype(int)
    flags["charlson_score_no_age"] = charlson_score(flags)
    flags["charlson_condition_count"] = flags[list(CHARLSON)].sum(axis=1)

    base = assignments.copy()
    base["onset_time"] = pd.to_datetime(base.onset_time)
    base["outtime"] = pd.to_datetime(base.outtime)
    time = intervals.merge(base[["stay_id", "onset_time"]], on="stay_id", how="inner", validate="many_to_one")
    time["active_pre2h"] = time.active & time.endtime.ge(time.onset_time - pd.Timedelta(hours=2)) & time.starttime.lt(time.onset_time)
    time["active_at_onset"] = time.active & time.starttime.le(time.onset_time) & time.endtime.ge(time.onset_time)
    time["active_post6h"] = time.active & time.endtime.ge(time.onset_time) & time.starttime.lt(time.onset_time + pd.Timedelta(hours=6))
    time["active_post24h"] = time.active & time.endtime.ge(time.onset_time) & time.starttime.lt(time.onset_time + pd.Timedelta(hours=24))
    time["nee_post6h"] = time.nee.where(time.endtime.ge(time.onset_time) & time.starttime.lt(time.onset_time + pd.Timedelta(hours=6)))
    time["nee_post24h"] = time.nee.where(time.endtime.ge(time.onset_time) & time.starttime.lt(time.onset_time + pd.Timedelta(hours=24)))
    vaso = time.groupby("stay_id", observed=True).agg(
        vasopressor_pre2h=("active_pre2h", "max"),
        vasopressor_at_onset=("active_at_onset", "max"),
        vasopressor_post6h=("active_post6h", "max"),
        vasopressor_post24h=("active_post24h", "max"),
        nee_max_post6h=("nee_post6h", "max"),
        nee_max_post24h=("nee_post24h", "max"),
    ).reset_index()
    base["hadm_id"] = pd.to_numeric(base.hadm_id, errors="coerce").astype("Int64")
    flags["hadm_id"] = pd.to_numeric(flags.hadm_id, errors="coerce").astype("Int64")
    out = base.merge(flags, on="hadm_id", how="left", validate="many_to_one")
    out = out.merge(vaso, on="stay_id", how="left", validate="one_to_one")
    vaso_flags = ["vasopressor_pre2h", "vasopressor_at_onset", "vasopressor_post6h", "vasopressor_post24h"]
    for variable in vaso_flags:
        out[variable] = out[variable].eq(True)
    for variable in ["nee_max_post6h", "nee_max_post24h"]:
        out[variable] = pd.to_numeric(out[variable], errors="coerce").fillna(0.0)
    out["observed_through_6h"] = out.outtime.ge(out.onset_time + pd.Timedelta(hours=6))
    out["observed_through_24h"] = out.outtime.ge(out.onset_time + pd.Timedelta(hours=24))
    out["new_vasopressor_post6h"] = out.vasopressor_post6h & ~out.vasopressor_pre2h
    out["high_dose_nee_post6h"] = out.nee_max_post6h.ge(0.2)
    out["high_dose_nee_post24h"] = out.nee_max_post24h.ge(0.2)
    return out


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


def rr_model(data: pd.DataFrame, outcome: str, adjusted: bool) -> dict:
    d = data[data[outcome].notna()].copy()
    y = pd.to_numeric(d[outcome], errors="coerce").astype(float).to_numpy()
    x, names = design_matrix(d, adjusted)
    beta, covariance, converged = modified_poisson(y, x)
    j = names.index("cluster_1"); se = float(np.sqrt(max(covariance[j, j], 0)))
    limits = np.exp(beta[j] + np.array([-1, 1]) * 1.96 * se)
    return {"n": len(d), "events": int(y.sum()), "risk_ratio_c1_vs_c0": float(np.exp(beta[j])),
            "ci95_low": float(limits[0]), "ci95_high": float(limits[1]),
            "p_value": float(2 * norm.sf(abs(beta[j] / se))) if se > 0 else np.nan,
            "converged": bool(converged)}


def binary_tables(enriched: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    prevalence_rows, model_rows = [], []
    outcomes = list(CHARLSON) + ["vasopressor_pre2h", "vasopressor_at_onset", "vasopressor_post6h",
                                 "new_vasopressor_post6h", "high_dose_nee_post6h",
                                 "vasopressor_post24h", "high_dose_nee_post24h"]
    for scope, source in [("validation_primary", enriched[enriched.split.eq("validation")]),
                          ("derivation_replication", enriched[enriched.split.eq("derivation")])]:
        for outcome in outcomes:
            data = source.copy()
            if outcome in {"vasopressor_post6h", "new_vasopressor_post6h", "high_dose_nee_post6h"}:
                data = data[data.observed_through_6h]
            elif outcome in {"vasopressor_post24h", "high_dose_nee_post24h"}:
                data = data[data.observed_through_24h]
            data = data[data[outcome].notna()]
            counts = []
            for cluster in [0, 1]:
                values = pd.to_numeric(data.loc[data.cluster_original.eq(cluster), outcome], errors="coerce").dropna()
                counts.append((len(values), int(values.sum()), float(values.mean()) if len(values) else np.nan))
                prevalence_rows.append({"scope": scope, "outcome": outcome, "cluster": cluster,
                                        "n": len(values), "events": int(values.sum()), "prevalence": float(values.mean())})
            table = [[counts[1][1], counts[1][0] - counts[1][1]], [counts[0][1], counts[0][0] - counts[0][1]]]
            odds_ratio, fisher_p = fisher_exact(table)
            for adjusted in [False, True]:
                model_rows.append({"scope": scope, "outcome": outcome,
                                   "family": "comorbidity" if outcome in CHARLSON else "vasopressor",
                                   "model": "adjusted" if adjusted else "unadjusted",
                                   "absolute_difference_c1_minus_c0": counts[1][2] - counts[0][2],
                                   "fisher_odds_ratio": float(odds_ratio), "fisher_p_value": float(fisher_p),
                                   **rr_model(data, outcome, adjusted)})
    models = pd.DataFrame(model_rows)
    # Multiplicity correction is applied separately by family, scope, and model.
    models["q_value_bh"] = np.nan
    for (_, _, family), idx in models.groupby(["scope", "model", "family"]).groups.items():
        models.loc[idx, "q_value_bh"] = bh_adjust(models.loc[idx, "p_value"])
    return pd.DataFrame(prevalence_rows), models


def continuous_table(enriched: pd.DataFrame) -> pd.DataFrame:
    rows = []
    variables = ["charlson_score_no_age", "charlson_condition_count", "nee_max_post6h", "nee_max_post24h"]
    for scope, source in [("validation_primary", enriched[enriched.split.eq("validation")]),
                          ("derivation_replication", enriched[enriched.split.eq("derivation")])]:
        for variable in variables:
            data = source.copy()
            if variable == "nee_max_post6h": data = data[data.observed_through_6h]
            if variable == "nee_max_post24h": data = data[data.observed_through_24h]
            values = [pd.to_numeric(data.loc[data.cluster_original.eq(c), variable], errors="coerce").dropna() for c in [0, 1]]
            test = mannwhitneyu(values[1], values[0], alternative="two-sided") if all(len(x) for x in values) else None
            for cluster, x in enumerate(values):
                rows.append({"scope": scope, "variable": variable, "cluster": cluster, "n": len(x),
                             "median": float(x.median()), "q1": float(x.quantile(.25)), "q3": float(x.quantile(.75)),
                             "mannwhitney_p_value": float(test.pvalue) if test else np.nan})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--mimic-root", type=Path, required=True)
    parser.add_argument("--charlson-sql", type=Path, required=True)
    parser.add_argument("--vaso-sql-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(); args.out_dir.mkdir(parents=True, exist_ok=True)
    assignments = pd.read_parquet(args.assignments)
    intervals = load_official_vasopressor(assignments, args.mimic_root, args.vaso_sql_dir)
    charlson = load_official_charlson(assignments, args.mimic_root, args.charlson_sql)
    enriched = aggregate(assignments, intervals, charlson)
    prevalence, models = binary_tables(enriched)
    continuous = continuous_table(enriched)
    enriched.to_parquet(args.out_dir / "patient_level_enrichment.parquet", index=False)
    prevalence.to_csv(args.out_dir / "absolute_prevalence_by_cluster.csv", index=False)
    models.to_csv(args.out_dir / "modified_poisson_associations.csv", index=False)
    continuous.to_csv(args.out_dir / "continuous_characteristics.csv", index=False)
    coverage = {
        "n_frozen_assignments": int(len(assignments)),
        "n_vasoactive_intervals": int(len(intervals)),
        "n_stays_with_vasoactive_intervals": int(intervals.stay_id.nunique()),
        "n_vasopressor_evaluable": int(len(enriched)),
        "n_charlson_matched": int(enriched.charlson_score_no_age.notna().sum()),
        "n_validation": int(enriched.split.eq("validation").sum()),
        "n_validation_observed_through_6h": int((enriched.split.eq("validation") & enriched.observed_through_6h).sum()),
        "n_validation_observed_through_24h": int((enriched.split.eq("validation") & enriched.observed_through_24h).sum()),
        "cluster_model_refit": False,
        "primary_scope": "frozen validation split",
        "interpretation": "clinical enrichment/association, not causal treatment effect",
        "vasopressor_circularity_warning": "vasopressor is a component of the circulatory-failure definition",
        "vasopressor_source": "official MIMIC-IV DuckDB medication concepts reconstructed locally from inputevents",
        "charlson_source": "official MIMIC-IV DuckDB concept applied directly to diagnoses_icd",
        "comorbidity_temporality_warning": "Charlson flags use diagnoses coded for the hospital admission and are not guaranteed to be known prospectively at onset",
        "post_onset_window_rule": "6h and 24h analyses require temporal observation through the full window",
        "adjustment": "age, sex, race, first ICU, log ICU-to-onset time",
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(coverage, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(coverage, indent=2, ensure_ascii=False))
    print("\nVALIDATION ADJUSTED ASSOCIATIONS")
    print(models[(models.scope.eq("validation_primary")) & models.model.eq("adjusted")].to_string(index=False))


if __name__ == "__main__":
    main()
