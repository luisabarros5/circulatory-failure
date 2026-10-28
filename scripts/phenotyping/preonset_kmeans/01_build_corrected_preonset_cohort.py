#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

from common import clinical_columns, read_parquet, subject_split, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Corrected first-failure cohort and fixed pre-onset tensor.")
    p.add_argument("--clean-dir", type=Path, required=True)
    p.add_argument("--labels-dir", type=Path, required=True)
    p.add_argument("--stay-subject-map", type=Path, required=True)
    p.add_argument("--mimic-root", type=Path, required=True)
    p.add_argument("--secondary-outcomes", type=Path)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--window-hours", type=float, default=2.0)
    p.add_argument("--step-minutes", type=int, default=5)
    p.add_argument("--max-files", type=int)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    clean_files = sorted(args.clean_dir.glob("batch_*.parquet"))
    if args.max_files:
        clean_files = clean_files[: args.max_files]
    if not clean_files:
        raise SystemExit("No clean batch files found.")
    label_files = [args.labels_dir / p.name for p in clean_files]
    missing = [str(p) for p in label_files if not p.exists()]
    if missing:
        raise SystemExit(f"Missing paired labels, first={missing[0]}")

    schema = pq.ParquetFile(clean_files[0]).schema_arrow.names
    features = clinical_columns(schema)
    mask_cols = [f"{c}_imputed" for c in features]
    if not all(c in schema for c in mask_cols):
        absent = [c for c in mask_cols if c not in schema]
        raise SystemExit(f"Missing imputation masks: {absent[:5]}")

    onset_parts = []
    for lp in tqdm(label_files, desc="Locate corrected onsets"):
        lab = read_parquet(lp, ["stay_id", "charttime", "failure_state_new_at_t"])
        lab["failure_state_new_at_t"] = lab["failure_state_new_at_t"].fillna(False).astype(bool)
        pos = lab.loc[lab["failure_state_new_at_t"], ["stay_id", "charttime"]]
        if len(pos):
            onset_parts.append(pos.groupby("stay_id", as_index=False)["charttime"].min())
    onset = pd.concat(onset_parts, ignore_index=True).drop_duplicates("stay_id")
    onset["stay_id"] = pd.to_numeric(onset["stay_id"], errors="raise").astype("int64")
    onset["onset_time"] = pd.to_datetime(onset.pop("charttime"))

    mapping = read_parquet(args.stay_subject_map, ["stay_id", "subject_id"])
    mapping = mapping.drop_duplicates("stay_id")
    onset = onset.merge(mapping, on="stay_id", how="left", validate="one_to_one")
    if onset["subject_id"].isna().any():
        raise RuntimeError(f"Missing subject_id for {onset.subject_id.isna().sum()} failure stays")
    onset["subject_id"] = onset["subject_id"].astype("int64")

    t_steps = int(round(args.window_hours * 60 / args.step_minutes))
    n, d = len(onset), len(features)
    row_for_stay = {int(s): i for i, s in enumerate(onset.stay_id)}
    values = np.full((n, t_steps, d), np.nan, dtype=np.float32)
    observed = np.zeros((n, t_steps, d), dtype=np.uint8)
    time_present = np.zeros((n, t_steps), dtype=np.uint8)

    for fp in tqdm(clean_files, desc="Build strict pre-onset windows"):
        cols = ["stay_id", "charttime", *features, *mask_cols]
        df = read_parquet(fp, cols)
        ids_here = set(pd.to_numeric(df["stay_id"], errors="coerce").dropna().astype(int).unique())
        wanted = ids_here.intersection(row_for_stay)
        if not wanted:
            continue
        df = df[df.stay_id.isin(wanted)].copy()
        df["charttime"] = pd.to_datetime(df["charttime"])
        df = df.merge(onset[["stay_id", "onset_time"]], on="stay_id", how="inner", validate="many_to_one")
        delta = (df["onset_time"] - df["charttime"]).dt.total_seconds() / 60
        df["slot"] = t_steps - (delta / args.step_minutes).round().astype(int)
        df = df[(delta > 0) & (delta <= args.window_hours * 60) & df.slot.between(0, t_steps - 1)]
        for sid, group in df.groupby("stay_id", sort=False):
            i = row_for_stay[int(sid)]
            slots = group["slot"].to_numpy(int)
            if len(np.unique(slots)) != len(slots):
                raise RuntimeError(f"Duplicate temporal slots for stay {sid}")
            values[i, slots, :] = group[features].to_numpy(np.float32)
            flags = group[mask_cols].to_numpy(float)
            observed[i, slots, :] = (flags == 0).astype(np.uint8)
            time_present[i, slots] = 1

    onset["n_timepoints"] = time_present.sum(axis=1)
    onset["full_preonset_window"] = onset["n_timepoints"].eq(t_steps)
    onset["observed_fraction"] = observed.mean(axis=(1, 2))

    eligible_order = onset.loc[onset.full_preonset_window].sort_values(["subject_id", "onset_time", "stay_id"])
    primary_ids = set(eligible_order.drop_duplicates("subject_id", keep="first").stay_id.astype(int))
    onset["primary_eligible"] = onset.stay_id.isin(primary_ids)
    onset["split"] = subject_split(onset.subject_id.to_numpy())

    icu = pd.read_csv(
        args.mimic_root / "icu/icustays.csv.gz",
        usecols=["subject_id", "hadm_id", "stay_id", "first_careunit", "intime", "outtime", "los"],
    )
    adm = pd.read_csv(
        args.mimic_root / "hosp/admissions.csv.gz",
        usecols=["subject_id", "hadm_id", "admittime", "dischtime", "deathtime", "race", "hospital_expire_flag"],
    )
    pat = pd.read_csv(
        args.mimic_root / "hosp/patients.csv.gz",
        usecols=["subject_id", "gender", "anchor_age", "anchor_year", "dod"],
    )
    cohort = onset.merge(icu, on=["stay_id", "subject_id"], how="left", validate="one_to_one")
    cohort = cohort.merge(adm, on=["subject_id", "hadm_id"], how="left", validate="many_to_one")
    cohort = cohort.merge(pat, on="subject_id", how="left", validate="many_to_one")
    for col in ["intime", "outtime", "admittime", "dischtime", "deathtime", "dod"]:
        cohort[col] = pd.to_datetime(cohort[col], errors="coerce")
    cohort["age"] = cohort["anchor_age"] + cohort["intime"].dt.year - cohort["anchor_year"]
    cohort["hours_icu_to_onset"] = (cohort.onset_time - cohort.intime).dt.total_seconds() / 3600
    cohort["icu_mortality"] = (
        cohort.deathtime.notna() & (cohort.deathtime >= cohort.intime) & (cohort.deathtime <= cohort.outtime)
    ).astype("int8")
    onset_day = cohort.onset_time.dt.normalize()
    cohort["mortality_28d"] = (
        cohort.dod.notna() & (cohort.dod >= onset_day) & (cohort.dod <= onset_day + pd.Timedelta(days=28))
    ).astype("int8")

    secondary_coverage = None
    if args.secondary_outcomes and args.secondary_outcomes.exists():
        sec = pd.read_csv(args.secondary_outcomes).drop_duplicates("stay_id")
        keep = [c for c in ["stay_id", "rrt_on", "vm_time_hours", "ventilation_hours"] if c in sec]
        cohort = cohort.merge(sec[keep], on="stay_id", how="left", validate="one_to_one")
        secondary_coverage = {c: float(cohort.loc[cohort.primary_eligible, c].notna().mean()) for c in keep if c != "stay_id"}

    cohort.insert(0, "tensor_row", np.arange(len(cohort), dtype=np.int64))
    np.save(args.out_dir / "preonset_values.npy", values)
    np.save(args.out_dir / "preonset_observed_mask.npy", observed)
    np.save(args.out_dir / "preonset_time_present.npy", time_present)
    cohort.to_parquet(args.out_dir / "cohort_all_failure_stays.parquet", index=False)
    cohort.loc[cohort.primary_eligible].to_parquet(args.out_dir / "cohort_primary.parquet", index=False)
    pd.DataFrame({"feature": features, "feature_index": range(d)}).to_csv(args.out_dir / "features_68.csv", index=False)

    audit = {
        "author_scope": "Lucas - phenotype inference v1",
        "n_clean_files": len(clean_files), "n_corrected_failure_stays": int(len(cohort)),
        "n_corrected_failure_subjects": int(cohort.subject_id.nunique()),
        "n_complete_preonset_window": int(cohort.full_preonset_window.sum()),
        "n_primary_one_stay_per_subject": int(cohort.primary_eligible.sum()),
        "n_features": d, "window_hours": args.window_hours, "step_minutes": args.step_minutes,
        "n_time_steps": t_steps, "secondary_outcome_coverage_primary": secondary_coverage,
        "exclusion_counts": {
            "incomplete_preonset_window": int((~cohort.full_preonset_window).sum()),
            "additional_failure_stay_same_subject": int(cohort.full_preonset_window.sum() - cohort.primary_eligible.sum()),
        },
    }
    write_json(args.out_dir / "cohort_manifest.json", audit)
    print(pd.Series(audit).to_string())


if __name__ == "__main__":
    main()

