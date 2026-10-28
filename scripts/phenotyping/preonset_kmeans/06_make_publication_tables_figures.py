#!/usr/bin/env python3
"""Create manuscript-ready tables and figures from the frozen outputs."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FEATURE_LABELS = {
    "heart_rate": "Heart rate", "mbp": "Mean arterial pressure",
    "resp_rate": "Respiratory rate", "spo2": "SpO₂", "lab_50813": "Lactate",
    "lab_51265": "Platelets", "lab_51274": "Prothrombin time",
    "lab_51275": "Partial thromboplastin time",
}
OUTCOME_LABELS = {
    "mortality_28d": "28-day mortality",
    "hospital_expire_flag": "Hospital mortality",
    "icu_mortality": "ICU mortality", "rrt_any": "Renal replacement therapy",
    "ventilation_any": "Mechanical ventilation",
    "high_dose_nee_post6h": "High-dose NEE at 6 h",
    "high_dose_nee_post24h": "High-dose NEE at 24 h",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_dir
    tables, figures = root / "tables", root / "figures"
    tables.mkdir(exist_ok=True); figures.mkdir(exist_ok=True)

    profiles = pd.read_csv(root / "model/clinical_profiles.csv")
    val = profiles.query("split == 'validation'").copy()
    val["feature"] = val.feature.map(FEATURE_LABELS)
    table1 = val.pivot(index="feature", columns="cluster", values=["median", "q25", "q75", "n"])
    table1.columns = [f"{metric}_C{cluster}" for metric, cluster in table1.columns]
    table1.reset_index().to_csv(tables / "table1_validation_phenotype_profiles.csv", index=False)

    risks = pd.read_csv(root / "results/primary_outcomes/absolute_risks.csv")
    poisson = pd.read_csv(root / "results/primary_outcomes/modified_poisson.csv")
    risk_wide = risks.query("scope == 'validation_primary'").pivot(
        index="outcome", columns="cluster", values="risk"
    ).rename(columns={0: "risk_C0", 1: "risk_C1"})
    adjusted = poisson.query("scope == 'validation_primary' and model == 'adjusted'").set_index("outcome")
    table2 = risk_wide.join(adjusted[["n", "events", "risk_ratio_c1_vs_c0", "ci95_low", "ci95_high", "p_value"]])
    table2.insert(0, "outcome_label", table2.index.map(OUTCOME_LABELS))
    table2.reset_index().to_csv(tables / "table2_validation_outcomes.csv", index=False)

    balanced = pd.read_csv(root / "results/confirmatory_inference/multi_outcome_balanced_associations.csv")
    table3 = balanced.query("scope == 'validation' and estimator == 'cross_fitted_AIPW'").copy()
    table3["outcome_label"] = table3.outcome.map(OUTCOME_LABELS)
    supported = {"mortality_28d", "hospital_expire_flag", "icu_mortality", "high_dose_nee_post6h", "high_dose_nee_post24h"}
    table3["interpretation"] = np.where(table3.outcome.isin(supported), "supported", "not_confirmatory")
    keep = ["outcome", "outcome_label", "n", "risk_C0", "risk_C1", "risk_difference",
            "ci95_low", "ci95_high", "risk_ratio", "p_value", "q_value_bh", "interpretation"]
    table3[keep].to_csv(tables / "table3_case_mix_balanced_validation.csv", index=False)

    incremental = pd.read_csv(root / "results/confirmatory_inference/incremental_prognostic_value.csv")
    table4 = incremental[incremental.metric.isin(["auroc", "auprc"])].copy()
    table4["outcome_label"] = table4.outcome.map(OUTCOME_LABELS)
    table4.to_csv(tables / "table4_incremental_prognostic_value.csv", index=False)

    artifact = joblib.load(root / "model/frozen_original_k2.joblib")
    centers = artifact["kmeans"].cluster_centers_
    mapping = artifact["label_mapping"]
    ordered = np.empty_like(centers)
    for old, new in mapping.items(): ordered[new] = centers[old]
    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    image = ax.imshow(ordered, cmap="RdBu_r", aspect="auto", vmin=-2.5, vmax=2.5)
    ax.set_yticks([0, 1], ["C0 compensated", "C1 decompensated"])
    ax.set_xticks(range(8), [FEATURE_LABELS[x] for x in artifact["features"]], rotation=35, ha="right")
    ax.set_title("Frozen phenotype centers in robust-scaled model space")
    fig.colorbar(image, ax=ax, label="Robust-scaled transformed value")
    fig.tight_layout(); fig.savefig(figures / "figure1_phenotype_centers.png", dpi=300); plt.close(fig)

    plot = table3.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(plot))
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.errorbar(plot.risk_difference * 100, y,
                xerr=np.vstack([(plot.risk_difference-plot.ci95_low)*100,
                                (plot.ci95_high-plot.risk_difference)*100]),
                fmt="o", color="#7b2cbf", capsize=3)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y, plot.outcome_label)
    ax.set_xlabel("Adjusted risk difference, C1 − C0 (percentage points)")
    ax.set_title("Case-mix-balanced associations in frozen validation")
    fig.tight_layout(); fig.savefig(figures / "figure2_adjusted_risk_differences.png", dpi=300); plt.close(fig)

    plot = table4.iloc[::-1].reset_index(drop=True)
    labels = plot.outcome_label + " — " + plot.metric.str.upper()
    y = np.arange(len(plot))
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.errorbar(plot.delta_plus_cluster, y,
                xerr=np.vstack([plot.delta_plus_cluster-plot.delta_ci95_low,
                                plot.delta_ci95_high-plot.delta_plus_cluster]),
                fmt="o", color="#0077b6", capsize=3)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Change after adding the frozen phenotype")
    ax.set_title("Incremental prognostic value in frozen validation")
    fig.tight_layout(); fig.savefig(figures / "figure3_incremental_value.png", dpi=300); plt.close(fig)


if __name__ == "__main__":
    main()
