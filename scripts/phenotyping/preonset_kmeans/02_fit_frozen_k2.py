#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import RobustScaler

from common import write_json


FEATURES = ["heart_rate", "mbp", "resp_rate", "spo2", "lab_50813", "lab_51265", "lab_51274", "lab_51275"]
LOG_FEATURES = {"lab_50813", "lab_51265", "lab_51274", "lab_51275"}


def last_observed(values: np.ndarray, observed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = observed & np.isfinite(values); count = valid.sum(axis=1)
    reverse = np.argmax(valid[:, ::-1, :], axis=1); index = values.shape[1] - 1 - reverse
    last = np.take_along_axis(values, index[:, None, :], axis=1)[:, 0, :]
    last[count == 0] = np.nan
    return last, count


def sampled_silhouette(x: np.ndarray, labels: np.ndarray, seed: int) -> float:
    rng = np.random.default_rng(seed); idx = rng.choice(len(x), min(3000, len(x)), replace=False)
    return float(silhouette_score(x[idx], labels[idx]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(); args.out_dir.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_parquet(args.cohort_dir / "cohort_primary.parquet").reset_index(drop=True)
    names = pd.read_csv(args.cohort_dir / "features_68.csv").feature.tolist(); indices = [names.index(f) for f in FEATURES]
    rows = cohort.tensor_row.to_numpy(int)
    values = np.asarray(np.load(args.cohort_dir / "preonset_values.npy", mmap_mode="r")[rows][:, :, indices], float)
    observed = np.asarray(np.load(args.cohort_dir / "preonset_observed_mask.npy", mmap_mode="r")[rows][:, :, indices], bool)
    raw, counts = last_observed(values, observed); eligible = np.isfinite(raw).all(axis=1)
    data = cohort.loc[eligible].reset_index(drop=True); raw = raw[eligible]; counts = counts[eligible]
    transformed = raw.copy()
    for j, feature in enumerate(FEATURES):
        if feature in LOG_FEATURES: transformed[:, j] = np.log1p(np.clip(transformed[:, j], 0, None))
    derivation = data.split.eq("derivation").to_numpy(); validation = ~derivation
    lower = np.quantile(transformed[derivation], .005, axis=0); upper = np.quantile(transformed[derivation], .995, axis=0)
    clipped = np.clip(transformed, lower, upper); scaler = RobustScaler().fit(clipped[derivation]); z = scaler.transform(clipped)
    model = KMeans(2, n_init=100, random_state=43).fit(z[derivation]); labels = model.predict(z)
    # Order by derivation coagulopathy severity (PT + PTT standardized), never by outcome.
    severity = [np.median(z[derivation & (labels == c), 6] + z[derivation & (labels == c), 7]) for c in range(2)]
    order = np.argsort(severity); mapping = {int(old): int(new) for new, old in enumerate(order)}
    labels = np.array([mapping[int(label)] for label in labels], int); data["cluster_original"] = labels
    data.to_parquet(args.out_dir / "assignments_original_k2.parquet", index=False)

    metrics = []
    for split, mask, seed in [("derivation", derivation, 43), ("validation", validation, 44)]:
        sizes = np.bincount(labels[mask], minlength=2) / mask.sum()
        metrics.append({"split": split, "n": int(mask.sum()), "silhouette_original": sampled_silhouette(z[mask], labels[mask], seed),
                        "cluster0_pct": 100 * sizes[0], "cluster1_pct": 100 * sizes[1], "minimum_cluster_pct": 100 * sizes.min()})
    pd.DataFrame(metrics).to_csv(args.out_dir / "primary_metrics.csv", index=False)

    cv = StratifiedKFold(5, shuffle=True, random_state=43)
    probabilities = cross_val_predict(LogisticRegression(max_iter=3000, class_weight="balanced"), counts[derivation], labels[derivation], cv=cv, method="predict_proba")
    measurement_auc = roc_auc_score(labels[derivation], probabilities[:, 1])

    profile_rows = []
    for split, mask in [("derivation", derivation), ("validation", validation), ("all", np.ones(len(data), bool))]:
        for j, feature in enumerate(FEATURES):
            for cluster in (0, 1):
                series = pd.Series(raw[mask & (labels == cluster), j])
                profile_rows.append({"split": split, "feature": feature, "cluster": cluster, "n": len(series),
                                     "median": series.median(), "q25": series.quantile(.25), "q75": series.quantile(.75)})
    pd.DataFrame(profile_rows).to_csv(args.out_dir / "clinical_profiles.csv", index=False)
    joblib.dump({"features": FEATURES, "log_features": LOG_FEATURES, "clip_lower": lower, "clip_upper": upper,
                 "scaler": scaler, "kmeans": model, "label_mapping": mapping, "window_hours": 2},
                args.out_dir / "frozen_original_k2.joblib", compress=3)
    write_json(args.out_dir / "manifest.json", {
        "method": "complete-case last observed values; robust scaling; k-means",
        "nonlinear_dimension_reduction": False, "window_hours": 2, "features": FEATURES, "k": 2,
        "n_complete_case": len(data), "n_derivation": int(derivation.sum()), "n_validation": int(validation.sum()),
        "measurement_count_auc_derivation": float(measurement_auc), "selection_used_outcomes": False,
        "validation_opened_after_freeze": True,
    })
    print(pd.DataFrame(metrics).to_string(index=False))
    print("measurement-count AUC derivation:", measurement_auc)
    print(pd.DataFrame(profile_rows).query("split == 'validation'").to_string(index=False))


if __name__ == "__main__":
    main()
