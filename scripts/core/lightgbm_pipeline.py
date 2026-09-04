#!/usr/bin/env python3
"""Leakage-guarded LightGBM interface and deterministic small smoke test.

This module deliberately does not implement definitive/full-cohort training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import feature_guard


@dataclass
class TrainOnlyMedianImputer:
    columns: list[str]
    medians_: pd.Series | None = None
    fit_split_: str | None = None

    def fit(self, frame: pd.DataFrame, *, split_name: str) -> "TrainOnlyMedianImputer":
        if split_name != "train":
            raise ValueError("Imputer may only be fitted on the training split")
        feature_guard.assert_no_forbidden_features(self.columns)
        clean = frame[self.columns].replace([np.inf, -np.inf], np.nan)
        medians = clean.median(axis=0, skipna=True).fillna(0.0)
        self.medians_ = medians.astype(np.float64)
        self.fit_split_ = split_name
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.medians_ is None or self.fit_split_ != "train":
            raise RuntimeError("Imputer must be fitted on train before transform")
        clean = frame[self.columns].replace([np.inf, -np.inf], np.nan)
        return clean.fillna(self.medians_).astype(np.float32)


class DeterministicCollector:
    def __init__(self, quotas: dict[tuple[str, int], int], seed: int):
        self.quotas = quotas
        self.rng = np.random.default_rng(seed)
        self.parts: list[pd.DataFrame] = []
        self.counts = {key: 0 for key in quotas}
        self.stays: dict[str, set[int]] = {"train": set(), "val": set()}

    @property
    def done(self) -> bool:
        return all(self.counts[key] >= quota for key, quota in self.quotas.items())

    def accept(self, features, labels, feature_names: list[str], target_col: str, eligibility_col: str) -> None:
        label_frame = labels.select(["stay_id", "split", target_col, eligibility_col]).to_pandas()
        target = label_frame[target_col]
        eligible = label_frame[eligibility_col].astype(bool) & target.notna()
        for key, quota in self.quotas.items():
            split, klass = key
            need = quota - self.counts[key]
            if need <= 0:
                continue
            candidates = np.flatnonzero(
                eligible.to_numpy()
                & (label_frame["split"].astype(str).to_numpy() == split)
                & (target.astype("boolean").fillna(False).to_numpy(dtype=bool) == bool(klass))
            )
            if candidates.size == 0:
                continue
            chosen = self.rng.choice(candidates, size=min(need, candidates.size), replace=False)
            x = features.select(["stay_id", *feature_names]).take(chosen).to_pandas()
            x["__split"] = split
            x["__target"] = klass
            self.parts.append(x)
            self.counts[key] += len(chosen)
            self.stays[split].update(x["stay_id"].astype(int).tolist())

    def frame(self) -> pd.DataFrame:
        if not self.done:
            raise RuntimeError(f"Insufficient smoke rows: {self.counts} versus {self.quotas}")
        return pd.concat(self.parts, ignore_index=True)


def collect_smoke_rows(dataset_dir: Path, feature_names: list[str], config: dict) -> tuple[pd.DataFrame, dict]:
    quotas = {
        ("train", 1): int(config["train_positive_rows"]),
        ("train", 0): int(config["train_negative_rows"]),
        ("val", 1): int(config["validation_positive_rows"]),
        ("val", 0): int(config["validation_negative_rows"]),
    }
    collector = DeterministicCollector(quotas, int(config["seed"]))
    target_col = config["endpoint_column"]
    eligibility_col = config["eligibility_column"]
    feature_files = sorted((dataset_dir / "features").glob("*_features_v2.parquet"))
    for feature_path in feature_files:
        label_name = feature_path.name.replace("_features_v2.parquet", "_labels_v2.parquet")
        label_path = dataset_dir / "labels" / label_name
        if not label_path.exists():
            raise FileNotFoundError(label_path)
        feature_batches = pq.ParquetFile(feature_path).iter_batches(
            batch_size=4096, columns=["stay_id", *feature_names]
        )
        label_batches = pq.ParquetFile(label_path).iter_batches(
            batch_size=4096,
            columns=["stay_id", "split", target_col, eligibility_col],
        )
        for feature_batch, label_batch in zip_longest(feature_batches, label_batches):
            if feature_batch is None or label_batch is None or feature_batch.num_rows != label_batch.num_rows:
                raise ValueError(f"Feature/label stream mismatch for {feature_path.name}")
            if not np.array_equal(
                feature_batch.column(0).to_numpy(zero_copy_only=False),
                label_batch.column(0).to_numpy(zero_copy_only=False),
            ):
                raise ValueError(f"stay_id mismatch for {feature_path.name}")
            collector.accept(feature_batch, label_batch, feature_names, target_col, eligibility_col)
            if collector.done:
                frame = collector.frame()
                return frame, {
                    "row_counts": {f"{s}_{k}": v for (s, k), v in collector.counts.items()},
                    "stay_counts": {s: len(v) for s, v in collector.stays.items()},
                    "stay_overlap": len(collector.stays["train"] & collector.stays["val"]),
                    "last_scanned_file": feature_path.name,
                }
    raise RuntimeError(f"Could not fill smoke quotas: {collector.counts}")


def run_smoke(dataset_dir: Path, config_path: Path, output_path: Path) -> dict:
    config = json.loads(config_path.read_text())
    feature_list_path = (config_path.parent / config["feature_list"]).resolve()
    feature_names = feature_list_path.read_text().splitlines()
    feature_guard.assert_no_forbidden_features(feature_names)
    if len(feature_names) != len(set(feature_names)):
        raise ValueError("Duplicate model feature names")

    frame, collection = collect_smoke_rows(dataset_dir, feature_names, config)
    train = frame[frame["__split"] == "train"].copy()
    val = frame[frame["__split"] == "val"].copy()
    if set(train.stay_id.astype(int)) & set(val.stay_id.astype(int)):
        raise ValueError("Smoke split has overlapping stay_id")
    imputer = TrainOnlyMedianImputer(feature_names).fit(train, split_name="train")
    x_train = imputer.transform(train)
    x_val = imputer.transform(val)
    y_train = train["__target"].astype(np.int8)
    y_val = val["__target"].astype(np.int8)
    if y_train.nunique() != 2 or y_val.nunique() != 2:
        raise ValueError("Smoke train/validation must each include both target classes")

    import lightgbm as lgb

    params = dict(config["params"])
    params["random_state"] = int(config["seed"])
    model = lgb.LGBMClassifier(**params)
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], callbacks=[lgb.log_evaluation(0)])
    prediction = model.predict_proba(x_val)[:, 1]
    if not np.isfinite(prediction).all() or not ((prediction >= 0) & (prediction <= 1)).all():
        raise ValueError("Smoke predictions are not finite probabilities")

    result = {
        "status": "passed",
        "scope": "technical smoke only; no definitive metric was calculated",
        "lightgbm_version": lgb.__version__,
        "seed": int(config["seed"]),
        "target_column": config["endpoint_column"],
        "endpoint_clinically_approved": False,
        "feature_count": len(feature_names),
        "feature_order_sha256": hashlib.sha256("\n".join(feature_names).encode()).hexdigest(),
        "imputer_fit_split": imputer.fit_split_,
        "normalization_applied": False,
        "scale_pos_weight": config["scale_pos_weight"],
        "train_rows": len(train),
        "validation_rows": len(val),
        "finite_predictions": int(np.isfinite(prediction).sum()),
        "collection": collection,
        "model_persisted": False,
        "definitive_training_performed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_smoke(args.dataset_dir, args.config, args.output), indent=2))


if __name__ == "__main__":
    main()
