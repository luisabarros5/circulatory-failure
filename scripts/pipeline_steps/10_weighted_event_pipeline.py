"""Source exported from the reviewed development notebook.

This file contains code only; patient-level data and notebook outputs are not distributed.
"""

# %% [source cell 1]
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

TARGET = "target_first_onset_0_8h"
EXPECTED_BATCHES = 285
SHORT_HISTORY_FLOOR_ROWS = 12  # 60 min em grade de 5 min


def canonical_batches(directory: Path) -> list[Path]:
    files = sorted(p for p in directory.glob("*.parquet") if re.fullmatch(r"batch_\d{4}\.parquet", p.name))
    expected = [f"batch_{i:04d}.parquet" for i in range(EXPECTED_BATCHES)]
    if [p.name for p in files] != expected:
        raise RuntimeError(f"Esperava batch_0000--batch_0284 em {directory}; encontrei {len(files)}")
    return files


def load_feature_names(project: Path) -> list[str]:
    registry = json.loads((project / "outputs/step07_aligned_xy_v2/feature_names.json").read_text())
    features = list(registry["feature_names"])
    if len(features) != int(registry["feature_count"]) or len(features) != len(set(features)):
        raise RuntimeError("Registro de features invalido")
    return features


def resolve_probability_column(frame: pd.DataFrame) -> str:
    found = [c for c in ("probability", "prediction") if c in frame.columns]
    if len(found) != 1:
        raise RuntimeError(f"Esperava exatamente uma coluna probability/prediction; encontrei {found}")
    return found[0]


def stable_keep_negative(frame: pd.DataFrame, fraction: float, seed: int) -> np.ndarray:
    if fraction >= 1.0:
        return np.ones(len(frame), dtype=bool)
    keys = frame[["stay_id", "label_time"]]
    hashes = pd.util.hash_pandas_object(keys, index=False).to_numpy(dtype="uint64")
    hashes ^= np.uint64(seed)
    threshold = np.uint64(fraction * np.iinfo(np.uint64).max)
    return hashes <= threshold


def classification_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "rows": int(len(y)),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()),
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
    }


def feature_variant(features: list[str], variant: str) -> list[str]:
    lactate = re.compile(r"(^|_)lab_50813($|_)|(^|_)lactate($|_)", re.I)
    pressure = re.compile(r"(^|_)(mbp|map|sbp|dbp)($|_)", re.I)
    if variant == "full":
        selected = features
    elif variant == "without_lactate":
        selected = [f for f in features if not lactate.search(f)]
    elif variant == "without_pressure":
        selected = [f for f in features if not pressure.search(f)]
    elif variant == "without_lactate_pressure":
        selected = [f for f in features if not lactate.search(f) and not pressure.search(f)]
    else:
        raise ValueError(variant)
    removed = sorted(set(features) - set(selected))
    if variant != "full" and not removed:
        raise RuntimeError(f"A variante {variant} nao removeu nenhuma feature")
    return selected


def weight_rows(frame: pd.DataFrame, counts: pd.DataFrame, negative_fraction: float) -> np.ndarray:
    info = frame[["stay_id", TARGET]].merge(
        counts[["stay_id", "positive_rows", "negative_rows"]],
        on="stay_id", how="left", validate="many_to_one",
    )
    if info[["positive_rows", "negative_rows"]].isna().any().any():
        raise RuntimeError("Stay sem contagens para ponderacao")
    y = info[TARGET].to_numpy(dtype="int8")
    n_pos = info["positive_rows"].to_numpy(dtype="float64")
    n_neg = info["negative_rows"].to_numpy(dtype="float64")
    weights = np.empty(len(info), dtype="float64")
    positive = y == 1
    # Cada evento completo soma 1. Eventos muito curtos continuam presentes,
    # mas o denominador minimo evita duas linhas receberem peso 1/2 cada.
    weights[positive] = 1.0 / np.maximum(n_pos[positive], SHORT_HISTORY_FLOOR_ROWS)
    weights[~positive] = 1.0 / np.maximum(n_neg[~positive], 1.0)
    if negative_fraction < 1.0:
        weights[~positive] /= negative_fraction
    weights *= len(weights) / weights.sum()
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise RuntimeError("Pesos invalidos")
    return weights.astype("float32")


def sha256_names(names: list[str]) -> str:
    return hashlib.sha256("\n".join(names).encode()).hexdigest()

# %% [source cell 2]
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def run_audit(project: Path, base_run_dir: Path) -> None:
    project = project.resolve()
    base_run_dir = (project / base_run_dir).resolve() if not base_run_dir.is_absolute() else base_run_dir
    files = canonical_batches(project / "outputs/step07_aligned_xy_v2/batches")
    split_file = base_run_dir / "stay_subject_split.parquet"
    if not split_file.exists():
        raise FileNotFoundError(split_file)
    splits = pd.read_parquet(split_file, columns=["stay_id", "subject_id", "split", "stay_positive"])
    if splits["subject_id"].groupby(splits["subject_id"]).size().empty:
        raise RuntimeError("Split vazio")
    if splits.groupby("subject_id")["split"].nunique().max() != 1:
        raise RuntimeError("Mesmo paciente em mais de um split")

    parts = []
    for index, path in enumerate(files, 1):
        f = pd.read_parquet(path, columns=["stay_id", "label_time", TARGET])
        f["label_time"] = pd.to_datetime(f["label_time"], errors="raise")
        f[TARGET] = f[TARGET].astype("int8")
        f["positive_time"] = f["label_time"].where(f[TARGET].eq(1))
        g = f.groupby("stay_id", as_index=False).agg(
            eligible_rows=(TARGET, "size"), positive_rows=(TARGET, "sum"),
            first_label_time=("label_time", "min"), last_label_time=("label_time", "max"),
            first_positive_time=("positive_time", "min"), last_positive_time=("positive_time", "max"),
        )
        g["negative_rows"] = g["eligible_rows"] - g["positive_rows"]
        parts.append(g)
        if index == 1 or index % 25 == 0 or index == 285:
            print(f"audit: {index}/285 batches", flush=True)

    audit = pd.concat(parts, ignore_index=True)
    if audit["stay_id"].duplicated().any():
        raise RuntimeError("Stay em mais de um batch")
    audit = audit.merge(splits, on="stay_id", how="left", validate="one_to_one")
    if audit["split"].isna().any():
        raise RuntimeError("Stay sem split")
    audit["observed_positive_minutes"] = np.where(
        audit["positive_rows"].gt(0), audit["positive_rows"] * 5.0, 0.0
    )
    audit["short_history_band"] = pd.cut(
        audit["observed_positive_minutes"],
        bins=[-0.1, 0, 30, 60, 120, 240, 480, np.inf],
        labels=["no_event", "0_30m", "30_60m", "1_2h", "2_4h", "4_8h", "over_8h"],
        right=True,
    ).astype(str)
    audit["positive_row_weight_raw"] = np.where(
        audit["positive_rows"].gt(0), 1.0 / np.maximum(audit["positive_rows"], 12), 0.0
    )
    audit["positive_event_total_weight_raw"] = audit["positive_rows"] * audit["positive_row_weight_raw"]
    audit["negative_row_weight_raw"] = np.where(
        audit["negative_rows"].gt(0), 1.0 / audit["negative_rows"], 0.0
    )
    audit["negative_stay_total_weight_raw"] = audit["negative_rows"] * audit["negative_row_weight_raw"]

    out = project / "outputs/step10_event_weight_audit"
    out.mkdir(parents=True, exist_ok=False)
    audit.to_parquet(out / "stay_event_weight_audit.parquet", index=False, compression="zstd")
    summary = (
        audit.groupby(["split", "short_history_band"], observed=True)
        .agg(stays=("stay_id", "nunique"), subjects=("subject_id", "nunique"),
             rows=("eligible_rows", "sum"), positive_rows=("positive_rows", "sum"))
        .reset_index()
    )
    summary.to_csv(out / "history_and_weight_summary.csv", index=False)
    manifest = {
        "status": "approved", "batches_scanned": len(files),
        "stays": int(audit.stay_id.nunique()), "subjects": int(audit.subject_id.nunique()),
        "positive_stays": int(audit.positive_rows.gt(0).sum()),
        "short_history_floor_rows": 12, "short_history_floor_minutes": 60,
        "all_events_retained": True,
        "split_counts": audit.groupby("split")["stay_id"].nunique().to_dict(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


PROJECT = Path("/path/to/circulatory_failure_prediction")
BASE_RUN_DIR = Path(
    "outputs/step09_lightgbm/full_primary_20260814T030218Z"
)

run_audit(PROJECT, BASE_RUN_DIR)

# %% [source cell 4]
from __future__ import annotations

import gc
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd


DEFAULT_PARAMS = {
    "objective": "binary", "n_estimators": 3000, "learning_rate": 0.03,
    "num_leaves": 63, "max_depth": -1, "min_child_samples": 200,
    "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
    "reg_alpha": 0.0, "reg_lambda": 1.0, "n_jobs": -1,
    "random_state": 42, "verbosity": -1,
}


def negative_fraction(files, train_stays, ratio: float) -> float:
    pos = neg = 0
    allowed = set(train_stays)
    for i, path in enumerate(files, 1):
        frame = pd.read_parquet(path, columns=["stay_id", TARGET])
        frame = frame.loc[frame.stay_id.isin(allowed)]
        pos += int(frame[TARGET].sum()); neg += int(len(frame) - frame[TARGET].sum())
        if i == 1 or i % 25 == 0 or i == 285:
            print(f"count train: {i}/285 batches", flush=True)
    if pos == 0 or neg == 0:
        raise RuntimeError("Treino sem ambas as classes")
    return min(1.0, ratio * pos / neg)


def collect(files, split_table, split, features, counts, neg_fraction, seed):
    allowed = split_table.loc[split_table.split.eq(split), ["stay_id", "subject_id"]]
    parts = []
    for i, path in enumerate(files, 1):
        frame = pd.read_parquet(path, columns=["stay_id", "label_time", TARGET, *features])
        frame = frame.merge(allowed, on="stay_id", how="inner", validate="many_to_one")
        if split == "train" and neg_fraction < 1 and not frame.empty:
            keep = stable_keep_negative(frame, neg_fraction, seed)
            frame = frame.loc[frame[TARGET].eq(1).to_numpy() | keep]
        if not frame.empty:
            parts.append(frame)
        if i == 1 or i % 25 == 0 or i == 285:
            print(f"{split}: {i}/285 batches; partes={len(parts)}", flush=True)
    if not parts:
        raise RuntimeError(f"Nenhuma linha em {split}")
    data = pd.concat(parts, ignore_index=True)
    data["label_time"] = pd.to_datetime(data["label_time"], errors="raise")
    y = data[TARGET].to_numpy(dtype="int8")
    meta = data[["stay_id", "subject_id", "label_time"]].copy()
    weights = weight_rows(data[["stay_id", TARGET]], counts, neg_fraction if split == "train" else 1.0)
    x = data[features].replace([np.inf, -np.inf], np.nan).astype("float32")
    if np.unique(y).size != 2:
        raise RuntimeError(f"{split} com uma classe")
    return x, y, weights, meta


def run(project: Path, base_run_dir: Path, run_name: str, variant: str, negative_ratio: float) -> dict:
    project = project.resolve()
    base_run_dir = (project / base_run_dir).resolve() if not base_run_dir.is_absolute() else base_run_dir
    files = canonical_batches(project / "outputs/step07_aligned_xy_v2/batches")
    split_table = pd.read_parquet(base_run_dir / "stay_subject_split.parquet")
    if split_table.groupby("subject_id")["split"].nunique().max() != 1:
        raise RuntimeError("Split por paciente violado")
    audit_dir = project / "outputs/step10_event_weight_audit"
    counts = pd.read_parquet(audit_dir / "stay_event_weight_audit.parquet")
    features_all = load_feature_names(project)
    features = feature_variant(features_all, variant)
    train_stays = split_table.loc[split_table.split.eq("train"), "stay_id"]
    neg_fraction = negative_fraction(files, train_stays, negative_ratio)
    out = project / "outputs/step09_lightgbm" / run_name
    out.mkdir(parents=True, exist_ok=False)
    split_table.to_parquet(out / "stay_subject_split.parquet", index=False, compression="zstd")
    (out / "features_used.json").write_text(json.dumps({
        "variant": variant, "features": features,
        "removed": sorted(set(features_all) - set(features)),
    }, indent=2) + "\n")

    x_train, y_train, w_train, train_meta = collect(
        files, split_table, "train", features, counts, neg_fraction, 42
    )
    expected_positive_stays = set(split_table.loc[
        split_table.split.eq("train") & split_table.stay_positive.eq(1), "stay_id"
    ])
    observed_positive_stays = set(train_meta.loc[y_train == 1, "stay_id"])
    if expected_positive_stays != observed_positive_stays:
        raise RuntimeError(
            f"Eventos de treino faltando/extras: faltam={len(expected_positive_stays-observed_positive_stays)}, "
            f"extras={len(observed_positive_stays-expected_positive_stays)}"
        )
    x_val, y_val, _, val_meta = collect(files, split_table, "validation", features, counts, 1.0, 42)

    model = lgb.LGBMClassifier(**DEFAULT_PARAMS)
    model.fit(
        x_train, y_train, sample_weight=w_train,
        eval_set=[(x_val, y_val)], eval_metric="average_precision",
        callbacks=[lgb.early_stopping(100, first_metric_only=True), lgb.log_evaluation(25)],
    )
    p_val = model.predict_proba(x_val, num_iteration=model.best_iteration_)[:, 1]
    val_predictions = val_meta.copy(); val_predictions["target"] = y_val; val_predictions["probability"] = p_val
    val_predictions.to_parquet(out / "validation_predictions.parquet", index=False, compression="zstd")
    val_metrics = classification_metrics(y_val, p_val)

    del x_train, y_train, w_train, train_meta, x_val, y_val, val_meta, p_val
    gc.collect()
    x_test, y_test, _, test_meta = collect(files, split_table, "test", features, counts, 1.0, 42)
    p_test = model.predict_proba(x_test, num_iteration=model.best_iteration_)[:, 1]
    test_predictions = test_meta.copy(); test_predictions["target"] = y_test; test_predictions["probability"] = p_test
    test_predictions.to_parquet(out / "test_predictions.parquet", index=False, compression="zstd")
    test_metrics = classification_metrics(y_test, p_test)

    joblib.dump(model, out / "model.joblib")
    model.booster_.save_model(out / "model.txt", num_iteration=model.best_iteration_)
    importance = pd.DataFrame({
        "feature": features,
        "gain": model.booster_.feature_importance("gain"),
        "split": model.booster_.feature_importance("split"),
    }).sort_values("gain", ascending=False)
    importance.to_csv(out / "feature_importance.csv", index=False)
    result = {
        "status": "completed", "run_name": run_name, "variant": variant,
        "batches_scanned_per_split": 285, "all_training_events_used": True,
        "patient_level_split_reused": str(base_run_dir / "stay_subject_split.parquet"),
        "short_history_treatment": "all retained; positive denominator floored at 12 rows/60 min",
        "negative_to_positive_sampling_ratio": negative_ratio,
        "negative_sampling_fraction": neg_fraction,
        "negative_sampling_inverse_probability_corrected": neg_fraction < 1,
        "feature_scaling": "not applied; LightGBM is split-based",
        "feature_count": len(features), "feature_sha256": sha256_names(features),
        "best_iteration": int(model.best_iteration_), "validation": val_metrics, "test": test_metrics,
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2)); return result


WEIGHTED_RUN_NAME = "weighted_event_primary"

weighted_result = run(
    project=PROJECT,
    base_run_dir=BASE_RUN_DIR,
    run_name=WEIGHTED_RUN_NAME,
    variant="full",
    negative_ratio=5.0,
)

weighted_result

# %% [source cell 5]
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def first_onsets(project: Path) -> pd.DataFrame:
    files = canonical_batches(project / "outputs/step05_labels_v2")
    parts = []
    for i, path in enumerate(files, 1):
        f = pd.read_parquet(path, columns=["stay_id", "charttime", "failure_state_new_at_t"])
        mask = f.failure_state_new_at_t.astype("boolean").fillna(False)
        if mask.any():
            g = f.loc[mask, ["stay_id", "charttime"]].groupby("stay_id", as_index=False).charttime.min()
            parts.append(g.rename(columns={"charttime": "first_onset_time"}))
        if i == 1 or i % 25 == 0 or i == 285:
            print(f"onsets: {i}/285 batches", flush=True)
    onsets = pd.concat(parts, ignore_index=True)
    if onsets.stay_id.duplicated().any():
        raise RuntimeError("Onset duplicado entre batches")
    onsets.first_onset_time = pd.to_datetime(onsets.first_onset_time, errors="raise")
    return onsets


def event_window_table(pred: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    prob = resolve_probability_column(pred)
    pred = pred.rename(columns={prob: "score"}).copy()
    pred.label_time = pd.to_datetime(pred.label_time, errors="raise")
    truth = pred.groupby("stay_id").target.max()
    positive_ids = set(truth[truth.eq(1)].index)
    negative_ids = set(truth[truth.eq(0)].index)
    positives = (
        pred.loc[pred.target.eq(1)]
        .groupby(["stay_id", "subject_id"], as_index=False)
        .agg(score=("score", "max"), window_rows=("score", "size"),
             window_start=("label_time", "min"), window_end=("label_time", "max"))
    )
    positives["target"] = 1
    if set(positives.stay_id) != positive_ids:
        raise RuntimeError("Evento positivo sem janela positiva")
    lengths = positives.window_rows.to_numpy(dtype=int)
    rng = np.random.default_rng(seed)
    controls = []
    for (stay_id, subject_id), group in pred.loc[pred.stay_id.isin(negative_ids)].groupby(
        ["stay_id", "subject_id"], sort=True
    ):
        group = group.sort_values("label_time").reset_index(drop=True)
        desired = int(lengths[rng.integers(0, len(lengths))])
        length = min(desired, len(group))
        start = int(rng.integers(0, len(group) - length + 1))
        window = group.iloc[start:start + length]
        controls.append({
            "stay_id": stay_id, "subject_id": subject_id, "score": float(window.score.max()),
            "window_rows": int(length), "desired_window_rows": desired,
            "window_start": window.label_time.min(), "window_end": window.label_time.max(), "target": 0,
        })
    controls = pd.DataFrame(controls)
    table = pd.concat([positives, controls], ignore_index=True, sort=False)
    if table.stay_id.duplicated().any():
        raise RuntimeError("Mais de uma janela por stay")
    return table


def ranking_with_cluster_bootstrap(table: pd.DataFrame, repetitions: int = 1000, seed: int = 42) -> dict:
    y = table.target.to_numpy(dtype="int8"); score = table.score.to_numpy(dtype="float64")
    result = {
        "events": int(len(table)), "positive_events": int(y.sum()), "prevalence": float(y.mean()),
        "auroc": float(roc_auc_score(y, score)), "auprc": float(average_precision_score(y, score)),
    }
    subjects = table.subject_id.unique(); by_subject = {s: np.flatnonzero(table.subject_id.to_numpy() == s) for s in subjects}
    rng = np.random.default_rng(seed); aucs = []; aps = []
    for _ in range(repetitions):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        idx = np.concatenate([by_subject[s] for s in sampled])
        if np.unique(y[idx]).size < 2:
            continue
        aucs.append(roc_auc_score(y[idx], score[idx])); aps.append(average_precision_score(y[idx], score[idx]))
    result["auroc_ci95"] = [float(np.quantile(aucs, .025)), float(np.quantile(aucs, .975))]
    result["auprc_ci95"] = [float(np.quantile(aps, .025)), float(np.quantile(aps, .975))]
    result["bootstrap_unit"] = "subject_id"; result["bootstrap_repetitions"] = len(aucs)
    return result


def threshold_curve(table: pd.DataFrame) -> pd.DataFrame:
    y = table.target.to_numpy(dtype=bool); score = table.score.to_numpy(dtype=float)
    thresholds = np.unique(np.r_[0.0, 1.0, np.quantile(score, np.linspace(0, 1, 2001))])
    rows = []
    for threshold in thresholds:
        alert = score >= threshold
        tp = int(np.sum(alert & y)); fp = int(np.sum(alert & ~y)); fn = int(np.sum(~alert & y)); tn = int(np.sum(~alert & ~y))
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"threshold": threshold, "tp_events": tp, "fp_control_windows": fp,
                     "fn_events": fn, "tn_control_windows": tn, "precision": precision,
                     "recall": recall, "specificity": specificity, "f1": f1})
    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


def choose_thresholds(curve: pd.DataFrame) -> dict:
    chosen = {}
    best = curve.sort_values(["f1", "threshold"], ascending=[False, False]).iloc[0]
    chosen["maximum_event_window_f1"] = float(best.threshold)
    for target in (0.70, 0.80, 0.90):
        candidates = curve.loc[curve.recall.ge(target)]
        if not candidates.empty:
            # maior threshold que ainda satisfaz o recall: tende a reduzir FP
            chosen[f"event_recall_{int(target*100)}"] = float(candidates.threshold.max())
    return chosen


def group_alerts(pred: pd.DataFrame, threshold: float, gap_minutes: int = 30) -> tuple[pd.DataFrame, dict]:
    prob = resolve_probability_column(pred)
    p = pred.loc[pred[prob].ge(threshold), ["stay_id", "subject_id", "label_time", "target", prob]].copy()
    p.label_time = pd.to_datetime(p.label_time, errors="raise")
    p = p.sort_values(["stay_id", "label_time"])
    new = p.groupby("stay_id").label_time.diff().gt(pd.Timedelta(minutes=gap_minutes)).fillna(True)
    p["alert_episode_number"] = new.groupby(p.stay_id).cumsum().astype(int)
    episodes = p.groupby(["stay_id", "subject_id", "alert_episode_number"], as_index=False).agg(
        alert_start=("label_time", "min"), alert_end=("label_time", "max"),
        alert_rows=("label_time", "size"), max_probability=(prob, "max"), true_event_window=("target", "max"),
    )
    false_patient_days = float(pred.loc[pred.target.eq(0)].shape[0] * 5 / 1440)
    false_episodes = int(episodes.true_event_window.eq(0).sum())
    summary = {
        "gap_minutes": gap_minutes, "alert_episodes": int(len(episodes)),
        "true_alert_episodes": int(episodes.true_event_window.eq(1).sum()),
        "false_alert_episodes": false_episodes, "eligible_negative_patient_days": false_patient_days,
        "false_alert_episodes_per_100_patient_days": 100 * false_episodes / false_patient_days if false_patient_days else None,
    }
    return episodes, summary


def lead_time_and_short_history(pred: pd.DataFrame, onsets: pd.DataFrame, threshold: float) -> pd.DataFrame:
    prob = resolve_probability_column(pred)
    positive = pred.loc[pred.target.eq(1), ["stay_id", "subject_id", "label_time", prob]].copy()
    positive.label_time = pd.to_datetime(positive.label_time, errors="raise")
    base = positive.groupby(["stay_id", "subject_id"], as_index=False).agg(
        positive_rows=("label_time", "size"), first_positive_time=("label_time", "min")
    ).merge(onsets, on="stay_id", how="left", validate="one_to_one")
    alerts = positive.loc[positive[prob].ge(threshold)].groupby("stay_id", as_index=False).label_time.min().rename(
        columns={"label_time": "first_alert_time"}
    )
    base = base.merge(alerts, on="stay_id", how="left", validate="one_to_one")
    base["detected"] = base.first_alert_time.notna()
    base["lead_time_hours"] = (base.first_onset_time - base.first_alert_time).dt.total_seconds() / 3600
    base["observed_history_minutes"] = base.positive_rows * 5
    base["history_band"] = pd.cut(
        base.observed_history_minutes, bins=[0, 30, 60, 120, 240, 480, np.inf],
        labels=["0_30m", "30_60m", "1_2h", "2_4h", "4_8h", "over_8h"], include_lowest=True,
    ).astype(str)
    return base


def evaluate_events(project: Path, run_dir: Path) -> None:
    project = project.resolve(); run_dir = (project / run_dir).resolve() if not run_dir.is_absolute() else run_dir
    out = run_dir / "event_evaluation_v2"; out.mkdir(parents=True, exist_ok=False)
    onsets = first_onsets(project)
    predictions = {}
    windows = {}
    rankings = {}
    for split in ("validation", "test"):
        predictions[split] = pd.read_parquet(run_dir / f"{split}_predictions.parquet")
        windows[split] = event_window_table(predictions[split], seed=42)
        windows[split].to_parquet(out / f"{split}_event_windows.parquet", index=False, compression="zstd")
        rankings[split] = ranking_with_cluster_bootstrap(windows[split])
    curve = threshold_curve(windows["validation"]); curve.to_csv(out / "validation_threshold_curve.csv", index=False)
    thresholds = choose_thresholds(curve)
    all_results = {}
    for name, threshold in thresholds.items():
        entry = {"threshold": threshold}
        for split in ("validation", "test"):
            row = threshold_curve(windows[split])
            nearest = row.iloc[(row.threshold - threshold).abs().argmin()].to_dict()
            # Recalculo exato no threshold congelado, sem aproximacao pela grade.
            y = windows[split].target.to_numpy(dtype=bool); alert = windows[split].score.to_numpy() >= threshold
            tp = int(np.sum(alert & y)); fp = int(np.sum(alert & ~y)); fn = int(np.sum(~alert & y)); tn = int(np.sum(~alert & ~y))
            precision = tp/(tp+fp) if tp+fp else 1.0; recall = tp/(tp+fn); specificity = tn/(tn+fp)
            f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
            episodes, burden = group_alerts(predictions[split], threshold, 30)
            if split == "test":
                episodes.to_parquet(out / f"test_alert_episodes_{name}.parquet", index=False, compression="zstd")
            leads = lead_time_and_short_history(predictions[split], onsets, threshold)
            strata = leads.groupby("history_band", observed=True).agg(
                events=("stay_id", "size"), detected=("detected", "sum"),
                recall=("detected", "mean"), median_lead_hours=("lead_time_hours", "median"),
            ).reset_index()
            if split == "test":
                leads.to_parquet(out / f"test_lead_times_{name}.parquet", index=False, compression="zstd")
                strata.to_csv(out / f"test_short_history_strata_{name}.csv", index=False)
            entry[split] = {"tp_events": tp, "fp_control_windows": fp, "fn_events": fn,
                            "tn_control_windows": tn, "precision": precision, "recall": recall,
                            "specificity": specificity, "f1": f1, "alert_burden": burden,
                            "median_lead_time_hours_detected": float(leads.lead_time_hours.median())}
        all_results[name] = entry
    result = {
        "status": "completed", "thresholds_selected_only_on_validation": thresholds,
        "event_window_ranking": rankings, "frozen_threshold_results": all_results,
        "control_window_rule": "one deterministic contiguous negative window per negative stay; length sampled from positive-event window lengths",
        "alert_grouping_rule": "same-stay alert rows separated by at most 30 minutes form one alert episode",
    }
    (out / "event_evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


WEIGHTED_RUN_DIR = (
    PROJECT
    / "outputs"
    / "step09_lightgbm"
    / WEIGHTED_RUN_NAME
)

evaluate_events(PROJECT, WEIGHTED_RUN_DIR)

# %% [source cell 6]
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


SENSITIVITY_RUNS = {
    "weighted_event_primary": "full",
    "weighted_event_without_lactate": "without_lactate",
    "weighted_event_without_pressure": "without_pressure",
    "weighted_event_without_lactate_pressure": "without_lactate_pressure",
}

sensitivity_rows = []

for run_name, variant in SENSITIVITY_RUNS.items():
    run_dir = (
        PROJECT
        / "outputs"
        / "step09_lightgbm"
        / run_name
    )

    if not run_dir.exists():
        print()
        print("TREINANDO SENSIBILIDADE:", variant)

        run(
            project=PROJECT,
            base_run_dir=BASE_RUN_DIR,
            run_name=run_name,
            variant=variant,
            negative_ratio=5.0,
        )
    else:
        print()
        print("Run existente, sem novo treinamento:", run_dir)

    event_dir = run_dir / "event_evaluation_v2"

    if not event_dir.exists():
        print("AVALIANDO EVENTOS:", variant)
        evaluate_events(PROJECT, run_dir)
    else:
        print("Avaliacao por evento existente:", event_dir)

    metrics = json.loads(
        (run_dir / "metrics.json").read_text()
    )

    features = json.loads(
        (run_dir / "features_used.json").read_text()
    )

    event_metrics = json.loads(
        (
            event_dir
            / "event_evaluation.json"
        ).read_text()
    )

    primary_threshold = event_metrics[
        "frozen_threshold_results"
    ]["maximum_event_window_f1"]

    event_ranking_test = event_metrics[
        "event_window_ranking"
    ]["test"]

    event_threshold_test = primary_threshold[
        "test"
    ]

    sensitivity_rows.append({
        "run_name": run_name,
        "variant": variant,
        "features": len(features["features"]),
        "removed_features": len(
            features["removed"]
        ),
        "timestep_validation_auroc": metrics[
            "validation"
        ]["auroc"],
        "timestep_validation_auprc": metrics[
            "validation"
        ]["auprc"],
        "timestep_test_auroc": metrics[
            "test"
        ]["auroc"],
        "timestep_test_auprc": metrics[
            "test"
        ]["auprc"],
        "event_test_auroc": event_ranking_test[
            "auroc"
        ],
        "event_test_auprc": event_ranking_test[
            "auprc"
        ],
        "event_test_recall": event_threshold_test[
            "recall"
        ],
        "event_test_precision": event_threshold_test[
            "precision"
        ],
        "event_test_specificity": event_threshold_test[
            "specificity"
        ],
        "false_alerts_per_100_patient_days": (
            event_threshold_test[
                "alert_burden"
            ][
                "false_alert_episodes_per_100_patient_days"
            ]
        ),
    })

sensitivity_table = pd.DataFrame(
    sensitivity_rows
)

SENSITIVITY_DIR = (
    PROJECT
    / "outputs"
    / "step13_target_feature_sensitivity"
)

SENSITIVITY_DIR.mkdir(
    parents=True,
    exist_ok=False,
)

sensitivity_table.to_csv(
    SENSITIVITY_DIR
    / "weighted_target_feature_sensitivity.csv",
    index=False,
)

print()
print("SENSIBILIDADE DAS FEATURES DO TARGET CONCLUIDA")
print("Resultados:", SENSITIVITY_DIR)

display(sensitivity_table)

# %% [source cell 7]
ORIGINAL_RUN_DIR = (
    PROJECT
    / "outputs"
    / "step09_lightgbm"
    / "full_primary_20260814T030218Z"
)

ORIGINAL_EVENT_DIR = (
    ORIGINAL_RUN_DIR
    / "event_evaluation_v2"
)

if not ORIGINAL_EVENT_DIR.exists():
    evaluate_events(
        PROJECT,
        ORIGINAL_RUN_DIR,
    )
else:
    print(
        "Avaliação existente:",
        ORIGINAL_EVENT_DIR,
    )

original_event_results = json.loads(
    (
        ORIGINAL_EVENT_DIR
        / "event_evaluation.json"
    ).read_text()
)

print(
    json.dumps(
        {
            "event_window_ranking": (
                original_event_results[
                    "event_window_ranking"
                ]["test"]
            ),
            "maximum_event_window_f1": (
                original_event_results[
                    "frozen_threshold_results"
                ][
                    "maximum_event_window_f1"
                ]["test"]
            ),
        },
        indent=2,
    )
)
