"""Source exported from the reviewed development notebook.

This file contains code only; patient-level data and notebook outputs are not distributed.
"""

# %% [source cell 1]
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch import nn

PROJECT = Path(os.environ.get(
    "CF_PROJECT", "/path/to/circulatory_failure_prediction"
)).resolve()

DATA_DIR = PROJECT / "outputs/step19_eicu_aligned_xy/eicu_v20"
BATCH_DIR = DATA_DIR / "batches"
FEATURE_FILE = PROJECT / "outputs/step07_aligned_xy_v2/feature_names.json"
TRANSFORMER_RUN = PROJECT / "outputs/step11_transformer/transformer_primary_seed42_v7"
FUSION_RUN = PROJECT / "outputs/step13_transformer_tabular_fusion/fusion_transformer_tabular_seed42_v1"

LIGHTGBM_CANDIDATES = [
    PROJECT / "outputs/step09_lightgbm/weighted_event_primary",
    Path("/path/to/circulatory_failure_prediction/outputs/step09_lightgbm/weighted_event_primary"),
]
LIGHTGBM_REQUIRED = ["model.joblib", "features_used.json", "metrics.json"]
complete_lightgbm_runs = [
    path for path in LIGHTGBM_CANDIDATES
    if all((path / name).is_file() for name in LIGHTGBM_REQUIRED)
]
if not complete_lightgbm_runs:
    raise FileNotFoundError(
        "Nenhum run completo do LightGBM foi encontrado: "
        + json.dumps({
            str(path): [name for name in LIGHTGBM_REQUIRED if not (path / name).is_file()]
            for path in LIGHTGBM_CANDIDATES
        }, indent=2)
    )
LIGHTGBM_RUN = complete_lightgbm_runs[0]

OUTPUT_DIR = PROJECT / "outputs/step20_external_validation_eicu/frozen_mimic_models_proxy"
LEGACY_OUTPUT_DIR = PROJECT / "outputs/step20_external_validation_eicu/frozen_mimic_models_proxy_old"
RESET_STEP20_OUTPUTS = False

if LEGACY_OUTPUT_DIR.exists():
    shutil.rmtree(LEGACY_OUTPUT_DIR)
if RESET_STEP20_OUTPUTS and OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "target_first_onset_0_8h"
SENSITIVITY_TARGET = "target_first_onset_map_only_0_8h"
HISTORY_ROWS = 96
BATCH_SIZE = 512
ALERT_GAP_MINUTES = 30
RUN_BOOTSTRAP = True
BOOTSTRAP_REPLICATES = 1000
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fusion_threshold_payload = json.loads(
    (FUSION_RUN / "thresholds_selected_on_validation.json").read_text()
)
FROZEN_THRESHOLDS = {
    "Transformer": 0.3960854142904282,
    "LightGBM": 0.5693604783826403,
    "FusionMLP": float(
        fusion_threshold_payload["FusionMLP"]["maximum_event_window_f1"]
    ),
}
PROBABILITY_COLUMNS = {
    "Transformer": "probability_transformer",
    "LightGBM": "probability_lightgbm",
    "FusionMLP": "probability_fusion",
}

manifest_path = DATA_DIR / "manifest.json"
assert manifest_path.is_file(), "Execute o Step 19 completo; o smoke não autoriza validação."
manifest = json.loads(manifest_path.read_text())
assert manifest["status"] == "completed"
assert manifest["cohort_kind"] == "fixed_external_validation_cohort"
assert manifest["cohort_subject_target"] == 10_000
assert manifest["external_data_used_for_training"] is False
assert manifest["external_scaler_refit"] is False
assert manifest["external_threshold_tuning"] is False
assert manifest.get("harmonization_revision") == 3

registry = json.loads(FEATURE_FILE.read_text())
FEATURES = list(registry["feature_names"])
FEATURE_HASH = hashlib.sha256("\n".join(FEATURES).encode()).hexdigest()
assert len(FEATURES) == 204
assert manifest["feature_sha256"] == FEATURE_HASH

SOURCE_FILES = sorted(BATCH_DIR.glob("batch_*.parquet"))
assert SOURCE_FILES, f"Nenhum batch completo encontrado em {BATCH_DIR}"

for required in (
    TRANSFORMER_RUN / "best_model.pt",
    TRANSFORMER_RUN / "config.json",
    TRANSFORMER_RUN / "train_scaler.npz",
    FUSION_RUN / "best_model.pt",
    FUSION_RUN / "config.json",
    FUSION_RUN / "thresholds_selected_on_validation.json",
):
    assert required.is_file(), required

print("Coorte externa:", manifest["produced_subjects"], "pacientes")
print("Stays externos:", manifest["produced_stays"])
print("Batches externos:", len(SOURCE_FILES))
print("Transformer:", TRANSFORMER_RUN)
print("LightGBM:", LIGHTGBM_RUN)
print("FusionMLP:", FUSION_RUN)
print("Dispositivo:", DEVICE)

# %% [source cell 3]
class FrozenConfig:
    def __init__(self, raw):
        for key, value in raw.items():
            setattr(self, key, value)


class CausalTransformer(nn.Module):
    def __init__(self, n_features, config):
        super().__init__()
        self.input_projection = nn.Linear(n_features, config.d_model)
        self.position = nn.Parameter(torch.zeros(1, config.history_rows, config.d_model))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=config.num_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, 1)

    def encode(self, x, padding_mask):
        length = x.shape[1]
        valid_lengths = (~padding_mask).sum(dim=1).clamp_min(1)
        offsets = length - valid_lengths
        position_ids = (
            torch.arange(length, device=x.device).unsqueeze(0) + offsets.unsqueeze(1)
        ).clamp_max(length - 1)
        hidden = self.input_projection(x) + self.position[0, position_ids]
        causal_mask = torch.triu(
            torch.ones(length, length, device=x.device, dtype=torch.bool), diagonal=1
        )
        hidden = self.encoder(hidden, mask=causal_mask)
        last_valid = valid_lengths - 1
        endpoint = hidden[torch.arange(len(hidden), device=x.device), last_valid]
        embedding = self.norm(endpoint)
        if not torch.isfinite(embedding).all():
            raise RuntimeError("O Transformer produziu embedding não finito.")
        return embedding

    def forward(self, x, padding_mask):
        logits = self.head(self.encode(x, padding_mask)).squeeze(-1)
        if not torch.isfinite(logits).all():
            raise RuntimeError("O Transformer produziu logits não finitos.")
        return logits


class FrozenTransformerTabularFusion(nn.Module):
    def __init__(self, transformer, fusion_config, transformer_config):
        super().__init__()
        self.transformer = transformer
        for parameter in self.transformer.parameters():
            parameter.requires_grad = False
        self.transformer.eval()

        self.tabular_branch = nn.Sequential(
            nn.Linear(len(FEATURES), int(fusion_config.tabular_hidden)),
            nn.GELU(),
            nn.Dropout(float(fusion_config.dropout)),
            nn.Linear(
                int(fusion_config.tabular_hidden),
                int(fusion_config.tabular_embedding),
            ),
            nn.GELU(),
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(
                int(transformer_config.d_model)
                + int(fusion_config.tabular_embedding),
                int(fusion_config.fusion_hidden),
            ),
            nn.GELU(),
            nn.Dropout(float(fusion_config.dropout)),
            nn.Linear(int(fusion_config.fusion_hidden), 1),
        )

    def train(self, mode=True):
        super().train(mode)
        self.transformer.eval()
        return self

    def forward(self, sequence, padding_mask, current_features):
        with torch.no_grad():
            temporal_embedding = self.transformer.encode(sequence, padding_mask)
        tabular_embedding = self.tabular_branch(current_features)
        combined = torch.cat([temporal_embedding, tabular_embedding], dim=1)
        logits = self.fusion_head(combined).squeeze(-1)
        if not torch.isfinite(logits).all():
            raise RuntimeError("O FusionMLP produziu logits não finitos.")
        return logits


lgbm = joblib.load(LIGHTGBM_RUN / "model.joblib")
lgbm_payload = json.loads((LIGHTGBM_RUN / "features_used.json").read_text())
if isinstance(lgbm_payload, dict):
    lgbm_features = lgbm_payload.get("features") or lgbm_payload.get("feature_names")
else:
    lgbm_features = lgbm_payload
assert list(lgbm_features) == FEATURES, "Ordem das features do LightGBM diverge."

transformer_config_raw = json.loads((TRANSFORMER_RUN / "config.json").read_text())
config = FrozenConfig(transformer_config_raw)
assert int(config.history_rows) == HISTORY_ROWS
checkpoint = torch.load(
    TRANSFORMER_RUN / "best_model.pt", map_location="cpu", weights_only=False
)
assert list(checkpoint.get("features", FEATURES)) == FEATURES
transformer = CausalTransformer(len(FEATURES), config)
transformer.load_state_dict(checkpoint["model_state"])
transformer.to(DEVICE).eval()

scaler = np.load(TRANSFORMER_RUN / "train_scaler.npz", allow_pickle=True)
mean = scaler["mean"].astype("float32")
scale = scaler["scale"].astype("float32")
assert len(mean) == len(scale) == len(FEATURES)
if "features" in scaler.files:
    assert [str(x) for x in scaler["features"].tolist()] == FEATURES
assert np.isfinite(mean).all() and np.isfinite(scale).all() and (scale > 0).all()

fusion_config_raw = json.loads((FUSION_RUN / "config.json").read_text())
fusion_config = FrozenConfig(fusion_config_raw)
assert int(fusion_config.history_rows) == HISTORY_ROWS
fusion_checkpoint = torch.load(
    FUSION_RUN / "best_model.pt", map_location="cpu", weights_only=False
)
assert list(fusion_checkpoint.get("features", FEATURES)) == FEATURES
if "config" in fusion_checkpoint:
    for key in ("history_rows", "tabular_hidden", "tabular_embedding", "fusion_hidden"):
        assert fusion_checkpoint["config"][key] == fusion_config_raw[key]
fusion_transformer = CausalTransformer(len(FEATURES), config)
fusion_model = FrozenTransformerTabularFusion(
    fusion_transformer, fusion_config, config
)
fusion_model.load_state_dict(fusion_checkpoint["model_state"], strict=True)
fusion_model.to(DEVICE).eval()
for name, tensor in transformer.state_dict().items():
    fusion_tensor = fusion_model.transformer.state_dict()[name]
    assert torch.equal(tensor.detach().cpu(), fusion_tensor.detach().cpu()), name

print("Transformer, LightGBM, FusionMLP, scaler, features e thresholds MIMIC carregados e congelados.")

# %% [source cell 5]
def frozen_models_predict_stay(group):
    x_all = (
        group[FEATURES]
        .replace([np.inf, -np.inf], np.nan)
        .to_numpy(dtype="float32")
    )
    x_all = (x_all - mean) / scale
    x_all = np.nan_to_num(x_all, nan=0.0, posinf=0.0, neginf=0.0)
    clip = float(getattr(config, "input_clip_z", 10.0))
    x_all = np.clip(x_all, -clip, clip)

    transformer_outputs = []
    fusion_outputs = []
    for batch_start in range(0, len(group), BATCH_SIZE):
        endpoints = np.arange(batch_start, min(batch_start + BATCH_SIZE, len(group)))
        sequences = np.zeros(
            (len(endpoints), HISTORY_ROWS, len(FEATURES)), dtype="float32"
        )
        padding = np.ones((len(endpoints), HISTORY_ROWS), dtype=bool)
        for row, endpoint in enumerate(endpoints):
            start = max(0, endpoint - HISTORY_ROWS + 1)
            valid = x_all[start:endpoint + 1]
            sequences[row, :len(valid)] = valid
            padding[row, :len(valid)] = False

        sequence_tensor = torch.from_numpy(sequences).to(DEVICE)
        padding_tensor = torch.from_numpy(padding).to(DEVICE)
        current_tensor = torch.from_numpy(x_all[endpoints]).to(DEVICE)
        with torch.no_grad():
            temporal_embedding = transformer.encode(sequence_tensor, padding_tensor)
            transformer_logits = transformer.head(temporal_embedding).squeeze(-1)
            tabular_embedding = fusion_model.tabular_branch(current_tensor)
            fusion_logits = fusion_model.fusion_head(
                torch.cat([temporal_embedding, tabular_embedding], dim=1)
            ).squeeze(-1)
        transformer_outputs.append(
            torch.sigmoid(transformer_logits).cpu().numpy()
        )
        fusion_outputs.append(torch.sigmoid(fusion_logits).cpu().numpy())

    return (
        np.concatenate(transformer_outputs).astype("float32"),
        np.concatenate(fusion_outputs).astype("float32"),
    )


prediction_contract = {
    "feature_sha256": FEATURE_HASH,
    "cohort_sha256": manifest["cohort_sha256"],
    "transformer_checkpoint": str(TRANSFORMER_RUN / "best_model.pt"),
    "transformer_checkpoint_size": (TRANSFORMER_RUN / "best_model.pt").stat().st_size,
    "lightgbm_model": str(LIGHTGBM_RUN / "model.joblib"),
    "lightgbm_model_size": (LIGHTGBM_RUN / "model.joblib").stat().st_size,
    "fusion_checkpoint": str(FUSION_RUN / "best_model.pt"),
    "fusion_checkpoint_size": (FUSION_RUN / "best_model.pt").stat().st_size,
    "data_harmonization_revision": manifest.get("harmonization_revision"),
    "thresholds": FROZEN_THRESHOLDS,
}
prediction_contract_path = OUTPUT_DIR / "prediction_contract.json"
if prediction_contract_path.exists():
    previous_prediction_contract = json.loads(prediction_contract_path.read_text())
    if previous_prediction_contract != prediction_contract:
        if "fusion_checkpoint" not in previous_prediction_contract:
            shutil.rmtree(OUTPUT_DIR)
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            prediction_contract_path = OUTPUT_DIR / "prediction_contract.json"
            print("Predições antigas de dois modelos removidas; iniciando validação com FusionMLP.")
        else:
            comparable_previous = dict(previous_prediction_contract)
            comparable_current = dict(prediction_contract)
            previous_revision = int(comparable_previous.pop("data_harmonization_revision", 0))
            current_revision = int(comparable_current.pop("data_harmonization_revision", 0))
            only_revision_advanced = (
                comparable_previous == comparable_current
                and previous_revision < current_revision
            )
            if only_revision_advanced:
                shutil.rmtree(OUTPUT_DIR)
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                prediction_contract_path = OUTPUT_DIR / "prediction_contract.json"
                print(
                    f"Harmonização eICU atualizada de {previous_revision} para "
                    f"{current_revision}; predições externas antigas removidas."
                )
            else:
                raise RuntimeError(
                    "Existem predições de outro contrato. Use a trava RESET do notebook uma única vez."
                )
else:
    old_predictions = list(OUTPUT_DIR.glob("predictions_*.parquet"))
    if old_predictions:
        raise RuntimeError(
            "Foram encontradas predições sem contrato. Use a trava RESET do notebook uma única vez."
        )
prediction_contract_path.write_text(json.dumps(prediction_contract, indent=2) + "\n")

prediction_files = []
for source_path in SOURCE_FILES:
    output_path = OUTPUT_DIR / f"predictions_{source_path.stem}.parquet"
    prediction_files.append(output_path)
    if output_path.exists():
        print("Já concluído:", output_path.name)
        continue

    frame = pd.read_parquet(source_path)
    if frame.empty:
        pd.DataFrame(columns=[
            "subject_id", "stay_id", "feature_time", "label_time", TARGET,
            SENSITIVITY_TARGET, "sensitivity_eligible",
            "probability_transformer", "probability_lightgbm", "probability_fusion",
        ]).to_parquet(output_path, index=False, compression="zstd")
        continue

    frame = frame.sort_values(["stay_id", "label_time"]).reset_index(drop=True)
    frame["probability_lightgbm"] = lgbm.predict_proba(frame[FEATURES])[:, 1].astype("float32")

    pieces = []
    for _, group in frame.groupby("stay_id", sort=False):
        group = group.copy()
        transformer_probability, fusion_probability = frozen_models_predict_stay(group)
        group["probability_transformer"] = transformer_probability
        group["probability_fusion"] = fusion_probability
        pieces.append(group[[
            "subject_id", "stay_id", "feature_time", "label_time", TARGET,
            SENSITIVITY_TARGET, "sensitivity_eligible",
            "probability_transformer", "probability_lightgbm", "probability_fusion",
        ]])

    pd.concat(pieces, ignore_index=True).to_parquet(
        output_path, index=False, compression="zstd"
    )
    print("Salvo:", output_path.name)

pred = pd.concat(
    [pd.read_parquet(path) for path in prediction_files], ignore_index=True
)
pred = pred.sort_values(["stay_id", "label_time"]).reset_index(drop=True)
assert len(pred) == manifest["rows"]
assert pred["stay_id"].nunique() == manifest["produced_stays"]
assert pred["subject_id"].nunique() == manifest["produced_subjects"]
assert pred[["probability_transformer", "probability_lightgbm", "probability_fusion"]].notna().all().all()

print("Linhas externas:", len(pred))
print("Pacientes externos:", pred["subject_id"].nunique())
print("Stays externos:", pred["stay_id"].nunique())

# %% [source cell 7]
def safe_ranking_metrics(y, probability):
    y = np.asarray(y, dtype="int8")
    probability = np.asarray(probability, dtype="float64")
    result = {
        "rows": int(len(y)),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()),
    }
    if np.unique(y).size < 2:
        result.update(auroc=np.nan, auprc=np.nan, log_loss=np.nan, brier=np.nan)
        return result
    clipped = np.clip(probability, 1e-7, 1 - 1e-7)
    result.update(
        auroc=float(roc_auc_score(y, probability)),
        auprc=float(average_precision_score(y, probability)),
        log_loss=float(log_loss(y, clipped, labels=[0, 1])),
        brier=float(brier_score_loss(y, probability)),
    )
    return result


timestep = pd.DataFrame([
    {
        "dataset": "eICU external validation cohort",
        "model": model,
        **safe_ranking_metrics(pred[TARGET], pred[column]),
    }
    for model, column in PROBABILITY_COLUMNS.items()
])
timestep.to_csv(OUTPUT_DIR / "timestep_metrics.csv", index=False)


def event_window_table(frame, seed=42):
    truth = frame.groupby("stay_id")[TARGET].max()
    positive_ids = set(truth.loc[truth.eq(1)].index)
    negative_ids = set(truth.loc[truth.eq(0)].index)
    if not positive_ids or not negative_ids:
        raise RuntimeError("A avaliação por evento exige stays positivos e negativos.")

    positive_rows = []
    for (stay_id, subject_id), group in frame.loc[
        frame["stay_id"].isin(positive_ids) & frame[TARGET].eq(1)
    ].groupby(["stay_id", "subject_id"], sort=True):
        row = {
            "stay_id": int(stay_id),
            "subject_id": str(subject_id),
            "target": 1,
            "window_rows": int(len(group)),
            "window_start": group["label_time"].min(),
            "window_end": group["label_time"].max(),
        }
        for model, column in PROBABILITY_COLUMNS.items():
            row[f"score_{model}"] = float(group[column].max())
        positive_rows.append(row)

    positives = pd.DataFrame(positive_rows)
    lengths = positives["window_rows"].to_numpy(dtype=int)
    rng = np.random.default_rng(seed)
    negative_rows = []
    for (stay_id, subject_id), group in frame.loc[
        frame["stay_id"].isin(negative_ids)
    ].groupby(["stay_id", "subject_id"], sort=True):
        group = group.sort_values("label_time").reset_index(drop=True)
        desired = int(lengths[rng.integers(0, len(lengths))])
        width = min(desired, len(group))
        start = int(rng.integers(0, len(group) - width + 1))
        window = group.iloc[start:start + width]
        row = {
            "stay_id": int(stay_id),
            "subject_id": str(subject_id),
            "target": 0,
            "window_rows": int(width),
            "window_start": window["label_time"].min(),
            "window_end": window["label_time"].max(),
        }
        for model, column in PROBABILITY_COLUMNS.items():
            row[f"score_{model}"] = float(window[column].max())
        negative_rows.append(row)

    result = pd.concat([positives, pd.DataFrame(negative_rows)], ignore_index=True)
    assert not result.duplicated("stay_id").any()
    return result


events = event_window_table(pred, SEED)
events.to_parquet(OUTPUT_DIR / "event_windows.parquet", index=False, compression="zstd")

event_ranking = pd.DataFrame([
    {
        "dataset": "eICU external validation cohort",
        "model": model,
        **safe_ranking_metrics(events["target"], events[f"score_{model}"]),
    }
    for model in PROBABILITY_COLUMNS
])

clinical_rows = []
for model, threshold in FROZEN_THRESHOLDS.items():
    y = events["target"].to_numpy(dtype="int8")
    prediction = events[f"score_{model}"].ge(threshold).astype("int8").to_numpy()
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, prediction, average="binary", zero_division=0
    )
    clinical_rows.append({
        "dataset": "eICU external validation cohort",
        "model": model,
        "threshold_source": "MIMIC-IV validation (frozen)",
        "threshold": threshold,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(tn / (tn + fp)),
        "f1": float(f1),
    })

clinical = pd.DataFrame(clinical_rows)
event_ranking.to_csv(OUTPUT_DIR / "event_ranking.csv", index=False)
clinical.to_csv(OUTPUT_DIR / "frozen_threshold_clinical_metrics.csv", index=False)

display(timestep)
display(event_ranking)
display(clinical)

# %% [source cell 9]
def count_alert_episodes(times, gap_minutes=30):
    times = pd.Series(pd.to_timedelta(times)).sort_values().drop_duplicates()
    if times.empty:
        return 0
    gaps = times.diff().dt.total_seconds().div(60)
    return int(1 + gaps.iloc[1:].gt(gap_minutes).sum())


evaluated_patient_days = len(pred) * 5 / (60 * 24)
burden_rows = []
for model, threshold in FROZEN_THRESHOLDS.items():
    column = PROBABILITY_COLUMNS[model]
    false_episodes = 0
    lead_hours = []
    detected_events = 0

    for _, group in pred.groupby("stay_id", sort=False):
        group = group.sort_values("label_time")
        false_times = group.loc[
            group[TARGET].eq(0) & group[column].ge(threshold), "label_time"
        ]
        false_episodes += count_alert_episodes(false_times, ALERT_GAP_MINUTES)

        positive = group.loc[group[TARGET].eq(1)]
        if positive.empty:
            continue
        onset = pd.to_timedelta(positive["label_time"].max()) + pd.Timedelta(minutes=5)
        alerts = positive.loc[positive[column].ge(threshold), "label_time"]
        if not alerts.empty:
            detected_events += 1
            first_alert = pd.to_timedelta(alerts.min())
            lead_hours.append((onset - first_alert).total_seconds() / 3600)

    burden_rows.append({
        "model": model,
        "threshold": threshold,
        "evaluated_patient_days": evaluated_patient_days,
        "false_alert_episodes": false_episodes,
        "false_alerts_per_100_patient_days": false_episodes / evaluated_patient_days * 100,
        "detected_positive_stays": detected_events,
        "median_lead_time_hours": float(np.median(lead_hours)) if lead_hours else np.nan,
        "lead_time_q1_hours": float(np.quantile(lead_hours, 0.25)) if lead_hours else np.nan,
        "lead_time_q3_hours": float(np.quantile(lead_hours, 0.75)) if lead_hours else np.nan,
    })

alert_burden = pd.DataFrame(burden_rows)
alert_burden.to_csv(OUTPUT_DIR / "alert_burden_and_lead_time.csv", index=False)
display(alert_burden)

# %% [source cell 11]
if RUN_BOOTSTRAP:
    rng = np.random.default_rng(SEED)
    subjects = np.sort(events["subject_id"].unique())
    records = []
    comparison_pairs = [
        ("LightGBM", "Transformer"),
        ("FusionMLP", "Transformer"),
        ("LightGBM", "FusionMLP"),
    ]

    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        multiplicity = pd.Series(sampled).value_counts()
        weights = events["subject_id"].map(multiplicity).fillna(0).to_numpy(dtype="float64")
        selected = weights > 0
        y = events.loc[selected, "target"].to_numpy(dtype="int8")
        w = weights[selected]
        if np.unique(y).size < 2:
            continue

        row = {"replicate": replicate}
        for model in PROBABILITY_COLUMNS:
            probability = events.loc[selected, f"score_{model}"].to_numpy(dtype="float64")
            row[f"auroc_{model}"] = roc_auc_score(y, probability, sample_weight=w)
            row[f"auprc_{model}"] = average_precision_score(y, probability, sample_weight=w)
        for first, second in comparison_pairs:
            slug = f"{first.lower()}_minus_{second.lower()}"
            row[f"delta_auroc_{slug}"] = row[f"auroc_{first}"] - row[f"auroc_{second}"]
            row[f"delta_auprc_{slug}"] = row[f"auprc_{first}"] - row[f"auprc_{second}"]
        records.append(row)

    bootstrap = pd.DataFrame(records)
    bootstrap.to_parquet(
        OUTPUT_DIR / "patient_bootstrap_replicates.parquet",
        index=False,
        compression="zstd",
    )

    ranking_index = event_ranking.set_index("model")
    point_estimates = {}
    for model in PROBABILITY_COLUMNS:
        point_estimates[f"auroc_{model}"] = float(ranking_index.loc[model, "auroc"])
        point_estimates[f"auprc_{model}"] = float(ranking_index.loc[model, "auprc"])
    for first, second in comparison_pairs:
        slug = f"{first.lower()}_minus_{second.lower()}"
        point_estimates[f"delta_auroc_{slug}"] = (
            point_estimates[f"auroc_{first}"] - point_estimates[f"auroc_{second}"]
        )
        point_estimates[f"delta_auprc_{slug}"] = (
            point_estimates[f"auprc_{first}"] - point_estimates[f"auprc_{second}"]
        )

    bootstrap_summary = pd.DataFrame([
        {
            "metric": column,
            "estimate": point_estimates[column],
            "ci95_low": float(bootstrap[column].quantile(0.025)),
            "ci95_high": float(bootstrap[column].quantile(0.975)),
            "valid_replicates": int(bootstrap[column].notna().sum()),
        }
        for column in bootstrap.columns if column != "replicate"
    ])
    bootstrap_summary.to_csv(OUTPUT_DIR / "patient_bootstrap_summary.csv", index=False)
    display(bootstrap_summary)
else:
    print("Bootstrap desativado.")

# %% [source cell 13]
sensitivity_rows = pred.loc[
    pred["sensitivity_eligible"].astype("boolean").fillna(False)
    & pred[SENSITIVITY_TARGET].notna()
].copy()
sensitivity_rows[SENSITIVITY_TARGET] = sensitivity_rows[SENSITIVITY_TARGET].astype("int8")

sensitivity_timestep = pd.DataFrame([
    {
        "dataset": "eICU external validation — MAP-only sensitivity",
        "model": model,
        **safe_ranking_metrics(sensitivity_rows[SENSITIVITY_TARGET], sensitivity_rows[column]),
    }
    for model, column in PROBABILITY_COLUMNS.items()
])
sensitivity_timestep.to_csv(
    OUTPUT_DIR / "timestep_metrics_map_only_sensitivity.csv", index=False
)

external_manifest = {
    "status": "completed",
    "external_dataset": "eICU-CRD v2.0",
    "validation_classification": "harmonized external validation with documented target proxies",
    "strict_semantic_equivalence": False,
    "cohort_kind": manifest["cohort_kind"],
    "cohort_subject_target": manifest["cohort_subject_target"],
    "cohort_seed": manifest["cohort_seed"],
    "cohort_sha256": manifest["cohort_sha256"],
    "external_rows": int(len(pred)),
    "external_stays": int(pred["stay_id"].nunique()),
    "external_subjects": int(pred["subject_id"].nunique()),
    "feature_count": len(FEATURES),
    "feature_sha256": FEATURE_HASH,
    "transformer_checkpoint": str(TRANSFORMER_RUN / "best_model.pt"),
    "transformer_scaler": str(TRANSFORMER_RUN / "train_scaler.npz"),
    "lightgbm_model": str(LIGHTGBM_RUN / "model.joblib"),
    "fusion_checkpoint": str(FUSION_RUN / "best_model.pt"),
    "thresholds": FROZEN_THRESHOLDS,
    "threshold_source": "MIMIC-IV validation",
    "external_retraining": False,
    "external_scaler_refit": False,
    "external_threshold_tuning": False,
}
(OUTPUT_DIR / "manifest.json").write_text(
    json.dumps(external_manifest, indent=2) + "\n"
)

display(sensitivity_timestep)
print(json.dumps(external_manifest, indent=2))
