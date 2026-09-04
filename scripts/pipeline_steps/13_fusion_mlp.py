"""Source exported from the reviewed development notebook.

This file contains code only; patient-level data and notebook outputs are not distributed.
"""

# %% [source cell 2]
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

PROJECT = Path(
    os.environ.get(
        "CF_PROJECT",
        "/path/to/circulatory_failure_prediction",
    )
).resolve()

TRANSFORMER_RUN = (
    PROJECT / "outputs/step11_transformer/transformer_primary_seed42_v7"
)
LIGHTGBM_RUN = PROJECT / "outputs/step09_lightgbm/weighted_event_primary"
BATCH_DIR = PROJECT / "outputs/step07_aligned_xy_v2/batches"
FEATURE_FILE = PROJECT / "outputs/step07_aligned_xy_v2/feature_names.json"
AUDIT_FILE = (
    PROJECT / "outputs/step10_event_weight_audit/stay_event_weight_audit.parquet"
)
SPLIT_FILE = LIGHTGBM_RUN / "stay_subject_split.parquet"
OUTPUT_ROOT = PROJECT / "outputs/step13_transformer_tabular_fusion"

TARGET = "target_first_onset_0_8h"
EXPECTED_BATCHES = 285
SEED = 42
SHORT_HISTORY_FLOOR_ROWS = 12

RUN_SMOKE = False
RUN_FULL = False
RESUME_FULL = False
RUN_BOOTSTRAP = True

FULL_RUN_NAME = "fusion_transformer_tabular_seed42_v1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Projeto:", PROJECT)
print("Dispositivo:", DEVICE)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# %% [source cell 4]
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def names_sha256(names: list[str]) -> str:
    return hashlib.sha256("\n".join(names).encode()).hexdigest()


def canonical_batches(directory: Path) -> list[Path]:
    files = sorted(
        p for p in directory.glob("*.parquet")
        if re.fullmatch(r"batch_\d{4}\.parquet", p.name)
    )
    expected = [f"batch_{index:04d}.parquet" for index in range(EXPECTED_BATCHES)]
    if [p.name for p in files] != expected:
        raise RuntimeError(
            f"Esperava batch_0000--batch_0284 em {directory}; encontrei {len(files)}"
        )
    return files


required_files = {
    "transformer_checkpoint": TRANSFORMER_RUN / "best_model.pt",
    "transformer_config": TRANSFORMER_RUN / "config.json",
    "transformer_scaler": TRANSFORMER_RUN / "train_scaler.npz",
    "lightgbm_features": LIGHTGBM_RUN / "features_used.json",
    "split": SPLIT_FILE,
    "weight_audit": AUDIT_FILE,
    "feature_registry": FEATURE_FILE,
}
missing = {name: str(path) for name, path in required_files.items() if not path.is_file()}
if missing:
    raise FileNotFoundError("Arquivos obrigatórios ausentes:\n" + json.dumps(missing, indent=2))

FILES = canonical_batches(BATCH_DIR)
transformer_config = json.loads(required_files["transformer_config"].read_text())
feature_registry = json.loads(required_files["feature_registry"].read_text())
lightgbm_feature_payload = json.loads(required_files["lightgbm_features"].read_text())
scaler_payload = np.load(required_files["transformer_scaler"], allow_pickle=True)

FEATURES = [str(value) for value in scaler_payload["features"].tolist()]
registry_features = list(feature_registry["feature_names"])
lightgbm_features = list(lightgbm_feature_payload["features"])

assert len(FEATURES) == 204
assert FEATURES == registry_features == lightgbm_features
assert TARGET not in FEATURES

MEAN = scaler_payload["mean"].astype("float32")
SCALE = scaler_payload["scale"].astype("float32")
assert MEAN.shape == SCALE.shape == (204,)
assert np.isfinite(MEAN).all() and np.isfinite(SCALE).all()
assert np.all(SCALE > 0)

SPLIT_TABLE = pd.read_parquet(SPLIT_FILE)
required_split_columns = {"stay_id", "subject_id", "split", "stay_positive"}
assert required_split_columns.issubset(SPLIT_TABLE.columns)
assert SPLIT_TABLE.groupby("subject_id")["split"].nunique().max() == 1

COUNTS = pd.read_parquet(
    AUDIT_FILE,
    columns=["stay_id", "positive_rows", "negative_rows"],
)
assert not COUNTS["stay_id"].duplicated().any()
assert set(SPLIT_TABLE.stay_id) == set(COUNTS.stay_id)

base_checkpoint = torch.load(
    required_files["transformer_checkpoint"],
    map_location="cpu",
    weights_only=False,
)
assert base_checkpoint["features"] == FEATURES

train_counts = (
    SPLIT_TABLE.loc[SPLIT_TABLE.split.eq("train"), ["stay_id"]]
    .merge(COUNTS, on="stay_id", how="left", validate="one_to_one")
)
train_positive_rows = int(train_counts.positive_rows.sum())
train_negative_rows = int(train_counts.negative_rows.sum())
NEGATIVE_FRACTION = min(1.0, 5.0 * train_positive_rows / train_negative_rows)

INPUT_AUDIT = {
    "feature_count": len(FEATURES),
    "feature_sha256": names_sha256(FEATURES),
    "transformer_checkpoint_sha256": sha256_file(required_files["transformer_checkpoint"]),
    "split_counts": SPLIT_TABLE.groupby("split").stay_id.nunique().to_dict(),
    "subject_counts": SPLIT_TABLE.groupby("split").subject_id.nunique().to_dict(),
    "negative_sampling_fraction": float(NEGATIVE_FRACTION),
}

print(json.dumps(INPUT_AUDIT, indent=2))

# %% [source cell 6]
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def stable_keep_negative(frame: pd.DataFrame, fraction: float, seed: int) -> np.ndarray:
    if fraction >= 1.0:
        return np.ones(len(frame), dtype=bool)
    hashes = pd.util.hash_pandas_object(
        frame[["stay_id", "label_time"]], index=False
    ).to_numpy(dtype="uint64")
    hashes ^= np.uint64(seed)
    threshold = np.uint64(fraction * np.iinfo(np.uint64).max)
    return hashes <= threshold


@dataclass(frozen=True)
class FusionConfig:
    run_name: str
    resume: bool = False
    seed: int = 42
    history_rows: int = 96
    negative_ratio: float = 5.0
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 15
    early_stopping_patience: int = 4
    num_workers: int = 2
    input_clip_z: float = 10.0
    tabular_hidden: int = 128
    tabular_embedding: int = 64
    fusion_hidden: int = 64
    dropout: float = 0.15
    grad_clip: float = 1.0
    smoke: bool = False
    smoke_batches: int = 4
    smoke_stays_per_worker: int = 100


class FusionWindowStream(IterableDataset):
    def __init__(
        self,
        files,
        split_name: str,
        config: FusionConfig,
        epoch: int = 0,
    ):
        self.files = list(files)
        self.allowed = SPLIT_TABLE.loc[
            SPLIT_TABLE.split.eq(split_name), ["stay_id", "subject_id"]
        ].copy()
        self.counts = COUNTS.set_index("stay_id")[["positive_rows", "negative_rows"]]
        self.split_name = split_name
        self.config = config
        self.epoch = epoch
        self.negative_fraction = NEGATIVE_FRACTION if split_name == "train" else 1.0

    def __iter__(self):
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        worker_count = 1 if worker is None else worker.num_workers
        files = self.files[worker_id::worker_count]
        rng = np.random.default_rng(
            self.config.seed + 10_000 * self.epoch + worker_id
        )
        yielded_stays = 0

        for path in files:
            frame = pd.read_parquet(
                path,
                columns=["stay_id", "label_time", TARGET, *FEATURES],
            )
            frame = frame.merge(
                self.allowed,
                on="stay_id",
                how="inner",
                validate="many_to_one",
            )
            if frame.empty:
                continue
            frame["label_time"] = pd.to_datetime(frame["label_time"], errors="raise")
            stay_ids = frame.stay_id.drop_duplicates().to_numpy()
            if self.split_name == "train":
                rng.shuffle(stay_ids)

            for stay_id in stay_ids:
                group = (
                    frame.loc[frame.stay_id.eq(stay_id)]
                    .sort_values("label_time")
                    .reset_index(drop=True)
                )
                y_all = group[TARGET].to_numpy(dtype="int8")
                if self.split_name == "train" and self.negative_fraction < 1.0:
                    keep_negative = stable_keep_negative(
                        group, self.negative_fraction, self.config.seed
                    )
                    endpoints = np.flatnonzero((y_all == 1) | keep_negative)
                else:
                    endpoints = np.arange(len(group))
                if self.split_name == "train":
                    rng.shuffle(endpoints)

                x_all = (
                    group[FEATURES]
                    .replace([np.inf, -np.inf], np.nan)
                    .to_numpy(dtype="float32")
                )
                x_all = (x_all - MEAN) / SCALE
                x_all = np.nan_to_num(x_all, nan=0.0, posinf=0.0, neginf=0.0)
                x_all = np.clip(
                    x_all, -self.config.input_clip_z, self.config.input_clip_z
                )
                row_counts = self.counts.loc[stay_id]

                for endpoint in endpoints:
                    start = max(0, endpoint - self.config.history_rows + 1)
                    valid = x_all[start : endpoint + 1]
                    length = len(valid)
                    sequence = np.zeros(
                        (self.config.history_rows, len(FEATURES)), dtype="float32"
                    )
                    padding = np.ones(self.config.history_rows, dtype=bool)
                    sequence[:length] = valid
                    padding[:length] = False
                    current_features = valid[-1].copy()
                    target = int(y_all[endpoint])

                    if target == 1:
                        weight = 1.0 / max(
                            float(row_counts.positive_rows), SHORT_HISTORY_FLOOR_ROWS
                        )
                    else:
                        weight = 1.0 / max(float(row_counts.negative_rows), 1.0)
                        if self.split_name == "train" and self.negative_fraction < 1.0:
                            weight /= self.negative_fraction

                    yield (
                        torch.from_numpy(sequence),
                        torch.from_numpy(padding),
                        torch.from_numpy(current_features),
                        torch.tensor(target, dtype=torch.float32),
                        torch.tensor(weight, dtype=torch.float32),
                        int(stay_id),
                        int(group.subject_id.iloc[0]),
                        int(group.label_time.iloc[endpoint].value),
                    )

                yielded_stays += 1
                if (
                    self.config.smoke
                    and yielded_stays >= self.config.smoke_stays_per_worker
                ):
                    return


class CausalTransformer(nn.Module):
    def __init__(self, n_features: int, config: dict):
        super().__init__()
        history_rows = int(config["history_rows"])
        d_model = int(config["d_model"])
        self.history_rows = history_rows
        self.input_projection = nn.Linear(n_features, d_model)
        self.position = nn.Parameter(torch.zeros(1, history_rows, d_model))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=int(config["nhead"]),
            dim_feedforward=int(config["dim_feedforward"]),
            dropout=float(config["dropout"]),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=int(config["num_layers"]),
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def encode(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        length = x.shape[1]
        valid_lengths = (~padding_mask).sum(dim=1).clamp_min(1)
        offsets = length - valid_lengths
        position_ids = (
            torch.arange(length, device=x.device).unsqueeze(0)
            + offsets.unsqueeze(1)
        ).clamp_max(length - 1)
        hidden = self.input_projection(x) + self.position[0, position_ids]
        causal_mask = torch.triu(
            torch.ones(length, length, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        hidden = self.encoder(hidden, mask=causal_mask)
        last_valid = valid_lengths - 1
        endpoint_hidden = hidden[
            torch.arange(len(hidden), device=x.device), last_valid
        ]
        embedding = self.norm(endpoint_hidden)
        if not torch.isfinite(embedding).all():
            raise RuntimeError("Embedding não finito produzido pelo Transformer")
        return embedding

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x, padding_mask)).squeeze(-1)


class FrozenTransformerTabularFusion(nn.Module):
    def __init__(self, transformer: CausalTransformer, config: FusionConfig):
        super().__init__()
        self.transformer = transformer
        for parameter in self.transformer.parameters():
            parameter.requires_grad = False
        self.transformer.eval()

        temporal_dimensions = int(transformer_config["d_model"])
        self.tabular_branch = nn.Sequential(
            nn.Linear(len(FEATURES), config.tabular_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.tabular_hidden, config.tabular_embedding),
            nn.GELU(),
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(
                temporal_dimensions + config.tabular_embedding,
                config.fusion_hidden,
            ),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fusion_hidden, 1),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.transformer.eval()
        return self

    def forward(self, sequence, padding_mask, current_features):
        with torch.no_grad():
            temporal_embedding = self.transformer.encode(sequence, padding_mask)
        tabular_embedding = self.tabular_branch(current_features)
        combined = torch.cat([temporal_embedding, tabular_embedding], dim=1)
        return self.fusion_head(combined).squeeze(-1)


def build_fusion_model(config: FusionConfig, device: torch.device):
    transformer = CausalTransformer(len(FEATURES), transformer_config)
    transformer.load_state_dict(base_checkpoint["model_state"], strict=True)
    model = FrozenTransformerTabularFusion(transformer, config).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Parâmetros treináveis: {trainable:,}")
    print(f"Parâmetros congelados: {frozen:,}")
    return model


def make_loader(files, split_name, config, epoch=0):
    dataset = FusionWindowStream(files, split_name, config, epoch)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

# %% [source cell 8]
def safe_ranking_metrics(y: np.ndarray, probability: np.ndarray) -> dict:
    y = np.asarray(y, dtype="int8")
    probability = np.asarray(probability, dtype="float64")
    result = {
        "rows": int(len(y)),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()),
    }
    if np.unique(y).size == 2:
        result.update(
            auroc=float(roc_auc_score(y, probability)),
            auprc=float(average_precision_score(y, probability)),
            log_loss=float(log_loss(y, probability, labels=[0, 1])),
            brier=float(brier_score_loss(y, probability)),
        )
    else:
        result.update(auroc=None, auprc=None, log_loss=None, brier=None)
    return result


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, probabilities, stays, subjects, times = [], [], [], [], []
    for sequence, padding, current, y, _, stay, subject, time_ns in loader:
        logits = model(
            sequence.to(device, non_blocking=True),
            padding.to(device, non_blocking=True),
            current.to(device, non_blocking=True),
        )
        ys.append(y.numpy().astype("int8"))
        probabilities.append(torch.sigmoid(logits).cpu().numpy().astype("float32"))
        stays.append(stay.numpy().astype("int64"))
        subjects.append(subject.numpy().astype("int64"))
        times.append(time_ns.numpy().astype("int64"))

    y_all = np.concatenate(ys)
    probability_all = np.concatenate(probabilities)
    prediction_frame = pd.DataFrame({
        "stay_id": np.concatenate(stays),
        "subject_id": np.concatenate(subjects),
        "label_time": pd.to_datetime(np.concatenate(times)),
        "target": y_all,
        "probability": probability_all,
    }).sort_values(["stay_id", "label_time"]).reset_index(drop=True)
    return safe_ranking_metrics(y_all, probability_all), prediction_frame


def atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def atomic_json_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def atomic_history_save(history: list[dict], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(history).to_csv(temporary, index=False)
    temporary.replace(path)


def create_run_directory(base: Path, smoke: bool, resume: bool) -> Path:
    if resume:
        if not base.is_dir():
            raise FileNotFoundError(base)
        if (base / "metrics.json").exists():
            raise RuntimeError(f"Execução já concluída: {base}")
        return base
    if not smoke:
        base.mkdir(parents=True, exist_ok=False)
        return base
    candidate = base
    suffix = 1
    while True:
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            suffix += 1
            candidate = base.parent / f"{base.name}_{suffix:02d}"

# %% [source cell 10]
def train_fusion(config: FusionConfig) -> dict:
    seed_everything(config.seed)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    selected_files = FILES[: config.smoke_batches] if config.smoke else FILES
    run_dir_base = OUTPUT_ROOT / config.run_name
    run_dir = create_run_directory(
        run_dir_base, smoke=config.smoke, resume=config.resume
    )

    model = build_fusion_model(config, DEVICE)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_function = nn.BCEWithLogitsLoss(reduction="none")

    best_ap = -math.inf
    best_epoch = 0
    patience = 0
    history = []
    start_epoch = 1
    latest_path = run_dir / "latest_checkpoint.pt"

    config_payload = {
        **asdict(config),
        "project": str(PROJECT),
        "base_transformer": str(required_files["transformer_checkpoint"]),
        "base_transformer_sha256": INPUT_AUDIT["transformer_checkpoint_sha256"],
        "feature_sha256": INPUT_AUDIT["feature_sha256"],
        "fusion_definition": "frozen transformer endpoint embedding + normalized current 204 features",
        "temporal_embedding_dimensions": int(transformer_config["d_model"]),
        "current_feature_dimensions": len(FEATURES),
    }

    if config.resume:
        state = torch.load(latest_path, map_location="cpu", weights_only=False)
        saved_config = dict(state["config"])
        current_config = dict(config_payload)
        saved_config.pop("resume", None)
        current_config.pop("resume", None)
        if saved_config != current_config or state["features"] != FEATURES:
            raise RuntimeError("Configuração ou features incompatíveis com o checkpoint")
        model.load_state_dict(state["model_state"], strict=True)
        optimizer.load_state_dict(state["optimizer_state"])
        best_ap = float(state["best_ap"])
        best_epoch = int(state["best_epoch"])
        patience = int(state["patience"])
        history = list(state["history"])
        start_epoch = int(state["epoch"]) + 1
        random.setstate(state["rng_state"]["python"])
        np.random.set_state(state["rng_state"]["numpy"])
        torch.set_rng_state(state["rng_state"]["torch_cpu"])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["rng_state"]["torch_cuda"])
        print(f"Retomando na época {start_epoch}")
    else:
        atomic_json_save(config_payload, run_dir / "config.json")
        atomic_json_save(INPUT_AUDIT, run_dir / "input_audit.json")
        (run_dir / "features_used.json").write_text(
            json.dumps({"features": FEATURES}, indent=2) + "\n"
        )

    started = time.perf_counter()
    for epoch in range(start_epoch, config.max_epochs + 1):
        model.train()
        weighted_loss_sum = 0.0
        weight_sum = 0.0
        train_rows = 0
        train_loader = make_loader(selected_files, "train", config, epoch)

        for sequence, padding, current, y, weight, *_ in train_loader:
            sequence = sequence.to(DEVICE, non_blocking=True)
            padding = padding.to(DEVICE, non_blocking=True)
            current = current.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            weight = weight.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(sequence, padding, current)
            individual_loss = loss_function(logits, y)
            loss = (individual_loss * weight).sum() / weight.sum()
            if not torch.isfinite(loss):
                raise RuntimeError("Loss não finita")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                config.grad_clip,
            )
            if not torch.isfinite(gradient_norm):
                raise RuntimeError("Gradiente não finito")
            optimizer.step()

            weighted_loss_sum += float(
                (individual_loss.detach() * weight).sum().cpu()
            )
            weight_sum += float(weight.sum().cpu())
            train_rows += len(y)

        validation_loader = make_loader(
            selected_files, "validation", config, epoch=0
        )
        validation_metrics, _ = predict(model, validation_loader, DEVICE)
        row = {
            "epoch": epoch,
            "train_weighted_loss": weighted_loss_sum / weight_sum,
            "train_rows": train_rows,
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        }
        history.append(row)
        print(json.dumps(row), flush=True)

        current_ap = validation_metrics.get("auprc")
        if current_ap is not None and current_ap > best_ap + 1e-5:
            best_ap = float(current_ap)
            best_epoch = epoch
            patience = 0
            atomic_torch_save(
                {
                    "model_state": model.state_dict(),
                    "config": config_payload,
                    "features": FEATURES,
                    "best_epoch": best_epoch,
                    "best_ap": best_ap,
                },
                run_dir / "best_model.pt",
            )
        else:
            patience += 1

        atomic_torch_save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": config_payload,
                "features": FEATURES,
                "best_ap": best_ap,
                "best_epoch": best_epoch,
                "patience": patience,
                "history": history,
                "rng_state": {
                    "python": random.getstate(),
                    "numpy": np.random.get_state(),
                    "torch_cpu": torch.get_rng_state(),
                    "torch_cuda": (
                        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
                    ),
                },
            },
            latest_path,
        )
        atomic_history_save(history, run_dir / "training_history.csv")
        print(f"Checkpoint salvo: época={epoch}; melhor={best_epoch}")

        if patience >= config.early_stopping_patience:
            print(f"Early stopping após a época {epoch}")
            break
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    best = torch.load(
        run_dir / "best_model.pt", map_location=DEVICE, weights_only=False
    )
    model.load_state_dict(best["model_state"], strict=True)

    split_metrics = {}
    for split_name in ("validation", "test"):
        loader = make_loader(selected_files, split_name, config, epoch=0)
        split_metrics[split_name], prediction_frame = predict(model, loader, DEVICE)
        prediction_frame.to_parquet(
            run_dir / f"{split_name}_predictions.parquet",
            index=False,
            compression="zstd",
        )
        del prediction_frame
        gc.collect()

    result = {
        "status": "completed",
        "run_name": run_dir.name,
        "output_directory": str(run_dir),
        "smoke": config.smoke,
        "best_epoch": int(best["best_epoch"]),
        "best_validation_auprc": float(best["best_ap"]),
        "timestep_metrics": split_metrics,
        "transformer_frozen": True,
        "feature_count": len(FEATURES),
        "negative_sampling_fraction": float(NEGATIVE_FRACTION),
        "elapsed_minutes": (time.perf_counter() - started) / 60,
        "warning": "Internal test is exploratory; confirm gains externally or in a new holdout.",
    }
    atomic_json_save(result, run_dir / "metrics.json")
    print(json.dumps(result, indent=2))
    return result

# %% [source cell 12]
if RUN_SMOKE:
    smoke_config = FusionConfig(
        run_name="fusion_transformer_tabular_smoke_v1",
        seed=SEED,
        smoke=True,
        max_epochs=2,
        early_stopping_patience=2,
        num_workers=0,
        batch_size=128,
    )
    smoke_result = train_fusion(smoke_config)
    assert smoke_result["status"] == "completed"
else:
    print("Smoke test desativado.")

# %% [source cell 14]
RUN_SMOKE = False
RUN_FULL = True
RESUME_FULL = True

full_run_dir = OUTPUT_ROOT / FULL_RUN_NAME

if (full_run_dir / "metrics.json").exists():
    print("O treinamento completo já terminou.")
else:
    assert (full_run_dir / "latest_checkpoint.pt").exists()
    assert len(FILES) == 285

    full_config = FusionConfig(
        run_name=FULL_RUN_NAME,
        resume=True,
        seed=SEED,
        smoke=False,
        smoke_batches=4,
        max_epochs=15,
        early_stopping_patience=4,
        batch_size=256,
        num_workers=2,
        learning_rate=1e-3,
        weight_decay=1e-4,
        dropout=0.15,
    )

    full_result = train_fusion(full_config)

# %% [source cell 16]
def load_prediction(path: Path, probability_name: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)

    required = {"stay_id", "subject_id", "label_time", "target", "probability"}
    if not required.issubset(frame.columns):
        raise RuntimeError(
            f"Colunas ausentes em {path}: {sorted(required - set(frame.columns))}"
        )

    frame = frame[
        ["stay_id", "subject_id", "label_time", "target", "probability"]
    ].copy()

    # Padroniza os tipos entre os três modelos
    frame["stay_id"] = pd.to_numeric(frame["stay_id"]).astype("int64")
    frame["subject_id"] = pd.to_numeric(frame["subject_id"]).astype("int64")
    frame["target"] = pd.to_numeric(frame["target"]).astype("int8")
    frame["probability"] = pd.to_numeric(frame["probability"]).astype("float32")
    frame["label_time"] = (
        pd.to_datetime(frame["label_time"], errors="raise", utc=True)
        .dt.tz_convert(None)
        .astype("datetime64[ns]")
    )

    frame = frame.rename(columns={"probability": probability_name})

    keys = ["stay_id", "subject_id", "label_time"]
    if frame.duplicated(keys).any():
        raise RuntimeError(f"Chaves duplicadas em {path}")

    return frame.sort_values(keys).reset_index(drop=True)


def align_three_models(split: str, fusion_run: Path) -> pd.DataFrame:
    paths = {
        "probability_fusion":
            fusion_run / f"{split}_predictions.parquet",
        "probability_transformer":
            TRANSFORMER_RUN / f"{split}_predictions.parquet",
        "probability_lightgbm":
            LIGHTGBM_RUN / f"{split}_predictions.parquet",
    }

    frames = {
        name: load_prediction(path, name)
        for name, path in paths.items()
    }

    keys = ["stay_id", "subject_id", "label_time"]
    base = frames["probability_fusion"].copy()

    for name in ("probability_transformer", "probability_lightgbm"):
        other = frames[name]

        merged = base.merge(
            other[keys + ["target", name]],
            on=keys,
            how="outer",
            suffixes=("", "_other"),
            indicator=True,
            validate="one_to_one",
        )

        if not merged["_merge"].eq("both").all():
            counts = merged["_merge"].value_counts().to_dict()
            raise RuntimeError(
                f"Conjuntos de previsões realmente diferentes para "
                f"{split}/{name}: {counts}"
            )

        if not merged["target"].eq(merged["target_other"]).all():
            raise RuntimeError(
                f"Targets divergentes para {split}/{name}"
            )

        base = (
            merged.drop(columns=["target_other", "_merge"])
            .sort_values(keys)
            .reset_index(drop=True)
        )

    return base


def shared_event_table(aligned: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    probability_columns = [
        "probability_fusion",
        "probability_transformer",
        "probability_lightgbm",
    ]
    truth = aligned.groupby("stay_id", sort=False).target.max()
    positive_ids = set(truth.index[truth.eq(1)])
    negative_ids = set(truth.index[truth.eq(0)])

    aggregation = {
        "subject_id": ("subject_id", "first"),
        "window_rows": ("target", "size"),
        "window_start": ("label_time", "min"),
        "window_end": ("label_time", "max"),
    }
    for column in probability_columns:
        aggregation[f"score_{column.removeprefix('probability_')}"] = (column, "max")

    positives = (
        aligned.loc[aligned.target.eq(1)]
        .groupby("stay_id", as_index=False)
        .agg(**aggregation)
    )
    positives["target"] = 1
    if set(positives.stay_id) != positive_ids:
        raise RuntimeError("Stay positivo sem janela positiva")

    lengths = positives.window_rows.to_numpy(dtype="int64")
    rng = np.random.default_rng(seed)
    controls = []
    negative = aligned.loc[aligned.stay_id.isin(negative_ids)].sort_values(
        ["stay_id", "label_time"]
    )
    for (stay_id, subject_id), group in negative.groupby(
        ["stay_id", "subject_id"], sort=True
    ):
        desired = int(lengths[rng.integers(0, len(lengths))])
        length = min(desired, len(group))
        start = int(rng.integers(0, len(group) - length + 1))
        window = group.iloc[start : start + length]
        row = {
            "stay_id": int(stay_id),
            "subject_id": int(subject_id),
            "target": 0,
            "window_rows": int(length),
            "window_start": window.label_time.min(),
            "window_end": window.label_time.max(),
        }
        for column in probability_columns:
            row[f"score_{column.removeprefix('probability_')}"] = float(
                window[column].max()
            )
        controls.append(row)

    table = pd.concat([positives, pd.DataFrame(controls)], ignore_index=True)
    if table.stay_id.duplicated().any() or len(table) != aligned.stay_id.nunique():
        raise RuntimeError("Tabela por evento inválida")
    return table.sort_values(["subject_id", "stay_id"]).reset_index(drop=True)


def confusion_metrics(y, probability, threshold):
    y = np.asarray(y, dtype=bool)
    alert = np.asarray(probability, dtype=float) >= float(threshold)
    tp = int(np.sum(alert & y)); fp = int(np.sum(alert & ~y))
    fn = int(np.sum(~alert & y)); tn = int(np.sum(~alert & ~y))
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall,
        "specificity": specificity, "f1": f1,
    }


def choose_event_thresholds(y, probability):
    y = np.asarray(y, dtype=bool)
    probability = np.asarray(probability, dtype=float)
    thresholds = np.unique(
        np.r_[0.0, 1.0, np.quantile(probability, np.linspace(0, 1, 2001))]
    )
    rows = []
    for threshold in thresholds:
        rows.append({"threshold": float(threshold), **confusion_metrics(y, probability, threshold)})
    curve = pd.DataFrame(rows)
    best = curve.sort_values(["f1", "threshold"], ascending=[False, False]).iloc[0]
    selected = {"maximum_event_window_f1": float(best.threshold)}
    candidates = curve.loc[curve.recall.ge(0.80)]
    if not candidates.empty:
        selected["event_recall_80"] = float(candidates.threshold.max())
    return selected, curve


def alert_burden(aligned, probability_column, threshold, gap_minutes=30):
    alerts = aligned.loc[
        aligned[probability_column].ge(threshold),
        ["stay_id", "subject_id", "label_time", "target"],
    ].sort_values(["stay_id", "label_time"])
    if alerts.empty:
        false_episodes = true_episodes = 0
    else:
        gap = alerts.groupby("stay_id").label_time.diff()
        new_episode = gap.isna() | gap.gt(pd.Timedelta(minutes=gap_minutes))
        alerts["episode"] = new_episode.groupby(alerts.stay_id).cumsum()
        episodes = alerts.groupby(
            ["stay_id", "subject_id", "episode"], as_index=False
        ).agg(true_event_window=("target", "max"))
        false_episodes = int(episodes.true_event_window.eq(0).sum())
        true_episodes = int(episodes.true_event_window.eq(1).sum())
    negative_patient_days = float(aligned.target.eq(0).sum() * 5 / 1440)
    return {
        "true_alert_episodes": true_episodes,
        "false_alert_episodes": false_episodes,
        "eligible_negative_patient_days": negative_patient_days,
        "false_alert_episodes_per_100_patient_days": (
            100 * false_episodes / negative_patient_days
            if negative_patient_days else None
        ),
    }


def lead_time_summary(aligned, probability_column, threshold):
    positive = aligned.loc[aligned.target.eq(1)].copy()
    # Na grade elegível, o último label positivo fica 5 min antes do primeiro onset.
    onsets = (
        positive.groupby("stay_id", as_index=False).label_time.max()
        .rename(columns={"label_time": "last_positive_time"})
    )
    onsets["first_onset_time"] = onsets.last_positive_time + pd.Timedelta(minutes=5)
    first_alert = (
        positive.loc[positive[probability_column].ge(threshold)]
        .groupby("stay_id", as_index=False).label_time.min()
        .rename(columns={"label_time": "first_alert_time"})
    )
    base = onsets.merge(first_alert, on="stay_id", how="left", validate="one_to_one")
    leads = (
        base.first_onset_time - base.first_alert_time
    ).dt.total_seconds().div(3600).dropna()
    return {
        "detected_events": int(len(leads)),
        "events": int(len(base)),
        "median_lead_time_hours": float(leads.median()) if len(leads) else None,
        "lead_time_q25_hours": float(leads.quantile(0.25)) if len(leads) else None,
        "lead_time_q75_hours": float(leads.quantile(0.75)) if len(leads) else None,
    }


fusion_run = OUTPUT_ROOT / FULL_RUN_NAME
if not (fusion_run / "metrics.json").is_file():
    print("Treinamento completo ainda não concluído; comparação não executada.")
else:
    aligned = {
        split: align_three_models(split, fusion_run)
        for split in ("validation", "test")
    }
    model_columns = {
        "FusionMLP": "probability_fusion",
        "Transformer": "probability_transformer",
        "LightGBM": "probability_lightgbm",
    }

    timestep_rows = []
    for split, frame in aligned.items():
        for model_name, column in model_columns.items():
            timestep_rows.append({
                "split": split,
                "model": model_name,
                **safe_ranking_metrics(frame.target.to_numpy(), frame[column].to_numpy()),
            })
    timestep_table = pd.DataFrame(timestep_rows)

    event_tables = {
        split: shared_event_table(frame, SEED)
        for split, frame in aligned.items()
    }
    for split, frame in event_tables.items():
        frame.to_parquet(
            fusion_run / f"{split}_shared_event_windows.parquet",
            index=False,
            compression="zstd",
        )

    score_columns = {
        "FusionMLP": "score_fusion",
        "Transformer": "score_transformer",
        "LightGBM": "score_lightgbm",
    }
    event_rows = []
    selected_thresholds = {}
    clinical_rows = []
    for model_name, score_column in score_columns.items():
        for split, table in event_tables.items():
            event_rows.append({
                "split": split,
                "model": model_name,
                **safe_ranking_metrics(
                    table.target.to_numpy(), table[score_column].to_numpy()
                ),
            })
        selected, curve = choose_event_thresholds(
            event_tables["validation"].target,
            event_tables["validation"][score_column],
        )
        selected_thresholds[model_name] = selected
        curve.to_csv(
            fusion_run / f"validation_threshold_curve_{model_name.lower()}.csv",
            index=False,
        )
        for operating_point, threshold in selected.items():
            confusion = confusion_metrics(
                event_tables["test"].target,
                event_tables["test"][score_column],
                threshold,
            )
            burden = alert_burden(
                aligned["test"], model_columns[model_name], threshold
            )
            lead = lead_time_summary(
                aligned["test"], model_columns[model_name], threshold
            )
            clinical_rows.append({
                "model": model_name,
                "operating_point": operating_point,
                "threshold_selected_on_validation": threshold,
                **confusion,
                **burden,
                **lead,
            })

    event_table = pd.DataFrame(event_rows)
    clinical_table = pd.DataFrame(clinical_rows)
    timestep_table.to_csv(fusion_run / "comparison_timestep.csv", index=False)
    event_table.to_csv(fusion_run / "comparison_event.csv", index=False)
    clinical_table.to_csv(fusion_run / "comparison_clinical_test.csv", index=False)
    atomic_json_save(
        selected_thresholds, fusion_run / "thresholds_selected_on_validation.json"
    )

    display(timestep_table)
    display(event_table)
    display(clinical_table)

# %% [source cell 18]
from IPython.display import display, Markdown

# Esta célula depende dos objetos criados pela célula 8.
required_objects = [
    "event_tables",
    "selected_thresholds",
    "score_columns",
    "safe_ranking_metrics",
    "confusion_metrics",
]

missing_objects = [
    name for name in required_objects
    if name not in globals()
]

if missing_objects:
    raise RuntimeError(
        "Execute primeiro a célula 8. Objetos ausentes: "
        + ", ".join(missing_objects)
    )

dataset_names = {
    "validation": "Validação interna",
    "test": "Teste interno",
}

model_order = ["FusionMLP", "Transformer", "LightGBM"]

performance_rows = []
confusion_rows = []

for split in ("validation", "test"):
    event_frame = event_tables[split]

    for model_name in model_order:
        score_column = score_columns[model_name]

        # Threshold de maior F1 escolhido na validação
        # e congelado para aplicação no teste.
        threshold = selected_thresholds[model_name][
            "maximum_event_window_f1"
        ]

        ranking = safe_ranking_metrics(
            event_frame["target"].to_numpy(),
            event_frame[score_column].to_numpy(),
        )

        confusion = confusion_metrics(
            event_frame["target"].to_numpy(),
            event_frame[score_column].to_numpy(),
            threshold,
        )

        performance_rows.append({
            "Dataset": dataset_names[split],
            "Modelo": model_name,
            "AUROC": ranking["auroc"],
            "AUPRC": ranking["auprc"],
            "F1": confusion["f1"],
            "Precisão": confusion["precision"],
            "Recall": confusion["recall"],
            "Threshold": threshold,
        })

        confusion_rows.append({
            "Dataset": dataset_names[split],
            "Modelo": model_name,
            "VP (TP)": confusion["tp"],
            "FP": confusion["fp"],
            "FN": confusion["fn"],
            "VN (TN)": confusion["tn"],
            "Total": (
                confusion["tp"]
                + confusion["fp"]
                + confusion["fn"]
                + confusion["tn"]
            ),
        })

performance_summary = pd.DataFrame(performance_rows)
confusion_summary = pd.DataFrame(confusion_rows)

display(Markdown(
    "### Tabela 1 — Comparação de desempenho por evento/stay\n"
    "*F1, precisão e recall calculados no threshold de maior F1 "
    "selecionado na validação e congelado no teste.*"
))

display(
    performance_summary.style
    .format({
        "AUROC": "{:.3f}",
        "AUPRC": "{:.3f}",
        "F1": "{:.3f}",
        "Precisão": "{:.3f}",
        "Recall": "{:.3f}",
        "Threshold": "{:.3f}",
    })
    .hide(axis="index")
)

display(Markdown(
    "### Tabela 2 — Matriz de confusão por evento/stay\n"
    "*VP: evento detectado; FP: falso alerta; FN: evento não "
    "detectado; VN: stay negativo corretamente classificado.*"
))

display(
    confusion_summary.style
    .format({
        "VP (TP)": "{:,}",
        "FP": "{:,}",
        "FN": "{:,}",
        "VN (TN)": "{:,}",
        "Total": "{:,}",
    })
    .hide(axis="index")
)

# %% [source cell 20]
def paired_bootstrap_difference(
    table: pd.DataFrame,
    reference_column: str,
    repetitions: int = 1000,
    seed: int = 42,
):
    subject_ids = table.subject_id.drop_duplicates().to_numpy()
    groups = {subject: group for subject, group in table.groupby("subject_id")}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(repetitions):
        sampled = rng.choice(subject_ids, size=len(subject_ids), replace=True)
        sample = pd.concat([groups[subject] for subject in sampled], ignore_index=True)
        if sample.target.nunique() < 2:
            continue
        fusion_ap = average_precision_score(sample.target, sample.score_fusion)
        reference_ap = average_precision_score(sample.target, sample[reference_column])
        draws.append(fusion_ap - reference_ap)
    values = np.asarray(draws, dtype=float)
    return {
        "median": float(np.median(values)),
        "ci95": [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ],
        "repetitions": int(len(values)),
        "unit": "subject_id",
    }


if RUN_BOOTSTRAP:
    if "event_tables" not in globals():
        raise RuntimeError("Execute primeiro a célula de comparação completa")
    bootstrap_summary = {
        "fusion_minus_transformer_auprc": paired_bootstrap_difference(
            event_tables["test"], "score_transformer"
        ),
        "fusion_minus_lightgbm_auprc": paired_bootstrap_difference(
            event_tables["test"], "score_lightgbm"
        ),
    }
    atomic_json_save(
        bootstrap_summary,
        (OUTPUT_ROOT / FULL_RUN_NAME) / "paired_bootstrap_summary.json",
    )
    print(json.dumps(bootstrap_summary, indent=2))
else:
    print("Bootstrap desativado.")

# %% [source cell 22]
import json
import pandas as pd

fusion_run = OUTPUT_ROOT / FULL_RUN_NAME

config_salva = json.loads(
    (fusion_run / "config.json").read_text()
)

metrics_salvas = json.loads(
    (fusion_run / "metrics.json").read_text()
)

history = pd.read_csv(
    fusion_run / "training_history.csv"
)

print("Diretório:", fusion_run)
print("Quantidade de Parquets disponíveis:", len(FILES))
print("RUN_SMOKE atual:", RUN_SMOKE)
print("RUN_FULL atual:", RUN_FULL)
print("config smoke salva:", config_salva["smoke"])
print("smoke_batches salvo:", config_salva["smoke_batches"])
print("Linhas de treino por época:")
print(history[["epoch", "train_rows", "validation_rows"]])

for split in ("validation", "test"):
    fusion_rows = len(
        pd.read_parquet(
            fusion_run / f"{split}_predictions.parquet",
            columns=["stay_id"]
        )
    )
    transformer_rows = len(
        pd.read_parquet(
            TRANSFORMER_RUN / f"{split}_predictions.parquet",
            columns=["stay_id"]
        )
    )

    print(
        split,
        "| Fusion:", fusion_rows,
        "| Transformer:", transformer_rows
    )
