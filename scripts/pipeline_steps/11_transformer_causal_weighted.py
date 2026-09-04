"""Source exported from the reviewed development notebook.

This file contains code only; patient-level data and notebook outputs are not distributed.
"""

# %% [source cell 1]
from __future__ import annotations

import gc
import hashlib
import json
import math
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

PROJECT = Path("/path/to/circulatory_failure_prediction")
BASE_RUN_DIR = PROJECT / "outputs/step09_lightgbm/weighted_event_primary"
BATCH_DIR = PROJECT / "outputs/step07_aligned_xy_v2/batches"
FEATURE_FILE = PROJECT / "outputs/step07_aligned_xy_v2/feature_names.json"
AUDIT_FILE = PROJECT / "outputs/step10_event_weight_audit/stay_event_weight_audit.parquet"

TARGET = "target_first_onset_0_8h"
EXPECTED_BATCHES = 285
SHORT_HISTORY_FLOOR_ROWS = 12

print("NOTEBOOK 11 — VERSÃO 7 — CHECKPOINT E RETOMADA SEGURA")
print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "nenhuma")

# %% [source cell 2]
@dataclass(frozen=True)
class RunConfig:
    run_name: str
    resume: bool = False
    seed: int = 42
    history_rows: int = 96
    negative_ratio: float = 5.0
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 3
    dim_feedforward: int = 256
    dropout: float = 0.10
    batch_size: int = 128
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    max_epochs: int = 15
    early_stopping_patience: int = 4
    num_workers: int = 2
    grad_clip: float = 1.0
    input_clip_z: float = 10.0
    smoke: bool = False
    smoke_batches: int = 4
    smoke_stays_per_worker: int = 100
    scaler_rows_per_batch: int = 50_000


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def canonical_batches(directory: Path) -> list[Path]:
    files = sorted(p for p in directory.glob("*.parquet") if re.fullmatch(r"batch_\d{4}\.parquet", p.name))
    expected = [f"batch_{i:04d}.parquet" for i in range(EXPECTED_BATCHES)]
    if [p.name for p in files] != expected:
        raise RuntimeError(f"Esperava batch_0000--batch_0284; encontrei {len(files)} em {directory}")
    return files


def load_features() -> list[str]:
    registry = json.loads(FEATURE_FILE.read_text())
    features = list(registry["feature_names"])
    if len(features) != int(registry["feature_count"]) or len(features) != len(set(features)):
        raise RuntimeError("Registro de features inválido")
    if len(features) != 204:
        raise RuntimeError(f"Esperava 204 features; encontrei {len(features)}")
    required = {
        "lab_50813", "lab_50813_imputed", "lab_50813_time_since_last_measurement_min",
        "sbp", "dbp", "mbp", "sbp_imputed", "dbp_imputed", "mbp_imputed",
    }
    missing = sorted(required - set(features))
    if missing:
        raise RuntimeError(f"Lactato/pressão ausentes do modelo principal: {missing}")
    vaso_pattern = re.compile(
        r"vasoactive|vasopressor|norepi|epineph|phenyleph|dopamine|dobutamine|milrinone|vasopressin",
        re.I,
    )
    forbidden = [name for name in features if vaso_pattern.search(name)]
    if forbidden:
        raise RuntimeError(f"Features vasoativas proibidas encontradas: {forbidden}")
    return features


def load_tables():
    split_file = BASE_RUN_DIR / "stay_subject_split.parquet"
    if not split_file.exists():
        raise FileNotFoundError(
            f"Split do LightGBM não encontrado: {split_file}. Ajuste BASE_RUN_DIR para a execução primária do notebook 10."
        )
    split = pd.read_parquet(split_file, columns=["stay_id", "subject_id", "split", "stay_positive"])
    if split.groupby("subject_id")["split"].nunique().max() != 1:
        raise RuntimeError("O mesmo paciente aparece em mais de um split")
    counts = pd.read_parquet(AUDIT_FILE, columns=["stay_id", "positive_rows", "negative_rows"])
    if counts.stay_id.duplicated().any():
        raise RuntimeError("Contagens duplicadas por stay")
    return split, counts


def stable_keep_negative(frame: pd.DataFrame, fraction: float, seed: int) -> np.ndarray:
    if fraction >= 1.0:
        return np.ones(len(frame), dtype=bool)
    hashes = pd.util.hash_pandas_object(frame[["stay_id", "label_time"]], index=False).to_numpy(dtype="uint64")
    hashes ^= np.uint64(seed)
    threshold = np.uint64(fraction * np.iinfo(np.uint64).max)
    return hashes <= threshold


def training_negative_fraction(split: pd.DataFrame, counts: pd.DataFrame, ratio: float) -> float:
    train_ids = split.loc[split.split.eq("train"), ["stay_id"]]
    selected = train_ids.merge(counts, on="stay_id", how="left", validate="one_to_one")
    if selected[["positive_rows", "negative_rows"]].isna().any().any():
        raise RuntimeError("Stay de treino sem contagens")
    pos = int(selected.positive_rows.sum())
    neg = int(selected.negative_rows.sum())
    if pos == 0 or neg == 0:
        raise RuntimeError("Treino sem ambas as classes")
    return min(1.0, ratio * pos / neg)


def fit_train_scaler(files, split, features, rows_per_batch, seed):
    train_ids = set(split.loc[split.split.eq("train"), "stay_id"])
    count = np.zeros(len(features), dtype=np.int64)
    total = np.zeros(len(features), dtype=np.float64)
    total_sq = np.zeros(len(features), dtype=np.float64)
    rng = np.random.default_rng(seed)
    for index, path in enumerate(files, 1):
        frame = pd.read_parquet(path, columns=["stay_id", *features])
        frame = frame.loc[frame.stay_id.isin(train_ids), features]
        if len(frame) > rows_per_batch:
            take = rng.choice(len(frame), rows_per_batch, replace=False)
            frame = frame.iloc[take]
        values = frame.replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float64)
        finite = np.isfinite(values)
        count += finite.sum(axis=0)
        clean = np.where(finite, values, 0.0)
        total += clean.sum(axis=0)
        total_sq += np.square(clean).sum(axis=0)
        if index == 1 or index % 25 == 0 or index == len(files):
            print(f"scaler: {index}/{len(files)} batches", flush=True)
    all_missing = count == 0
    if np.any(all_missing):
        missing = [features[i] for i in np.flatnonzero(all_missing)]
        print(f"AVISO: {len(missing)} features sem valor finito neste recorte; serão neutralizadas: {missing}", flush=True)
    safe_count = np.where(all_missing, 1, count)
    mean = total / safe_count
    variance = np.maximum(total_sq / safe_count - np.square(mean), 1e-12)
    scale = np.sqrt(variance)
    scale[scale < 1e-6] = 1.0
    mean[all_missing] = 0.0
    scale[all_missing] = 1.0
    return mean.astype("float32"), scale.astype("float32"), count


fit_train_scaler.notebook_version = 7
FEATURES = load_features()
FILES = canonical_batches(BATCH_DIR)
SPLIT_TABLE, COUNTS = load_tables()
print("Features:", len(FEATURES))
print("Split por paciente:", SPLIT_TABLE.groupby("split").stay_id.nunique().to_dict())

# %% [source cell 3]
class WindowStream(IterableDataset):
    def __init__(self, files, split_table, counts, features, mean, scale, split_name, config, epoch=0):
        self.files = list(files)
        self.allowed = split_table.loc[split_table.split.eq(split_name), ["stay_id", "subject_id"]].copy()
        self.counts = counts.set_index("stay_id")[["positive_rows", "negative_rows"]]
        self.features = list(features)
        self.mean = mean
        self.scale = scale
        self.split_name = split_name
        self.config = config
        self.epoch = epoch
        self.neg_fraction = training_negative_fraction(split_table, counts, config.negative_ratio) if split_name == "train" else 1.0

    def __iter__(self):
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        worker_count = 1 if worker is None else worker.num_workers
        files = self.files[worker_id::worker_count]
        rng = np.random.default_rng(self.config.seed + 10_000 * self.epoch + worker_id)
        yielded_stays = 0
        allowed = self.allowed
        for path in files:
            frame = pd.read_parquet(path, columns=["stay_id", "label_time", TARGET, *self.features])
            frame = frame.merge(allowed, on="stay_id", how="inner", validate="many_to_one")
            if frame.empty:
                continue
            frame["label_time"] = pd.to_datetime(frame["label_time"], errors="raise")
            stay_ids = frame.stay_id.drop_duplicates().to_numpy()
            if self.split_name == "train":
                rng.shuffle(stay_ids)
            for stay_id in stay_ids:
                group = frame.loc[frame.stay_id.eq(stay_id)].sort_values("label_time").reset_index(drop=True)
                y = group[TARGET].to_numpy(dtype="int8")
                if self.split_name == "train" and self.neg_fraction < 1.0:
                    keep_neg = stable_keep_negative(group, self.neg_fraction, self.config.seed)
                    endpoint_indices = np.flatnonzero((y == 1) | keep_neg)
                else:
                    endpoint_indices = np.arange(len(group))
                if self.split_name == "train":
                    rng.shuffle(endpoint_indices)
                x_all = group[self.features].replace([np.inf, -np.inf], np.nan).to_numpy(dtype="float32")
                x_all = (x_all - self.mean) / self.scale
                x_all = np.nan_to_num(x_all, nan=0.0, posinf=0.0, neginf=0.0)
                # LightGBM tolera extremos diretamente; a atenção neural não. O clipping
                # após normalização evita overflow sem usar informação de validação/teste.
                x_all = np.clip(x_all, -self.config.input_clip_z, self.config.input_clip_z)
                row_counts = self.counts.loc[stay_id]
                for endpoint in endpoint_indices:
                    start = max(0, endpoint - self.config.history_rows + 1)
                    valid = x_all[start:endpoint + 1]
                    length = len(valid)
                    sequence = np.zeros((self.config.history_rows, len(self.features)), dtype="float32")
                    padding = np.ones(self.config.history_rows, dtype=bool)
                    # Padding à direita: todo timestep clínico válido consegue observar
                    # pelo menos a si próprio sob a máscara causal. Padding à esquerda
                    # criava queries totalmente mascaradas e propagava NaN na atenção.
                    sequence[:length] = valid
                    padding[:length] = False
                    target = int(y[endpoint])
                    if target == 1:
                        weight = 1.0 / max(float(row_counts.positive_rows), SHORT_HISTORY_FLOOR_ROWS)
                    else:
                        weight = 1.0 / max(float(row_counts.negative_rows), 1.0)
                        if self.split_name == "train" and self.neg_fraction < 1.0:
                            weight /= self.neg_fraction
                    yield (
                        torch.from_numpy(sequence), torch.from_numpy(padding),
                        torch.tensor(target, dtype=torch.float32), torch.tensor(weight, dtype=torch.float32),
                        int(stay_id), int(group.subject_id.iloc[0]), group.label_time.iloc[endpoint].value,
                    )
                yielded_stays += 1
                if self.config.smoke and yielded_stays >= self.config.smoke_stays_per_worker:
                    return


class CausalTransformer(nn.Module):
    def __init__(self, n_features, config):
        super().__init__()
        self.input_projection = nn.Linear(n_features, config.d_model)
        self.position = nn.Parameter(torch.zeros(1, config.history_rows, config.d_model))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model, nhead=config.nhead,
            dim_feedforward=config.dim_feedforward, dropout=config.dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=config.num_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, 1)

    def forward(self, x, padding_mask):
        length = x.shape[1]
        valid_lengths = (~padding_mask).sum(dim=1).clamp_min(1)
        # As posições válidas são alinhadas ao fim da janela posicional: o timestep
        # atual recebe sempre a posição history_rows-1, inclusive em históricos curtos.
        offsets = length - valid_lengths
        position_ids = torch.arange(length, device=x.device).unsqueeze(0) + offsets.unsqueeze(1)
        position_ids = position_ids.clamp_max(length - 1)
        hidden = self.input_projection(x) + self.position[0, position_ids]
        causal_mask = torch.triu(torch.ones(length, length, device=x.device, dtype=torch.bool), diagonal=1)
        # Com padding à direita, os tokens clínicos formam um prefixo válido. A máscara
        # causal já impede esse prefixo de observar o padding futuro; uma padding mask
        # adicional é desnecessária e pode criar linhas totalmente mascaradas.
        hidden = self.encoder(hidden, mask=causal_mask)
        last_valid = valid_lengths - 1
        endpoint_hidden = hidden[torch.arange(len(hidden), device=x.device), last_valid]
        logits = self.head(self.norm(endpoint_hidden)).squeeze(-1)
        if not torch.isfinite(logits).all():
            input_max = float(x.detach().abs().max().cpu())
            raise RuntimeError(f"Logits não finitos produzidos pelo Transformer; max_abs_input={input_max}")
        return logits


def make_loader(files, split_name, config, mean, scale, epoch=0):
    dataset = WindowStream(files, SPLIT_TABLE, COUNTS, FEATURES, mean, scale, split_name, config, epoch)
    return DataLoader(
        dataset, batch_size=config.batch_size, num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(), drop_last=False,
    )


def safe_ranking_metrics(y, p):
    result = {"rows": int(len(y)), "positives": int(y.sum()), "prevalence": float(y.mean())}
    if np.unique(y).size == 2:
        result.update(
            auroc=float(roc_auc_score(y, p)), auprc=float(average_precision_score(y, p)),
            log_loss=float(log_loss(y, p, labels=[0, 1])), brier=float(brier_score_loss(y, p)),
        )
    else:
        result.update(auroc=None, auprc=None, log_loss=None, brier=None)
    return result


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, ps, stays, subjects, times = [], [], [], [], []
    for x, padding, y, _, stay, subject, time_ns in loader:
        logits = model(x.to(device, non_blocking=True), padding.to(device, non_blocking=True))
        ys.append(y.numpy().astype("int8")); ps.append(torch.sigmoid(logits).cpu().numpy())
        stays.append(stay.numpy()); subjects.append(subject.numpy()); times.append(time_ns.numpy())
    if not ys:
        raise RuntimeError("Nenhuma observação produzida")
    y = np.concatenate(ys); p = np.concatenate(ps)
    meta = pd.DataFrame({
        "stay_id": np.concatenate(stays), "subject_id": np.concatenate(subjects),
        "label_time": pd.to_datetime(np.concatenate(times)), "target": y, "probability": p,
    })
    return safe_ranking_metrics(y, p), meta

# %% [source cell 4]
def event_window_table(pred: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    pred = pred.copy()
    truth = pred.groupby("stay_id").target.max()
    positive_ids = set(truth[truth.eq(1)].index)
    negative_ids = set(truth[truth.eq(0)].index)
    positives = (
        pred.loc[pred.target.eq(1)]
        .groupby(["stay_id", "subject_id"], as_index=False)
        .agg(score=("probability", "max"), window_rows=("probability", "size"),
             window_start=("label_time", "min"), window_end=("label_time", "max"))
    )
    positives["target"] = 1
    if not positive_ids or set(positives.stay_id) != positive_ids:
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
            "stay_id": stay_id, "subject_id": subject_id, "score": float(window.probability.max()),
            "window_rows": int(length), "window_start": window.label_time.min(),
            "window_end": window.label_time.max(), "target": 0,
        })
    table = pd.concat([positives, pd.DataFrame(controls)], ignore_index=True, sort=False)
    if table.stay_id.duplicated().any():
        raise RuntimeError("Mais de uma janela por stay")
    return table


def threshold_curve(table: pd.DataFrame) -> pd.DataFrame:
    y = table.target.to_numpy(dtype=bool)
    score = table.score.to_numpy(dtype=float)
    thresholds = np.unique(np.r_[0.0, 1.0, np.quantile(score, np.linspace(0, 1, 2001))])
    rows = []
    for threshold in thresholds:
        alert = score >= threshold
        tp = int(np.sum(alert & y)); fp = int(np.sum(alert & ~y))
        fn = int(np.sum(~alert & y)); tn = int(np.sum(~alert & ~y))
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(dict(threshold=float(threshold), tp_events=tp, fp_control_windows=fp,
                         fn_events=fn, tn_control_windows=tn, precision=precision,
                         recall=recall, specificity=specificity, f1=f1))
    return pd.DataFrame(rows)


def event_metrics_at(table, threshold):
    row = threshold_curve(table.loc[:, ["target", "score"]])
    y = table.target.to_numpy(dtype=bool); alert = table.score.to_numpy() >= threshold
    tp = int(np.sum(alert & y)); fp = int(np.sum(alert & ~y))
    fn = int(np.sum(~alert & y)); tn = int(np.sum(~alert & ~y))
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return dict(tp_events=tp, fp_control_windows=fp, fn_events=fn, tn_control_windows=tn,
                precision=precision, recall=recall, specificity=specificity, f1=f1)


def create_run_directory(base: Path, smoke: bool, resume: bool) -> Path:
    if resume:
        if not base.is_dir():
            raise FileNotFoundError(f"Diretório para retomada não encontrado: {base}")
        if (base / "metrics.json").exists():
            raise RuntimeError(f"A execução já foi concluída: {base}")
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


def atomic_torch_save(payload: dict, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def atomic_history_save(history: list[dict], destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    pd.DataFrame(history).to_csv(temporary, index=False)
    temporary.replace(destination)


def atomic_json_save(payload: dict, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(destination)


def train_run(config: RunConfig):
    seed_everything(config.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA não disponível; este treinamento foi preparado para a RTX 3090 Ti")
    files = FILES[:config.smoke_batches] if config.smoke else FILES
    run_dir_base = PROJECT / "outputs/step11_transformer" / config.run_name
    if not config.resume and not config.smoke and run_dir_base.exists():
        raise FileExistsError(f"Execução já existe; use a célula de retomada: {run_dir_base}")
    if config.resume:
        run_dir = create_run_directory(run_dir_base, config.smoke, resume=True)
        scaler_path = run_dir / "train_scaler.npz"
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler ausente para retomada: {scaler_path}")
        saved_scaler = np.load(scaler_path, allow_pickle=True)
        mean = saved_scaler["mean"].astype("float32")
        scale = saved_scaler["scale"].astype("float32")
        scaler_count = saved_scaler["count"]
        if list(saved_scaler["features"]) != FEATURES:
            raise RuntimeError("Features do scaler diferem do registro atual")
    else:
        mean, scale, scaler_count = fit_train_scaler(
            files, SPLIT_TABLE, FEATURES, config.scaler_rows_per_batch, config.seed
        )
        run_dir = create_run_directory(run_dir_base, config.smoke, resume=False)
        scaler_path = run_dir / "train_scaler.npz"
        atomic_json_save(asdict(config), run_dir / "config.json")
        np.savez(scaler_path, mean=mean, scale=scale, count=scaler_count,
                 features=np.asarray(FEATURES, dtype=object))
    print(f"Saída desta execução: {run_dir}", flush=True)

    device = torch.device("cuda")
    model = CausalTransformer(len(FEATURES), config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    best_ap = -math.inf
    best_epoch = 0
    patience = 0
    history = []
    start_epoch = 1
    latest_path = run_dir / "latest_checkpoint.pt"
    if config.resume:
        if not latest_path.exists():
            raise FileNotFoundError(
                f"Checkpoint completo ausente: {latest_path}. A tentativa v6 é piloto e não pode ser retomada exatamente."
            )
        state = torch.load(latest_path, map_location="cpu", weights_only=False)
        saved_config = dict(state["config"]); current_config = asdict(config)
        saved_config.pop("resume", None); current_config.pop("resume", None)
        if saved_config != current_config or state["features"] != FEATURES:
            raise RuntimeError("Configuração/features da retomada não coincidem com a execução original")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        random.setstate(state["rng_state"]["python"])
        np.random.set_state(state["rng_state"]["numpy"])
        torch.set_rng_state(state["rng_state"]["torch_cpu"])
        torch.cuda.set_rng_state_all(state["rng_state"]["torch_cuda"])
        best_ap = float(state["best_ap"]); best_epoch = int(state["best_epoch"])
        patience = int(state["patience"]); history = list(state["history"])
        start_epoch = int(state["epoch"]) + 1
        print(f"Retomando após a época {state['epoch']}; próxima época: {start_epoch}", flush=True)
        if patience >= config.early_stopping_patience:
            start_epoch = config.max_epochs + 1
            print("Early stopping já havia sido atingido; retomando diretamente na avaliação final.", flush=True)

    for epoch in range(start_epoch, config.max_epochs + 1):
        model.train()
        weighted_loss_sum = 0.0; weight_sum = 0.0; train_rows = 0
        loader = make_loader(files, "train", config, mean, scale, epoch)
        for x, padding, y, weight, *_ in loader:
            x = x.to(device, non_blocking=True); padding = padding.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True); weight = weight.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if not torch.isfinite(x).all():
                raise RuntimeError("Input não finito após normalização e clipping")
            logits = model(x, padding)
            individual = loss_fn(logits, y)
            loss = (individual * weight).sum() / weight.sum()
            if not torch.isfinite(loss):
                raise RuntimeError("Loss não finita antes do backward")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            if not torch.isfinite(grad_norm):
                raise RuntimeError("Gradientes não finitos antes do optimizer.step")
            optimizer.step()
            weighted_loss_sum += float((individual.detach() * weight).sum().cpu())
            weight_sum += float(weight.sum().cpu()); train_rows += len(y)

        val_loader = make_loader(files, "validation", config, mean, scale, epoch=0)
        val_metrics, _ = predict(model, val_loader, device)
        row = {"epoch": epoch, "train_weighted_loss": weighted_loss_sum / weight_sum,
               "train_rows": train_rows, **{f"validation_{k}": v for k, v in val_metrics.items()}}
        history.append(row); print(json.dumps(row), flush=True)
        current_ap = val_metrics.get("auprc")
        if current_ap is None:
            current_ap = -row["train_weighted_loss"]
        if current_ap > best_ap + 1e-5:
            best_ap = current_ap; best_epoch = epoch; patience = 0
            atomic_torch_save({
                "model_state": model.state_dict(), "config": asdict(config),
                "features": FEATURES, "best_epoch": best_epoch, "best_ap": best_ap,
            }, run_dir / "best_model.pt")
        else:
            patience += 1
        atomic_torch_save({
            "epoch": epoch, "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(), "config": asdict(config),
            "features": FEATURES, "best_ap": best_ap, "best_epoch": best_epoch,
            "patience": patience, "history": history,
            "rng_state": {"python": random.getstate(), "numpy": np.random.get_state(),
                          "torch_cpu": torch.get_rng_state(),
                          "torch_cuda": torch.cuda.get_rng_state_all()},
        }, latest_path)
        atomic_history_save(history, run_dir / "training_history.csv")
        print(f"CHECKPOINT_COMPLETO_SALVO época={epoch} melhor_época={best_epoch}", flush=True)
        if patience >= config.early_stopping_patience:
            print(f"Early stopping após a época {epoch}", flush=True)
            break
        gc.collect(); torch.cuda.empty_cache()

    checkpoint = torch.load(run_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])

    predictions = {}
    timestep_metrics = {}
    for split_name in ("validation", "test"):
        loader = make_loader(files, split_name, config, mean, scale, epoch=0)
        timestep_metrics[split_name], predictions[split_name] = predict(model, loader, device)
        predictions[split_name].to_parquet(
            run_dir / f"{split_name}_predictions.parquet", index=False, compression="zstd"
        )

    event_result = None
    if all(predictions[s].target.nunique() == 2 for s in ("validation", "test")):
        val_windows = event_window_table(predictions["validation"], config.seed)
        test_windows = event_window_table(predictions["test"], config.seed)
        curve = threshold_curve(val_windows)
        best = curve.sort_values(["f1", "threshold"], ascending=[False, False]).iloc[0]
        threshold = float(best.threshold)
        curve.to_csv(run_dir / "validation_event_threshold_curve.csv", index=False)
        val_windows.to_parquet(run_dir / "validation_event_windows.parquet", index=False)
        test_windows.to_parquet(run_dir / "test_event_windows.parquet", index=False)
        event_result = {
            "threshold_selected_on_validation": threshold,
            "validation_ranking": safe_ranking_metrics(val_windows.target.to_numpy(), val_windows.score.to_numpy()),
            "test_ranking": safe_ranking_metrics(test_windows.target.to_numpy(), test_windows.score.to_numpy()),
            "validation_at_frozen_threshold": event_metrics_at(val_windows, threshold),
            "test_at_frozen_threshold": event_metrics_at(test_windows, threshold),
        }

    result = {
        "status": "completed", "smoke": config.smoke, "run_name": run_dir.name,
        "output_directory": str(run_dir),
        "feature_count": len(FEATURES),
        "feature_sha256": hashlib.sha256("\n".join(FEATURES).encode()).hexdigest(),
        "lactate_and_pressure_retained": True, "vasoactives_excluded": True,
        "history_rows": config.history_rows, "history_hours": config.history_rows * 5 / 60,
        "causal_attention_mask": True, "post_onset_rows": "excluded upstream in Step 7",
        "split_reused_from": str(BASE_RUN_DIR / "stay_subject_split.parquet"),
        "all_positive_training_endpoints_retained": True,
        "positive_weight_rule": "1/max(positive_rows_in_stay,12)",
        "negative_weight_rule": "1/negative_rows_in_stay; inverse sampling correction in train",
        "numeric_precision": "float32; autocast disabled",
        "normalized_input_clip_z": config.input_clip_z,
        "best_epoch": best_epoch, "timestep_metrics": timestep_metrics,
        "event_evaluation": event_result,
    }
    atomic_json_save(result, run_dir / "metrics.json")
    print(json.dumps(result, indent=2))
    return result

# %% [source cell 6]
RUN_SMOKE_V7 = False  # o smoke V7 já foi aprovado; mantenha False para retomar o treino completo

if RUN_SMOKE_V7:
    assert getattr(fit_train_scaler, "notebook_version", None) == 7, (
        "Definições da versão 7 ausentes. Reinicie o kernel e execute as células 1 a 4."
    )
    assert "train_run" in globals(), (
        "A célula que define train_run não foi executada. Execute as células 1 a 4 primeiro."
    )
    SMOKE_CONFIG = RunConfig(
        run_name="transformer_primary_seed42_smoke_v7",
        seed=42,
        smoke=True,
        max_epochs=2,
        early_stopping_patience=2,
        num_workers=0,
    )
    smoke_result = train_run(SMOKE_CONFIG)
    assert smoke_result["status"] == "completed"
else:
    print("Smoke V7 não executado: ele já foi aprovado. Use a seção de treinamento robusto para retomar.")

# %% [source cell 8]
from pathlib import Path
import base64
import shlex
import subprocess
import time

INICIAR_OU_RETOMAR_V7 = True  # altere para True e execute esta célula uma única vez

PROJECT_BG = Path("/path/to/circulatory_failure_prediction")
PYTHON_BG = Path("/path/to/venv/bin/python")
NOTEBOOK_BG = PROJECT_BG / "notebooks/11_transformer_causal_weighted.ipynb"
RUNNER_BG = PROJECT_BG / "resume_transformer_v7.py"
RUN_BG = PROJECT_BG / "outputs/step11_transformer/transformer_primary_seed42_v7"
CHECKPOINT_BG = RUN_BG / "latest_checkpoint.pt"
LOG_BG = RUN_BG / "resume_console.log"
SESSION_BG = "transformer_v7"
RUNNER_B64_BG = "IyEvdXNyL2Jpbi9lbnYgcHl0aG9uMwoiIiJSZXRvbWEgZXhjbHVzaXZhbWVudGUgbyBUcmFuc2Zvcm1lciB2NyBhIHBhcnRpciBkbyBjaGVja3BvaW50IGNvbXBsZXRvLiIiIgoKZnJvbSBfX2Z1dHVyZV9fIGltcG9ydCBhbm5vdGF0aW9ucwoKaW1wb3J0IGFyZ3BhcnNlCmltcG9ydCBqc29uCmltcG9ydCBzeXMKaW1wb3J0IHR5cGVzCmZyb20gcGF0aGxpYiBpbXBvcnQgUGF0aAoKClBST0pFQ1QgPSBQYXRoKCIvc2NyYXRjaDIvbHVpc2FiYXJyb3MvRkFMRU5DSUEvY2lyY3VsYXRvcnlfZmFpbHVyZV9wcmVkaWN0aW9uIikKUlVOX05BTUUgPSAidHJhbnNmb3JtZXJfcHJpbWFyeV9zZWVkNDJfdjciClJVTl9ESVIgPSBQUk9KRUNUIC8gIm91dHB1dHMvc3RlcDExX3RyYW5zZm9ybWVyIiAvIFJVTl9OQU1FCgoKZGVmIGxvYWRfbm90ZWJvb2tfZGVmaW5pdGlvbnMobm90ZWJvb2tfcGF0aDogUGF0aCkgLT4gZGljdDoKICAgIG5vdGVib29rID0ganNvbi5sb2Fkcyhub3RlYm9va19wYXRoLnJlYWRfdGV4dChlbmNvZGluZz0idXRmLTgiKSkKICAgIG1vZHVsZV9uYW1lID0gInRyYW5zZm9ybWVyX3Y3X2RlZmluaXRpb25zIgogICAgbW9kdWxlID0gdHlwZXMuTW9kdWxlVHlwZShtb2R1bGVfbmFtZSkKICAgIG1vZHVsZS5fX2ZpbGVfXyA9IHN0cihub3RlYm9va19wYXRoKQogICAgc3lzLm1vZHVsZXNbbW9kdWxlX25hbWVdID0gbW9kdWxlCiAgICBuYW1lc3BhY2UgPSBtb2R1bGUuX19kaWN0X18KICAgIHJlYWNoZWRfc21va2UgPSBGYWxzZQoKICAgIGZvciBpbmRleCwgY2VsbCBpbiBlbnVtZXJhdGUobm90ZWJvb2suZ2V0KCJjZWxscyIsIFtdKSk6CiAgICAgICAgc291cmNlID0gIiIuam9pbihjZWxsLmdldCgic291cmNlIiwgW10pKQogICAgICAgIGlmIGNlbGwuZ2V0KCJjZWxsX3R5cGUiKSA9PSAibWFya2Rvd24iIGFuZCBzb3VyY2UubHN0cmlwKCkuc3RhcnRzd2l0aCgiIyMgU21va2UgdGVzdCIpOgogICAgICAgICAgICByZWFjaGVkX3Ntb2tlID0gVHJ1ZQogICAgICAgICAgICBicmVhawogICAgICAgIGlmIGNlbGwuZ2V0KCJjZWxsX3R5cGUiKSA9PSAiY29kZSI6CiAgICAgICAgICAgIGV4ZWMoY29tcGlsZShzb3VyY2UsIGYie25vdGVib29rX3BhdGh9OmNlbGxfe2luZGV4fSIsICJleGVjIiksIG5hbWVzcGFjZSkKCiAgICBpZiBub3QgcmVhY2hlZF9zbW9rZToKICAgICAgICByYWlzZSBSdW50aW1lRXJyb3IoIlNlw6fDo28gU21va2UgdGVzdCBuw6NvIGVuY29udHJhZGE7IG5vdGVib29rIGluY29tcGF0w612ZWwiKQogICAgc2NhbGVyX2Z1bmN0aW9uID0gbmFtZXNwYWNlLmdldCgiZml0X3RyYWluX3NjYWxlciIpCiAgICB2ZXJzaW9uID0gZ2V0YXR0cihzY2FsZXJfZnVuY3Rpb24sICJub3RlYm9va192ZXJzaW9uIiwgTm9uZSkKICAgIGlmIHZlcnNpb24gIT0gNzoKICAgICAgICByYWlzZSBSdW50aW1lRXJyb3IoZiJFc3BlcmF2YSBub3RlYm9vayB2ZXJzw6NvIDc7IGVuY29udHJlaSB2ZXJzw6NvIHt2ZXJzaW9ufSIpCiAgICBpZiAiUnVuQ29uZmlnIiBub3QgaW4gbmFtZXNwYWNlIG9yICJ0cmFpbl9ydW4iIG5vdCBpbiBuYW1lc3BhY2U6CiAgICAgICAgcmFpc2UgUnVudGltZUVycm9yKCJEZWZpbmnDp8O1ZXMgUnVuQ29uZmlnL3RyYWluX3J1biBhdXNlbnRlcyIpCiAgICByZXR1cm4gbmFtZXNwYWNlCgoKZGVmIG1haW4oKSAtPiBOb25lOgogICAgcGFyc2VyID0gYXJncGFyc2UuQXJndW1lbnRQYXJzZXIoKQogICAgcGFyc2VyLmFkZF9hcmd1bWVudCgKICAgICAgICAiLS1ub3RlYm9vayIsCiAgICAgICAgdHlwZT1QYXRoLAogICAgICAgIHJlcXVpcmVkPVRydWUsCiAgICAgICAgaGVscD0iQ2FtaW5obyBkbyBub3RlYm9vayAxMSB2ZXJzw6NvIDcgbmEgVk0iLAogICAgKQogICAgYXJncyA9IHBhcnNlci5wYXJzZV9hcmdzKCkKICAgIG5vdGVib29rX3BhdGggPSBhcmdzLm5vdGVib29rLmV4cGFuZHVzZXIoKS5yZXNvbHZlKCkKICAgIGlmIG5vdCBub3RlYm9va19wYXRoLmlzX2ZpbGUoKToKICAgICAgICByYWlzZSBGaWxlTm90Rm91bmRFcnJvcihub3RlYm9va19wYXRoKQoKICAgIGxhdGVzdCA9IFJVTl9ESVIgLyAibGF0ZXN0X2NoZWNrcG9pbnQucHQiCiAgICBpZiBub3QgbGF0ZXN0LmlzX2ZpbGUoKToKICAgICAgICByYWlzZSBGaWxlTm90Rm91bmRFcnJvcihmIkNoZWNrcG9pbnQgdjcgbsOjbyBlbmNvbnRyYWRvOiB7bGF0ZXN0fSIpCiAgICBpZiAoUlVOX0RJUiAvICJtZXRyaWNzLmpzb24iKS5leGlzdHMoKToKICAgICAgICByYWlzZSBSdW50aW1lRXJyb3IoZiJBIGV4ZWN1w6fDo28gdjcgasOhIGVzdMOhIGNvbmNsdcOtZGE6IHtSVU5fRElSIC8gJ21ldHJpY3MuanNvbid9IikKCiAgICBuYW1lc3BhY2UgPSBsb2FkX25vdGVib29rX2RlZmluaXRpb25zKG5vdGVib29rX3BhdGgpCiAgICB0b3JjaCA9IG5hbWVzcGFjZVsidG9yY2giXQogICAgY2hlY2twb2ludCA9IHRvcmNoLmxvYWQobGF0ZXN0LCBtYXBfbG9jYXRpb249ImNwdSIsIHdlaWdodHNfb25seT1GYWxzZSkKICAgIGVwb2NoID0gaW50KGNoZWNrcG9pbnRbImVwb2NoIl0pCiAgICByZWNvcmRlZCA9IFtpbnQocm93WyJlcG9jaCJdKSBmb3Igcm93IGluIGNoZWNrcG9pbnRbImhpc3RvcnkiXV0KICAgIGlmIHJlY29yZGVkICE9IGxpc3QocmFuZ2UoMSwgZXBvY2ggKyAxKSk6CiAgICAgICAgcmFpc2UgUnVudGltZUVycm9yKGYiSGlzdMOzcmljbyBkZXNjb250w61udW8gbm8gY2hlY2twb2ludDoge3JlY29yZGVkfSIpCgogICAgcHJpbnQoZiJDSEVDS1BPSU5UIFY3IENPTkZJUk1BRE86IMOpcG9jYT17ZXBvY2h9OyBoaXN0w7NyaWNvPXtyZWNvcmRlZH0iLCBmbHVzaD1UcnVlKQogICAgcHJpbnQoZiJSRVRPTUFEQSBFWENMVVNJVkE6IHByw7N4aW1hIMOpcG9jYT17ZXBvY2ggKyAxfTsgcGFzdGE9e1JVTl9ESVJ9IiwgZmx1c2g9VHJ1ZSkKCiAgICBSdW5Db25maWcgPSBuYW1lc3BhY2VbIlJ1bkNvbmZpZyJdCiAgICBjb25maWcgPSBSdW5Db25maWcoCiAgICAgICAgcnVuX25hbWU9UlVOX05BTUUsCiAgICAgICAgcmVzdW1lPVRydWUsCiAgICAgICAgc2VlZD00MiwKICAgICAgICBzbW9rZT1GYWxzZSwKICAgICAgICBtYXhfZXBvY2hzPTE1LAogICAgICAgIGVhcmx5X3N0b3BwaW5nX3BhdGllbmNlPTQsCiAgICAgICAgYmF0Y2hfc2l6ZT0xMjgsCiAgICAgICAgbnVtX3dvcmtlcnM9MiwKICAgICkKICAgIHJlc3VsdCA9IG5hbWVzcGFjZVsidHJhaW5fcnVuIl0oY29uZmlnKQogICAgaWYgcmVzdWx0LmdldCgic3RhdHVzIikgIT0gImNvbXBsZXRlZCI6CiAgICAgICAgcmFpc2UgUnVudGltZUVycm9yKGYiRXhlY3XDp8OjbyB0ZXJtaW5vdSBzZW0gc3RhdHVzIGNvbXBsZXRlZDoge3Jlc3VsdC5nZXQoJ3N0YXR1cycpfSIpCgoKaWYgX19uYW1lX18gPT0gIl9fbWFpbl9fIjoKICAgIG1haW4oKQo="

if INICIAR_OU_RETOMAR_V7:
    runner_bytes = base64.b64decode(RUNNER_B64_BG)
    runner_tmp = RUNNER_BG.with_suffix(".py.tmp")
    runner_tmp.write_bytes(runner_bytes)
    runner_tmp.replace(RUNNER_BG)
    print(f"Executor auxiliar preparado automaticamente: {RUNNER_BG}")

    for required in (PYTHON_BG, NOTEBOOK_BG, CHECKPOINT_BG):
        if not required.exists():
            raise FileNotFoundError(f"Arquivo obrigatório não encontrado: {required}")

    process_table = subprocess.run(
        ["ps", "-eo", "pid=,args="], capture_output=True, text=True, check=True
    ).stdout.splitlines()
    runner_processes = [
        line.strip() for line in process_table
        if str(RUNNER_BG) in line and "--notebook" in line
    ]
    session_exists = subprocess.run(
        ["tmux", "has-session", "-t", SESSION_BG],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0
    if runner_processes:
        print("O processo real do treinamento V7 já está ativo:")
        print("\n".join(runner_processes))
    else:
        if session_exists:
            subprocess.run(["tmux", "kill-session", "-t", SESSION_BG], check=True)
            print(f"Sessão tmux vazia removida: {SESSION_BG}")
        command = [str(PYTHON_BG), str(RUNNER_BG), "--notebook", str(NOTEBOOK_BG)]
        shell_command = (
            f"cd {shlex.quote(str(PROJECT_BG))} && "
            f"exec {shlex.join(command)} >> {shlex.quote(str(LOG_BG))} 2>&1"
        )
        subprocess.run(
            ["tmux", "new-session", "-d", "-s", SESSION_BG, shell_command],
            check=True,
        )
        time.sleep(3)
        process_table = subprocess.run(
            ["ps", "-eo", "pid=,args="], capture_output=True, text=True, check=True
        ).stdout.splitlines()
        runner_processes = [
            line.strip() for line in process_table
            if str(RUNNER_BG) in line and "--notebook" in line
        ]
        if not runner_processes:
            tail = LOG_BG.read_text(errors="replace").splitlines()[-30:] if LOG_BG.exists() else []
            raise RuntimeError("A sessão encerrou ao iniciar. Final do log:\n" + "\n".join(tail))
        print("Treinamento V7 iniciado/retomado em segundo plano.")
        print(f"Sessão: {SESSION_BG}")
        print(f"Log: {LOG_BG}")
        print("Agora devolva INICIAR_OU_RETOMAR_V7 para False.")
else:
    print("Nenhum processo iniciado. Mude INICIAR_OU_RETOMAR_V7 para True para iniciar/retomar.")

# %% [source cell 10]
import subprocess
from pathlib import Path
import torch

PROJECT_BG = Path("/path/to/circulatory_failure_prediction")
RUN_BG = PROJECT_BG / "outputs/step11_transformer/transformer_primary_seed42_v7"
CHECKPOINT_BG = RUN_BG / "latest_checkpoint.pt"
METRICS_BG = RUN_BG / "metrics.json"
LOG_BG = RUN_BG / "resume_console.log"
SESSION_BG = "transformer_v7"
RUNNER_BG = PROJECT_BG / "resume_transformer_v7.py"

process_table = subprocess.run(
    ["ps", "-eo", "pid=,args="], capture_output=True, text=True, check=True
).stdout.splitlines()
runner_processes = [
    line.strip() for line in process_table
    if str(RUNNER_BG) in line and "--notebook" in line
]
session_exists = subprocess.run(
    ["tmux", "has-session", "-t", SESSION_BG],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
).returncode == 0

if METRICS_BG.exists():
    status = "CONCLUÍDO: métricas finais já foram gravadas"
elif runner_processes:
    status = "EM EXECUÇÃO: processo real confirmado"
elif session_exists:
    status = "SESSÃO VAZIA: tmux existe, mas o treinamento não está rodando"
elif CHECKPOINT_BG.exists():
    status = "INTERROMPIDO/RETOMÁVEL: checkpoint preservado"
else:
    status = "NÃO INICIADO: checkpoint ausente"
print(f"STATUS: {status}")
if runner_processes:
    print("Processo do treinamento:")
    print("\n".join(runner_processes))

if CHECKPOINT_BG.exists():
    checkpoint = torch.load(CHECKPOINT_BG, map_location="cpu", weights_only=False)
    epochs = [int(row["epoch"]) for row in checkpoint.get("history", [])]
    print(f"Última época completamente salva: {checkpoint.get('epoch')}")
    print(f"Épocas registradas: {epochs}")
    print(f"Melhor época: {checkpoint.get('best_epoch')}")
    print(f"Melhor AUPRC de validação: {checkpoint.get('best_ap')}")

gpu = subprocess.run(
    ["nvidia-smi", "--query-compute-apps=pid,used_memory,process_name", "--format=csv,noheader"],
    capture_output=True, text=True,
)
print("Processos na GPU:")
print(gpu.stdout.strip() or "nenhum processo de computação encontrado")

print("\nÚltimas linhas do log:")
if LOG_BG.exists():
    print("\n".join(LOG_BG.read_text(errors="replace").splitlines()[-40:]))
else:
    print("log ainda não criado")

# %% [source cell 12]
# for seed in (43, 44):
#     result = train_run(RunConfig(
#         run_name=f"transformer_primary_seed{seed}",
#         seed=seed,
#         smoke=False,
#         max_epochs=15,
#         early_stopping_patience=4,
#         batch_size=128,
#         num_workers=2,
#     ))
