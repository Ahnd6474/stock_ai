from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
from statistics import mean
from typing import Any

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - optional dependency
    torch = None
    nn = None

from .predictor import (
    DEFAULT_TEMPORAL_DELTA_FEATURE_DIM,
    DEFAULT_TEMPORAL_NUMERIC_FEATURE_KEYS,
    DEFAULT_TEMPORAL_SEQUENCE_KEY,
    DEFAULT_TEMPORAL_TIME_FEATURE_DIM,
    DEFAULT_VECTOR_FEATURE_DIMS,
    REGIME_LABELS,
    TemporalStateAttentionModel,
    build_temporal_model_inputs,
)
from .text_encoder import DEFAULT_KOREAN_ROBERTA_MODEL_ID, TrainableRobertaEncoder


@dataclass(frozen=True)
class FoldMetric:
    fold_index: int
    label_key: str
    mae: float


@dataclass(frozen=True)
class Fold:
    train_start: date
    train_end: date
    valid_start: date
    valid_end: date


@dataclass(frozen=True)
class EpochLoss:
    epoch: int
    loss: float


class RobertaTextRegressor(nn.Module if nn is not None else object):
    def __init__(
        self,
        *,
        encoder: TrainableRobertaEncoder | None = None,
        model_id: str = DEFAULT_KOREAN_ROBERTA_MODEL_ID,
        projection_dim: int | None = 128,
        dropout: float = 0.1,
        device: str = "cpu",
        train_backbone: bool = True,
    ) -> None:
        if nn is None or torch is None:
            raise RuntimeError("torch is required for RoBERTa fine-tuning")
        super().__init__()
        self.encoder = encoder or TrainableRobertaEncoder(
            model_id=model_id,
            device=device,
            projection_dim=projection_dim,
            dropout=dropout,
            train_backbone=train_backbone,
        )
        self.head = nn.Linear(self.encoder.output_dim, 1)

    def forward_texts(self, texts: list[str]) -> Any:
        embeddings = self.encoder.forward_texts(texts)
        return self.head(embeddings).squeeze(-1)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _resolve_temporal_vector_feature_dims(payload: object) -> dict[str, int]:
    if not isinstance(payload, dict):
        return dict(DEFAULT_VECTOR_FEATURE_DIMS)
    out: dict[str, int] = {}
    for key, value in payload.items():
        try:
            dim = int(value)
        except (TypeError, ValueError):
            continue
        if dim > 0:
            out[str(key)] = dim
    return out or dict(DEFAULT_VECTOR_FEATURE_DIMS)


def _left_pad_sequence_batch(
    sequences: list[list[list[float]]],
    *,
    input_dim: int,
    max_seq_len: int,
    device: str,
) -> tuple[Any, Any]:
    if torch is None:
        raise RuntimeError("torch is required for temporal transformer training")
    trimmed = [sequence[-max_seq_len:] if max_seq_len > 0 else sequence for sequence in sequences]
    batch_seq_len = max((len(sequence) for sequence in trimmed), default=1)
    batch = torch.zeros((len(trimmed), batch_seq_len, input_dim), dtype=torch.float32, device=device)
    padding_mask = torch.ones((len(trimmed), batch_seq_len), dtype=torch.bool, device=device)
    for row_index, sequence in enumerate(trimmed):
        if not sequence:
            continue
        seq_tensor = torch.tensor(sequence, dtype=torch.float32, device=device)
        seq_len = int(seq_tensor.size(0))
        start = batch_seq_len - seq_len
        batch[row_index, start:, :] = seq_tensor
        padding_mask[row_index, start:] = False
    return batch, padding_mask


class WalkForwardSplitter:
    def build(self, days: list[date], train_size: int = 120, valid_size: int = 20, step: int = 20) -> list[Fold]:
        folds: list[Fold] = []
        i = 0
        while i + train_size + valid_size <= len(days):
            tr = days[i:i+train_size]
            va = days[i+train_size:i+train_size+valid_size]
            folds.append(Fold(tr[0], tr[-1], va[0], va[-1]))
            i += step
        return folds


class TrainingPipeline:
    """Lightweight scaffold for walk-forward training orchestration."""

    def __init__(self) -> None:
        self.splitter = WalkForwardSplitter()

    def build_dataset(self, rows: list[dict], feature_keys: list[str], label_key: str) -> list[dict]:
        dataset: list[dict] = []
        for r in rows:
            if label_key not in r:
                continue
            dataset.append(
                {
                    "x": {k: float(r.get(k, 0.0)) for k in feature_keys},
                    "y": float(r[label_key]),
                    "date": r["date"],
                }
            )
        return dataset

    def leakage_safe_join(self, feature_rows: list[dict], label_rows: list[dict], label_key: str) -> list[dict]:
        labels = {(r["symbol"], r["date"]): r for r in label_rows}
        joined: list[dict] = []
        for fr in feature_rows:
            if fr.get("as_of_date", fr["date"]) > fr["date"]:
                continue
            lab = labels.get((fr["symbol"], fr["date"]))
            if not lab or label_key not in lab:
                continue
            row = dict(fr)
            row[label_key] = lab[label_key]
            joined.append(row)
        return joined

    def fit_linear_model(self, dataset: list[dict], feature_keys: list[str], *, lr: float = 0.03, epochs: int = 300) -> dict:
        weights = {k: 0.0 for k in feature_keys}
        bias = 0.0
        if not dataset:
            return {"weights": weights, "bias": bias}
        n = float(len(dataset))
        for _ in range(epochs):
            grad_w = {k: 0.0 for k in feature_keys}
            grad_b = 0.0
            for row in dataset:
                y = float(row["y"])
                pred = bias + sum(weights[k] * float(row["x"].get(k, 0.0)) for k in feature_keys)
                err = pred - y
                grad_b += err
                for k in feature_keys:
                    grad_w[k] += err * float(row["x"].get(k, 0.0))
            bias -= lr * (grad_b / n)
            for k in feature_keys:
                weights[k] -= lr * (grad_w[k] / n)
        return {"weights": weights, "bias": bias}

    def _mae(self, model: dict, rows: list[dict], feature_keys: list[str]) -> float:
        if not rows:
            return 0.0
        errs = []
        for r in rows:
            pred = float(model["bias"]) + sum(float(model["weights"].get(k, 0.0)) * float(r["x"].get(k, 0.0)) for k in feature_keys)
            errs.append(abs(pred - float(r["y"])))
        return mean(errs)

    def train_walk_forward(
        self,
        rows: list[dict],
        feature_keys: list[str],
        label_key: str,
        artifact_dir: str | Path,
        model_version: str = "linear_baseline_v1",
    ) -> tuple[Path, Path]:
        dataset = self.build_dataset(rows, feature_keys, label_key)
        days = sorted({r["date"] for r in dataset})
        folds = self.splitter.build(days, train_size=120, valid_size=20, step=20)

        metrics: list[FoldMetric] = []
        final_model = {"weights": {k: 0.0 for k in feature_keys}, "bias": 0.0}
        for i, f in enumerate(folds):
            train_rows = [r for r in dataset if f.train_start <= r["date"] <= f.train_end]
            valid_rows = [r for r in dataset if f.valid_start <= r["date"] <= f.valid_end]
            if not train_rows:
                continue
            model = self.fit_linear_model(train_rows, feature_keys)
            final_model = model
            metrics.append(FoldMetric(i, label_key, self._mae(model, valid_rows, feature_keys)))

        out_dir = Path(artifact_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = out_dir / f"{label_key}_artifact.json"
        metrics_path = out_dir / f"{label_key}_fold_metrics.json"
        artifact_payload = {
            "model_version": model_version,
            "schema_version": "v1",
            "feature_keys": feature_keys,
            "label_key": label_key,
            "weights": final_model["weights"],
            "bias": final_model["bias"],
        }
        artifact_path.write_text(json.dumps(artifact_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        metrics_path.write_text(
            json.dumps([m.__dict__ for m in metrics], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return artifact_path, metrics_path

    def train_multi_head(
        self,
        rows: list[dict],
        feature_keys: list[str],
        artifact_dir: str | Path,
        label_keys: tuple[str, str, str] = ("er_20d", "dd_20d", "p_up_20d"),
        model_version: str = "linear_multihead_v1",
    ) -> Path:
        head_payload: dict[str, dict] = {}
        for label_key in label_keys:
            dataset = self.build_dataset(rows, feature_keys, label_key)
            model = self.fit_linear_model(dataset, feature_keys)
            head_payload[label_key] = model

        artifact = {
            "model_version": model_version,
            "schema_version": "v1",
            "feature_keys": feature_keys,
            "heads": head_payload,
            "uncertainty_base": 0.30,
            "uncertainty_extension_mult": 0.30,
        }
        out = Path(artifact_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "multi_head_artifact.json"
        path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def _temporal_regime_index(self, row: dict, sequence_key: str) -> int:
        label = row.get("regime_final", row.get("regime"))
        if label in REGIME_LABELS:
            return REGIME_LABELS.index(str(label))

        sequence = row.get(sequence_key)
        latest = sequence[-1] if isinstance(sequence, list) and sequence else row
        numeric = latest.get("numeric_features") if isinstance(latest, dict) else None
        numeric_source = numeric if isinstance(numeric, dict) else latest if isinstance(latest, dict) else row

        if bool(numeric_source.get("market_risk_off", row.get("market_risk_off", False))):
            return REGIME_LABELS.index("risk_off")

        event_score = _safe_float(numeric_source.get("event_score", row.get("event_score", 0.0)), 0.0)
        trend_120m = _safe_float(numeric_source.get("trend_120m", row.get("trend_120m", 0.0)), 0.0)
        if abs(event_score) >= 0.35:
            return REGIME_LABELS.index("event")
        if trend_120m > 0:
            return REGIME_LABELS.index("trend")
        return REGIME_LABELS.index("chop")

    def _build_temporal_training_dataset(
        self,
        rows: list[dict],
        *,
        numeric_feature_keys: list[str],
        sequence_key: str,
        vector_feature_dims: dict[str, int],
        max_seq_len: int,
    ) -> list[dict]:
        dataset: list[dict] = []
        for row in rows:
            if not all(label_key in row for label_key in ("er_20d", "dd_20d", "p_up_20d")):
                continue
            model_inputs = build_temporal_model_inputs(
                row,
                numeric_feature_keys=numeric_feature_keys,
                vector_feature_dims=vector_feature_dims,
                as_of_time=None,
                sequence_key=sequence_key,
                max_seq_len=max_seq_len,
                include_time_features=True,
                include_delta_features=True,
            )
            sequence = row.get(sequence_key)
            latest = sequence[-1] if isinstance(sequence, list) and sequence else row
            numeric = latest.get("numeric_features") if isinstance(latest, dict) else None
            numeric_source = numeric if isinstance(numeric, dict) else latest if isinstance(latest, dict) else row
            extension = _safe_float(numeric_source.get("extension_60m", row.get("extension_60m", 0.0)), 0.0)
            flow_strength = _safe_float(numeric_source.get("flow_strength", row.get("flow_strength", 0.0)), 0.0)
            dataset.append(
                {
                    "sequence": model_inputs["state_sequence"],
                    "time_sequence": model_inputs["time_sequence"],
                    "social_sequence": model_inputs["social_sequence"],
                    "delta_sequence": model_inputs["delta_sequence"],
                    "targets": {
                        "er_20d": _safe_float(row["er_20d"], 0.0),
                        "dd_20d": _clamp(_safe_float(row["dd_20d"], 0.0), 0.0, 1.0),
                        "p_up_20d": _clamp(_safe_float(row["p_up_20d"], 0.5), 0.0, 1.0),
                        "uncertainty": _clamp(
                            _safe_float(row.get("uncertainty"), 0.35 + 0.25 * abs(extension)),
                            0.0,
                            1.0,
                        ),
                        "flow_persist": _clamp(
                            _safe_float(row.get("flow_persist"), 0.5 + 0.4 * flow_strength),
                            0.0,
                            1.0,
                        ),
                        "regime_index": self._temporal_regime_index(row, sequence_key),
                    },
                    "date": row.get("date"),
                }
            )
        return dataset

    def train_temporal_transformer(
        self,
        rows: list[dict],
        *,
        numeric_feature_keys: list[str] | None = None,
        artifact_dir: str | Path,
        sequence_key: str = DEFAULT_TEMPORAL_SEQUENCE_KEY,
        vector_feature_dims: dict[str, int] | None = None,
        max_seq_len: int = 32,
        embedding_hidden_dim: int = 128,
        d_model: int = 128,
        context_num_layers: int = 1,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        batch_size: int = 8,
        epochs: int = 8,
        lr: float = 1e-3,
        device: str = "cpu",
        model_version: str = "temporal_transformer_v1",
    ) -> tuple[Path, Path, Path]:
        if nn is None or torch is None:
            raise RuntimeError("torch is required for temporal transformer training")
        numeric_keys = list(numeric_feature_keys or DEFAULT_TEMPORAL_NUMERIC_FEATURE_KEYS)
        resolved_vector_dims = _resolve_temporal_vector_feature_dims(vector_feature_dims)
        time_feature_dim = DEFAULT_TEMPORAL_TIME_FEATURE_DIM
        delta_feature_dim = DEFAULT_TEMPORAL_DELTA_FEATURE_DIM
        social_feature_dim = int(resolved_vector_dims.get("z_social", 0))
        dataset = self._build_temporal_training_dataset(
            rows,
            numeric_feature_keys=numeric_keys,
            sequence_key=sequence_key,
            vector_feature_dims=resolved_vector_dims,
            max_seq_len=max_seq_len,
        )
        if not dataset:
            raise ValueError("temporal transformer dataset is empty")

        input_dim = len(numeric_keys) + sum(resolved_vector_dims.values())
        model = TemporalStateAttentionModel(
            input_dim=input_dim,
            embedding_hidden_dim=embedding_hidden_dim,
            d_model=d_model,
            context_num_layers=context_num_layers,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            max_seq_len=max_seq_len,
            time_feature_dim=time_feature_dim,
            social_feature_dim=social_feature_dim,
            delta_feature_dim=delta_feature_dim,
        )
        model.to(device)
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        mse_loss = nn.MSELoss()
        ce_loss = nn.CrossEntropyLoss()
        metrics: list[EpochLoss] = []

        for epoch in range(epochs):
            epoch_losses: list[float] = []
            for start in range(0, len(dataset), max(1, batch_size)):
                batch_rows = dataset[start : start + max(1, batch_size)]
                state_batch, padding_mask = _left_pad_sequence_batch(
                    [row["sequence"] for row in batch_rows],
                    input_dim=input_dim,
                    max_seq_len=max_seq_len,
                    device=device,
                )
                time_batch, _ = _left_pad_sequence_batch(
                    [row["time_sequence"] for row in batch_rows],
                    input_dim=time_feature_dim,
                    max_seq_len=max_seq_len,
                    device=device,
                )
                social_batch = None
                if social_feature_dim > 0:
                    social_batch, _ = _left_pad_sequence_batch(
                        [row["social_sequence"] for row in batch_rows],
                        input_dim=social_feature_dim,
                        max_seq_len=max_seq_len,
                        device=device,
                    )
                delta_batch, _ = _left_pad_sequence_batch(
                    [row["delta_sequence"] for row in batch_rows],
                    input_dim=delta_feature_dim,
                    max_seq_len=max_seq_len,
                    device=device,
                )
                outputs = model(
                    state_batch,
                    padding_mask=padding_mask,
                    time_features=time_batch,
                    social_sequence=social_batch,
                    delta_features=delta_batch,
                )
                er_targets = torch.tensor(
                    [row["targets"]["er_20d"] for row in batch_rows],
                    dtype=torch.float32,
                    device=device,
                )
                dd_targets = torch.tensor(
                    [row["targets"]["dd_20d"] for row in batch_rows],
                    dtype=torch.float32,
                    device=device,
                )
                pup_targets = torch.tensor(
                    [row["targets"]["p_up_20d"] for row in batch_rows],
                    dtype=torch.float32,
                    device=device,
                )
                uncertainty_targets = torch.tensor(
                    [row["targets"]["uncertainty"] for row in batch_rows],
                    dtype=torch.float32,
                    device=device,
                )
                flow_targets = torch.tensor(
                    [row["targets"]["flow_persist"] for row in batch_rows],
                    dtype=torch.float32,
                    device=device,
                )
                regime_targets = torch.tensor(
                    [row["targets"]["regime_index"] for row in batch_rows],
                    dtype=torch.long,
                    device=device,
                )

                optimizer.zero_grad()
                loss = (
                    mse_loss(outputs["er_20d"], er_targets)
                    + mse_loss(outputs["dd_20d"], dd_targets)
                    + mse_loss(outputs["p_up_20d"], pup_targets)
                    + 0.5 * mse_loss(outputs["uncertainty"], uncertainty_targets)
                    + 0.5 * mse_loss(outputs["flow_persist"], flow_targets)
                    + 0.2 * ce_loss(outputs["regime_logits"], regime_targets)
                )
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu()))
            metrics.append(EpochLoss(epoch=epoch + 1, loss=mean(epoch_losses) if epoch_losses else 0.0))

        model.eval()
        out_dir = Path(artifact_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        weights_path = out_dir / "temporal_transformer_weights.pt"
        artifact_path = out_dir / "temporal_transformer_artifact.json"
        metrics_path = out_dir / "temporal_transformer_metrics.json"
        with weights_path.open("wb") as fh:
            torch.save({"model_state_dict": model.state_dict()}, fh)
        artifact_payload = {
            "model_type": "temporal_transformer_v1",
            "model_version": model_version,
            "schema_version": "v4",
            "sequence_key": sequence_key,
            "numeric_feature_keys": numeric_keys,
            "feature_keys": numeric_keys,
            "vector_feature_dims": resolved_vector_dims,
            "time_feature_dim": time_feature_dim,
            "delta_feature_dim": delta_feature_dim,
            "max_seq_len": max_seq_len,
            "embedding_hidden_dim": embedding_hidden_dim,
            "d_model": d_model,
            "context_num_layers": context_num_layers,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "dropout": dropout,
            "weights_path": weights_path.name,
            "calibrator_version": "affine_v1",
        }
        artifact_path.write_text(json.dumps(artifact_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        metrics_path.write_text(
            json.dumps([metric.__dict__ for metric in metrics], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return weights_path, artifact_path, metrics_path

    def train_text_regressor(
        self,
        rows: list[dict],
        *,
        text_key: str,
        label_key: str,
        artifact_dir: str | Path,
        model: Any | None = None,
        model_id: str = DEFAULT_KOREAN_ROBERTA_MODEL_ID,
        projection_dim: int | None = 128,
        batch_size: int = 4,
        epochs: int = 1,
        lr: float = 2e-5,
        device: str = "cpu",
        train_backbone: bool = True,
        model_version: str = "roberta_text_regressor_v1",
    ) -> tuple[Path, Path, Path]:
        if nn is None or torch is None:
            raise RuntimeError("torch is required for RoBERTa fine-tuning")

        dataset = [
            {"text": str(row[text_key]), "y": float(row[label_key])}
            for row in rows
            if text_key in row and label_key in row
        ]
        if not dataset:
            raise ValueError("text regression dataset is empty")

        regressor = model or RobertaTextRegressor(
            model_id=model_id,
            projection_dim=projection_dim,
            device=device,
            train_backbone=train_backbone,
        )
        if hasattr(regressor, "to"):
            regressor.to(device)
        if hasattr(regressor, "train"):
            regressor.train()

        params = [param for param in regressor.parameters() if param.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=lr)
        loss_fn = nn.MSELoss()
        metrics: list[EpochLoss] = []

        for epoch in range(epochs):
            epoch_losses: list[float] = []
            for start in range(0, len(dataset), batch_size):
                batch = dataset[start : start + batch_size]
                texts = [row["text"] for row in batch]
                targets = torch.tensor([row["y"] for row in batch], dtype=torch.float32, device=device)
                optimizer.zero_grad()
                predictions = regressor.forward_texts(texts)
                if predictions.ndim > 1:
                    predictions = predictions.squeeze(-1)
                loss = loss_fn(predictions, targets)
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu()))
            metrics.append(EpochLoss(epoch=epoch + 1, loss=mean(epoch_losses) if epoch_losses else 0.0))

        out_dir = Path(artifact_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        weights_path = out_dir / "text_regressor_weights.pt"
        artifact_path = out_dir / "text_regressor_artifact.json"
        metrics_path = out_dir / "text_regressor_metrics.json"
        with weights_path.open("wb") as fh:
            torch.save(regressor.state_dict(), fh)
        artifact_payload = {
            "model_version": model_version,
            "text_key": text_key,
            "label_key": label_key,
            "encoder_model_id": getattr(getattr(regressor, "encoder", None), "model_id", model_id),
            "projection_dim": projection_dim,
            "train_backbone": train_backbone,
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": lr,
            "weights_path": str(weights_path),
        }
        artifact_path.write_text(json.dumps(artifact_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        metrics_path.write_text(
            json.dumps([metric.__dict__ for metric in metrics], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return weights_path, artifact_path, metrics_path
