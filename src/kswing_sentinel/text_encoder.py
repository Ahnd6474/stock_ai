from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Any

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - optional dependency
    torch = None
    nn = None

DEFAULT_KOREAN_ROBERTA_MODEL_ID = "klue/roberta-base"


def _tokenize_ko_en(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+|[\uac00-\ud7a3]+", text.lower())


def _normalize_vector(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in values))
    if norm <= 0:
        return values
    return [v / norm for v in values]


def _hashing_embedding(text: str, dim: int, ngram: int = 2) -> list[float]:
    tokens = _tokenize_ko_en(text)
    if not tokens:
        return [0.0] * dim
    vec = [0.0] * dim
    features = list(tokens)
    if ngram >= 2:
        features.extend(" ".join(tokens[i : i + 2]) for i in range(max(0, len(tokens) - 1)))
    if ngram >= 3:
        features.extend(" ".join(tokens[i : i + 3]) for i in range(max(0, len(tokens) - 2)))
    for feat in features:
        h = hashlib.sha256(feat.encode("utf-8")).digest()
        idx = int.from_bytes(h[:4], "little") % dim
        sign = -1.0 if (h[4] & 1) else 1.0
        vec[idx] += sign
    return _normalize_vector(vec)


def _project_vector(values: list[float], dim: int) -> list[float]:
    if not values:
        return [0.0] * dim
    if len(values) == dim:
        return _normalize_vector(list(values))
    bucketed = [0.0] * dim
    counts = [0] * dim
    for idx, value in enumerate(values):
        target = idx % dim
        bucketed[target] += float(value)
        counts[target] += 1
    for idx, count in enumerate(counts):
        if count > 0:
            bucketed[idx] /= count
    return _normalize_vector(bucketed)


def download_roberta_model(
    model_id: str = DEFAULT_KOREAN_ROBERTA_MODEL_ID,
    *,
    cache_dir: str | Path | None = None,
) -> str:
    try:
        from transformers import AutoModel, AutoTokenizer
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("transformers is required to download RoBERTa models") from exc

    kwargs: dict[str, Any] = {}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    AutoTokenizer.from_pretrained(model_id, **kwargs)
    AutoModel.from_pretrained(model_id, **kwargs)
    return model_id


def _mean_pool(last_hidden_state: Any, attention_mask: Any) -> Any:
    if torch is None:  # pragma: no cover - defensive
        raise RuntimeError("torch is required for RoBERTa pooling")
    mask = attention_mask.unsqueeze(-1).float()
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


class TrainableRobertaEncoder(nn.Module if nn is not None else object):
    def __init__(
        self,
        *,
        model_id: str = DEFAULT_KOREAN_ROBERTA_MODEL_ID,
        device: str = "cpu",
        max_length: int = 256,
        projection_dim: int | None = None,
        dropout: float = 0.1,
        normalize: bool = True,
        train_backbone: bool = True,
        cache_dir: str | Path | None = None,
    ) -> None:
        if nn is None or torch is None:
            raise RuntimeError("torch is required for trainable RoBERTa encoders")
        super().__init__()
        try:
            from transformers import AutoModel, AutoTokenizer
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("transformers is required for trainable RoBERTa encoders") from exc

        kwargs: dict[str, Any] = {}
        if cache_dir is not None:
            kwargs["cache_dir"] = str(cache_dir)
        self.model_id = model_id
        self.device_name = device
        self.max_length = max_length
        self.normalize = normalize
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
        self.backbone = AutoModel.from_pretrained(model_id, **kwargs)
        hidden_size = int(getattr(self.backbone.config, "hidden_size", 768))
        self.dropout = nn.Dropout(dropout)
        if projection_dim is None or projection_dim == hidden_size:
            self.projection = nn.Identity()
            self.output_dim = hidden_size
        else:
            self.projection = nn.Linear(hidden_size, projection_dim)
            self.output_dim = projection_dim
        self.set_train_backbone(train_backbone)
        self.to(device)

    def set_train_backbone(self, enabled: bool) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = enabled

    def tokenize(self, texts: list[str]) -> dict[str, Any]:
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {key: value.to(self.device_name) for key, value in tokens.items()}

    def forward(self, input_ids: Any, attention_mask: Any, token_type_ids: Any | None = None) -> Any:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled = _mean_pool(outputs.last_hidden_state, attention_mask)
        pooled = self.dropout(pooled)
        projected = self.projection(pooled)
        if self.normalize:
            projected = torch.nn.functional.normalize(projected, p=2, dim=-1)
        return projected

    def forward_texts(self, texts: list[str]) -> Any:
        tokens = self.tokenize(texts)
        return self.forward(
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
            token_type_ids=tokens.get("token_type_ids"),
        )


class KoreanTextEncoder:
    def __init__(
        self,
        *,
        model_id: str = DEFAULT_KOREAN_ROBERTA_MODEL_ID,
        encoder_version: str = "ko_roberta_v2",
        tokenizer_version: str = "auto",
        backend: str = "auto",
        device: str = "cpu",
        max_length: int = 256,
    ) -> None:
        self.model_id = model_id
        self.encoder_version = encoder_version
        self.tokenizer_version = tokenizer_version
        self.backend = backend
        self.device = device
        self.max_length = max_length
        self._runtime_backend = "hashing_bow_v1"
        self._model_family = "hashing"
        self._tokenizer: Any = None
        self._model: Any = None
        self._torch: Any = None
        if self.backend in {"auto", "transformers", "roberta"}:
            self._load_transformers()

    @staticmethod
    def ensure_model_downloaded(
        model_id: str = DEFAULT_KOREAN_ROBERTA_MODEL_ID,
        *,
        cache_dir: str | Path | None = None,
    ) -> str:
        return download_roberta_model(model_id=model_id, cache_dir=cache_dir)

    def build_trainable_encoder(
        self,
        *,
        projection_dim: int | None = None,
        dropout: float = 0.1,
        normalize: bool = True,
        train_backbone: bool = True,
        cache_dir: str | Path | None = None,
    ) -> TrainableRobertaEncoder:
        return TrainableRobertaEncoder(
            model_id=self.model_id,
            device=self.device,
            max_length=self.max_length,
            projection_dim=projection_dim,
            dropout=dropout,
            normalize=normalize,
            train_backbone=train_backbone,
            cache_dir=cache_dir,
        )

    def _load_transformers(self) -> None:
        try:
            from transformers import AutoModel, AutoTokenizer
        except Exception:
            self._runtime_backend = "hashing_bow_v1"
            self._model_family = "hashing"
            return
        if torch is None:
            self._runtime_backend = "hashing_bow_v1"
            self._model_family = "hashing"
            return

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModel.from_pretrained(self.model_id)
            self._model.eval()
            if self.device:
                self._model.to(self.device)
            self._torch = torch
            model_type = str(getattr(getattr(self._model, "config", None), "model_type", "")).lower()
            if model_type == "roberta":
                self._runtime_backend = "roberta_mean_pool_v2"
                self._model_family = "roberta"
            else:
                self._runtime_backend = "transformers_mean_pool_v1"
                self._model_family = model_type or "transformers"
            if self.tokenizer_version == "auto":
                self.tokenizer_version = str(self.model_id)
        except Exception:
            self._tokenizer = None
            self._model = None
            self._torch = None
            self._runtime_backend = "hashing_bow_v1"
            self._model_family = "hashing"

    def _encode_transformers_batch(self, texts: list[str], dim: int) -> list[list[float]]:
        if self._model is None or self._tokenizer is None or self._torch is None:
            raise RuntimeError("transformers backend not available")
        if not texts:
            return []
        with self._torch.no_grad():
            tokens = self._tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            if self.device:
                tokens = {k: v.to(self.device) for k, v in tokens.items()}
            outputs = self._model(**tokens)
            pooled = _mean_pool(outputs.last_hidden_state, tokens["attention_mask"])
            pooled = pooled.detach().cpu().tolist()
        return [_project_vector([float(v) for v in row], dim) for row in pooled]

    def encode(self, text: str, dim: int) -> list[float]:
        return self.batch_encode([text], dim=dim)[0]

    def batch_encode(self, texts: list[str], dim: int) -> list[list[float]]:
        if self._runtime_backend in {"transformers_mean_pool_v1", "roberta_mean_pool_v2"}:
            try:
                return self._encode_transformers_batch(texts, dim)
            except Exception:
                self._runtime_backend = "hashing_bow_v1"
                self._model_family = "hashing"
        return [_hashing_embedding(text, dim) for text in texts]

    def metadata(self) -> dict:
        return {
            "encoder_version": self.encoder_version,
            "tokenizer_version": self.tokenizer_version,
            "model_id": self.model_id,
            "model_family": self._model_family,
            "embedding_backend": self._runtime_backend,
        }
