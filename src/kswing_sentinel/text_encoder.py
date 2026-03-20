from __future__ import annotations

import hashlib
import math
import re
from typing import Any

DEFAULT_KOREAN_ROBERTA_MODEL_ID = "klue/roberta-base"


def _tokenize_ko_en(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+|[가-힣]+", text.lower())


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


class KoreanTextEncoder:
    def __init__(
        self,
        *,
        model_id: str = DEFAULT_KOREAN_ROBERTA_MODEL_ID,
        encoder_version: str = "ko_roberta_v1",
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

    def _load_transformers(self) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except Exception:
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
                self._runtime_backend = "roberta_mean_pool_v1"
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
            hidden = outputs.last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            pooled = pooled.detach().cpu().tolist()
        return [_project_vector([float(v) for v in row], dim) for row in pooled]

    def encode(self, text: str, dim: int) -> list[float]:
        return self.batch_encode([text], dim=dim)[0]

    def batch_encode(self, texts: list[str], dim: int) -> list[list[float]]:
        if self._runtime_backend in {"transformers_mean_pool_v1", "roberta_mean_pool_v1"}:
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
