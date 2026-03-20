from __future__ import annotations

import hashlib
import math
import re
from datetime import datetime, timezone


def _hash_to_vec(text: str, dim: int) -> list[float]:
    out = []
    seed = text.encode("utf-8")
    for i in range(dim):
        h = hashlib.sha256(seed + i.to_bytes(2, "little")).digest()
        v = int.from_bytes(h[:4], "little") / 2**32
        out.append(2.0 * v - 1.0)
    return out


def _tokenize_ko_en(text: str) -> list[str]:
    # 한글/영문/숫자 토큰을 보존하는 최소 토크나이저 (외부 모델 fallback 용).
    return re.findall(r"[가-힣A-Za-z0-9_]+", text.lower())


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
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


class VectorizationPipeline:
    def __init__(
        self,
        encoder_version: str = "ko_bert_v1",
        tokenizer_version: str = "ko_bert_tokenizer_v1",
        attention_aggregator_version: str = "hier_attn_v1",
        prompt_version: str = "prompt_v1",
        embedding_backend: str = "hashing_bow_v1",
    ) -> None:
        self.encoder_version = encoder_version
        self.tokenizer_version = tokenizer_version
        self.attention_aggregator_version = attention_aggregator_version
        self.prompt_version = prompt_version
        self.embedding_backend = embedding_backend

    def build(
        self,
        summary: str,
        social: str = "",
        macro: str = "",
        source_doc_ids: list[str] | None = None,
        cluster_ids: list[str] | None = None,
        as_of_time: datetime | None = None,
        session_type: str = "OFF_MARKET",
    ) -> dict:
        now = datetime.now(timezone.utc)
        asof = as_of_time or now
        z_event = _hashing_embedding(summary, 64)
        z_social = _hashing_embedding(social or summary, 32)
        z_macro = _hashing_embedding(macro or "KOSPI", 16)
        return {
            "z_event": z_event,
            "z_social": z_social,
            "z_macro": z_macro,
            "metadata": {
                "encoder_version": self.encoder_version,
                "tokenizer_version": self.tokenizer_version,
                "attention_aggregator_version": self.attention_aggregator_version,
                "prompt_version": self.prompt_version,
                "embedding_backend": self.embedding_backend,
                "source_doc_ids": source_doc_ids or [],
                "cluster_ids": cluster_ids or [],
                "generated_at": now.isoformat(),
                "as_of_time": asof.isoformat(),
                "session_type": session_type,
            },
        }
