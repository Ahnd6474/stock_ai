from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from .attention_aggregator import HierarchicalAttentionAggregator
from .text_encoder import KoreanTextEncoder


def _hash_to_vec(text: str, dim: int) -> list[float]:
    out = []
    seed = text.encode("utf-8")
    for i in range(dim):
        h = hashlib.sha256(seed + i.to_bytes(2, "little")).digest()
        v = int.from_bytes(h[:4], "little") / 2**32
        out.append(2.0 * v - 1.0)
    return out


class VectorizationPipeline:
    def __init__(
        self,
        encoder_version: str = "ko_bert_v1",
        tokenizer_version: str = "ko_bert_tokenizer_v1",
        attention_aggregator_version: str = "hier_attn_v1",
        prompt_version: str = "prompt_v1",
    ) -> None:
        self.encoder_version = encoder_version
        self.tokenizer_version = tokenizer_version
        self.attention_aggregator_version = attention_aggregator_version
        self.prompt_version = prompt_version
        self.encoder = KoreanTextEncoder(encoder_version=encoder_version, tokenizer_version=tokenizer_version)
        self.aggregator = HierarchicalAttentionAggregator()

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
        event_vec = self.encoder.encode(summary, 64)
        social_vec = self.encoder.encode(social or summary, 32)
        macro_vec = self.encoder.encode(macro or "KOSPI", 16)

        # lightweight hierarchical weighting scaffold
        pooled_event = self.aggregator.aggregate(
            [
                {
                    "vec": event_vec,
                    "freshness_score": 0.7,
                    "source_quality_score": 0.7,
                    "novelty_score": 0.6,
                    "semantic_confidence": 0.7,
                }
            ],
            dim=64,
        )
        return {
            "z_event": pooled_event,
            "z_social": social_vec,
            "z_macro": macro_vec,
            "metadata": {
                "encoder_version": self.encoder_version,
                "tokenizer_version": self.tokenizer_version,
                "attention_aggregator_version": self.attention_aggregator_version,
                "prompt_version": self.prompt_version,
                "source_doc_ids": source_doc_ids or [],
                "cluster_ids": cluster_ids or [],
                "generated_at": now.isoformat(),
                "as_of_time": asof.isoformat(),
                "session_type": session_type,
            },
        }
