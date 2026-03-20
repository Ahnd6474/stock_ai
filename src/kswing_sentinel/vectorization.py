from __future__ import annotations

from datetime import datetime

from .attention_aggregator import HierarchicalAttentionAggregator
from .text_encoder import KoreanTextEncoder


class VectorizationPipeline:
    def __init__(self) -> None:
        self.encoder = KoreanTextEncoder()
        self.agg = HierarchicalAttentionAggregator()

    def build(self, canonical_summary: str, *, source_doc_ids: list[str] | None = None,
              cluster_ids: list[str] | None = None, as_of_time: datetime | None = None,
              prompt_version: str = "normalizer_prompt_v1") -> dict:
        source_doc_ids = source_doc_ids or []
        cluster_ids = cluster_ids or []
        now = as_of_time or datetime.utcnow()

        event_items = [{
            "vec": self.encoder.encode(canonical_summary, 64),
            "freshness_score": 0.8,
            "source_quality_score": 0.8,
            "novelty_score": 0.7,
            "semantic_confidence": 0.7,
        }]
        social_items = [{
            "vec": self.encoder.encode(canonical_summary[:256], 32),
            "freshness_score": 0.7,
            "source_quality_score": 0.5,
            "novelty_score": 0.6,
            "semantic_confidence": 0.6,
        }]
        macro_items = [{
            "vec": self.encoder.encode("KOSPI KOSDAQ USDKRW", 16),
            "freshness_score": 0.6,
            "source_quality_score": 0.9,
            "novelty_score": 0.4,
            "semantic_confidence": 0.8,
        }]

        return {
            "z_event": self.agg.aggregate(event_items, 64),
            "z_social": self.agg.aggregate(social_items, 32),
            "z_macro": self.agg.aggregate(macro_items, 16),
            "metadata": {
                "encoder_version": self.encoder.encoder_version,
                "tokenizer_version": self.encoder.tokenizer_version,
                "attention_aggregator_version": "hier_attn_v1",
                "prompt_version": prompt_version,
                "source_doc_ids": source_doc_ids,
                "cluster_ids": cluster_ids,
                "generated_at": now.isoformat(),
                "as_of_time": now.isoformat(),
            },
        }
