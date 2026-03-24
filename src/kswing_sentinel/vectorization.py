from __future__ import annotations

import re
from datetime import datetime, timezone

from .attention_aggregator import HierarchicalAttentionAggregator
from .text_encoder import DEFAULT_KOREAN_ROBERTA_MODEL_ID, KoreanTextEncoder


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|[\r\n]+")


def _chunk_long_sentence(text: str, chunk_chars: int = 220, overlap_chars: int = 40) -> list[str]:
    normalized = " ".join(text.split())
    if not normalized:
        return []
    if len(normalized) <= chunk_chars:
        return [normalized]
    chunks: list[str] = []
    step = max(1, chunk_chars - overlap_chars)
    for start in range(0, len(normalized), step):
        chunk = normalized[start : start + chunk_chars]
        if chunk:
            chunks.append(chunk)
        if start + chunk_chars >= len(normalized):
            break
    return chunks


def _sentence_units(text: str, max_sentence_chars: int = 220, overlap_chars: int = 40) -> list[str]:
    normalized = " ".join(text.split())
    if not normalized:
        return []
    raw_sentences = [item.strip() for item in _SENTENCE_SPLIT_RE.split(normalized) if item.strip()]
    if not raw_sentences:
        raw_sentences = [normalized]

    units: list[str] = []
    for sentence in raw_sentences:
        if len(sentence) <= max_sentence_chars:
            units.append(sentence)
            continue
        units.extend(_chunk_long_sentence(sentence, chunk_chars=max_sentence_chars, overlap_chars=overlap_chars))
    return units


class VectorizationPipeline:
    def __init__(
        self,
        *,
        encoder: KoreanTextEncoder | None = None,
        aggregator: HierarchicalAttentionAggregator | None = None,
        model_id: str = DEFAULT_KOREAN_ROBERTA_MODEL_ID,
        encoder_version: str = "ko_roberta_v2",
        tokenizer_version: str = "auto",
        encoder_backend: str = "roberta",
        attention_aggregator_version: str = "hier_transformer_v3",
        prompt_version: str = "prompt_v1",
    ) -> None:
        self.encoder = encoder or KoreanTextEncoder(
            model_id=model_id,
            encoder_version=encoder_version,
            tokenizer_version=tokenizer_version,
            backend=encoder_backend,
        )
        self.aggregator = aggregator or HierarchicalAttentionAggregator()
        self.attention_aggregator_version = attention_aggregator_version or getattr(
            self.aggregator,
            "version",
            "hier_transformer_v3",
        )
        self.prompt_version = prompt_version
        metadata = self.encoder.metadata()
        self.encoder_version = str(metadata["encoder_version"])
        self.tokenizer_version = str(metadata["tokenizer_version"])
        self.embedding_backend = str(metadata["embedding_backend"])

    def _encode_single_item(self, item: dict, dim: int) -> dict:
        text = str(item.get("text", ""))
        sentences = _sentence_units(text)
        if not sentences:
            return {
                "vec": [0.0] * dim,
                "cluster_id": item.get("cluster_id", "default"),
                "freshness_score": item.get("freshness_score", 0.0),
                "source_quality_score": item.get("source_quality_score", 0.0),
                "novelty_score": item.get("novelty_score", 0.0),
                "semantic_confidence": item.get("semantic_confidence", 0.0),
            }
        sentence_vectors = self.encoder.batch_encode(sentences, dim)
        sentence_items = [
            {
                "vec": vec,
                "freshness_score": item.get("freshness_score", 0.8),
                "source_quality_score": item.get("source_quality_score", 0.8),
                "novelty_score": item.get("novelty_score", 0.5),
                "semantic_confidence": item.get("semantic_confidence", 0.7),
            }
            for vec in sentence_vectors
        ]
        return {
            "vec": self.aggregator.aggregate(sentence_items, dim),
            "cluster_id": item.get("cluster_id", "default"),
            "freshness_score": item.get("freshness_score", 0.8),
            "source_quality_score": item.get("source_quality_score", 0.8),
            "novelty_score": item.get("novelty_score", 0.5),
            "semantic_confidence": item.get("semantic_confidence", 0.7),
        }

    def _aggregate_stream(self, items: list[dict], dim: int) -> list[float]:
        if not items:
            return [0.0] * dim
        cluster_items: dict[str, list[dict]] = {}
        for item in items:
            encoded = self._encode_single_item(item, dim)
            cluster_id = str(encoded.get("cluster_id", "default"))
            cluster_items.setdefault(cluster_id, []).append(encoded)
        return self.aggregator.aggregate_by_cluster(cluster_items, dim)

    def build_from_items(
        self,
        *,
        event_items: list[dict] | None = None,
        social_items: list[dict] | None = None,
        macro_items: list[dict] | None = None,
        source_doc_ids: list[str] | None = None,
        cluster_ids: list[str] | None = None,
        as_of_time: datetime | None = None,
        session_type: str = "OFF_MARKET",
    ) -> dict:
        now = datetime.now(timezone.utc)
        asof = as_of_time or now
        self.embedding_backend = str(self.encoder.metadata()["embedding_backend"])
        z_event = self._aggregate_stream(event_items or [], 64)
        z_social = self._aggregate_stream(social_items or [], 32)
        z_macro = self._aggregate_stream(macro_items or [], 16)
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
        event_items = [
            {
                "text": summary,
                "cluster_id": (cluster_ids or ["event"])[0],
                "freshness_score": 0.9,
                "source_quality_score": 0.9,
                "novelty_score": 0.7,
                "semantic_confidence": 0.8,
            }
        ]
        social_items = [
            {
                "text": social or summary,
                "cluster_id": (cluster_ids or ["social"])[0],
                "freshness_score": 0.8,
                "source_quality_score": 0.6,
                "novelty_score": 0.8,
                "semantic_confidence": 0.6,
            }
        ]
        macro_items = [
            {
                "text": macro or "KOSPI market context",
                "cluster_id": "macro",
                "freshness_score": 0.7,
                "source_quality_score": 0.7,
                "novelty_score": 0.4,
                "semantic_confidence": 0.7,
            }
        ]
        return self.build_from_items(
            event_items=event_items,
            social_items=social_items,
            macro_items=macro_items,
            source_doc_ids=source_doc_ids,
            cluster_ids=cluster_ids,
            as_of_time=as_of_time,
            session_type=session_type,
        )
