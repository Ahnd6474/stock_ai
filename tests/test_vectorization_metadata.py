from datetime import datetime, timezone

from kswing_sentinel.attention_aggregator import HierarchicalAttentionAggregator
from kswing_sentinel.text_encoder import DEFAULT_KOREAN_ROBERTA_MODEL_ID, KoreanTextEncoder
from kswing_sentinel.vectorization import VectorizationPipeline


def test_roberta_encoder_metadata_defaults_to_roberta_or_hashing_fallback():
    encoder = KoreanTextEncoder(backend="roberta")
    metadata = encoder.metadata()

    assert metadata["model_id"] == DEFAULT_KOREAN_ROBERTA_MODEL_ID
    assert metadata["encoder_version"].startswith("ko_roberta")
    assert metadata["embedding_backend"] in {"hashing_bow_v1", "roberta_mean_pool_v2"}


def test_vectorization_returns_metadata_versions():
    v = VectorizationPipeline().build(
        "Earnings guidance improved and order backlog expanded.",
        source_doc_ids=["doc1"],
        cluster_ids=["cl1"],
        as_of_time=datetime(2026, 3, 20, tzinfo=timezone.utc),
    )
    assert len(v["z_event"]) == 64
    assert v["metadata"]["embedding_backend"] in {
        "hashing_bow_v1",
        "roberta_mean_pool_v2",
        "transformers_mean_pool_v1",
    }
    assert v["metadata"]["encoder_version"].startswith("ko_roberta")
    assert v["metadata"]["tokenizer_version"] in {"auto", DEFAULT_KOREAN_ROBERTA_MODEL_ID}


class RecordingSentenceEncoder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def metadata(self) -> dict:
        return {
            "encoder_version": "sentence_roberta_stub_v1",
            "tokenizer_version": "stub",
            "model_id": "stub-roberta",
            "model_family": "stub",
            "embedding_backend": "sentence_stub_backend",
        }

    def batch_encode(self, texts: list[str], dim: int) -> list[list[float]]:
        self.calls.append(list(texts))
        rows: list[list[float]] = []
        for index, text in enumerate(texts):
            vec = [0.0] * dim
            token = text.strip().lower()
            if "alpha" in token or "first" in token:
                vec[0] = 1.0
            if "beta" in token or "second" in token:
                vec[1] = 1.0
            if "third" in token:
                vec[2] = 1.0
            vec[(index + 3) % dim] += 0.25
            rows.append(vec)
        return rows


def test_vectorization_splits_text_into_sentences_before_encoding():
    encoder = RecordingSentenceEncoder()
    pipeline = VectorizationPipeline(
        encoder=encoder,
        aggregator=HierarchicalAttentionAggregator(num_layers=1, num_heads=2),
    )
    pipeline.build_from_items(
        event_items=[
            {
                "text": "First sentence. Second sentence!\nThird sentence?",
                "cluster_id": "event",
            }
        ],
        as_of_time=datetime(2026, 3, 20, tzinfo=timezone.utc),
    )
    assert encoder.calls[0] == ["First sentence.", "Second sentence!", "Third sentence?"]


def test_hierarchical_transformer_aggregation_is_order_sensitive():
    pipeline = VectorizationPipeline(
        encoder=RecordingSentenceEncoder(),
        aggregator=HierarchicalAttentionAggregator(num_layers=2, num_heads=2),
    )
    v1 = pipeline.build_from_items(event_items=[{"text": "Alpha. Beta.", "cluster_id": "event"}])
    v2 = pipeline.build_from_items(event_items=[{"text": "Beta. Alpha.", "cluster_id": "event"}])
    dot = sum(a * b for a, b in zip(v1["z_event"], v2["z_event"]))
    assert v1["z_event"] != v2["z_event"]
    assert dot < sum(a * a for a in v1["z_event"])


def test_vectorization_semantic_similarity_signal():
    pipeline = VectorizationPipeline()
    v1 = pipeline.build("Samsung Electronics earnings outlook improved after memory pricing recovery.")
    v2 = pipeline.build("Samsung Electronics profit expectations improved on stronger memory demand.")
    v3 = pipeline.build("Oil prices surged after a supply disruption in the Middle East.")
    dot_similar = sum(a * b for a, b in zip(v1["z_event"], v2["z_event"]))
    dot_different = sum(a * b for a, b in zip(v1["z_event"], v3["z_event"]))
    assert dot_similar > dot_different
