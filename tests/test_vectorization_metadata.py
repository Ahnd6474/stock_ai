from datetime import datetime, timezone

from kswing_sentinel.text_encoder import DEFAULT_KOREAN_ROBERTA_MODEL_ID, KoreanTextEncoder
from kswing_sentinel.vectorization import VectorizationPipeline


def test_roberta_encoder_metadata_defaults_to_roberta_or_hashing_fallback():
    encoder = KoreanTextEncoder(backend="roberta")
    metadata = encoder.metadata()

    assert metadata["model_id"] == DEFAULT_KOREAN_ROBERTA_MODEL_ID
    assert metadata["encoder_version"].startswith("ko_roberta")
    assert metadata["embedding_backend"] in {"hashing_bow_v1", "roberta_mean_pool_v1"}


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
        "roberta_mean_pool_v1",
        "transformers_mean_pool_v1",
    }
    assert v["metadata"]["encoder_version"].startswith("ko_roberta")
    assert v["metadata"]["tokenizer_version"] in {"auto", DEFAULT_KOREAN_ROBERTA_MODEL_ID}


def test_vectorization_semantic_similarity_signal():
    pipeline = VectorizationPipeline()
    v1 = pipeline.build("Samsung Electronics earnings outlook improved after memory pricing recovery.")
    v2 = pipeline.build("Samsung Electronics profit expectations improved on stronger memory demand.")
    v3 = pipeline.build("Oil prices surged after a supply disruption in the Middle East.")
    dot_similar = sum(a * b for a, b in zip(v1["z_event"], v2["z_event"]))
    dot_different = sum(a * b for a, b in zip(v1["z_event"], v3["z_event"]))
    assert dot_similar > dot_different
