from datetime import datetime, timezone

from kswing_sentinel.vectorization import VectorizationPipeline


def test_vectorization_returns_metadata_versions():
    v = VectorizationPipeline().build(
        "실적 개선 공시",
        source_doc_ids=["doc1"],
        cluster_ids=["cl1"],
        as_of_time=datetime(2026, 3, 20, tzinfo=timezone.utc),
    )
    assert len(v["z_event"]) == 64
    assert v["metadata"]["embedding_backend"] == "hashing_bow_v1"
    assert v["metadata"]["encoder_version"].startswith("ko_bert")


def test_vectorization_semantic_similarity_signal():
    pipeline = VectorizationPipeline()
    v1 = pipeline.build("삼성전자 실적 개선 전망")
    v2 = pipeline.build("삼성전자 실적 호전 전망")
    v3 = pipeline.build("유가 급등으로 항공주 약세")
    dot_similar = sum(a * b for a, b in zip(v1["z_event"], v2["z_event"]))
    dot_different = sum(a * b for a, b in zip(v1["z_event"], v3["z_event"]))
    assert dot_similar > dot_different
