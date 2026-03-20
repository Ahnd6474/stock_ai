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
    assert v["metadata"]["encoder_version"].startswith("ko_bert")
