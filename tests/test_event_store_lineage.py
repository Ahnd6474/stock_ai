from datetime import datetime, timedelta, timezone

from kswing_sentinel.event_store import EventDoc, EventStore


def test_event_store_tracks_cluster_lineage_and_novelty():
    store = EventStore()
    t0 = datetime(2026, 3, 20, 0, 0, tzinfo=timezone.utc)
    store.add(
        EventDoc(
            doc_id="doc-1",
            symbol="005930",
            published_at=t0,
            first_seen_at=t0,
            retrieved_at=t0,
            available_at=t0,
            body="Samsung guidance improved on strong memory demand",
            source_type="NEWS",
            source_quality_score=0.9,
        )
    )
    store.add(
        EventDoc(
            doc_id="doc-2",
            symbol="005930",
            published_at=t0 + timedelta(minutes=5),
            first_seen_at=t0 + timedelta(minutes=5),
            retrieved_at=t0 + timedelta(minutes=5),
            available_at=t0 + timedelta(minutes=5),
            body="Samsung guidance improved with stronger memory demand outlook",
            source_type="NEWS",
            source_quality_score=0.8,
        )
    )

    docs = store.available_docs("005930", t0 + timedelta(minutes=10))
    assert docs[1].cluster_id == docs[0].cluster_id
    assert docs[1].source_lineage == ("doc-1",)
    assert docs[1].novelty_score < 1.0


def test_event_store_dedup_available_docs_keeps_latest_canonical():
    store = EventStore()
    t0 = datetime(2026, 3, 20, 0, 0, tzinfo=timezone.utc)
    body = "Order backlog improved materially."
    store.add(
        EventDoc(
            doc_id="doc-a",
            symbol="005930",
            published_at=t0,
            first_seen_at=t0,
            retrieved_at=t0,
            available_at=t0,
            body=body,
            source_type="FILING",
            source_quality_score=1.0,
        )
    )
    store.add(
        EventDoc(
            doc_id="doc-b",
            symbol="005930",
            published_at=t0 + timedelta(minutes=2),
            first_seen_at=t0 + timedelta(minutes=2),
            retrieved_at=t0 + timedelta(minutes=2),
            available_at=t0 + timedelta(minutes=2),
            body=body,
            source_type="FILING",
            source_quality_score=1.0,
        )
    )

    deduped = store.dedup_available_docs("005930", t0 + timedelta(minutes=3))
    assert len(deduped) == 1
    assert deduped[0].doc_id == "doc-b"


def test_event_store_delta_docs_focus_on_new_and_updated_clusters():
    store = EventStore()
    t0 = datetime(2026, 3, 20, 9, 0, tzinfo=timezone.utc)
    store.add(
        EventDoc(
            doc_id="doc-1",
            symbol="005930",
            published_at=t0,
            first_seen_at=t0,
            retrieved_at=t0,
            available_at=t0,
            body="Initial backlog update",
            source_type="NEWS",
            source_quality_score=0.8,
            canonical_event_id="evt-backlog",
            cluster_id="cluster-backlog",
        )
    )
    store.add(
        EventDoc(
            doc_id="doc-2",
            symbol="005930",
            published_at=t0 + timedelta(minutes=10),
            first_seen_at=t0 + timedelta(minutes=10),
            retrieved_at=t0 + timedelta(minutes=10),
            available_at=t0 + timedelta(minutes=10),
            body="Initial backlog update revised with stronger demand",
            source_type="NEWS",
            source_quality_score=0.9,
            canonical_event_id="evt-backlog",
            cluster_id="cluster-backlog",
        )
    )
    store.add(
        EventDoc(
            doc_id="doc-3",
            symbol="005930",
            published_at=t0 + timedelta(minutes=12),
            first_seen_at=t0 + timedelta(minutes=12),
            retrieved_at=t0 + timedelta(minutes=12),
            available_at=t0 + timedelta(minutes=12),
            body="New buyback headline",
            source_type="NEWS",
            source_quality_score=0.7,
            canonical_event_id="evt-buyback",
            cluster_id="cluster-buyback",
        )
    )

    delta_docs = store.delta_docs("005930", t0 + timedelta(minutes=15), since_time=t0 + timedelta(minutes=5))
    delta_summary = store.delta_summary("005930", t0 + timedelta(minutes=15), since_time=t0 + timedelta(minutes=5))

    assert [doc.doc_id for doc in delta_docs] == ["doc-2", "doc-3"]
    assert delta_summary["delta_doc_count"] == 2.0
    assert delta_summary["new_doc_count"] == 1.0
    assert delta_summary["updated_doc_count"] == 1.0
