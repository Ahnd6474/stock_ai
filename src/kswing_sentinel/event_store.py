from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib


@dataclass(frozen=True)
class EventDoc:
    doc_id: str
    symbol: str
    published_at: datetime
    first_seen_at: datetime
    retrieved_at: datetime
    available_at: datetime
    body: str
    source_type: str
    source_quality_score: float
    canonical_event_id: str = ""
    cluster_id: str = ""
    source_lineage: list[str] | None = None
    novelty_score: float = 0.5


class EventStore:
    def __init__(self) -> None:
        self.docs: list[EventDoc] = []

    def add(self, doc: EventDoc) -> None:
        canonical = doc.canonical_event_id or self._canonical_event_id(doc.symbol, doc.body)
        cluster = doc.cluster_id or canonical[:12]
        self.docs.append(
            EventDoc(
                doc_id=doc.doc_id,
                symbol=doc.symbol,
                published_at=doc.published_at,
                first_seen_at=doc.first_seen_at,
                retrieved_at=doc.retrieved_at,
                available_at=doc.available_at,
                body=doc.body,
                source_type=doc.source_type,
                source_quality_score=doc.source_quality_score,
                canonical_event_id=canonical,
                cluster_id=cluster,
                source_lineage=doc.source_lineage or [doc.source_type],
                novelty_score=doc.novelty_score,
            )
        )

    def available_docs(self, symbol: str, as_of_time: datetime) -> list[EventDoc]:
        return [d for d in self.docs if d.symbol == symbol and d.available_at <= as_of_time]

    def dedup_available_docs(self, symbol: str, as_of_time: datetime) -> list[EventDoc]:
        docs = self.available_docs(symbol, as_of_time)
        latest_by_canonical: dict[str, EventDoc] = {}
        for d in docs:
            cur = latest_by_canonical.get(d.canonical_event_id)
            if cur is None or d.available_at > cur.available_at:
                latest_by_canonical[d.canonical_event_id] = d
        return list(latest_by_canonical.values())

    @staticmethod
    def _canonical_event_id(symbol: str, body: str) -> str:
        digest = hashlib.sha256(f"{symbol}:{body.strip().lower()}".encode("utf-8")).hexdigest()
        return f"evt_{digest[:24]}"
