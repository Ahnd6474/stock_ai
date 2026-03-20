from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import re


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
    source_lineage: tuple[str, ...] = ()
    novelty_score: float = 1.0


class EventStore:
    def __init__(self) -> None:
        self.docs: list[EventDoc] = []

    def add(self, doc: EventDoc) -> None:
        canonical = doc.canonical_event_id or self._canonical_event_id(doc.symbol, doc.body)
        best_match, best_similarity = self._best_cluster_match(doc.symbol, doc.body)
        cluster = doc.cluster_id or (best_match.cluster_id if best_match and best_similarity >= 0.45 else canonical[:12])
        lineage = doc.source_lineage or tuple(d.doc_id for d in self.docs if d.symbol == doc.symbol and d.cluster_id == cluster)
        novelty = doc.novelty_score if doc.novelty_score != 1.0 else max(0.0, 1.0 - best_similarity)
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
                source_lineage=lineage,
                novelty_score=novelty,
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

    def cluster_available_docs(self, symbol: str, as_of_time: datetime) -> dict[str, list[EventDoc]]:
        out: dict[str, list[EventDoc]] = {}
        for doc in self.available_docs(symbol, as_of_time):
            out.setdefault(doc.cluster_id, []).append(doc)
        return out

    def lineage_for_cluster(self, symbol: str, cluster_id: str) -> list[EventDoc]:
        return sorted(
            [d for d in self.docs if d.symbol == symbol and d.cluster_id == cluster_id],
            key=lambda d: d.available_at,
        )

    @staticmethod
    def _canonical_event_id(symbol: str, body: str) -> str:
        digest = hashlib.sha256(f"{symbol}:{body.strip().lower()}".encode("utf-8")).hexdigest()
        return f"evt_{digest[:24]}"

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return set(re.findall(r"[가-힣A-Za-z0-9_]+", text.lower()))

    def _best_cluster_match(self, symbol: str, body: str) -> tuple[EventDoc | None, float]:
        target = self._tokens(body)
        best_doc: EventDoc | None = None
        best_similarity = 0.0
        for doc in self.docs:
            if doc.symbol != symbol:
                continue
            base = self._tokens(doc.body)
            union = base | target
            if not union:
                continue
            similarity = len(base & target) / len(union)
            if similarity > best_similarity:
                best_doc = doc
                best_similarity = similarity
        return best_doc, best_similarity
