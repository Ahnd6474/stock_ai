from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import re
from statistics import mean


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
        for doc in docs:
            current = latest_by_canonical.get(doc.canonical_event_id)
            if current is None or doc.available_at > current.available_at:
                latest_by_canonical[doc.canonical_event_id] = doc
        return list(latest_by_canonical.values())

    def cluster_available_docs(self, symbol: str, as_of_time: datetime) -> dict[str, list[EventDoc]]:
        out: dict[str, list[EventDoc]] = {}
        for doc in self.available_docs(symbol, as_of_time):
            out.setdefault(doc.cluster_id, []).append(doc)
        return out

    def latest_docs_by_cluster(self, symbol: str, as_of_time: datetime) -> dict[str, EventDoc]:
        latest_by_cluster: dict[str, EventDoc] = {}
        for cluster_id, docs in self.cluster_available_docs(symbol, as_of_time).items():
            latest_by_cluster[cluster_id] = max(docs, key=lambda doc: (doc.available_at, doc.retrieved_at, doc.first_seen_at))
        return latest_by_cluster

    def delta_docs(self, symbol: str, as_of_time: datetime, since_time: datetime | None = None) -> list[EventDoc]:
        current_by_cluster = self.latest_docs_by_cluster(symbol, as_of_time)
        if since_time is None:
            return sorted(current_by_cluster.values(), key=lambda doc: (doc.available_at, doc.retrieved_at))
        previous_by_cluster = self.latest_docs_by_cluster(symbol, since_time)
        delta: list[EventDoc] = []
        for cluster_id, current in current_by_cluster.items():
            previous = previous_by_cluster.get(cluster_id)
            if previous is None:
                delta.append(current)
                continue
            body_changed = current.body.strip() != previous.body.strip()
            lineage_changed = current.source_lineage != previous.source_lineage
            metadata_changed = (
                current.doc_id != previous.doc_id
                and (
                    current.available_at > previous.available_at
                    or current.retrieved_at > previous.retrieved_at
                    or current.first_seen_at > previous.first_seen_at
                )
            )
            if body_changed or lineage_changed or metadata_changed:
                delta.append(current)
        return sorted(delta, key=lambda doc: (doc.available_at, doc.retrieved_at))

    def delta_summary(self, symbol: str, as_of_time: datetime, since_time: datetime | None = None) -> dict[str, float]:
        delta_docs = self.delta_docs(symbol, as_of_time, since_time=since_time)
        if not delta_docs:
            return {
                "delta_doc_count": 0.0,
                "new_doc_count": 0.0,
                "updated_doc_count": 0.0,
                "delta_novelty_mean": 0.0,
                "delta_source_quality_mean": 0.0,
                "delta_freshness_mean": 0.0,
                "time_since_last_collection_sec": max(0.0, (as_of_time - since_time).total_seconds()) if since_time is not None else 0.0,
            }
        previous_by_cluster = self.latest_docs_by_cluster(symbol, since_time) if since_time is not None else {}
        new_doc_count = 0
        updated_doc_count = 0
        freshness_scores: list[float] = []
        for doc in delta_docs:
            if doc.cluster_id in previous_by_cluster:
                updated_doc_count += 1
            else:
                new_doc_count += 1
            age_seconds = max(0.0, (as_of_time - doc.available_at).total_seconds())
            freshness_scores.append(max(0.0, 1.0 - min(1.0, age_seconds / 86400.0)))
        return {
            "delta_doc_count": float(len(delta_docs)),
            "new_doc_count": float(new_doc_count),
            "updated_doc_count": float(updated_doc_count),
            "delta_novelty_mean": mean(doc.novelty_score for doc in delta_docs),
            "delta_source_quality_mean": mean(doc.source_quality_score for doc in delta_docs),
            "delta_freshness_mean": mean(freshness_scores),
            "time_since_last_collection_sec": max(0.0, (as_of_time - since_time).total_seconds()) if since_time is not None else 0.0,
        }

    def lineage_for_cluster(self, symbol: str, cluster_id: str) -> list[EventDoc]:
        return sorted(
            [doc for doc in self.docs if doc.symbol == symbol and doc.cluster_id == cluster_id],
            key=lambda doc: doc.available_at,
        )

    @staticmethod
    def _canonical_event_id(symbol: str, body: str) -> str:
        digest = hashlib.sha256(f"{symbol}:{body.strip().lower()}".encode("utf-8")).hexdigest()
        return f"evt_{digest[:24]}"

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return set(re.findall(r"[A-Za-z0-9_]+|[\uac00-\ud7a3]+", text.lower()))

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
