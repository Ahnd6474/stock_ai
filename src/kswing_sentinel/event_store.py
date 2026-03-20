from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


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


class EventStore:
    def __init__(self) -> None:
        self.docs: list[EventDoc] = []

    def add(self, doc: EventDoc) -> None:
        self.docs.append(doc)

    def available_docs(self, symbol: str, as_of_time: datetime) -> list[EventDoc]:
        return [d for d in self.docs if d.symbol == symbol and d.available_at <= as_of_time]
