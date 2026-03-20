from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class DecisionAuditEntry:
    symbol: str
    decision_time: datetime
    model_version: str
    prompt_version: str
    vectorizer_version: str
    source_doc_ids: list[str]
    cluster_ids: list[str]
    selected_venue: str
    rationale_codes: list[str]


class AuditLogStore:
    def __init__(self) -> None:
        self.entries: list[DecisionAuditEntry] = []

    def append(self, entry: DecisionAuditEntry) -> None:
        self.entries.append(entry)

    def latest_for_symbol(self, symbol: str) -> DecisionAuditEntry | None:
        rows = [e for e in self.entries if e.symbol == symbol]
        if not rows:
            return None
        return max(rows, key=lambda e: e.decision_time)
