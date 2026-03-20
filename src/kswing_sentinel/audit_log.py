from __future__ import annotations

import json
from dataclasses import dataclass
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any


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


@dataclass(frozen=True)
class RuntimeAuditEvent:
    event_type: str
    event_time: datetime
    payload: dict[str, Any]


class AuditLogStore:
    def __init__(self, jsonl_path: str | Path | None = None) -> None:
        self.entries: list[DecisionAuditEntry] = []
        self.jsonl_path = Path(jsonl_path) if jsonl_path else None
        if self.jsonl_path is not None:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, entry: DecisionAuditEntry) -> None:
        self.entries.append(entry)
        self._write_jsonl(
            {
                "record_type": "decision",
                **self._serialize(asdict(entry)),
            }
        )

    def latest_for_symbol(self, symbol: str) -> DecisionAuditEntry | None:
        rows = [e for e in self.entries if e.symbol == symbol]
        if not rows:
            return None
        return max(rows, key=lambda e: e.decision_time)

    def append_runtime_event(self, event: RuntimeAuditEvent) -> None:
        self._write_jsonl(
            {
                "record_type": "runtime_event",
                **self._serialize(asdict(event)),
            }
        )

    def is_writable(self) -> bool:
        if self.jsonl_path is None:
            return True
        try:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with self.jsonl_path.open("a", encoding="utf-8"):
                pass
            return True
        except OSError:
            return False

    def _write_jsonl(self, payload: dict[str, Any]) -> None:
        if self.jsonl_path is None:
            return
        with self.jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _serialize(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {key: self._serialize_value(value) for key, value in payload.items()}

    def _serialize_value(self, value: Any) -> Any:
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._serialize_value(v) for v in value]
        return value
