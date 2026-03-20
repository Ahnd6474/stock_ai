from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .live import LiveInferenceService


@dataclass(frozen=True)
class AnchorRunRequest:
    symbol: str
    anchor_time: datetime
    venue_eligibility: str
    event_payload: dict
    idempotency_key: str


class TemporalLikeOrchestrator:
    def __init__(self) -> None:
        self.live = LiveInferenceService()
        self._seen: set[str] = set()

    def run_anchor(self, req: AnchorRunRequest):
        if req.idempotency_key in self._seen:
            return None
        self._seen.add(req.idempotency_key)
        return self.live.run_for_symbol(
            req.symbol,
            req.anchor_time,
            req.event_payload,
            req.venue_eligibility,
        )
