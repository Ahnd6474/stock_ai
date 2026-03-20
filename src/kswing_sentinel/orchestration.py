from __future__ import annotations

from datetime import datetime

from .live import LiveInferenceService


class TemporalLikeOrchestrator:
    def __init__(self) -> None:
        self.live = LiveInferenceService()

    def run_anchor(self, symbol: str, anchor_time: datetime, payload: dict, features: dict, venue_eligibility: str):
        # idempotency/retry hooks would be attached here in real Temporal workflow
        return self.live.run_for_symbol(symbol, anchor_time, payload, features, venue_eligibility)
