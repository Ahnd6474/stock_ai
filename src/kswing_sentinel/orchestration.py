from __future__ import annotations

from datetime import datetime

from .live import LiveInferenceService


class TemporalLikeOrchestrator:
    def __init__(self) -> None:
        self.live = LiveInferenceService()
        self._idempotency_cache: dict[str, object] = {}
        self._consecutive_failures = 0
        self._breaker_open = False

    def run_anchor(self, symbol: str, anchor_time: datetime, payload: dict, features: dict, venue_eligibility: str, retry: int = 1):
        if self._breaker_open:
            raise RuntimeError("circuit breaker is open")
        idem_key = f"{symbol}:{anchor_time.isoformat()}"
        if idem_key in self._idempotency_cache:
            return self._idempotency_cache[idem_key]

        last_exc: Exception | None = None
        for _ in range(max(1, retry + 1)):
            try:
                result = self.live.run_for_symbol(symbol, anchor_time, payload, features, venue_eligibility)
                self._idempotency_cache[idem_key] = result
                self._consecutive_failures = 0
                return result
            except Exception as exc:  # pragma: no cover - defensive runtime path
                last_exc = exc
                self._consecutive_failures += 1
                if self._consecutive_failures >= 3:
                    self._breaker_open = True

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("anchor run failed without explicit exception")

    def reset_circuit_breaker(self) -> None:
        self._breaker_open = False
        self._consecutive_failures = 0
