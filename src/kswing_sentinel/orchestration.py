from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from typing import Callable

from .live import LiveInferenceService


@dataclass(frozen=True)
class DeadLetterRecord:
    symbol: str
    anchor_time: datetime
    payload: dict
    features: dict
    venue_eligibility: str
    error_message: str
    attempts: int
    failed_at: datetime


class CircuitBreakerOpen(RuntimeError):
    pass


class TemporalLikeOrchestrator:
    def __init__(
        self,
        live: LiveInferenceService | None = None,
        *,
        backoff_schedule_sec: tuple[float, ...] = (0.0, 0.25, 1.0),
        circuit_breaker_threshold: int = 3,
        circuit_breaker_cooldown_sec: float = 300.0,
        semantic_refresh_interval_sec: float = 1800.0,
        sleep_fn: Callable[[float], None] | None = None,
        now_fn: Callable[[], datetime] | None = None,
    ) -> None:
        self.live = live or LiveInferenceService()
        self._idempotency_cache: dict[str, object] = {}
        self.backoff_schedule_sec = backoff_schedule_sec
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.circuit_breaker_cooldown = timedelta(seconds=circuit_breaker_cooldown_sec)
        self.semantic_refresh_interval = timedelta(seconds=semantic_refresh_interval_sec)
        self.sleep_fn = sleep_fn or (lambda _: None)
        self.now_fn = now_fn or datetime.utcnow
        self.dead_letter_queue: list[DeadLetterRecord] = []
        self._failure_streak = 0
        self._circuit_open_until: datetime | None = None
        self._last_semantic_refresh: dict[str, datetime] = {}

    def _assert_circuit_closed(self, anchor_time: datetime) -> None:
        if self._circuit_open_until is None:
            return
        if anchor_time >= self._circuit_open_until:
            self._circuit_open_until = None
            self._failure_streak = 0
            return
        raise CircuitBreakerOpen(f"circuit open until {self._circuit_open_until.isoformat()}")

    def _semantic_refresh_required(self, symbol: str, anchor_time: datetime) -> bool:
        last = self._last_semantic_refresh.get(symbol)
        return last is None or (anchor_time - last) >= self.semantic_refresh_interval

    def _record_failure(
        self,
        *,
        symbol: str,
        anchor_time: datetime,
        payload: dict,
        features: dict,
        venue_eligibility: str,
        attempts: int,
        error: Exception,
    ) -> None:
        self._failure_streak += 1
        if self._failure_streak >= self.circuit_breaker_threshold:
            self._circuit_open_until = anchor_time + self.circuit_breaker_cooldown
        self.dead_letter_queue.append(
            DeadLetterRecord(
                symbol=symbol,
                anchor_time=anchor_time,
                payload=dict(payload),
                features=dict(features),
                venue_eligibility=venue_eligibility,
                error_message=str(error),
                attempts=attempts,
                failed_at=self.now_fn(),
            )
        )

    def drain_dead_letters(self) -> list[DeadLetterRecord]:
        drained = list(self.dead_letter_queue)
        self.dead_letter_queue.clear()
        return drained

    def run_anchor(self, symbol: str, anchor_time: datetime, payload: dict, features: dict, venue_eligibility: str, retry: int = 1):
        idem_key = f"{symbol}:{anchor_time.isoformat()}"
        if idem_key in self._idempotency_cache:
            return self._idempotency_cache[idem_key]
        self._assert_circuit_closed(anchor_time)

        last_exc: Exception | None = None
        attempts = max(1, retry + 1)
        request_payload = dict(payload)
        request_payload.setdefault("semantic_refresh_required", self._semantic_refresh_required(symbol, anchor_time))
        for attempt in range(attempts):
            try:
                result = self.live.run_for_symbol(symbol, anchor_time, request_payload, features, venue_eligibility)
                self._idempotency_cache[idem_key] = result
                self._failure_streak = 0
                if request_payload.get("semantic_refresh_required"):
                    self._last_semantic_refresh[symbol] = anchor_time
                return result
            except Exception as exc:  # pragma: no cover - defensive runtime path
                last_exc = exc
                if attempt < attempts - 1:
                    backoff = self.backoff_schedule_sec[min(attempt, len(self.backoff_schedule_sec) - 1)]
                    if backoff > 0:
                        self.sleep_fn(backoff)

        if last_exc is not None:
            self._record_failure(
                symbol=symbol,
                anchor_time=anchor_time,
                payload=request_payload,
                features=features,
                venue_eligibility=venue_eligibility,
                attempts=attempts,
                error=last_exc,
            )
            raise last_exc
        raise RuntimeError("anchor run failed without explicit exception")
