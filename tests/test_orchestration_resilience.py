from datetime import datetime, timedelta, timezone

import pytest

from kswing_sentinel.orchestration import CircuitBreakerOpen, TemporalLikeOrchestrator


class FlakyLiveService:
    def __init__(self, failures_before_success: int) -> None:
        self.failures_before_success = failures_before_success
        self.calls = 0
        self.payloads: list[dict] = []

    def run_for_symbol(self, symbol, as_of_time, raw_event_payload, features, venue_eligibility):
        self.calls += 1
        self.payloads.append(dict(raw_event_payload))
        if self.calls <= self.failures_before_success:
            raise RuntimeError("transient failure")
        return {"symbol": symbol, "ok": True, "attempts": self.calls}


def test_orchestrator_retries_and_marks_semantic_refresh():
    service = FlakyLiveService(failures_before_success=1)
    waits: list[float] = []
    orchestrator = TemporalLikeOrchestrator(
        live=service,
        backoff_schedule_sec=(0.1, 0.2),
        sleep_fn=waits.append,
        semantic_refresh_interval_sec=60,
    )
    anchor = datetime(2026, 3, 20, 0, 0, tzinfo=timezone.utc)

    result = orchestrator.run_anchor("005930", anchor, {}, {"flow_strength": 0.1}, "KRX_ONLY", retry=1)
    assert result["ok"] is True
    assert service.calls == 2
    assert waits == [0.1]
    assert service.payloads[0]["semantic_refresh_required"] is True


def test_orchestrator_opens_circuit_and_writes_dead_letter():
    service = FlakyLiveService(failures_before_success=10)
    now = datetime(2026, 3, 20, 0, 0, tzinfo=timezone.utc)
    orchestrator = TemporalLikeOrchestrator(
        live=service,
        circuit_breaker_threshold=1,
        circuit_breaker_cooldown_sec=300,
        now_fn=lambda: now,
    )

    with pytest.raises(RuntimeError):
        orchestrator.run_anchor("005930", now, {}, {"flow_strength": 0.1}, "KRX_ONLY", retry=0)

    assert len(orchestrator.dead_letter_queue) == 1

    with pytest.raises(CircuitBreakerOpen):
        orchestrator.run_anchor("005930", now + timedelta(minutes=1), {}, {"flow_strength": 0.1}, "KRX_ONLY", retry=0)

    drained = orchestrator.drain_dead_letters()
    assert drained[0].attempts == 1
    assert orchestrator.dead_letter_queue == []
