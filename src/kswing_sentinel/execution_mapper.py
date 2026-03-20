from __future__ import annotations

from datetime import timedelta
from zoneinfo import ZoneInfo

from .calendar import TradingCalendar
from .cost_model import SessionCostModel
from .schemas import ExecutionPlan, ExecutionRequest
from .session_rules import classify_session, round_to_next_5m

KST = ZoneInfo("Asia/Seoul")


def _phase_end_minutes(session_type: str) -> int:
    return {
        "NXT_PRE": 8 * 60 + 50,
        "CORE_DAY": 15 * 60 + 20,
        "CLOSE_PRICE": 16 * 60,
        "NXT_AFTER": 20 * 60,
    }.get(session_type, 24 * 60)


class ExecutionMapper:
    def __init__(
        self,
        calendar: TradingCalendar | None = None,
        cost_model: SessionCostModel | None = None,
        broker_cutoff_minutes: int = 3,
    ) -> None:
        self.calendar = calendar or TradingCalendar()
        self.cost_model = cost_model or SessionCostModel()
        self.broker_cutoff_minutes = broker_cutoff_minutes

    def map_execution(self, req: ExecutionRequest) -> ExecutionPlan:
        session = classify_session(req.decision_timestamp, self.calendar)
        ts = req.decision_timestamp
        selected_venue = "KRX"
        rollover_reason = None

        if session == "OFF_MARKET":
            ts = ts + timedelta(hours=12)
            while classify_session(ts, self.calendar) == "OFF_MARKET":
                ts += timedelta(minutes=5)
            session = classify_session(ts, self.calendar)
            rollover_reason = "OFF_MARKET_ROLLOVER"

        local = ts.astimezone(KST)
        minutes = local.hour * 60 + local.minute
        phase_end = _phase_end_minutes(session)
        if phase_end - minutes <= self.broker_cutoff_minutes:
            ts += timedelta(minutes=5)
            while classify_session(ts, self.calendar) == session:
                ts += timedelta(minutes=5)
            session = classify_session(ts, self.calendar)
            rollover_reason = "BROKER_CUTOFF_ROLLOVER"

        if req.venue_eligibility == "KRX_PLUS_NXT" and req.broker_supports_nxt and req.venue_freshness_ok and req.session_liquidity_ok:
            if session in {"NXT_PRE", "NXT_AFTER", "CLOSE_PRICE", "CORE_DAY"}:
                selected_venue = "NXT"
        if not req.venue_freshness_ok:
            selected_venue = "KRX"
            rollover_reason = rollover_reason or "VENUE_STALE_FALLBACK"
        elif not req.session_liquidity_ok:
            selected_venue = "KRX"
            rollover_reason = rollover_reason or "LOW_LIQUIDITY_FALLBACK"
        elif req.venue_eligibility == "KRX_ONLY":
            selected_venue = "KRX"
            rollover_reason = rollover_reason or "KRX_ONLY_POLICY"
        elif not req.broker_supports_nxt:
            selected_venue = "KRX"
            rollover_reason = rollover_reason or "BROKER_NXT_UNSUPPORTED"

        exec_time = round_to_next_5m(ts)
        cost_bps = self.cost_model.estimate(selected_venue, session, participation=0.03).total_bps
        return ExecutionPlan(
            symbol=req.symbol,
            selected_venue=selected_venue,
            selected_session_type=session,
            scheduled_exec_time=exec_time,
            expected_cost_bps=cost_bps,
            rollover_reason=rollover_reason,
        )
