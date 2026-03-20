from __future__ import annotations

from datetime import datetime, time, timedelta

from .calendar import KST, TradingCalendar
from .cost_model import SessionCostModel
from .schemas import ExecutionPlan, ExecutionRequest
from .session_rules import classify_session, round_to_next_5m


def _phase_end_minutes(session_type: str) -> int:
    return {
        "NXT_PRE": 8 * 60 + 50,
        "CORE_DAY": 15 * 60 + 20,
        "CLOSE_PRICE": 16 * 60,
        "NXT_AFTER": 20 * 60,
    }.get(session_type, 24 * 60)


class ExecutionMapper:
    def __init__(self, calendar: TradingCalendar | None = None, cost_model: SessionCostModel | None = None) -> None:
        self.calendar = calendar or TradingCalendar()
        self.cost_model = cost_model or SessionCostModel()

    def _roll_to_next_open(self, ts: datetime) -> datetime:
        local = ts.astimezone(KST)
        d = local.date()
        if not self.calendar.is_trading_day(d):
            d = self.calendar.next_trading_day(d)
            return datetime.combine(d, time(8, 0), tzinfo=KST)
        if local.time() >= time(20, 0):
            d = self.calendar.next_trading_day(d)
            return datetime.combine(d, time(8, 0), tzinfo=KST)
        return local

    def map_execution(self, req: ExecutionRequest) -> ExecutionPlan:
        ts = self._roll_to_next_open(req.decision_timestamp)
        session = classify_session(ts)
        selected_venue = "KRX"
        rollover_reason = None

        if session == "OFF_MARKET":
            ts = self._roll_to_next_open(ts + timedelta(minutes=5))
            while classify_session(ts) == "OFF_MARKET":
                ts += timedelta(minutes=5)
            session = classify_session(ts)
            rollover_reason = "OFF_MARKET_ROLLOVER"

        local = ts.astimezone(KST)
        minutes = local.hour * 60 + local.minute
        if _phase_end_minutes(session) - minutes <= 3:
            ts += timedelta(minutes=5)
            while classify_session(ts) == session:
                ts += timedelta(minutes=5)
            session = classify_session(ts)
            rollover_reason = "PHASE_END_ROLLOVER"

        if req.venue_eligibility == "KRX_PLUS_NXT" and req.broker_supports_nxt and req.venue_freshness_ok and req.session_liquidity_ok:
            if session in {"NXT_PRE", "NXT_AFTER", "CLOSE_PRICE"}:
                selected_venue = "NXT"

        if not req.venue_freshness_ok:
            selected_venue = "KRX"
            rollover_reason = rollover_reason or "VENUE_STALE_FALLBACK"

        exec_time = round_to_next_5m(ts)
        cost = self.cost_model.estimate(selected_venue, session, participation=0.03).total_bps
        return ExecutionPlan(
            symbol=req.symbol,
            selected_venue=selected_venue,
            selected_session_type=session,
            scheduled_exec_time=exec_time,
            expected_cost_bps=cost,
            rollover_reason=rollover_reason,
        )
