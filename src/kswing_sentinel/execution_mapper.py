from __future__ import annotations

from datetime import datetime, time, timedelta
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
        "CLOSE_PRICE": 15 * 60 + 40,
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

    def _next_tradable_minute(self, ts: datetime) -> datetime:
        probe = ts.astimezone(KST).replace(second=0, microsecond=0)
        if self.calendar.is_tradable_minute(probe):
            return probe
        probe += timedelta(minutes=1)
        while not self.calendar.is_tradable_minute(probe):
            probe += timedelta(minutes=1)
        return probe

    def _roll_to_next_session(self, ts: datetime, current_session: str) -> datetime:
        probe = ts.astimezone(KST).replace(second=0, microsecond=0) + timedelta(minutes=1)
        while True:
            if self.calendar.is_tradable_minute(probe) and classify_session(probe, self.calendar) != current_session:
                return probe
            probe += timedelta(minutes=1)

    def _next_core_day_start(self, ts: datetime) -> datetime:
        local = ts.astimezone(KST)
        cur_date = local.date()
        if self.calendar.is_trading_day(cur_date) and local.time() < time(9, 0):
            target_date = cur_date
        else:
            target_date = self.calendar.next_trading_day(cur_date)
        return datetime.combine(target_date, time(9, 0), tzinfo=KST)

    def _nxt_allowed(self, req: ExecutionRequest) -> bool:
        return (
            req.venue_eligibility == "KRX_PLUS_NXT"
            and req.broker_supports_nxt
            and req.venue_freshness_ok
            and req.session_liquidity_ok
            and req.venue_availability_ok
            and req.venue_clock_ok
        )

    def map_execution(self, req: ExecutionRequest) -> ExecutionPlan:
        session = classify_session(req.decision_timestamp, self.calendar)
        ts = req.decision_timestamp
        selected_venue = "KRX"
        rollover_reason = None
        venue_uncertain = not (
            req.venue_freshness_ok
            and req.session_liquidity_ok
            and req.venue_availability_ok
            and req.venue_clock_ok
        )
        cutoff_minutes = max(self.broker_cutoff_minutes, int(req.broker_cutoff_minutes))

        if session == "OFF_MARKET":
            ts = self._next_tradable_minute(ts)
            session = classify_session(ts, self.calendar)
            rollover_reason = "OFF_MARKET_ROLLOVER"

        local = ts.astimezone(KST)
        minutes = local.hour * 60 + local.minute
        phase_end = _phase_end_minutes(session)
        if phase_end - minutes <= cutoff_minutes:
            ts = self._roll_to_next_session(ts, session)
            session = classify_session(ts, self.calendar)
            rollover_reason = "BROKER_CUTOFF_ROLLOVER"

        nxt_allowed = self._nxt_allowed(req)
        if session == "CORE_DAY":
            selected_venue = "KRX"
            if nxt_allowed:
                krx_cost = self.cost_model.estimate(
                    "KRX",
                    session,
                    participation=0.03,
                    liquidity_bucket=req.liquidity_bucket,
                ).total_bps
                nxt_cost = self.cost_model.estimate(
                    "NXT",
                    session,
                    participation=0.03,
                    liquidity_bucket=req.liquidity_bucket,
                ).total_bps
                if nxt_cost + 1.0 < krx_cost:
                    selected_venue = "NXT"
            if venue_uncertain:
                selected_venue = "KRX"
                rollover_reason = rollover_reason or "VENUE_UNCERTAIN_FAIL_CLOSED"
        else:
            if nxt_allowed:
                selected_venue = "NXT"
            else:
                ts = self._next_core_day_start(ts)
                session = "CORE_DAY"
                selected_venue = "KRX"
                if venue_uncertain:
                    rollover_reason = rollover_reason or "VENUE_UNCERTAIN_FAIL_CLOSED"
                elif req.venue_eligibility == "KRX_ONLY":
                    rollover_reason = rollover_reason or "KRX_ONLY_OFFCORE_ROLLOVER"
                elif not req.broker_supports_nxt:
                    rollover_reason = rollover_reason or "BROKER_NXT_UNSUPPORTED"
                else:
                    rollover_reason = rollover_reason or "NXT_UNAVAILABLE_ROLL_TO_KRX_CORE"

        exec_time = round_to_next_5m(ts)
        exec_session = classify_session(exec_time, self.calendar)
        if exec_session == "OFF_MARKET" or exec_session != session:
            ts = self._next_tradable_minute(exec_time)
            session = classify_session(ts, self.calendar)
            exec_time = round_to_next_5m(ts)

        cost_bps = self.cost_model.estimate(
            selected_venue,
            session,
            participation=0.03,
            liquidity_bucket=req.liquidity_bucket,
        ).total_bps
        available_order_modes = {
            "NXT_PRE": ["REGULAR"],
            "CORE_DAY": ["REGULAR"],
            "CLOSE_PRICE": ["CLOSE_PRICE"],
            "NXT_AFTER": ["AFTER_MARKET"],
        }.get(session, ["REGULAR"])
        return ExecutionPlan(
            symbol=req.symbol,
            selected_venue=selected_venue,
            selected_session_type=session,
            scheduled_exec_time=exec_time,
            expected_cost_bps=cost_bps,
            rollover_reason=rollover_reason,
            available_order_modes=available_order_modes,
            venue_uncertain=venue_uncertain,
            cost_model_version=self.cost_model.cost_model_version,
        )
