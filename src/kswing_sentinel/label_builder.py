from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

from .calendar import TradingCalendar
from .cost_model import SessionCostModel
from .execution_mapper import ExecutionMapper
from .schemas import ExecutionRequest
from .session_rules import classify_session


@dataclass(frozen=True)
class PricePoint:
    timestamp: datetime
    close: float


@dataclass(frozen=True)
class LabelBundle:
    er_5d: float | None
    er_20d: float | None
    dd_20d: float | None
    p_up_20d: float | None
    horizon_interrupted: bool
    selected_venue: str
    entry_session_type: str
    execution_timestamp: datetime
    entry_cost_bps: float
    exit_cost_bps: float


class LabelBuilder:
    def __init__(self, calendar: TradingCalendar, mapper: ExecutionMapper, costs: SessionCostModel) -> None:
        self.calendar = calendar
        self.mapper = mapper
        self.costs = costs

    @staticmethod
    def _sorted_prices(prices: list[PricePoint]) -> list[PricePoint]:
        return sorted(prices, key=lambda p: p.timestamp)

    def _entry_point(self, prices: list[PricePoint], exec_ts: datetime) -> PricePoint | None:
        return next((p for p in self._sorted_prices(prices) if p.timestamp >= exec_ts), None)

    def _exit_point(self, prices: list[PricePoint], trading_day: date) -> PricePoint | None:
        candidates = [p for p in self._sorted_prices(prices) if p.timestamp.date() >= trading_day]
        if not candidates:
            return None
        same_day = [p for p in candidates if p.timestamp.date() == candidates[0].timestamp.date()]
        if same_day:
            return max(same_day, key=lambda p: p.timestamp)
        return candidates[0]

    def _net_return(
        self,
        entry_px: float,
        exit_px: float,
        entry_venue: str,
        entry_session: str,
        exit_session: str,
        liquidity_bucket: str = "mid",
    ) -> tuple[float, float, float]:
        gross = (exit_px - entry_px) / entry_px
        entry_cost = self.costs.estimate_side(
            entry_venue,
            entry_session,
            participation=0.03,
            side="BUY",
            liquidity_bucket=liquidity_bucket,
        ).total_bps / 1e4
        exit_cost = self.costs.estimate_side(
            "KRX",
            exit_session,
            participation=0.03,
            side="SELL",
            liquidity_bucket=liquidity_bucket,
        ).total_bps / 1e4
        return gross - entry_cost - exit_cost, entry_cost * 1e4, exit_cost * 1e4

    def er_20d(self, symbol: str, decision_ts: datetime, prices: list[PricePoint], req: ExecutionRequest) -> float | None:
        plan = self.mapper.map_execution(req)
        entry_point = self._entry_point(prices, plan.scheduled_exec_time)
        if entry_point is None:
            return None
        entry = entry_point.close
        d20 = self.calendar.add_trading_days(decision_ts.date(), 20)
        exit_point = self._exit_point(prices, d20)
        if exit_point is None:
            return None
        exit_session = classify_session(exit_point.timestamp, self.calendar)
        if exit_session == "OFF_MARKET":
            exit_session = "CORE_DAY"
        value, _, _ = self._net_return(
            entry,
            exit_point.close,
            entry_venue=plan.selected_venue,
            entry_session=plan.selected_session_type,
            exit_session=exit_session,
            liquidity_bucket=req.liquidity_bucket,
        )
        return value

    def build(self, symbol: str, decision_ts: datetime, prices: list[PricePoint], req: ExecutionRequest) -> LabelBundle:
        plan = self.mapper.map_execution(req)
        sorted_prices = self._sorted_prices(prices)
        entry_point = self._entry_point(sorted_prices, plan.scheduled_exec_time)
        if entry_point is None:
            return LabelBundle(None, None, None, None, True, plan.selected_venue, plan.selected_session_type, plan.scheduled_exec_time, 0.0, 0.0)

        entry = entry_point.close
        d5 = self.calendar.add_trading_days(decision_ts.date(), 5)
        d20 = self.calendar.add_trading_days(decision_ts.date(), 20)
        exit_5d = self._exit_point(sorted_prices, d5)
        exit_20d = self._exit_point(sorted_prices, d20)

        er5 = None
        exit_cost_bps = 0.0
        if exit_5d is not None:
            exit_5d_session = classify_session(exit_5d.timestamp, self.calendar)
            if exit_5d_session == "OFF_MARKET":
                exit_5d_session = "CORE_DAY"
            er5, _, _ = self._net_return(
                entry,
                exit_5d.close,
                entry_venue=plan.selected_venue,
                entry_session=plan.selected_session_type,
                exit_session=exit_5d_session,
                liquidity_bucket=req.liquidity_bucket,
            )

        er20 = None
        dd20 = None
        p_up = None
        entry_cost_bps = self.costs.estimate_side(
            plan.selected_venue,
            plan.selected_session_type,
            participation=0.03,
            side="BUY",
            liquidity_bucket=req.liquidity_bucket,
        ).total_bps
        if exit_20d is not None:
            exit_20d_session = classify_session(exit_20d.timestamp, self.calendar)
            if exit_20d_session == "OFF_MARKET":
                exit_20d_session = "CORE_DAY"
            er20, _, exit_cost_bps = self._net_return(
                entry,
                exit_20d.close,
                entry_venue=plan.selected_venue,
                entry_session=plan.selected_session_type,
                exit_session=exit_20d_session,
                liquidity_bucket=req.liquidity_bucket,
            )
            window20 = [p.close for p in sorted_prices if plan.scheduled_exec_time <= p.timestamp and p.timestamp.date() <= d20]
            if window20:
                dd20 = max(0.0, (entry - min(window20)) / entry)
            p_up = 1.0 if er20 > 0 else 0.0
        return LabelBundle(
            er_5d=er5,
            er_20d=er20,
            dd_20d=dd20,
            p_up_20d=p_up,
            horizon_interrupted=exit_20d is None,
            selected_venue=plan.selected_venue,
            entry_session_type=plan.selected_session_type,
            execution_timestamp=plan.scheduled_exec_time,
            entry_cost_bps=entry_cost_bps,
            exit_cost_bps=exit_cost_bps,
        )
