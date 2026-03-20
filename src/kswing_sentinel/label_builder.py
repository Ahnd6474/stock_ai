from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .calendar import TradingCalendar
from .cost_model import SessionCostModel
from .execution_mapper import ExecutionMapper
from .schemas import ExecutionRequest


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

    def er_20d(self, symbol: str, decision_ts: datetime, prices: list[PricePoint], req: ExecutionRequest) -> float | None:
        plan = self.mapper.map_execution(req)
        future = [p for p in prices if p.timestamp >= plan.scheduled_exec_time]
        if not future:
            return None
        entry = future[0].close
        d20 = self.calendar.add_trading_days(decision_ts.date(), 20)
        horizon = [p for p in prices if p.timestamp.date() >= d20]
        if not horizon:
            return None
        exit_px = horizon[0].close
        gross = (exit_px - entry) / entry
        cost = self.costs.estimate(plan.selected_venue, plan.selected_session_type, participation=0.03).total_bps / 1e4
        return gross - cost

    def build(self, symbol: str, decision_ts: datetime, prices: list[PricePoint], req: ExecutionRequest) -> LabelBundle:
        plan = self.mapper.map_execution(req)
        future = [p for p in prices if p.timestamp >= plan.scheduled_exec_time]
        if not future:
            return LabelBundle(None, None, None, None, True, plan.selected_venue, plan.selected_session_type, plan.scheduled_exec_time, 0.0, 0.0)

        entry = future[0].close
        d5 = self.calendar.add_trading_days(decision_ts.date(), 5)
        d20 = self.calendar.add_trading_days(decision_ts.date(), 20)
        px5 = next((p.close for p in prices if p.timestamp.date() >= d5), None)
        px20 = next((p.close for p in prices if p.timestamp.date() >= d20), None)

        er5 = None if px5 is None else (px5 - entry) / entry
        er20 = None if px20 is None else (px20 - entry) / entry
        dd20 = None
        p_up = None
        if px20 is not None:
            window20 = [p.close for p in prices if plan.scheduled_exec_time <= p.timestamp and p.timestamp.date() <= d20]
            if window20:
                dd20 = max(0.0, (entry - min(window20)) / entry)
            p_up = 1.0 if px20 > entry else 0.0

        entry_cost_bps = self.costs.estimate(plan.selected_venue, plan.selected_session_type, participation=0.03).total_bps
        exit_cost_bps = self.costs.estimate(plan.selected_venue, plan.selected_session_type, participation=0.03).total_bps
        return LabelBundle(
            er_5d=er5,
            er_20d=er20,
            dd_20d=dd20,
            p_up_20d=p_up,
            horizon_interrupted=px20 is None,
            selected_venue=plan.selected_venue,
            entry_session_type=plan.selected_session_type,
            execution_timestamp=plan.scheduled_exec_time,
            entry_cost_bps=entry_cost_bps,
            exit_cost_bps=exit_cost_bps,
        )
