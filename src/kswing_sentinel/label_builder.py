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
