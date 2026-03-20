from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .cost_model import SessionCostModel
from .execution_mapper import ExecutionMapper
from .schemas import ExecutionRequest


@dataclass
class FeatureRow:
    symbol: str
    timestamp: datetime
    as_of_time: datetime


class NoLookaheadError(RuntimeError):
    pass


@dataclass(frozen=True)
class Bar:
    symbol: str
    timestamp: datetime
    close: float
    session_type: str


@dataclass(frozen=True)
class TradeResult:
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    gross_return: float
    net_return: float
    horizon_interrupted: bool


class Backtester:
    def __init__(self, mapper: ExecutionMapper | None = None, costs: SessionCostModel | None = None) -> None:
        self.mapper = mapper or ExecutionMapper()
        self.costs = costs or SessionCostModel()

    def validate_no_lookahead(self, rows: list[FeatureRow]) -> None:
        bad = [r for r in rows if r.timestamp > r.as_of_time]
        if bad:
            raise NoLookaheadError(f"Found {len(bad)} leaking rows")

    def run_trade(
        self,
        req: ExecutionRequest,
        as_of_time: datetime,
        bars: list[Bar],
        horizon_bars: int = 20,
    ) -> TradeResult | None:
        self.validate_no_lookahead([FeatureRow(req.symbol, as_of_time, as_of_time)])
        plan = self.mapper.map_execution(req)

        tradable = [b for b in bars if b.symbol == req.symbol and b.timestamp >= plan.scheduled_exec_time]
        if len(tradable) < 2:
            return None

        entry_bar = tradable[0]
        target_ix = min(len(tradable) - 1, horizon_bars)
        exit_bar = tradable[target_ix]
        interrupted = target_ix < horizon_bars

        gross = (exit_bar.close - entry_bar.close) / entry_bar.close
        entry_cost = self.costs.estimate(plan.selected_venue, plan.selected_session_type, participation=0.03).total_bps / 1e4
        exit_cost = self.costs.estimate(plan.selected_venue, exit_bar.session_type, participation=0.03).total_bps / 1e4
        net = gross - entry_cost - exit_cost

        return TradeResult(
            symbol=req.symbol,
            entry_time=entry_bar.timestamp,
            exit_time=exit_bar.timestamp,
            entry_price=entry_bar.close,
            exit_price=exit_bar.close,
            gross_return=gross,
            net_return=net,
            horizon_interrupted=interrupted,
        )
