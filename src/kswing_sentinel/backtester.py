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


@dataclass(frozen=True)
class Bar:
    symbol: str
    timestamp: datetime
    close: float
    session_type: str
    tradable: bool = True


@dataclass(frozen=True)
class BacktestTrade:
    symbol: str
    decision_ts: datetime
    entry_ts: datetime
    exit_ts: datetime
    entry_px: float
    exit_px: float
    net_return: float
    horizon_interrupted: bool


class NoLookaheadError(RuntimeError):
    pass


class Backtester:
    def __init__(self, mapper: ExecutionMapper | None = None, costs: SessionCostModel | None = None) -> None:
        self.mapper = mapper or ExecutionMapper()
        self.costs = costs or SessionCostModel()

    def validate_no_lookahead(self, rows: list[FeatureRow]) -> None:
        bad = [r for r in rows if r.timestamp > r.as_of_time]
        if bad:
            raise NoLookaheadError(f"Found {len(bad)} leaking rows")

    def run_trade(self, req: ExecutionRequest, decision_ts: datetime, bars: list[Bar], horizon_bars: int = 20) -> BacktestTrade | None:
        plan = self.mapper.map_execution(req)
        tradable = [b for b in bars if b.symbol == req.symbol and b.tradable]
        future = [b for b in tradable if b.timestamp >= plan.scheduled_exec_time]
        if not future:
            return None
        entry_bar = future[0]

        from_entry = [b for b in tradable if b.timestamp > entry_bar.timestamp]
        interrupted = len(from_entry) < horizon_bars
        if not from_entry:
            return None
        exit_idx = min(horizon_bars - 1, len(from_entry) - 1)
        exit_bar = from_entry[exit_idx]

        gross = (exit_bar.close - entry_bar.close) / entry_bar.close
        c = self.costs.estimate(plan.selected_venue, entry_bar.session_type, participation=0.03).total_bps / 1e4
        net = gross - c
        return BacktestTrade(
            symbol=req.symbol,
            decision_ts=decision_ts,
            entry_ts=entry_bar.timestamp,
            exit_ts=exit_bar.timestamp,
            entry_px=entry_bar.close,
            exit_px=exit_bar.close,
            net_return=net,
            horizon_interrupted=interrupted,
        )
