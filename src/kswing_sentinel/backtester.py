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
    entry_session_type: str
    gross_return: float
    net_return: float
    filled_ratio: float
    horizon_interrupted: bool


@dataclass(frozen=True)
class PortfolioBacktestResult:
    results_by_symbol: dict[str, TradeResult]
    gross_return: float
    net_return: float
    session_breakdown: dict[str, dict[str, float]]


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
        target_participation: float = 0.03,
        max_participation: float = 0.1,
    ) -> TradeResult | None:
        self.validate_no_lookahead([FeatureRow(req.symbol, as_of_time, as_of_time)])
        plan = self.mapper.map_execution(req)

        tradable = [
            b
            for b in bars
            if b.symbol == req.symbol and b.timestamp >= plan.scheduled_exec_time and b.timestamp > req.decision_timestamp
        ]
        if len(tradable) < 2:
            return None

        entry_bar = tradable[0]
        target_ix = min(len(tradable) - 1, horizon_bars)
        exit_bar = tradable[target_ix]
        interrupted = target_ix < horizon_bars
        filled_ratio = min(1.0, max(0.0, target_participation / max(max_participation, 1e-6)))

        gross = (exit_bar.close - entry_bar.close) / entry_bar.close
        entry_cost = self.costs.estimate_side(
            plan.selected_venue,
            plan.selected_session_type,
            participation=target_participation,
            side="BUY",
        ).total_bps / 1e4
        exit_cost = self.costs.estimate_side(
            plan.selected_venue,
            exit_bar.session_type,
            participation=target_participation,
            side="SELL",
        ).total_bps / 1e4
        net = gross * filled_ratio - entry_cost - exit_cost

        return TradeResult(
            symbol=req.symbol,
            entry_time=entry_bar.timestamp,
            exit_time=exit_bar.timestamp,
            entry_price=entry_bar.close,
            exit_price=exit_bar.close,
            entry_session_type=plan.selected_session_type,
            gross_return=gross,
            net_return=net,
            filled_ratio=filled_ratio,
            horizon_interrupted=interrupted,
        )

    def run_portfolio(
        self,
        requests: list[ExecutionRequest],
        as_of_time: datetime,
        bars: list[Bar],
        horizon_bars: int = 20,
    ) -> PortfolioBacktestResult:
        results_by_symbol: dict[str, TradeResult] = {}
        session_breakdown: dict[str, dict[str, float]] = {}
        gross_total = 0.0
        net_total = 0.0

        for req in requests:
            result = self.run_trade(req, as_of_time, bars, horizon_bars=horizon_bars)
            if result is None:
                continue
            results_by_symbol[req.symbol] = result
            gross_total += result.gross_return
            net_total += result.net_return
            session_stats = session_breakdown.setdefault(
                result.entry_session_type,
                {"gross_return": 0.0, "net_return": 0.0, "count": 0.0},
            )
            session_stats["gross_return"] += result.gross_return
            session_stats["net_return"] += result.net_return
            session_stats["count"] += 1.0

        return PortfolioBacktestResult(
            results_by_symbol=results_by_symbol,
            gross_return=gross_total,
            net_return=net_total,
            session_breakdown=session_breakdown,
        )
