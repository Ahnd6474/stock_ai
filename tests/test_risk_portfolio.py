from datetime import datetime, timezone

from kswing_sentinel.risk_engine import RiskEngine, MarketRiskState
from kswing_sentinel.portfolio_engine import PortfolioEngine
from kswing_sentinel.schemas import TradeDecision


def _decision(weight: float) -> TradeDecision:
    return TradeDecision(
        symbol="005930",
        action="BUY",
        target_weight=weight,
        tranche_ratio=0.4,
        session_type="CORE_DAY",
        selected_venue="KRX",
        rationale_codes=[],
        as_of_time=datetime(2026, 3, 20, tzinfo=timezone.utc),
        execution_time=datetime(2026, 3, 20, tzinfo=timezone.utc),
    )


def test_market_risk_off_vetoes_buy():
    eng = RiskEngine()
    d = eng.apply(_decision(0.03), MarketRiskState(market_risk_off=True, portfolio_beta=0.8, beta_cap=1.0))
    assert d.action == "NO_TRADE"


def test_portfolio_caps_single_name():
    pe = PortfolioEngine()
    [d] = pe.apply([_decision(0.2)], current_gross=0.1)
    assert d.target_weight <= 0.08
