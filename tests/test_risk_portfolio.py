from datetime import datetime, timezone

from kswing_sentinel.risk_engine import RiskEngine, MarketRiskState
from kswing_sentinel.portfolio_engine import PortfolioEngine
from kswing_sentinel.schemas import TradeDecision


def _decision(weight: float, symbol: str = "005930") -> TradeDecision:
    return TradeDecision(
        symbol=symbol,
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


def test_risk_engine_hard_stop_and_uncertainty_shrink():
    eng = RiskEngine()
    d1 = eng.apply(
        _decision(0.03),
        MarketRiskState(market_risk_off=False, portfolio_beta=0.8, beta_cap=1.0, hard_stop_triggered=True),
    )
    assert d1.action == "SELL"

    d2 = eng.apply(
        _decision(0.05),
        MarketRiskState(
            market_risk_off=False,
            portfolio_beta=0.8,
            beta_cap=1.0,
            uncertainty=0.9,
            session_type="NXT_AFTER",
        ),
    )
    assert d2.target_weight < 0.05
    assert "UNCERTAINTY_SIZE_SHRINK" in d2.rationale_codes


def test_portfolio_sector_corr_and_liquidity_caps():
    pe = PortfolioEngine()
    d1 = _decision(0.08, symbol="005930")
    d2 = _decision(0.08, symbol="000660")
    out = pe.apply(
        [d1, d2],
        current_gross=0.1,
        sector_of={"005930": "IT", "000660": "IT"},
        sector_exposure={"IT": 0.27},
        correlation_to_book={"005930": 0.9, "000660": 0.2},
        liquidity_cap={"005930": 0.03, "000660": 0.05},
    )
    out_map = {d.symbol: d for d in out}
    assert out_map["005930"].target_weight <= 0.03
    assert out_map["000660"].target_weight <= 0.03
