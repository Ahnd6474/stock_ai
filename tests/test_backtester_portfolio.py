from datetime import datetime
from zoneinfo import ZoneInfo

from kswing_sentinel.backtester import Backtester, Bar
from kswing_sentinel.schemas import ExecutionRequest

KST = ZoneInfo("Asia/Seoul")


def test_backtester_can_run_portfolio_summary():
    backtester = Backtester()
    as_of = datetime(2026, 3, 20, 9, 35, tzinfo=KST)
    requests = [
        ExecutionRequest(
            symbol="005930",
            decision_timestamp=as_of,
            venue_eligibility="KRX_ONLY",
            broker_supports_nxt=False,
            venue_freshness_ok=True,
            session_liquidity_ok=True,
        ),
        ExecutionRequest(
            symbol="000660",
            decision_timestamp=as_of,
            venue_eligibility="KRX_ONLY",
            broker_supports_nxt=False,
            venue_freshness_ok=True,
            session_liquidity_ok=True,
        ),
    ]
    bars = [
        Bar("005930", datetime(2026, 3, 20, 9, 40, tzinfo=KST), 100.0, "CORE_DAY"),
        Bar("005930", datetime(2026, 3, 21, 9, 40, tzinfo=KST), 102.0, "CORE_DAY"),
        Bar("000660", datetime(2026, 3, 20, 9, 40, tzinfo=KST), 50.0, "CORE_DAY"),
        Bar("000660", datetime(2026, 3, 21, 9, 40, tzinfo=KST), 49.0, "CORE_DAY"),
    ]

    result = backtester.run_portfolio(requests, as_of, bars, horizon_bars=1)
    assert set(result.results_by_symbol) == {"005930", "000660"}
    assert "CORE_DAY" in result.session_breakdown
    assert result.session_breakdown["CORE_DAY"]["count"] == 2.0
