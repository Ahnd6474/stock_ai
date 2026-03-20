from datetime import datetime
from zoneinfo import ZoneInfo

from kswing_sentinel.backtester import Backtester, Bar
from kswing_sentinel.schemas import ExecutionRequest

KST = ZoneInfo("Asia/Seoul")


def test_backtester_trade_has_cost_and_horizon_flag():
    bt = Backtester()
    req = ExecutionRequest(
        symbol="005930",
        decision_timestamp=datetime(2026, 3, 20, 9, 35, tzinfo=KST),
        venue_eligibility="KRX_ONLY",
        broker_supports_nxt=False,
        venue_freshness_ok=True,
        session_liquidity_ok=True,
    )
    bars = [
        Bar("005930", datetime(2026, 3, 20, 9, 40, tzinfo=KST), 100.0, "CORE_DAY"),
        Bar("005930", datetime(2026, 3, 21, 9, 40, tzinfo=KST), 101.0, "CORE_DAY"),
        Bar("005930", datetime(2026, 3, 22, 9, 40, tzinfo=KST), 102.0, "CORE_DAY"),
    ]
    tr = bt.run_trade(req, req.decision_timestamp, bars, horizon_bars=5)
    assert tr is not None
    assert tr.horizon_interrupted is True
    assert tr.net_return < (102.0 - 100.0) / 100.0
