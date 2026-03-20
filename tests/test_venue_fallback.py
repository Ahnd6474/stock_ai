from datetime import datetime
from zoneinfo import ZoneInfo

from kswing_sentinel.execution_mapper import ExecutionMapper
from kswing_sentinel.schemas import ExecutionRequest

KST = ZoneInfo("Asia/Seoul")


def test_fallback_to_krx_when_nxt_stale():
    mapper = ExecutionMapper()
    req = ExecutionRequest(
        symbol="005930",
        decision_timestamp=datetime(2026,3,20,15,45,tzinfo=KST),
        venue_eligibility="KRX_PLUS_NXT",
        broker_supports_nxt=True,
        venue_freshness_ok=False,
        session_liquidity_ok=True,
    )
    plan = mapper.map_execution(req)
    assert plan.selected_venue == "KRX"
