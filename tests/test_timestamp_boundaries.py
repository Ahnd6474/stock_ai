from datetime import datetime
from zoneinfo import ZoneInfo

from kswing_sentinel.execution_mapper import ExecutionMapper
from kswing_sentinel.schemas import ExecutionRequest

KST = ZoneInfo("Asia/Seoul")


def test_rollover_near_phase_end():
    mapper = ExecutionMapper()
    req = ExecutionRequest(
        symbol="005930",
        decision_timestamp=datetime(2026,3,20,15,18,tzinfo=KST),
        venue_eligibility="KRX_PLUS_NXT",
        broker_supports_nxt=True,
        venue_freshness_ok=True,
        session_liquidity_ok=True,
    )
    plan = mapper.map_execution(req)
    assert plan.rollover_reason == "PHASE_END_ROLLOVER"
