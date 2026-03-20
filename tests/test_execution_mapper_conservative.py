from datetime import datetime
from zoneinfo import ZoneInfo

from kswing_sentinel.execution_mapper import ExecutionMapper
from kswing_sentinel.schemas import ExecutionRequest

KST = ZoneInfo("Asia/Seoul")


def test_core_day_prefers_krx_when_nxt_cost_gap_is_small():
    mapper = ExecutionMapper()
    plan = mapper.map_execution(
        ExecutionRequest(
            symbol="005930",
            decision_timestamp=datetime(2026, 3, 20, 9, 35, tzinfo=KST),
            venue_eligibility="KRX_PLUS_NXT",
            broker_supports_nxt=True,
            venue_freshness_ok=True,
            session_liquidity_ok=True,
        )
    )
    assert plan.selected_session_type == "CORE_DAY"
    assert plan.selected_venue == "KRX"


def test_offcore_krx_only_rolls_to_next_core_day():
    mapper = ExecutionMapper()
    plan = mapper.map_execution(
        ExecutionRequest(
            symbol="005930",
            decision_timestamp=datetime(2026, 3, 20, 15, 45, tzinfo=KST),
            venue_eligibility="KRX_ONLY",
            broker_supports_nxt=False,
            venue_freshness_ok=True,
            session_liquidity_ok=True,
        )
    )
    assert plan.selected_venue == "KRX"
    assert plan.selected_session_type == "CORE_DAY"
    assert plan.rollover_reason == "KRX_ONLY_OFFCORE_ROLLOVER"
    assert plan.scheduled_exec_time.hour == 9
    assert plan.scheduled_exec_time.minute == 5
