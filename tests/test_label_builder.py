from datetime import datetime
from zoneinfo import ZoneInfo

from kswing_sentinel.calendar import TradingCalendar
from kswing_sentinel.execution_mapper import ExecutionMapper
from kswing_sentinel.label_builder import LabelBuilder, PricePoint
from kswing_sentinel.cost_model import SessionCostModel
from kswing_sentinel.schemas import ExecutionRequest

KST = ZoneInfo("Asia/Seoul")


def test_er20_label_uses_execution_mapper_and_costs():
    cal = TradingCalendar()
    mapper = ExecutionMapper(calendar=cal, cost_model=SessionCostModel())
    lb = LabelBuilder(cal, mapper, SessionCostModel())
    decision_ts = datetime(2026, 3, 20, 9, 35, tzinfo=KST)
    req = ExecutionRequest(
        symbol="005930",
        decision_timestamp=decision_ts,
        venue_eligibility="KRX_ONLY",
        broker_supports_nxt=False,
        venue_freshness_ok=True,
        session_liquidity_ok=True,
    )
    prices = [
        PricePoint(datetime(2026, 3, 20, 9, 40, tzinfo=KST), 100.0),
        PricePoint(datetime(2026, 4, 17, 15, 20, tzinfo=KST), 108.0),
    ]
    v = lb.er_20d("005930", decision_ts, prices, req)
    assert v is not None
    assert v < 0.08  # costs deducted from gross 8%


def test_build_returns_net_labels_and_cost_metadata():
    cal = TradingCalendar()
    mapper = ExecutionMapper(calendar=cal, cost_model=SessionCostModel())
    lb = LabelBuilder(cal, mapper, SessionCostModel())
    decision_ts = datetime(2026, 3, 20, 9, 35, tzinfo=KST)
    req = ExecutionRequest(
        symbol="005930",
        decision_timestamp=decision_ts,
        venue_eligibility="KRX_ONLY",
        broker_supports_nxt=False,
        venue_freshness_ok=True,
        session_liquidity_ok=True,
    )
    prices = [
        PricePoint(datetime(2026, 3, 20, 9, 40, tzinfo=KST), 100.0),
        PricePoint(datetime(2026, 3, 27, 15, 20, tzinfo=KST), 103.0),
        PricePoint(datetime(2026, 4, 17, 15, 20, tzinfo=KST), 108.0),
    ]
    bundle = lb.build("005930", decision_ts, prices, req)
    assert bundle.er_5d is not None
    assert bundle.er_20d is not None
    assert bundle.er_5d < 0.03
    assert bundle.er_20d < 0.08
    assert bundle.entry_cost_bps > 0
    assert bundle.exit_cost_bps > 0
