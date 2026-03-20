from datetime import datetime
from zoneinfo import ZoneInfo

from kswing_sentinel.calendar import TradingCalendar
from kswing_sentinel.cost_model import SessionCostModel
from kswing_sentinel.execution_mapper import ExecutionMapper
from kswing_sentinel.label_builder import LabelBuilder, PricePoint
from kswing_sentinel.schemas import ExecutionRequest

KST = ZoneInfo("Asia/Seoul")


def test_label_builder_marks_corporate_action_adjustment():
    cal = TradingCalendar()
    mapper = ExecutionMapper(calendar=cal, cost_model=SessionCostModel())

    def adjuster(symbol: str, prices: list[PricePoint]) -> list[PricePoint]:
        return [
            PricePoint(p.timestamp, p.close / 10.0 if p.timestamp.date().month == 3 else p.close)
            for p in prices
        ]

    lb = LabelBuilder(cal, mapper, SessionCostModel(), corporate_action_adjuster=adjuster)
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
        PricePoint(datetime(2026, 3, 20, 9, 40, tzinfo=KST), 1000.0),
        PricePoint(datetime(2026, 3, 27, 15, 20, tzinfo=KST), 103.0),
        PricePoint(datetime(2026, 4, 17, 15, 20, tzinfo=KST), 108.0),
    ]

    bundle = lb.build("005930", decision_ts, prices, req)
    assert bundle.corporate_action_adjusted is True


def test_label_builder_censors_halted_horizon():
    cal = TradingCalendar()
    mapper = ExecutionMapper(calendar=cal, cost_model=SessionCostModel())

    def halt_checker(symbol: str, start: datetime, end: datetime) -> bool:
        return True

    lb = LabelBuilder(cal, mapper, SessionCostModel(), halt_suspension_checker=halt_checker)
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
    assert bundle.horizon_interrupted is True
    assert bundle.censored_reason == "TRADING_HALT_OR_SUSPENSION"
    assert bundle.er_20d is None
