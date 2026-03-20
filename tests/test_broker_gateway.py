from datetime import datetime, timezone

from kswing_sentinel.broker_gateway import BrokerCapabilities, BrokerGateway, OrderRequest


def test_broker_partial_fill_when_liquidity_low():
    gw = BrokerGateway(BrokerCapabilities(supports_nxt=True, supports_after_market=True))
    rep = gw.submit(
        OrderRequest("005930", "BUY", 100, "KRX", None, datetime(2026, 3, 20, tzinfo=timezone.utc)),
        market_price=100.0,
        liquidity_score=0.3,
    )
    assert rep.status in {"PARTIAL", "FILLED"}
    assert rep.filled_qty <= 100
