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


def test_broker_replace_and_reconcile():
    now = datetime(2026, 3, 20, tzinfo=timezone.utc)
    gw = BrokerGateway(BrokerCapabilities(supports_nxt=True, supports_after_market=True))
    rep = gw.submit(
        OrderRequest("005930", "BUY", 100, "KRX", None, now),
        market_price=100.0,
        liquidity_score=0.3,
    )
    assert rep.status == "PARTIAL"

    rep2 = gw.replace(rep.order_id, new_qty=120, at=now)
    assert rep2.status == "REPLACED"

    rep3 = gw.reconcile(rep.order_id, at=now)
    assert rep3.status in {"PARTIAL", "NEW"}


def test_broker_rejects_unknown_tif():
    now = datetime(2026, 3, 20, tzinfo=timezone.utc)
    gw = BrokerGateway(BrokerCapabilities(supports_nxt=True, supports_after_market=True))
    rep = gw.submit(
        OrderRequest("005930", "BUY", 100, "KRX", None, now, tif="GTC"),
        market_price=100.0,
        liquidity_score=1.0,
    )
    assert rep.status == "REJECTED"
