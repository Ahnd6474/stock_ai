from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class BrokerCapabilities:
    supports_nxt: bool
    supports_after_market: bool
    order_cutoff_minutes: int = 1


@dataclass
class OrderRequest:
    symbol: str
    side: str
    qty: int
    venue: str
    limit_price: float | None
    submitted_at: datetime
    tif: str = "DAY"


@dataclass
class ExecutionReport:
    order_id: str
    filled_qty: int
    avg_price: float
    status: str
    venue: str
    received_at: datetime


class BrokerGateway:
    def __init__(self, capabilities: BrokerCapabilities) -> None:
        self.capabilities = capabilities
        self._counter = 0
        self._open_orders: dict[str, OrderRequest] = {}

    def submit(self, order: OrderRequest, market_price: float, liquidity_score: float) -> ExecutionReport:
        self._counter += 1
        oid = f"ORD-{self._counter:07d}"
        self._open_orders[oid] = order
        if order.venue == "NXT" and not self.capabilities.supports_nxt:
            return ExecutionReport(oid, 0, 0.0, "REJECTED", order.venue, order.submitted_at)
        fill_ratio = 1.0 if liquidity_score >= 0.6 else max(0.2, liquidity_score)
        filled = int(order.qty * fill_ratio)
        if filled == 0:
            return ExecutionReport(oid, 0, 0.0, "NEW", order.venue, order.submitted_at)
        slip = 0.0006 if order.venue == "KRX" else 0.0012
        avg = market_price * (1 + slip if order.side == "BUY" else 1 - slip)
        status = "FILLED" if filled == order.qty else "PARTIAL"
        if status == "FILLED":
            self._open_orders.pop(oid, None)
        return ExecutionReport(oid, filled, avg, status, order.venue, order.submitted_at)

    def cancel(self, order_id: str, at: datetime) -> ExecutionReport:
        order = self._open_orders.pop(order_id, None)
        if order is None:
            return ExecutionReport(order_id, 0, 0.0, "REJECTED", "KRX", at)
        return ExecutionReport(order_id, 0, 0.0, "CANCELED", order.venue, at)
