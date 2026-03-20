from __future__ import annotations

from dataclasses import dataclass

from .cost_model import SessionCostModel
from .nxt_eligibility_store import NXTEligibilityStore
from .schemas import SessionType


@dataclass(frozen=True)
class VenueContext:
    symbol: str
    session_type: SessionType
    eligibility_version: str
    broker_supports_nxt: bool
    venue_freshness_ok: bool
    session_liquidity_ok: bool


class VenueRouter:
    def __init__(self, eligibility_store: NXTEligibilityStore, cost_model: SessionCostModel) -> None:
        self.eligibility_store = eligibility_store
        self.cost_model = cost_model

    def choose(self, ctx: VenueContext) -> str:
        elig = self.eligibility_store.get(ctx.eligibility_version, ctx.symbol)
        if elig != "KRX_PLUS_NXT" or not ctx.broker_supports_nxt:
            return "KRX"
        if not ctx.venue_freshness_ok or not ctx.session_liquidity_ok:
            return "KRX"

        krx_cost = self.cost_model.estimate("KRX", ctx.session_type, participation=0.03).total_bps
        nxt_cost = self.cost_model.estimate("NXT", ctx.session_type, participation=0.03).total_bps

        if ctx.session_type in {"NXT_PRE", "NXT_AFTER", "CLOSE_PRICE"}:
            return "NXT" if nxt_cost <= krx_cost + 2.0 else "KRX"
        return "KRX" if abs(krx_cost - nxt_cost) <= 1.0 else ("NXT" if nxt_cost < krx_cost else "KRX")
