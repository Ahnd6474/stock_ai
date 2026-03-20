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
        return self.choose_with_rationale(ctx)[0]

    def choose_with_rationale(self, ctx: VenueContext) -> tuple[str, list[str]]:
        rationale: list[str] = []
        elig = self.eligibility_store.get(ctx.eligibility_version, ctx.symbol)
        if elig != "KRX_PLUS_NXT" or not ctx.broker_supports_nxt:
            rationale.append("NXT_NOT_ELIGIBLE_OR_UNSUPPORTED")
            return "KRX", rationale
        if not ctx.venue_freshness_ok or not ctx.session_liquidity_ok:
            rationale.append("VENUE_STATE_UNCERTAIN_FAIL_CLOSED")
            return "KRX", rationale

        krx_cost = self.cost_model.estimate("KRX", ctx.session_type, participation=0.03).total_bps
        nxt_cost = self.cost_model.estimate("NXT", ctx.session_type, participation=0.03).total_bps

        if ctx.session_type in {"NXT_PRE", "NXT_AFTER", "CLOSE_PRICE"}:
            if nxt_cost <= krx_cost + 2.0:
                rationale.append("NXT_SESSION_ADVANTAGE")
                return "NXT", rationale
            rationale.append("KRX_COST_PREFERRED")
            return "KRX", rationale

        if abs(krx_cost - nxt_cost) <= 1.0:
            rationale.append("COST_GAP_SMALL_DEFAULT_KRX")
            return "KRX", rationale
        chosen = "NXT" if nxt_cost < krx_cost else "KRX"
        rationale.append("LOWER_EXPECTED_COST")
        return chosen, rationale
