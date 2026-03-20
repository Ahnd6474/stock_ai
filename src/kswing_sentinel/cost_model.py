from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SessionType = Literal["NXT_PRE", "CORE_DAY", "CLOSE_PRICE", "NXT_AFTER", "OFF_MARKET"]
VenueType = Literal["KRX", "NXT"]


@dataclass(frozen=True)
class CostComponents:
    commission_bps: float
    tax_bps: float
    spread_bps: float
    slippage_bps: float
    impact_bps: float

    @property
    def total_bps(self) -> float:
        return self.commission_bps + self.tax_bps + self.spread_bps + self.slippage_bps + self.impact_bps


class SessionCostModel:
    """Conservative default curves; to be replaced by broker/vendor calibrated curves."""

    def __init__(self, cost_model_version: str = "v2") -> None:
        self.cost_model_version = cost_model_version

    def estimate(
        self,
        venue: VenueType,
        session: SessionType,
        participation: float,
        liquidity_bucket: str = "mid",
    ) -> CostComponents:
        base = {
            ("KRX", "CORE_DAY"): CostComponents(1.0, 18.0, 4.0, 3.0, 2.0),
            ("NXT", "NXT_PRE"): CostComponents(1.2, 18.0, 8.0, 7.0, 5.0),
            ("NXT", "NXT_AFTER"): CostComponents(1.2, 18.0, 9.0, 8.0, 6.0),
            ("NXT", "CLOSE_PRICE"): CostComponents(1.2, 18.0, 6.0, 5.0, 4.0),
            ("NXT", "CORE_DAY"): CostComponents(1.2, 18.0, 5.0, 4.0, 3.0),
            ("KRX", "NXT_PRE"): CostComponents(1.0, 18.0, 10.0, 9.0, 7.0),
            ("KRX", "NXT_AFTER"): CostComponents(1.0, 18.0, 10.0, 9.0, 7.0),
            ("KRX", "CLOSE_PRICE"): CostComponents(1.0, 18.0, 7.0, 6.0, 4.0),
        }
        c = base.get((venue, session), CostComponents(1.5, 18.0, 10.0, 10.0, 8.0))
        liq_mult = {"high": 0.85, "mid": 1.0, "low": 1.25}.get(liquidity_bucket, 1.0)
        part = max(0.0, participation)
        # non-linear impact penalty for aggressive participation
        part_mult = 1.0 + 4.0 * (part**1.3)
        return CostComponents(
            commission_bps=c.commission_bps,
            tax_bps=c.tax_bps,
            spread_bps=c.spread_bps * liq_mult * part_mult,
            slippage_bps=c.slippage_bps * liq_mult * part_mult,
            impact_bps=c.impact_bps * liq_mult * (1.0 + 6.0 * (part**1.5)),
        )

    def estimate_side(
        self,
        venue: VenueType,
        session: SessionType,
        participation: float,
        side: str,
        liquidity_bucket: str = "mid",
    ) -> CostComponents:
        base = self.estimate(venue, session, participation, liquidity_bucket=liquidity_bucket)
        sell_tax = base.tax_bps if side.upper() == "SELL" else max(0.0, base.tax_bps - 18.0)
        return CostComponents(
            commission_bps=base.commission_bps,
            tax_bps=sell_tax,
            spread_bps=base.spread_bps,
            slippage_bps=base.slippage_bps,
            impact_bps=base.impact_bps,
        )
