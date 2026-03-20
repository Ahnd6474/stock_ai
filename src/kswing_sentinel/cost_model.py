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

    def estimate(self, venue: VenueType, session: SessionType, participation: float) -> CostComponents:
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
        mult = 1.0 + max(0.0, participation - 0.05) * 8.0
        return CostComponents(
            commission_bps=c.commission_bps,
            tax_bps=c.tax_bps,
            spread_bps=c.spread_bps * mult,
            slippage_bps=c.slippage_bps * mult,
            impact_bps=c.impact_bps * mult,
        )
