from __future__ import annotations

from dataclasses import dataclass

from .schemas import TradeDecision


@dataclass
class PortfolioConstraints:
    single_name_cap: float = 0.08
    sector_cap: float = 0.30
    gross_cap: float = 0.95
    turnover_cap: float = 0.25


class PortfolioEngine:
    def __init__(self, constraints: PortfolioConstraints | None = None) -> None:
        self.constraints = constraints or PortfolioConstraints()

    def apply(
        self,
        decisions: list[TradeDecision],
        current_gross: float,
        current_turnover: float = 0.0,
        sector_of: dict[str, str] | None = None,
        sector_exposure: dict[str, float] | None = None,
        correlation_to_book: dict[str, float] | None = None,
        liquidity_cap: dict[str, float] | None = None,
    ) -> list[TradeDecision]:
        out: list[TradeDecision] = []
        sector_of = sector_of or {}
        sector_exposure = sector_exposure or {}
        correlation_to_book = correlation_to_book or {}
        liquidity_cap = liquidity_cap or {}

        remaining_gross = max(0.0, self.constraints.gross_cap - current_gross)
        remaining_turnover = max(0.0, self.constraints.turnover_cap - current_turnover)
        prioritized = sorted(decisions, key=lambda d: d.target_weight, reverse=True)
        for d in prioritized:
            d.target_weight = min(d.target_weight, self.constraints.single_name_cap)
            d.target_weight = min(d.target_weight, liquidity_cap.get(d.symbol, self.constraints.single_name_cap))
            if correlation_to_book.get(d.symbol, 0.0) > 0.8:
                d.target_weight *= 0.7
                d.rationale_codes.append("CORRELATION_CAP_THROTTLE")
            sec = sector_of.get(d.symbol)
            if sec is not None:
                sec_used = sector_exposure.get(sec, 0.0)
                sec_room = max(0.0, self.constraints.sector_cap - sec_used)
                if d.target_weight > sec_room:
                    d.target_weight = sec_room
                    d.rationale_codes.append("SECTOR_CAP_THROTTLE")
                sector_exposure[sec] = sec_used + d.target_weight
            if d.target_weight > remaining_gross:
                d.target_weight = remaining_gross
                if d.target_weight <= 0:
                    d.action = "NO_TRADE"
            if d.target_weight > remaining_turnover:
                d.target_weight = remaining_turnover
                d.rationale_codes.append("TURNOVER_CAP_THROTTLE")
            remaining_gross -= d.target_weight
            remaining_turnover -= d.target_weight
            out.append(d)
        return out
