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

    def apply(self, decisions: list[TradeDecision], current_gross: float, current_turnover: float = 0.0) -> list[TradeDecision]:
        out: list[TradeDecision] = []
        remaining_gross = max(0.0, self.constraints.gross_cap - current_gross)
        remaining_turnover = max(0.0, self.constraints.turnover_cap - current_turnover)
        for d in decisions:
            d.target_weight = min(d.target_weight, self.constraints.single_name_cap)
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
