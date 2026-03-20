from __future__ import annotations

from dataclasses import dataclass

from .schemas import TradeDecision


@dataclass
class MarketRiskState:
    market_risk_off: bool
    portfolio_beta: float
    beta_cap: float
    liquidity_deteriorated: bool = False
    venue_uncertain: bool = False
    predicted_drawdown: float = 0.0


class RiskEngine:
    def apply(self, decision: TradeDecision, state: MarketRiskState) -> TradeDecision:
        if state.market_risk_off and decision.action in {"BUY", "BUY_HALF"}:
            decision.action = "NO_TRADE"
            decision.target_weight = 0.0
            decision.rationale_codes.append("MARKET_RISK_OFF_VETO")
        if state.liquidity_deteriorated and decision.action in {"BUY", "BUY_HALF"}:
            decision.action = "WAIT_PULLBACK"
            decision.target_weight = min(decision.target_weight, 0.003)
            decision.rationale_codes.append("LIQUIDITY_DETERIORATION_VETO")
        if state.venue_uncertain and decision.action in {"BUY", "BUY_HALF"}:
            decision.action = "NO_TRADE"
            decision.target_weight = 0.0
            decision.rationale_codes.append("VENUE_UNCERTAINTY_VETO")
        if state.portfolio_beta > state.beta_cap and decision.action in {"BUY", "BUY_HALF"}:
            decision.action = "WAIT_PULLBACK"
            decision.target_weight = min(decision.target_weight, 0.005)
            decision.rationale_codes.append("BETA_CAP_THROTTLE")
        if state.predicted_drawdown > 0.08 and decision.action in {"BUY", "BUY_HALF", "WAIT_PULLBACK"}:
            decision.target_weight *= 0.5
            decision.rationale_codes.append("PREDICTED_DD_SIZE_SHRINK")
        return decision
