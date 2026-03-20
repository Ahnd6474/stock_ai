from __future__ import annotations

from .schemas import ExecutionPlan, FusedPrediction, TradeDecision


class DecisionEngine:
    def decide(self, pred: FusedPrediction, plan: ExecutionPlan, *, trend_120m_ok: bool, tech_extension_high: bool,
               market_risk_off: bool, no_position: bool) -> TradeDecision:
        vetoes: list[str] = []
        if market_risk_off or pred.regime_final == "risk_off":
            action = "NO_TRADE" if no_position else "REDUCE"
            size = 0.0
            rationale = ["RISK_VETO"]
            vetoes.append("MARKET_RISK_OFF")
        elif pred.flow_persist >= 0.62 and pred.event_score >= 0.15 and trend_120m_ok:
            if tech_extension_high:
                action = "WAIT_PULLBACK" if no_position else "BUY_HALF"
                size = 0.01
                rationale = ["FLOW_EVENT_TREND_OK", "SHORT_TERM_EXTENSION_HIGH"]
            else:
                action = "BUY"
                size = 0.02
                rationale = ["FLOW_EVENT_TREND_OK"]
            size *= max(0.2, 1.0 - pred.dd_20d)
            size *= max(0.2, 1.0 - pred.uncertainty)
        else:
            action = "NO_TRADE"
            size = 0.0
            rationale = ["DIRECTION_NOT_CONFIRMED"]
            vetoes.append("DIRECTION_GATE_FAILED")

        tranche = {"BUY": 0.40, "BUY_HALF": 0.30}.get(action, 0.0)
        return TradeDecision(
            symbol=pred.symbol,
            action=action,
            target_weight=size,
            tranche_ratio=tranche,
            session_type=plan.selected_session_type,
            selected_venue=plan.selected_venue,
            rationale_codes=rationale,
            as_of_time=pred.as_of_time,
            execution_time=plan.scheduled_exec_time,
            vetoes_triggered=vetoes,
            risk_budget_used=size,
            expected_slippage_bps=plan.expected_cost_bps,
            exit_policy_hint="time_stop_or_thesis_break",
        )
