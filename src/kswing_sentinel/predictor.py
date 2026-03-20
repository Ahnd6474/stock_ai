from __future__ import annotations

from datetime import datetime

from .schemas import FusedPrediction


class NumericFirstPredictor:
    def predict(self, symbol: str, session_type: str, as_of_time: datetime, features: dict) -> FusedPrediction:
        flow_strength = float(features.get("flow_strength", 0.0))
        trend_120m = float(features.get("trend_120m", 0.0))
        extension = float(features.get("extension_60m", 0.0))
        event_score = float(features.get("event_score", 0.0))
        uncertainty = min(1.0, max(0.0, 0.35 + 0.25 * abs(extension)))
        er20 = 0.01 + 0.02 * flow_strength + 0.015 * trend_120m + 0.01 * event_score - 0.01 * max(extension, 0)
        er5 = er20 * 0.45
        dd20 = max(0.01, 0.05 - 0.02 * trend_120m + 0.015 * max(extension, 0))
        pup = min(0.95, max(0.05, 0.5 + er20 * 2.0))
        flow_persist = min(1.0, max(0.0, 0.5 + flow_strength * 0.4))
        regime = "risk_off" if features.get("market_risk_off") else ("trend" if trend_120m > 0 else "chop")
        return FusedPrediction(
            symbol=symbol,
            as_of_time=as_of_time,
            session_type=session_type,
            er_5d=er5,
            er_20d=er20,
            dd_20d=dd20,
            p_up_20d=pup,
            flow_persist=flow_persist,
            uncertainty=uncertainty,
            regime_final=regime,
            event_score=event_score,
        )
