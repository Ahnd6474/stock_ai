from __future__ import annotations

from datetime import datetime

from .decision_engine import DecisionEngine
from .execution_mapper import ExecutionMapper
from .llm_event_normalizer import LLMEventNormalizer
from .predictor import NumericFirstPredictor
from .schemas import ExecutionRequest
from .session_rules import classify_session
from .vectorization import VectorizationPipeline


class LiveInferenceService:
    def __init__(self) -> None:
        self.mapper = ExecutionMapper()
        self.normalizer = LLMEventNormalizer()
        self.vectorizer = VectorizationPipeline()
        self.predictor = NumericFirstPredictor()
        self.decider = DecisionEngine()

    def run_for_symbol(self, symbol: str, as_of_time: datetime, raw_event_payload: dict, features: dict,
                       venue_eligibility: str, broker_supports_nxt: bool = True, venue_freshness_ok: bool = True,
                       session_liquidity_ok: bool = True, no_position: bool = True):
        session = classify_session(as_of_time)
        sem = self.normalizer.normalize(raw_event_payload)
        _ = self.vectorizer.build(sem.canonical_summary)
        features = dict(features)
        features["event_score"] = sem.event_score
        pred = self.predictor.predict(symbol, session, as_of_time, features)
        req = ExecutionRequest(
            symbol=symbol,
            decision_timestamp=as_of_time,
            venue_eligibility=venue_eligibility,
            broker_supports_nxt=broker_supports_nxt,
            venue_freshness_ok=venue_freshness_ok,
            session_liquidity_ok=session_liquidity_ok,
        )
        plan = self.mapper.map_execution(req)
        return self.decider.decide(
            pred,
            plan,
            trend_120m_ok=features.get("trend_120m", 0) > 0,
            tech_extension_high=features.get("extension_60m", 0) > 0.7,
            market_risk_off=bool(features.get("market_risk_off", False)),
            no_position=no_position,
        )
