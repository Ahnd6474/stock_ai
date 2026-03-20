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
        self._semantic_cache: dict[str, tuple[datetime, float]] = {}
        self.anchor_schedule = ["08:10", "08:40", "09:35", "12:30", "15:10", "15:35", "15:45", "18:30", "19:40", "20:05"]

    def run_for_symbol(self, symbol: str, as_of_time: datetime, raw_event_payload: dict, features: dict,
                       venue_eligibility: str, broker_supports_nxt: bool = True, venue_freshness_ok: bool = True,
                       session_liquidity_ok: bool = True, no_position: bool = True):
        session = classify_session(as_of_time)
        sem = self.normalizer.normalize(raw_event_payload)
        _ = self.vectorizer.build(sem.canonical_summary, source_doc_ids=[], cluster_ids=[], as_of_time=as_of_time, session_type=session)
        features = dict(features)
        features.setdefault("flow_strength", 0.0)
        features.setdefault("trend_120m", 0.0)
        features.setdefault("extension_60m", 0.0)
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

    def run_anchor_batch(self, symbols: list[str], as_of_time: datetime, payload_by_symbol: dict[str, dict], features_by_symbol: dict[str, dict], venue_eligibility_by_symbol: dict[str, str]) -> dict:
        out = {}
        for symbol in symbols:
            out[symbol] = self.run_for_symbol(
                symbol,
                as_of_time,
                payload_by_symbol.get(symbol, {}),
                features_by_symbol.get(symbol, {}),
                venue_eligibility_by_symbol.get(symbol, "KRX_ONLY"),
            )
        return out
