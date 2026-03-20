from __future__ import annotations

from datetime import datetime

from .decision_engine import DecisionEngine
from .execution_mapper import ExecutionMapper
from .feature_store import FeatureStore
from .flow_snapshot_store import FlowSnapshotStore
from .llm_event_normalizer import LLMEventNormalizer
from .portfolio_engine import PortfolioEngine
from .predictor import NumericFirstPredictor
from .risk_engine import MarketRiskState, RiskEngine
from .schemas import ExecutionRequest
from .session_rules import classify_session
from .vectorization import VectorizationPipeline


class LiveInferenceService:
    def __init__(self, feature_store: FeatureStore | None = None, flow_store: FlowSnapshotStore | None = None) -> None:
        self.mapper = ExecutionMapper()
        self.normalizer = LLMEventNormalizer()
        self.vectorizer = VectorizationPipeline()
        self.predictor = NumericFirstPredictor()
        self.decider = DecisionEngine()
        self.risk = RiskEngine()
        self.portfolio = PortfolioEngine()
        self.feature_store = feature_store or FeatureStore()
        self.flow_store = flow_store or FlowSnapshotStore()

    def run_for_symbol(
        self,
        symbol: str,
        as_of_time: datetime,
        raw_event_payload: dict,
        venue_eligibility: str,
        *,
        broker_supports_nxt: bool = True,
        venue_freshness_ok: bool = True,
        session_liquidity_ok: bool = True,
        no_position: bool = True,
        market_risk_off: bool = False,
        portfolio_beta: float = 0.8,
        beta_cap: float = 1.0,
        current_gross: float = 0.3,
    ):
        session = classify_session(as_of_time)

        sem = self.normalizer.normalize(raw_event_payload)
        vectors = self.vectorizer.build(
            sem.canonical_summary,
            source_doc_ids=[e.get("doc_id", "") for e in sem.evidence_spans],
            cluster_ids=[raw_event_payload.get("cluster_id", "")],
            as_of_time=as_of_time,
        )

        features = self.feature_store.get_latest(symbol, as_of_time)
        features = dict(features)
        features["event_score"] = sem.event_score
        features["vector_energy"] = sum(abs(x) for x in vectors["z_event"][:8]) / 8.0

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

        decision = self.decider.decide(
            pred,
            plan,
            trend_120m_ok=features.get("trend_120m", 0) > 0,
            tech_extension_high=features.get("extension_60m", 0) > 0.7,
            market_risk_off=market_risk_off,
            no_position=no_position,
        )
        decision = self.risk.apply(decision, MarketRiskState(market_risk_off, portfolio_beta, beta_cap))
        [decision] = self.portfolio.apply([decision], current_gross=current_gross)
        return {"decision": decision, "prediction": pred, "vector_metadata": vectors["metadata"]}
