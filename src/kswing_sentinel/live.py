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
    def __init__(
        self,
        *,
        normalizer: LLMEventNormalizer | None = None,
        vectorizer: VectorizationPipeline | None = None,
        predictor: NumericFirstPredictor | None = None,
    ) -> None:
        self.mapper = ExecutionMapper()
        self.normalizer = normalizer or LLMEventNormalizer(prompt_version="disabled")
        self.vectorizer = vectorizer or VectorizationPipeline()
        self.predictor = predictor or NumericFirstPredictor()
        self.decider = DecisionEngine()
        self._semantic_cache: dict[str, tuple[datetime, float]] = {}
        self.anchor_schedule = ["08:10", "08:40", "09:35", "12:30", "15:10", "15:35", "15:45", "18:30", "19:40", "20:05"]
        self.semantic_refresh_anchors = {"08:10", "09:35", "15:45", "20:05"}
        self.uses_llm_normalizer = False
        self.audit_prompt_version = "disabled"

    @staticmethod
    def _payload_text(raw_event_payload: dict) -> str:
        fields = []
        for key in ("headline", "title", "body", "text", "summary", "canonical_summary"):
            value = str(raw_event_payload.get(key, "")).strip()
            if value and value not in fields:
                fields.append(value)
        return "\n\n".join(fields)

    @staticmethod
    def _payload_list(raw_event_payload: dict, key: str) -> list[str]:
        value = raw_event_payload.get(key, [])
        if not isinstance(value, list):
            return []
        return [str(item) for item in value if str(item).strip()]

    @staticmethod
    def _payload_event_score(raw_event_payload: dict) -> float:
        value = raw_event_payload.get("event_score", 0.0)
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _payload_bool(raw_event_payload: dict, key: str) -> bool:
        value = raw_event_payload.get(key, False)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on", "y"}
        return False

    def describe_semantic_refresh(self, as_of_time: datetime, raw_event_payload: dict) -> tuple[bool, str | None]:
        explicit = raw_event_payload.get("semantic_refresh_required")
        if explicit is not None:
            if self._payload_bool(raw_event_payload, "semantic_refresh_required"):
                return True, "PAYLOAD_OVERRIDE"
            return False, "PAYLOAD_OVERRIDE_DISABLED"

        reasons: list[str] = []
        anchor_label = as_of_time.strftime("%H:%M")
        if anchor_label in self.semantic_refresh_anchors:
            reasons.append("ANCHOR_SCHEDULE")

        event_score = abs(self._payload_event_score(raw_event_payload))
        if self._payload_bool(raw_event_payload, "event_burst") or self._payload_bool(raw_event_payload, "breaking"):
            reasons.append("EVENT_BURST")
        elif event_score >= 0.7:
            reasons.append("HIGH_EVENT_SCORE")
        elif event_score >= 0.5 and self._payload_bool(raw_event_payload, "fresh"):
            reasons.append("FRESH_HIGH_EVENT")

        if not reasons:
            return False, None
        return True, "+".join(reasons)

    def run_for_symbol(self, symbol: str, as_of_time: datetime, raw_event_payload: dict, features: dict,
                       venue_eligibility: str, broker_supports_nxt: bool = True, venue_freshness_ok: bool = True,
                       session_liquidity_ok: bool = True, no_position: bool = True):
        session = classify_session(as_of_time)
        features = dict(features)
        semantic_refresh_required, semantic_refresh_reason = self.describe_semantic_refresh(as_of_time, raw_event_payload)
        features["semantic_refresh_required"] = semantic_refresh_required
        if semantic_refresh_reason is not None:
            features["semantic_refresh_reason"] = semantic_refresh_reason
        raw_text = self._payload_text(raw_event_payload)
        if raw_text:
            try:
                vector_payload = self.vectorizer.build(
                    raw_text,
                    source_doc_ids=self._payload_list(raw_event_payload, "source_doc_ids"),
                    cluster_ids=self._payload_list(raw_event_payload, "cluster_ids"),
                    as_of_time=as_of_time,
                    session_type=session,
                )
                features["vector_payload"] = vector_payload
                state_sequence = features.get("state_sequence")
                if isinstance(state_sequence, list) and state_sequence:
                    normalized_sequence: list[dict] = []
                    last_index = max((index for index, state in enumerate(state_sequence) if isinstance(state, dict)), default=-1)
                    for index, state in enumerate(state_sequence):
                        if not isinstance(state, dict):
                            continue
                        normalized_state = dict(state)
                        if index == last_index and "vector_payload" not in normalized_state and "vectors" not in normalized_state:
                            normalized_state["vector_payload"] = vector_payload
                        normalized_sequence.append(normalized_state)
                    features["state_sequence"] = normalized_sequence
                features.setdefault("text_branch_enabled", True)
            except Exception:
                features["text_branch_enabled"] = False
        else:
            features["text_branch_enabled"] = False
        features.setdefault("flow_strength", 0.0)
        features.setdefault("trend_120m", 0.0)
        features.setdefault("extension_60m", 0.0)
        features["semantic_branch_enabled"] = False
        features.setdefault("event_score", self._payload_event_score(raw_event_payload))
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
