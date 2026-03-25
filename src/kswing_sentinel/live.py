from __future__ import annotations

from datetime import datetime, timezone

from .decision_engine import DecisionEngine
from .event_store import EventStore
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
        event_store: EventStore | None = None,
    ) -> None:
        self.mapper = ExecutionMapper()
        self.normalizer = normalizer or LLMEventNormalizer(prompt_version="disabled")
        self.vectorizer = vectorizer or VectorizationPipeline()
        self.predictor = predictor or NumericFirstPredictor()
        self.event_store = event_store
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

    @staticmethod
    def _parse_datetime(value: object) -> datetime | None:
        if isinstance(value, datetime):
            return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        if not isinstance(value, str):
            return None
        text = value.strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)

    @classmethod
    def _state_timestamp(cls, state: dict) -> datetime | None:
        for source in (
            state,
            state.get("numeric_features"),
            state.get("metadata"),
            (state.get("vector_payload") or {}).get("metadata") if isinstance(state.get("vector_payload"), dict) else None,
            (state.get("vectors") or {}).get("metadata") if isinstance(state.get("vectors"), dict) else None,
        ):
            if not isinstance(source, dict):
                continue
            for key in (
                "as_of_time",
                "timestamp",
                "observed_at",
                "collected_at",
                "available_at",
                "retrieved_at",
                "generated_at",
            ):
                parsed = cls._parse_datetime(source.get(key))
                if parsed is not None:
                    return parsed
        return None

    @staticmethod
    def _dedup_strings(values: list[str]) -> list[str]:
        out: list[str] = []
        for value in values:
            text = str(value).strip()
            if text and text not in out:
                out.append(text)
        return out

    def _reference_collection_time(self, raw_event_payload: dict, features: dict) -> datetime | None:
        for source in (raw_event_payload, features):
            for key in ("previous_collection_time", "last_collection_time", "last_social_refresh_at", "previous_refresh_time"):
                parsed = self._parse_datetime(source.get(key))
                if parsed is not None:
                    return parsed
        state_sequence = features.get("state_sequence")
        if isinstance(state_sequence, list) and state_sequence:
            for state in reversed(state_sequence[:-1]):
                if isinstance(state, dict):
                    parsed = self._state_timestamp(state)
                    if parsed is not None:
                        return parsed
            for state in reversed(state_sequence):
                if isinstance(state, dict):
                    parsed = self._state_timestamp(state)
                    if parsed is not None:
                        return parsed
        return None

    def _delta_docs(self, symbol: str, as_of_time: datetime, raw_event_payload: dict, features: dict) -> tuple[list[object], dict[str, float], datetime | None]:
        since_time = self._reference_collection_time(raw_event_payload, features)
        if self.event_store is None:
            return [], {
                "delta_doc_count": 0.0,
                "new_doc_count": 0.0,
                "updated_doc_count": 0.0,
                "delta_novelty_mean": 0.0,
                "delta_source_quality_mean": 0.0,
                "delta_freshness_mean": 0.0,
                "time_since_last_collection_sec": max(0.0, (as_of_time - since_time).total_seconds()) if since_time is not None else 0.0,
            }, since_time
        return (
            self.event_store.delta_docs(symbol, as_of_time, since_time=since_time),
            self.event_store.delta_summary(symbol, as_of_time, since_time=since_time),
            since_time,
        )

    @staticmethod
    def _delta_social_items(delta_docs: list[object], as_of_time: datetime) -> list[dict]:
        items: list[dict] = []
        for doc in delta_docs:
            age_seconds = max(0.0, (as_of_time - doc.available_at).total_seconds())
            freshness = max(0.0, 1.0 - min(1.0, age_seconds / 86400.0))
            items.append(
                {
                    "text": doc.body,
                    "cluster_id": doc.cluster_id or "social",
                    "freshness_score": freshness,
                    "source_quality_score": float(doc.source_quality_score),
                    "novelty_score": float(doc.novelty_score),
                    "semantic_confidence": 0.7,
                }
            )
        return items

    def _build_vector_payload(
        self,
        *,
        symbol: str,
        as_of_time: datetime,
        session: str,
        raw_event_payload: dict,
        features: dict,
        raw_text: str,
        delta_docs: list[object],
        delta_features: dict[str, float],
        previous_collection_time: datetime | None,
    ) -> dict:
        payload_doc_ids = self._payload_list(raw_event_payload, "source_doc_ids")
        payload_cluster_ids = self._payload_list(raw_event_payload, "cluster_ids")
        merged_doc_ids = self._dedup_strings(payload_doc_ids + [getattr(doc, "doc_id", "") for doc in delta_docs])
        merged_cluster_ids = self._dedup_strings(payload_cluster_ids + [getattr(doc, "cluster_id", "") for doc in delta_docs])
        social_text = "\n\n".join(getattr(doc, "body", "") for doc in delta_docs if str(getattr(doc, "body", "")).strip())
        build_from_items = getattr(self.vectorizer, "build_from_items", None)
        if callable(build_from_items):
            event_items = None
            if raw_text:
                event_items = [
                    {
                        "text": raw_text,
                        "cluster_id": (merged_cluster_ids or ["event"])[0],
                        "freshness_score": 1.0,
                        "source_quality_score": 0.8,
                        "novelty_score": max(0.0, abs(self._payload_event_score(raw_event_payload))),
                        "semantic_confidence": 0.8,
                    }
                ]
            social_items = self._delta_social_items(delta_docs, as_of_time)
            if not social_items and raw_text:
                social_items = [
                    {
                        "text": raw_text,
                        "cluster_id": (merged_cluster_ids or ["social"])[0],
                        "freshness_score": 0.8,
                        "source_quality_score": 0.6,
                        "novelty_score": 0.5,
                        "semantic_confidence": 0.6,
                    }
                ]
            vector_payload = build_from_items(
                event_items=event_items,
                social_items=social_items or None,
                source_doc_ids=merged_doc_ids,
                cluster_ids=merged_cluster_ids,
                as_of_time=as_of_time,
                session_type=session,
            )
        else:
            vector_payload = self.vectorizer.build(
                raw_text or social_text,
                social=social_text,
                source_doc_ids=merged_doc_ids,
                cluster_ids=merged_cluster_ids,
                as_of_time=as_of_time,
                session_type=session,
            )
        metadata = vector_payload.setdefault("metadata", {}) if isinstance(vector_payload, dict) else {}
        if isinstance(metadata, dict):
            metadata.update(delta_features)
            if previous_collection_time is not None:
                metadata["previous_collection_time"] = previous_collection_time.isoformat()
        return vector_payload

    def _attach_temporal_metadata(
        self,
        *,
        features: dict,
        as_of_time: datetime,
        vector_payload: dict | None,
        delta_features: dict[str, float],
    ) -> None:
        features["as_of_time"] = as_of_time.isoformat()
        features["delta_features"] = dict(delta_features)
        state_sequence = features.get("state_sequence")
        if isinstance(state_sequence, list) and state_sequence:
            normalized_sequence: list[dict] = []
            last_index = max((index for index, state in enumerate(state_sequence) if isinstance(state, dict)), default=-1)
            for index, state in enumerate(state_sequence):
                if not isinstance(state, dict):
                    continue
                normalized_state = dict(state)
                if index == last_index:
                    normalized_state.setdefault("as_of_time", as_of_time.isoformat())
                    normalized_state["delta_features"] = dict(delta_features)
                    if vector_payload is not None and "vector_payload" not in normalized_state and "vectors" not in normalized_state:
                        normalized_state["vector_payload"] = vector_payload
                normalized_sequence.append(normalized_state)
            features["state_sequence"] = normalized_sequence

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

    def run_for_symbol(
        self,
        symbol: str,
        as_of_time: datetime,
        raw_event_payload: dict,
        features: dict,
        venue_eligibility: str,
        broker_supports_nxt: bool = True,
        venue_freshness_ok: bool = True,
        session_liquidity_ok: bool = True,
        no_position: bool = True,
    ):
        session = classify_session(as_of_time)
        features = dict(features)
        semantic_refresh_required, semantic_refresh_reason = self.describe_semantic_refresh(as_of_time, raw_event_payload)
        features["semantic_refresh_required"] = semantic_refresh_required
        if semantic_refresh_reason is not None:
            features["semantic_refresh_reason"] = semantic_refresh_reason

        raw_text = self._payload_text(raw_event_payload)
        delta_docs, delta_features, previous_collection_time = self._delta_docs(symbol, as_of_time, raw_event_payload, features)
        self._attach_temporal_metadata(
            features=features,
            as_of_time=as_of_time,
            vector_payload=None,
            delta_features=delta_features,
        )

        if raw_text or delta_docs:
            try:
                vector_payload = self._build_vector_payload(
                    symbol=symbol,
                    as_of_time=as_of_time,
                    session=session,
                    raw_event_payload=raw_event_payload,
                    features=features,
                    raw_text=raw_text,
                    delta_docs=delta_docs,
                    delta_features=delta_features,
                    previous_collection_time=previous_collection_time,
                )
                features["vector_payload"] = vector_payload
                self._attach_temporal_metadata(
                    features=features,
                    as_of_time=as_of_time,
                    vector_payload=vector_payload,
                    delta_features=delta_features,
                )
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

    def run_anchor_batch(
        self,
        symbols: list[str],
        as_of_time: datetime,
        payload_by_symbol: dict[str, dict],
        features_by_symbol: dict[str, dict],
        venue_eligibility_by_symbol: dict[str, str],
    ) -> dict:
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
