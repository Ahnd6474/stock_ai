from __future__ import annotations

from .schemas import LLMNormalizedEvent


NEUTRAL_EVENT = {
    "canonical_summary": "No reliable new event. Semantic branch neutralized.",
    "event_score": 0.0,
    "event_half_life": "3d",
    "regime_hint": "chop",
    "red_flag": False,
    "flow_vs_tech_resolution": "NO_TRADE",
    "source_quality_score": 0.0,
    "freshness_score": 0.0,
    "semantic_confidence": 0.0,
    "evidence_spans": [],
    "entity_tags": [],
    "reasons": ["DEGRADED_MODE"],
}


class LLMEventNormalizer:
    def __init__(self, prompt_version: str = "v1", provider: str = "fallback") -> None:
        self.prompt_version = prompt_version
        self.provider = provider
        self.schema_violations = 0

    def normalize(self, payload: dict, retry_once: bool = True) -> LLMNormalizedEvent:
        try:
            return LLMNormalizedEvent(**payload)
        except Exception:
            self.schema_violations += 1
            if retry_once:
                try:
                    return LLMNormalizedEvent(**payload)
                except Exception:
                    self.schema_violations += 1
                    return LLMNormalizedEvent(**NEUTRAL_EVENT)
            return LLMNormalizedEvent(**NEUTRAL_EVENT)
