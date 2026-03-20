from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from kswing_sentinel.schemas import LLMNormalizedEvent


def test_llm_schema_validation_success():
    obj = LLMNormalizedEvent(
        canonical_summary="유효한 이벤트 요약입니다.",
        event_score=0.1,
        event_half_life="1w",
        regime_hint="event",
        red_flag=False,
        flow_vs_tech_resolution="FLOW_DOMINANT_CONTINUATION",
        source_quality_score=0.8,
        freshness_score=0.7,
        semantic_confidence=0.6,
        evidence_spans=[],
        entity_tags=["earnings"],
        reasons=["OK"],
    )
    assert obj.event_score == 0.1


def test_llm_schema_validation_fail():
    with pytest.raises(ValidationError):
        LLMNormalizedEvent(
            canonical_summary="short",
            event_score=1.5,
            event_half_life="bad",
            regime_hint="bad",
            red_flag=False,
            flow_vs_tech_resolution="NO_TRADE",
            source_quality_score=2.0,
            freshness_score=0.7,
            semantic_confidence=0.6,
            evidence_spans=[],
            entity_tags=[],
            reasons=[],
        )
