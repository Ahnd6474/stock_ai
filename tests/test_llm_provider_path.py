from kswing_sentinel.llm_event_normalizer import LLMEventNormalizer, StaticLLMProvider


def test_llm_provider_normalizes_raw_payload():
    provider = StaticLLMProvider(
        {
            "canonical_summary": "Guidance improved and order visibility strengthened.",
            "event_score": 0.55,
            "event_half_life": "2w",
            "regime_hint": "event",
            "red_flag": False,
            "flow_vs_tech_resolution": "FLOW_DOMINANT_CONTINUATION",
            "source_quality_score": 0.9,
            "freshness_score": 0.9,
            "semantic_confidence": 0.8,
            "evidence_spans": [],
            "entity_tags": ["guidance"],
            "reasons": ["GUIDANCE_UP"],
        }
    )
    normalizer = LLMEventNormalizer(provider=provider, prompt_version="v2")
    out = normalizer.normalize(
        {
            "symbol": "005930",
            "source_type": "NEWS",
            "headline": "Samsung Electronics outlook improves",
            "body": "Management pointed to stronger memory pricing and better shipment visibility.",
        }
    )
    assert out.event_score == 0.55
    assert out.provider_name == "static"
    assert out.prompt_version == "v2"
