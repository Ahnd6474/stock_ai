from kswing_sentinel.llm_event_normalizer import (
    EnsembleLLMProvider,
    InternalContextSearchClient,
    LLMEventNormalizer,
    StaticLLMProvider,
    StaticSearchClient,
)


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


def test_llm_provider_can_use_search_context():
    provider = StaticLLMProvider(
        {
            "canonical_summary": "Search-backed summary.",
            "event_score": 0.4,
            "event_half_life": "1w",
            "regime_hint": "event",
            "red_flag": False,
            "flow_vs_tech_resolution": "WAIT_FOR_PULLBACK",
            "source_quality_score": 0.8,
            "freshness_score": 0.7,
            "semantic_confidence": 0.75,
            "evidence_spans": [],
            "entity_tags": ["search"],
            "reasons": ["SEARCH_CONTEXT"],
        }
    )
    search_client = StaticSearchClient(
        [
            {
                "title": "Samsung filing",
                "url": "https://example.com/doc1",
                "snippet": "Filing suggests stronger demand.",
                "published_at": "2026-03-20T09:00:00+09:00",
            }
        ]
    )
    normalizer = LLMEventNormalizer(provider=provider, search_client=search_client)
    out = normalizer.normalize(
        {
            "symbol": "005930",
            "headline": "Demand improves",
            "body": "Body text",
        }
    )
    assert out.canonical_summary == "Search-backed summary."


def test_llm_provider_prefers_internal_search_results():
    provider = StaticLLMProvider(
        {
            "canonical_summary": "Internal context summary.",
            "event_score": 0.3,
            "event_half_life": "3d",
            "regime_hint": "event",
            "red_flag": False,
            "flow_vs_tech_resolution": "NO_TRADE",
            "source_quality_score": 0.7,
            "freshness_score": 0.8,
            "semantic_confidence": 0.6,
            "evidence_spans": [],
            "entity_tags": ["internal"],
            "reasons": ["INTERNAL_CONTEXT"],
        }
    )
    search_client = InternalContextSearchClient(
        [{"title": "unused external fallback", "snippet": "unused", "url": "https://example.com"}]
    )
    normalizer = LLMEventNormalizer(provider=provider, search_client=search_client)
    out = normalizer.normalize(
        {
            "headline": "Internal only",
            "body": "Body text",
            "internal_search_results": [
                {"title": "Internal note", "url": "memory://1", "snippet": "internal snippet"}
            ],
        }
    )
    assert out.canonical_summary == "Internal context summary."


def test_ensemble_provider_merges_multiple_outputs():
    provider = EnsembleLLMProvider(
        [
            StaticLLMProvider(
                {
                    "canonical_summary": "Primary summary.",
                    "event_score": 0.6,
                    "event_half_life": "1w",
                    "regime_hint": "event",
                    "red_flag": False,
                    "flow_vs_tech_resolution": "FLOW_DOMINANT_CONTINUATION",
                    "source_quality_score": 0.9,
                    "freshness_score": 0.8,
                    "semantic_confidence": 0.7,
                    "evidence_spans": [{"doc_id": "a"}],
                    "entity_tags": ["earnings"],
                    "reasons": ["A"],
                }
            ),
            StaticLLMProvider(
                {
                    "canonical_summary": "Secondary summary.",
                    "event_score": 0.2,
                    "event_half_life": "1w",
                    "regime_hint": "event",
                    "red_flag": True,
                    "flow_vs_tech_resolution": "FLOW_DOMINANT_CONTINUATION",
                    "source_quality_score": 0.7,
                    "freshness_score": 0.6,
                    "semantic_confidence": 0.9,
                    "evidence_spans": [{"doc_id": "b"}],
                    "entity_tags": ["guidance"],
                    "reasons": ["B"],
                }
            ),
        ]
    )
    out = provider.generate_structured_json("system", "user")
    assert out["canonical_summary"] == "Primary summary."
    assert out["red_flag"] is True
    assert len(out["evidence_spans"]) == 2
