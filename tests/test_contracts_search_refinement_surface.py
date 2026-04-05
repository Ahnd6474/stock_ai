from kswing_sentinel.contracts import (
    BaseSearchClient,
    EventNormalizerService,
    LLMEventNormalizerService,
    SearchRefinementPolicy,
    SearchRefinementService,
    SearchRefinementTrace,
)
from kswing_sentinel.schemas import LLMNormalizedEvent


def test_contracts_expose_search_refinement_surface():
    policy = SearchRefinementPolicy()
    assert policy.max_rounds >= 1


def test_base_search_client_protocol_runtime_checkable():
    class DummySearchClient:
        def search(self, query: str, *, max_results: int = 10, as_of_time=None, symbols=None):
            return [
                {
                    "title": "Example",
                    "url": "https://example.com",
                    "snippet": "Snippet",
                    "source": "example",
                    "published_at": None,
                }
            ]

    assert isinstance(DummySearchClient(), BaseSearchClient)


def test_search_refinement_service_protocol_runtime_checkable():
    class DummyRefinementService:
        def refine(self, query: str, *, policy=None, as_of_time=None, symbols=None):
            return SearchRefinementTrace(initial_query=query, rounds=[])

    assert isinstance(DummyRefinementService(), SearchRefinementService)


def test_event_normalizer_protocols_do_not_require_llm_specific_methods():
    class DummyNormalizer:
        def normalize(self, payload, retry_once: bool = True):
            return LLMNormalizedEvent(
                canonical_summary="example",
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

    normalizer = DummyNormalizer()
    assert isinstance(normalizer, EventNormalizerService)
    assert isinstance(normalizer, LLMEventNormalizerService)

