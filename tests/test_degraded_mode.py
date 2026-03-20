from kswing_sentinel.llm_event_normalizer import LLMEventNormalizer


def test_llm_fallback_to_neutral():
    n = LLMEventNormalizer()
    out = n.normalize({"bad": "payload"})
    assert out.event_score == 0.0
    assert out.regime_hint == "chop"
