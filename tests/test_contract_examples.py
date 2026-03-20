from kswing_sentinel.example_payloads import all_example_payloads


def test_example_payloads_have_required_shapes():
    payloads = all_example_payloads()
    assert payloads["llm_structured_output"]["event_score"] == 0.62
    assert payloads["event_metadata"]["source_type"] == "DART"
    assert len(payloads["vector_payload"]["z_event"]) == 64
    assert len(payloads["vector_payload"]["z_social"]) == 32
    assert len(payloads["vector_payload"]["z_macro"]) == 16
    assert payloads["fused_prediction"]["semantic_branch_enabled"] is True
    assert payloads["trade_decision"]["expected_slippage_bps"] == 9.5
