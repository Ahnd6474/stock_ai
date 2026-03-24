from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from .schemas import (
    EventMetadata,
    FusedPrediction,
    LLMNormalizedEvent,
    TradeDecision,
    VectorPayload,
)

KST = ZoneInfo("Asia/Seoul")


def example_llm_structured_output() -> dict:
    return LLMNormalizedEvent(
        canonical_summary="DART filing indicates improved order backlog and firmer near-term earnings guidance.",
        event_score=0.62,
        event_half_life="2w",
        regime_hint="event",
        red_flag=False,
        flow_vs_tech_resolution="FLOW_DOMINANT_CONTINUATION",
        source_quality_score=0.86,
        freshness_score=0.91,
        semantic_confidence=0.79,
        evidence_spans=[
            {
                "doc_id": "dart_20260319_001",
                "start_char": 142,
                "end_char": 218,
                "text_snippet": "Order backlog expanded and management raised near-term shipment guidance.",
            }
        ],
        entity_tags=["earnings_guidance", "order_backlog", "semiconductor"],
        reasons=["GUIDANCE_UP", "ORDER_BOOK_STRENGTH"],
        canonical_event_id="evt_005899",
        cluster_id="clu_22014",
        prompt_version="normalizer_prompt_v1.2",
        provider_name="grok-fast",
        source_doc_ids=["dart_20260319_001", "news_a_991"],
    ).model_dump()


def example_event_metadata() -> dict:
    return EventMetadata(
        canonical_event_id="evt_005899",
        cluster_id="clu_22014",
        source_lineage=["news_a_991", "news_b_113", "dart_20260319_001"],
        source_type="DART",
        source_quality_score=0.95,
        novelty_score=0.74,
        published_at=datetime(2026, 3, 19, 11, 58, 10, tzinfo=KST),
        first_seen_at=datetime(2026, 3, 19, 11, 59, 3, tzinfo=KST),
        retrieved_at=datetime(2026, 3, 19, 12, 0, 0, tzinfo=KST),
        available_at=datetime(2026, 3, 19, 11, 59, 3, tzinfo=KST),
        as_of_time=datetime(2026, 3, 19, 12, 30, 0, tzinfo=KST),
        source_doc_ids=["dart_20260319_001", "news_a_991"],
        entity_tags=["earnings_guidance", "order_backlog", "semiconductor"],
    ).model_dump()


def example_vector_payload() -> dict:
    return VectorPayload(
        symbol="005930",
        session_type="CORE_DAY",
        z_event=[0.01] * 64,
        z_social=[0.02] * 32,
        z_macro=[0.03] * 16,
        metadata={
            "encoder_version": "ko_roberta_v2",
            "tokenizer_version": "klue/roberta-base",
            "attention_aggregator_version": "hier_transformer_v3",
            "prompt_version": "normalizer_prompt_v1.2",
            "embedding_backend": "roberta_mean_pool_v2",
            "source_doc_ids": ["dart_20260319_001", "news_a_991"],
            "cluster_ids": ["clu_22014"],
            "generated_at": datetime(2026, 3, 19, 12, 31, 10, tzinfo=KST).isoformat(),
            "as_of_time": datetime(2026, 3, 19, 12, 30, 0, tzinfo=KST).isoformat(),
            "session_type": "CORE_DAY",
        },
    ).model_dump()


def example_fused_prediction() -> dict:
    return FusedPrediction(
        symbol="005930",
        as_of_time=datetime(2026, 3, 19, 12, 30, 0, tzinfo=KST),
        session_type="CORE_DAY",
        er_5d=0.018,
        er_20d=0.043,
        dd_20d=0.035,
        p_up_20d=0.66,
        flow_persist=0.71,
        uncertainty=0.28,
        regime_final="trend",
        event_score=0.62,
        semantic_branch_enabled=True,
        text_branch_enabled=True,
        model_version="lgbm_numeric_fusion_v1.2",
        calibrator_version="session_cal_v1",
        missing_flags={"provisional_flow_missing": False},
        stale_flags={"nxt_snapshot_stale": False},
    ).model_dump()


def example_trade_decision() -> dict:
    return TradeDecision(
        symbol="005930",
        action="BUY_HALF",
        target_weight=0.022,
        tranche_ratio=0.30,
        session_type="CORE_DAY",
        selected_venue="KRX",
        rationale_codes=["FLOW_EVENT_TREND_OK", "SHORT_TERM_EXTENSION_HIGH"],
        as_of_time=datetime(2026, 3, 19, 12, 30, 0, tzinfo=KST),
        execution_time=datetime(2026, 3, 19, 12, 35, 0, tzinfo=KST),
        vetoes_triggered=[],
        risk_budget_used=0.004,
        expected_slippage_bps=9.5,
        degraded_mode_flags=[],
        exit_policy_hint="time_stop_or_thesis_break",
    ).model_dump()


def all_example_payloads() -> dict[str, dict]:
    return {
        "llm_structured_output": example_llm_structured_output(),
        "event_metadata": example_event_metadata(),
        "vector_payload": example_vector_payload(),
        "fused_prediction": example_fused_prediction(),
        "trade_decision": example_trade_decision(),
    }
