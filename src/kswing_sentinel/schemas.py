from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

SessionType = Literal["NXT_PRE", "CORE_DAY", "CLOSE_PRICE", "NXT_AFTER", "OFF_MARKET"]
VenueEligibility = Literal["KRX_ONLY", "KRX_PLUS_NXT"]
VenueType = Literal["KRX", "NXT"]
FlowStatus = Literal["PROVISIONAL", "CONFIRMED"]
DecisionAction = Literal["BUY", "BUY_HALF", "WAIT_PULLBACK", "NO_TRADE", "REDUCE", "SELL"]
RegimeHint = Literal["trend", "event", "chop", "risk_off"]
EventHalfLife = Literal["intraday", "3d", "1w", "2w", "1m"]
FlowTechResolution = Literal[
    "FLOW_DOMINANT_CONTINUATION",
    "TECH_DOMINANT_EXHAUSTION",
    "WAIT_FOR_PULLBACK",
    "NO_TRADE",
]
SourceType = Literal["DART", "IR", "NEWS", "SOCIAL", "RESEARCH"]
OrderMode = Literal["REGULAR", "CLOSE_PRICE", "AFTER_MARKET"]


class VersionBundle(BaseModel):
    feature_definition_version: str = "v1"
    model_version: str = "v1"
    calibrator_version: str = "v1"
    prompt_version: str = "v1"
    vectorizer_version: str = "v1"
    attention_aggregator_version: str = "v1"
    retrieval_policy_version: str = "v1"
    universe_filter_version: str = "v1"
    market_cap_source_version: str = "v1"
    liquidity_rule_version: str = "v1"
    corporate_action_adjustment_version: str = "v1"
    cost_model_version: str = "v1"
    session_calendar_version: str = "v1"
    nxt_eligibility_snapshot_version: str = "v1"
    broker_capabilities_version: str = "v1"


class AuditStamp(BaseModel):
    as_of_time: datetime
    generated_at: datetime
    run_id: str
    trace_id: str


class EvidenceSpan(BaseModel):
    doc_id: str
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    text_snippet: str = Field(min_length=1, max_length=400)


class EventMetadata(BaseModel):
    canonical_event_id: str
    cluster_id: str
    source_lineage: list[str]
    source_type: SourceType
    source_quality_score: float = Field(ge=0, le=1)
    novelty_score: float = Field(ge=0, le=1)
    published_at: datetime
    first_seen_at: datetime
    retrieved_at: datetime
    available_at: datetime
    as_of_time: datetime
    source_doc_ids: list[str] | None = None
    entity_tags: list[str] | None = None


class LLMNormalizedEvent(BaseModel):
    canonical_summary: str = Field(min_length=10, max_length=1200)
    event_score: float = Field(ge=-1, le=1)
    event_half_life: EventHalfLife
    regime_hint: RegimeHint
    red_flag: bool
    flow_vs_tech_resolution: FlowTechResolution
    source_quality_score: float = Field(ge=0, le=1)
    freshness_score: float = Field(ge=0, le=1)
    semantic_confidence: float = Field(ge=0, le=1)
    evidence_spans: list[dict]
    entity_tags: list[str]
    reasons: list[str]
    canonical_event_id: str | None = None
    cluster_id: str | None = None
    degraded_mode: bool = False
    provider_name: str = "fallback"
    prompt_version: str = "v1"
    source_doc_ids: list[str] | None = None


class FlowSnapshot(BaseModel):
    symbol: str
    window: Literal["INTRADAY", "1D", "3D", "5D"]
    foreign_net: float
    institutional_net: float
    program_net: float
    preliminary_or_final: FlowStatus
    vendor: str = "UNKNOWN"
    revision_id: str = "r0"
    snapshot_at: datetime
    as_of_time: datetime


class VectorPayloadMetadata(BaseModel):
    encoder_version: str
    tokenizer_version: str
    attention_aggregator_version: str
    prompt_version: str
    embedding_backend: str = "hashing_bow_v1"
    source_doc_ids: list[str]
    cluster_ids: list[str]
    generated_at: datetime
    as_of_time: datetime
    session_type: SessionType


class VectorPayload(BaseModel):
    symbol: str
    session_type: SessionType
    z_event: list[float]
    z_social: list[float]
    z_macro: list[float]
    metadata: dict


class ExecutionRequest(BaseModel):
    symbol: str
    decision_timestamp: datetime
    venue_eligibility: VenueEligibility
    broker_supports_nxt: bool
    venue_freshness_ok: bool
    session_liquidity_ok: bool
    broker_capabilities_version: str = "v1"
    session_calendar_version: str = "v1"
    nxt_eligibility_snapshot_version: str = "v1"
    venue_availability_ok: bool = True
    venue_clock_ok: bool = True
    broker_cutoff_minutes: int = 3
    liquidity_bucket: Literal["high", "mid", "low"] = "mid"
    decision_anchor: str | None = None


class ExecutionPlan(BaseModel):
    symbol: str
    selected_venue: VenueType
    selected_session_type: SessionType
    scheduled_exec_time: datetime
    fill_proxy: Literal["NEXT_COMPLETED_5M_VWAP"] = "NEXT_COMPLETED_5M_VWAP"
    expected_cost_bps: float
    rollover_reason: str | None = None
    available_order_modes: list[OrderMode] | None = None
    venue_uncertain: bool = False
    cost_model_version: str = "v1"


class FusedPrediction(BaseModel):
    symbol: str
    as_of_time: datetime
    session_type: SessionType
    er_5d: float
    er_20d: float
    dd_20d: float
    p_up_20d: float = Field(ge=0, le=1)
    flow_persist: float = Field(ge=0, le=1)
    uncertainty: float = Field(ge=0, le=1)
    regime_final: RegimeHint
    event_score: float = Field(ge=-1, le=1, default=0)
    semantic_branch_enabled: bool = True
    text_branch_enabled: bool = True
    model_version: str = "v1"
    calibrator_version: str = "v1"
    missing_flags: dict | None = None
    stale_flags: dict | None = None


class TradeDecision(BaseModel):
    symbol: str
    action: DecisionAction
    target_weight: float
    tranche_ratio: float
    session_type: SessionType
    selected_venue: VenueType
    rationale_codes: list[str]
    as_of_time: datetime
    execution_time: datetime
    vetoes_triggered: list[str] | None = None
    risk_budget_used: float = 0.0
    expected_slippage_bps: float = 0.0
    degraded_mode_flags: list[str] | None = None
    exit_policy_hint: str | None = None


class MonitoringSnapshot(BaseModel):
    feature_freshness_lag_sec: float = 0.0
    retrieval_success_rate: float = Field(ge=0, le=1, default=1.0)
    llm_schema_violation_rate: float = Field(ge=0, default=0.0)
    semantic_coverage_rate: float = Field(ge=0, le=1, default=1.0)
    degraded_mode_rate: float = Field(ge=0, default=0.0)
    venue_selection_error_rate: float = Field(ge=0, default=0.0)
    slippage_vs_expected_bps: float = 0.0
    realized_dd_vs_predicted_dd_gap: float = 0.0
