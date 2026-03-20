from __future__ import annotations

from datetime import datetime
from typing import Literal

from kswing_sentinel.pydantic_compat import BaseModel, Field

SessionType = Literal["NXT_PRE", "CORE_DAY", "CLOSE_PRICE", "NXT_AFTER", "OFF_MARKET"]
VenueEligibility = Literal["KRX_ONLY", "KRX_PLUS_NXT"]
VenueType = Literal["KRX", "NXT"]
FlowStatus = Literal["PROVISIONAL", "CONFIRMED"]


class VersionBundle(BaseModel):
    feature_definition_version: str = "v1"
    model_version: str = "v1"
    prompt_version: str = "v1"
    vectorizer_version: str = "v1"
    attention_aggregator_version: str = "v1"
    universe_filter_version: str = "v1"
    cost_model_version: str = "v1"
    session_calendar_version: str = "v1"
    nxt_eligibility_snapshot_version: str = "v1"


class LLMNormalizedEvent(BaseModel):
    canonical_summary: str = Field(min_length=10, max_length=1200)
    event_score: float = Field(ge=-1, le=1)
    event_half_life: Literal["intraday", "3d", "1w", "2w", "1m"]
    regime_hint: Literal["trend", "event", "chop", "risk_off"]
    red_flag: bool
    flow_vs_tech_resolution: Literal[
        "FLOW_DOMINANT_CONTINUATION",
        "TECH_DOMINANT_EXHAUSTION",
        "WAIT_FOR_PULLBACK",
        "NO_TRADE",
    ]
    source_quality_score: float = Field(ge=0, le=1)
    freshness_score: float = Field(ge=0, le=1)
    semantic_confidence: float = Field(ge=0, le=1)
    evidence_spans: list[dict]
    entity_tags: list[str]
    reasons: list[str]


class FlowSnapshot(BaseModel):
    symbol: str
    window: Literal["INTRADAY", "1D", "3D", "5D"]
    foreign_net: float
    institutional_net: float
    program_net: float
    preliminary_or_final: FlowStatus
    snapshot_at: datetime
    as_of_time: datetime


class ExecutionRequest(BaseModel):
    symbol: str
    decision_timestamp: datetime
    venue_eligibility: VenueEligibility
    broker_supports_nxt: bool
    venue_freshness_ok: bool
    session_liquidity_ok: bool


class ExecutionPlan(BaseModel):
    symbol: str
    selected_venue: VenueType
    selected_session_type: SessionType
    scheduled_exec_time: datetime
    fill_proxy: Literal["NEXT_COMPLETED_5M_VWAP"] = "NEXT_COMPLETED_5M_VWAP"
    expected_cost_bps: float
    rollover_reason: str | None = None


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
    regime_final: Literal["trend", "event", "chop", "risk_off"]
    event_score: float = Field(ge=-1, le=1, default=0)


class TradeDecision(BaseModel):
    symbol: str
    action: Literal["BUY", "BUY_HALF", "WAIT_PULLBACK", "NO_TRADE", "REDUCE", "SELL"]
    target_weight: float
    tranche_ratio: float
    session_type: SessionType
    selected_venue: VenueType
    rationale_codes: list[str]
    as_of_time: datetime
    execution_time: datetime
