# K-Swing Sentinel v1.2 (Production-Oriented LONG-ONLY KRX/NXT Swing System)

## 1) Concise Architecture Document

### 1.1 Objective and Operating Constraints
- **Primary objective**: maximize **net** return after commission, tax, slippage, spread, and liquidity/impact constraints.
- **Secondary objective**: explicit drawdown control, operational robustness, reproducibility, and full auditability.
- **Stance**:
  - Numeric model is primary alpha/risk engine.
  - LLM is restricted to normalization, semantic factors, regime hints, veto flags, and evidence traceability.
  - Execution assumes **next tradable opportunity only** (no same-bar fill).
  - NXT is first-class with separate phase behavior.

### 1.2 System Layers
1. **Ingestion Layer**
   - KRX/NXT market bars + quote proxies
   - Flow snapshots (provisional + confirmed separated)
   - Event/doc retrieval (DART/news/IR/social)
2. **Normalization & Representation Layer**
   - Event dedup/clustering
   - LLM structured normalizer (strict JSON schema)
   - Korean BERT encoder
   - Hierarchical attention aggregator to produce z_event/z_social/z_macro
3. **Feature & Modeling Layer**
   - Session-aware feature store (60m/120m + phase features)
   - Numeric-first predictor (LightGBM/CatBoost)
   - Calibration pipeline
4. **Decision & Risk Layer**
   - Decision engine (BUY/BUY_HALF/WAIT_PULLBACK/NO_TRADE/REDUCE/SELL)
   - Risk engine + portfolio constraints + market throttle
   - Venue router + ExecutionMapper
5. **Execution & Ops Layer**
   - Broker gateway
   - Backtester sharing execution logic with live
   - Monitoring/audit + orchestration (Temporal preferred)

### 1.3 Time and Session Model (KST)
- Decision anchors: 08:10, 08:40, 09:35, 12:30, 15:10, 15:35, 15:45, 18:30, 19:40, 20:05.
- Full numeric inference at all anchors.
- LLM semantic refresh only at 08:10, 09:35, 15:45, 20:05 and event-triggered bursts.
- Session phases explicitly modeled:
  - NXT_PRE (08:00–08:50)
  - CORE_DAY (KRX regular / NXT main overlap)
  - CLOSE_PRICE (NXT closing-price window)
  - NXT_AFTER (15:40–20:00)

### 1.4 Execution Realism Requirements
- Default fill: next fully completed 5-minute VWAP proxy after decision timestamp.
- If within 3 minutes of phase end, roll to next eligible phase.
- If venue freshness uncertain: fail closed and fallback conservatively.
- Outside core day liquidity: aggressiveness and size materially reduced.

### 1.5 No-Lookahead Contract
Every dataset row, feature, document, vector, and decision carries `as_of_time`.
- All joins are `timestamp <= as_of_time`.
- Labeling uses same `ExecutionMapper` class as live/backtest.
- Doc availability uses `available_at`/`first_seen_at`, not just `published_at`.

---

## 2) Repository Tree (Implementation-Ready)

```text
stock_ai/
  pyproject.toml
  README.md
  docs/
    k_swing_sentinel_v1_2.md
    runbooks/
      degraded_mode.md
      venue_outage.md
      model_rollout.md
  configs/
    universe/
      universe_filters_v1.yml
    session/
      krx_nxt_calendar_v1.yml
    costs/
      cost_curves_v1.yml
    llm/
      normalizer_prompt_v1.yml
    models/
      predictor_v1.yml
      calibrator_v1.yml
    routing/
      broker_capabilities_v1.yml
  schemas/
    common.py
    market.py
    flow.py
    events.py
    vectors.py
    features.py
    predictions.py
    decisions.py
    execution.py
    monitoring.py
  src/
    market_data_ingestion/
      service.py
      adapters/
        krx_adapter.py
        nxt_adapter.py
    flow_snapshot_store/
      service.py
      repository.py
    event_retrieval/
      service.py
      connectors/
        dart_connector.py
        news_connector.py
        social_connector.py
    event_dedup_and_clustering/
      service.py
      clustering.py
    llm_event_normalizer/
      service.py
      providers/
        grok_adapter.py
        deepseek_adapter.py
        qwen_adapter.py
    text_encoder_service/
      service.py
      ko_bert_encoder.py
    attention_aggregator/
      service.py
      hierarchical.py
    vectorization_pipeline/
      pipeline.py
    feature_store/
      service.py
      offline_store.py
      online_store.py
    predictor_training/
      pipeline.py
      dataset_builder.py
      train_lightgbm.py
    calibration_pipeline/
      pipeline.py
      calibrators.py
    live_inference/
      service.py
      anchor_scheduler.py
    venue_router_and_execution_mapper/
      execution_mapper.py
      venue_router.py
      session_rules.py
    decision_engine/
      service.py
      policy.py
    risk_engine/
      service.py
      vetoes.py
    portfolio_engine/
      service.py
      constraints.py
    backtester/
      engine.py
      fills.py
      costs.py
    broker_gateway/
      service.py
      order_router.py
    experiment_tracking/
      service.py
    monitoring_and_audit/
      service.py
      metrics.py
      audit_log.py
    orchestration/
      temporal_workflows.py
      temporal_activities.py
  tests/
    test_no_lookahead.py
    test_schema_validation.py
    test_degraded_mode.py
    test_timestamp_boundaries.py
    test_flow_leakage.py
    test_session_mapping.py
    test_venue_fallback.py
```

---

## 3) Pydantic Schemas for Major Inputs/Outputs

```python
# schemas/common.py
from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Dict
from datetime import datetime

SessionType = Literal["NXT_PRE", "CORE_DAY", "CLOSE_PRICE", "NXT_AFTER"]
VenueType = Literal["KRX", "NXT"]
VenueEligibility = Literal["KRX_ONLY", "KRX_PLUS_NXT"]
DecisionAction = Literal["BUY", "BUY_HALF", "WAIT_PULLBACK", "NO_TRADE", "REDUCE", "SELL"]

class VersionBundle(BaseModel):
    feature_definition_version: str
    model_version: str
    calibrator_version: str
    prompt_version: str
    vectorizer_version: str
    attention_aggregator_version: str
    retrieval_policy_version: str
    universe_filter_version: str
    cost_model_version: str
    session_calendar_version: str
    nxt_eligibility_snapshot_version: str

class AuditStamp(BaseModel):
    as_of_time: datetime
    generated_at: datetime
    run_id: str
    trace_id: str
```

```python
# schemas/events.py
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
from datetime import datetime

class EvidenceSpan(BaseModel):
    doc_id: str
    start_char: int
    end_char: int
    text_snippet: str = Field(max_length=400)

class EventMetadata(BaseModel):
    canonical_event_id: str
    cluster_id: str
    source_lineage: List[str]
    source_type: Literal["DART", "IR", "NEWS", "SOCIAL", "RESEARCH"]
    source_quality_score: float = Field(ge=0.0, le=1.0)
    novelty_score: float = Field(ge=0.0, le=1.0)
    published_at: datetime
    first_seen_at: datetime
    retrieved_at: datetime
    available_at: datetime
    as_of_time: datetime

class LLMNormalizedEvent(BaseModel):
    canonical_summary: str = Field(min_length=20, max_length=1200)
    event_score: float = Field(ge=-1.0, le=1.0)
    event_half_life: Literal["intraday", "3d", "1w", "2w", "1m"]
    regime_hint: Literal["trend", "event", "chop", "risk_off"]
    red_flag: bool
    flow_vs_tech_resolution: Literal[
        "FLOW_DOMINANT_CONTINUATION",
        "TECH_DOMINANT_EXHAUSTION",
        "WAIT_FOR_PULLBACK",
        "NO_TRADE"
    ]
    source_quality_score: float = Field(ge=0.0, le=1.0)
    freshness_score: float = Field(ge=0.0, le=1.0)
    semantic_confidence: float = Field(ge=0.0, le=1.0)
    evidence_spans: List[EvidenceSpan]
    entity_tags: List[str]
    reasons: List[str] = Field(max_items=8)
```

```python
# schemas/flow.py
from pydantic import BaseModel
from typing import Literal, Optional
from datetime import datetime

class FlowSnapshot(BaseModel):
    symbol: str
    window: Literal["INTRADAY", "1D", "3D", "5D"]
    foreign_net: float
    institutional_net: float
    program_net: float
    preliminary_or_final: Literal["PROVISIONAL", "CONFIRMED"]
    vendor: str
    revision_id: str
    snapshot_at: datetime
    as_of_time: datetime
```

```python
# schemas/vectors.py
from pydantic import BaseModel, Field
from typing import List
from datetime import datetime

class VectorPayload(BaseModel):
    symbol: str
    session_type: str
    z_event: List[float] = Field(min_items=64, max_items=64)
    z_social: List[float] = Field(min_items=32, max_items=32)
    z_macro: List[float] = Field(min_items=16, max_items=16)
    encoder_version: str
    tokenizer_version: str
    attention_aggregator_version: str
    prompt_version: str
    source_doc_ids: List[str]
    cluster_ids: List[str]
    generated_at: datetime
    as_of_time: datetime
```

```python
# schemas/execution.py
from pydantic import BaseModel, Field
from typing import Literal, Optional
from datetime import datetime

class ExecutionRequest(BaseModel):
    symbol: str
    decision_timestamp: datetime
    broker_capabilities_version: str
    venue_eligibility: Literal["KRX_ONLY", "KRX_PLUS_NXT"]
    session_calendar_version: str
    venue_freshness_ok: bool
    session_liquidity_ok: bool

class ExecutionPlan(BaseModel):
    symbol: str
    selected_venue: Literal["KRX", "NXT"]
    selected_session_type: Literal["NXT_PRE", "CORE_DAY", "CLOSE_PRICE", "NXT_AFTER"]
    scheduled_exec_time: datetime
    fill_proxy: Literal["NEXT_COMPLETED_5M_VWAP"]
    expected_cost_bps: float
    rollover_reason: Optional[str]
```

```python
# schemas/predictions.py
from pydantic import BaseModel, Field
from datetime import datetime

class FusedPrediction(BaseModel):
    symbol: str
    as_of_time: datetime
    session_type: str
    er_5d: float
    er_20d: float
    dd_20d: float
    p_up_20d: float = Field(ge=0.0, le=1.0)
    flow_persist: float = Field(ge=0.0, le=1.0)
    uncertainty: float = Field(ge=0.0, le=1.0)
    regime_final: str
    semantic_branch_enabled: bool
    text_branch_enabled: bool
```

```python
# schemas/decisions.py
from pydantic import BaseModel, Field
from typing import List
from datetime import datetime

class TradeDecision(BaseModel):
    symbol: str
    action: str
    target_weight: float
    tranche_ratio: float
    session_type: str
    selected_venue: str
    vetoes_triggered: List[str]
    rationale_codes: List[str]
    risk_budget_used: float
    expected_slippage_bps: float
    as_of_time: datetime
    execution_time: datetime
```

---

## 4) Service Contracts / Interfaces by Module

```python
class MarketDataIngestionService:
    def ingest_session_bars(self, as_of_time) -> None: ...
    def ingest_quote_proxies(self, as_of_time) -> None: ...

class FlowSnapshotStoreService:
    def upsert_snapshot(self, snapshot: FlowSnapshot) -> None: ...
    def get_latest(self, symbol: str, as_of_time, provisional_only: bool): ...

class EventRetrievalService:
    def retrieve(self, symbols: list[str], as_of_time) -> list[dict]: ...

class EventDedupClusteringService:
    def normalize_and_cluster(self, docs: list[dict], as_of_time) -> list[EventMetadata]: ...

class LLMEventNormalizerService:
    def normalize(self, event_batch: list[EventMetadata], as_of_time) -> list[LLMNormalizedEvent]: ...

class TextEncoderService:
    def encode_items(self, summaries: list[str], spans: list[str]) -> list[list[float]]: ...

class AttentionAggregatorService:
    def aggregate(self, encoded_items, context) -> VectorPayload: ...

class VectorizationPipeline:
    def run(self, symbol: str, as_of_time) -> VectorPayload: ...

class FeatureStoreService:
    def build_online_features(self, symbol: str, as_of_time): ...
    def build_offline_features(self, start, end): ...

class PredictorTrainingPipeline:
    def train(self, dataset_id: str) -> str: ...

class CalibrationPipeline:
    def calibrate(self, model_id: str, validation_set_id: str) -> str: ...

class LiveInferenceService:
    def infer_anchor(self, anchor_time) -> list[FusedPrediction]: ...

class VenueRouterExecutionMapper:
    def map_execution(self, req: ExecutionRequest) -> ExecutionPlan: ...

class DecisionEngineService:
    def decide(self, pred: FusedPrediction, exec_plan: ExecutionPlan) -> TradeDecision: ...

class RiskEngineService:
    def apply_vetoes(self, decision: TradeDecision, context) -> TradeDecision: ...

class PortfolioEngineService:
    def rebalance(self, decisions: list[TradeDecision], positions, constraints): ...

class Backtester:
    def run(self, start, end, config_version) -> dict: ...

class BrokerGateway:
    def submit_orders(self, orders) -> list[str]: ...

class MonitoringAuditService:
    def emit_metrics(self, payload: dict) -> None: ...
    def log_decision(self, decision: TradeDecision, lineage: dict) -> None: ...

class OrchestrationService:
    def run_anchor_workflow(self, anchor_time) -> None: ...
```

---

## 5) Training Pipeline Pseudocode

```python
def training_pipeline(train_start, train_end, versions):
    # 0) Freeze version bundle
    vb = lock_versions(versions)

    # 1) Build universe snapshots by day
    universe = build_universe(
        min_adv_60d_krw=3e9,
        min_mcap_krw=1.5e11,
        exclude_spac_pref_etf_etn=True,
        require_fresh_metadata=True,
        version=vb.universe_filter_version
    )

    # 2) Ingest historical market/session bars (KRX + NXT) with as_of guarantees
    bars = load_session_aware_bars(train_start, train_end, asof_strict=True)

    # 3) Load flow snapshots (PROVISIONAL separate from CONFIRMED)
    flow = load_flow_snapshots(train_start, train_end)
    assert_no_confirmed_flow_in_intraday_models(flow)

    # 4) Retrieve docs with availability timestamps
    docs = load_event_docs(train_start, train_end)
    docs = filter(lambda d: d.available_at <= d.as_of_time, docs)

    # 5) Dedup/cluster + LLM normalization (schema validated)
    clustered = dedup_cluster(docs)
    normalized = llm_normalize_with_retry(clustered, max_retry=1)
    normalized = fallback_neutral_on_failure(normalized)

    # 6) Vectorization stage (BERT + hierarchical attention)
    vectors = build_vectors(normalized, vb.vectorizer_version, vb.attention_aggregator_version)

    # 7) Build features per decision anchor and session_type
    X = build_features(
        bars=bars,
        flow=flow,
        vectors=vectors,
        include_missingness_flags=True,
        session_aware=True
    )

    # 8) Label generation using EXACT same ExecutionMapper as live
    y = generate_labels_with_execution_mapper(
        X,
        fill_proxy="NEXT_COMPLETED_5M_VWAP",
        horizons=[5, 20],
        include_costs=True,
        include_session_cost_curves=True
    )

    # 9) Drop/flag interrupted horizons (halts/suspensions)
    X, y = apply_censoring_policy(X, y)

    # 10) Train model heads
    model = train_lightgbm_multitask(X, y)

    # 11) Calibrate p_up_20d / uncertainty / dd outputs by session_type
    calibrators = fit_calibrators(model, X_val, y_val, by_session=True)

    # 12) Persist model package with full lineage
    package_id = register_model_package(model, calibrators, vb)

    return package_id
```

---

## 6) Live Inference Pipeline Pseudocode

```python
def live_anchor_run(anchor_time):
    vb = get_active_versions()

    # 1) Determine session and tradability
    session_type = session_classifier(anchor_time)
    universe = load_universe_snapshot(anchor_time, vb.universe_filter_version)

    # 2) Ingest latest market + flow snapshots with freshness checks
    market_ok, market_data = load_market_data(anchor_time)
    flow_data = load_provisional_flow(anchor_time)

    # 3) Semantic refresh policy
    do_semantic = is_semantic_refresh_anchor(anchor_time) or event_burst_detected(anchor_time)

    if do_semantic:
        docs = retrieve_new_docs(universe.top_candidates, anchor_time)
        clustered = dedup_cluster(docs)
        sem = llm_normalize_with_retry(clustered, max_retry=1)
        sem = fallback_neutral_on_failure(sem)
        vectors = vectorize(sem)
    else:
        sem = load_last_valid_semantics(anchor_time)
        vectors = load_last_valid_vectors(anchor_time)

    # 4) Build online features
    feats = build_online_features(market_data, flow_data, sem, vectors, anchor_time)

    # 5) Predict + calibrate
    pred = model_predict(feats)
    pred = calibrate(pred, session_type)

    # 6) Map execution per symbol via venue-aware mapper
    exec_plans = [execution_mapper(p.symbol, anchor_time, vb) for p in pred]

    # 7) Decision + risk veto + portfolio constraints
    dec = [decision_engine(p, e) for p, e in zip(pred, exec_plans)]
    dec = [risk_engine(d) for d in dec]
    orders = portfolio_engine(dec)

    # 8) Submit orders where allowed
    order_ids = broker_gateway_submit(orders)

    # 9) Audit and monitoring
    log_all(anchor_time, feats, pred, dec, order_ids, vb)
    emit_metrics(anchor_time)
```

---

## 7) Decision Engine Pseudocode

```python
def decision_engine(pred, exec_plan, ctx):
    # Hard veto layer
    if ctx.market_risk_off or pred.regime_final == "risk_off" or ctx.red_flag:
        return make_decision("NO_TRADE" if ctx.no_position else "REDUCE", size=0.0, reason="RISK_VETO")

    # Direction gate: flow + event + 120m trend
    long_direction_ok = (
        pred.flow_persist >= ctx.th.flow_persist_min and
        ctx.event_score >= ctx.th.event_score_min and
        ctx.trend_120m_ok
    )

    if not long_direction_ok:
        return make_decision("NO_TRADE", size=0.0, reason="DIRECTION_NOT_CONFIRMED")

    # Timing and sizing gate with 60m extension
    if ctx.tech_extension_high:
        action = "WAIT_PULLBACK" if ctx.no_position else "BUY_HALF"
    else:
        action = "BUY"

    # Size modifiers
    raw_size = base_risk_budget(ctx.equity, ctx.risk_per_trade)
    raw_size *= drawdown_shrink(pred.dd_20d)
    raw_size *= uncertainty_shrink(pred.uncertainty)
    raw_size *= liquidity_shrink(ctx.liquidity_score, ctx.session_type)

    # Session aggressiveness
    if ctx.session_type in ["NXT_PRE", "NXT_AFTER", "CLOSE_PRICE"]:
        raw_size *= ctx.session_aggr_mult  # e.g. 0.3~0.7

    # Venue uncertainty fallback
    if not ctx.venue_freshness_ok:
        action = "NO_TRADE"
        raw_size = 0.0

    # tranche mapping
    tranche = tranche_ratio_for_action(action)  # BUY=0.4/0.3/0.3 staged statefully

    return TradeDecision(
        symbol=pred.symbol,
        action=action,
        target_weight=clip_by_portfolio_caps(raw_size),
        tranche_ratio=tranche,
        session_type=ctx.session_type,
        selected_venue=exec_plan.selected_venue,
        vetoes_triggered=ctx.vetoes,
        rationale_codes=ctx.reason_codes,
        risk_budget_used=raw_size,
        expected_slippage_bps=ctx.expected_slippage_bps,
        as_of_time=ctx.as_of_time,
        execution_time=exec_plan.scheduled_exec_time,
    )
```

---

## 8) Backtest Rules with Explicit No-Lookahead Guarantees

1. **Single execution logic parity**: `ExecutionMapper` imported from live module, no alternative implementation.
2. **As-of filtering**:
   - Feature query: only records with `timestamp <= decision_timestamp`.
   - Event docs: require `available_at <= decision_timestamp`.
   - Flow: use provisional snapshots for intraday anchors.
3. **Session-aware bars**:
   - No synthetic continuous bars across closed phases.
   - Dedicated phase features for NXT_PRE/CLOSE_PRICE/NXT_AFTER.
4. **Fill model**:
   - Next completed 5m VWAP in selected eligible venue/session.
   - If phase end <3 min, roll forward.
5. **Cost model**:
   - Separate cost curves by session type and venue.
   - Include commission, taxes, spread, slippage, impact, participation cap.
6. **Universe versioning**:
   - Universe frozen daily with explicit versions.
7. **Corporate action handling**:
   - Features may use adjusted series; labels/execution prices use raw executable prices + explicit cashflow handling.
8. **Suspension/halt policy**:
   - If no next tradable opportunity, label unavailable.
   - If horizon interrupted: `horizon_interrupted=true`, excluded from default alpha training.

---

## 9) Example JSON Payloads

### 9.1 LLM Structured Output
```json
{
  "canonical_summary": "실적 가이던스 상향 및 주요 고객사 수주 공시가 확인되며, 단기 실적 모멘텀이 강화됨.",
  "event_score": 0.62,
  "event_half_life": "2w",
  "regime_hint": "event",
  "red_flag": false,
  "flow_vs_tech_resolution": "FLOW_DOMINANT_CONTINUATION",
  "source_quality_score": 0.86,
  "freshness_score": 0.91,
  "semantic_confidence": 0.79,
  "evidence_spans": [
    {"doc_id": "dart_20260319_001", "start_char": 142, "end_char": 218, "text_snippet": "...수주 금액은 최근 매출 대비..."}
  ],
  "entity_tags": ["earnings_guidance", "order_backlog", "semiconductor"],
  "reasons": ["GUIDANCE_UP", "ORDER_BOOK_STRENGTH"]
}
```

### 9.2 Event Metadata
```json
{
  "canonical_event_id": "evt_005899",
  "cluster_id": "clu_22014",
  "source_lineage": ["news_a_991", "news_b_113", "dart_20260319_001"],
  "source_type": "DART",
  "source_quality_score": 0.95,
  "novelty_score": 0.74,
  "published_at": "2026-03-19T11:58:10+09:00",
  "first_seen_at": "2026-03-19T11:59:03+09:00",
  "retrieved_at": "2026-03-19T12:00:00+09:00",
  "available_at": "2026-03-19T11:59:03+09:00",
  "as_of_time": "2026-03-19T12:30:00+09:00"
}
```

### 9.3 Vector Payload Metadata
```json
{
  "symbol": "005930",
  "session_type": "CORE_DAY",
  "z_event": [0.01, -0.02, 0.03, "... 64 dims ..."],
  "z_social": [0.04, -0.01, "... 32 dims ..."],
  "z_macro": [0.02, "... 16 dims ..."],
  "encoder_version": "ko_bert_fin_v3",
  "tokenizer_version": "spm_kr_fin_v2",
  "attention_aggregator_version": "hier_attn_v2",
  "prompt_version": "normalizer_prompt_v1.2",
  "source_doc_ids": ["dart_20260319_001", "news_a_991"],
  "cluster_ids": ["clu_22014"],
  "generated_at": "2026-03-19T12:31:10+09:00",
  "as_of_time": "2026-03-19T12:30:00+09:00"
}
```

### 9.4 Fused Prediction Output
```json
{
  "symbol": "005930",
  "as_of_time": "2026-03-19T12:30:00+09:00",
  "session_type": "CORE_DAY",
  "er_5d": 0.018,
  "er_20d": 0.043,
  "dd_20d": 0.035,
  "p_up_20d": 0.66,
  "flow_persist": 0.71,
  "uncertainty": 0.28,
  "regime_final": "trend",
  "semantic_branch_enabled": true,
  "text_branch_enabled": true
}
```

### 9.5 Trade Decision Output
```json
{
  "symbol": "005930",
  "action": "BUY_HALF",
  "target_weight": 0.022,
  "tranche_ratio": 0.30,
  "session_type": "CORE_DAY",
  "selected_venue": "KRX",
  "vetoes_triggered": [],
  "rationale_codes": ["FLOW_EVENT_TREND_OK", "SHORT_TERM_EXTENSION_HIGH"],
  "risk_budget_used": 0.004,
  "expected_slippage_bps": 9.5,
  "as_of_time": "2026-03-19T12:30:00+09:00",
  "execution_time": "2026-03-19T12:35:00+09:00"
}
```

---

## 10) Practical MVP Plan (8–12 weeks)

### Phase 0 (Week 1–2): Data and session correctness
- Implement session calendar + phase mapping engine.
- Build universe snapshot with versioning + exclusions.
- Build KRX/NXT ingestion skeleton and freshness monitors.
- **Exit criterion**: deterministic `ExecutionMapper` test suite passes.

### Phase 1 (Week 3–4): Numeric baseline (no semantics)
- Feature store with 60m/120m/session features.
- Flow snapshot store with provisional/confirmed segregation.
- LightGBM baseline for `er_5d`, `er_20d`, `dd_20d`, `p_up_20d`.
- Cost-aware backtester parity with live execution mapper.
- **Exit criterion**: baseline live-paper simulation with full audit logs.

### Phase 2 (Week 5–7): Event normalization + vector pipeline
- Event retrieval + dedup/cluster.
- LLM normalizer strict JSON and fallback.
- Korean BERT encoder + hierarchical attention outputs.
- Integrate scalar semantics + vectors into predictor.
- **Exit criterion**: schema violation rate <1%, degraded-mode operational.

### Phase 3 (Week 8–10): Decision/risk hardening
- Decision policy + explicit exit categories.
- Portfolio constraints, beta cap, sector cap, exposure throttle.
- Venue routing and after-market restrictions.
- **Exit criterion**: stress backtests + outage drills pass.

### Phase 4 (Week 11–12): controlled production shadow
- Run live shadow + paper execution at anchors.
- Daily model and slippage diagnostics.
- Gate to capital only after calibration/slippage stability.

---

## 11) V2 Upgrade Plan

1. Multi-model ensembling by session_type (e.g., dedicated NXT_AFTER model).
2. Enhanced venue IS model using richer quote/depth microstructure.
3. Better censoring/survival modeling for halt/interruption cases.
4. Cross-asset regime integration (KOSPI200 futures term structure, FX vol).
5. Dynamic risk budget allocator using uncertainty regime buckets.
6. Active learning loop for low-confidence semantic events.
7. Multi-broker smart routing abstraction with failover.

---

## 12) Critical Data Gaps and Assumptions

### Production-grade available (if contracted/licensed)
- Real-time KRX market data and official calendar.
- Broker order routing APIs and execution reports.
- DART filings timestamps and retrieval logs.

### Research-mode approximation
- Synthetic provisional-flow reconstruction when only confirmed EOD flows are archived.
- Approximate session cost curves without full quote/depth history.
- Event availability approximated by retrieval timestamp when `available_at` missing.

### Critical data gaps (blockers for production claims)
1. **Archived provisional intraday flow snapshots missing**
   - Consequence: intraday flow models cannot be certified production-grade; leakage risk.
2. **NXT historical depth/quote coverage incomplete**
   - Consequence: venue IS model weak; routing confidence reduced.
3. **Broker NXT routing support not confirmed per symbol/session**
   - Consequence: must run KRX-only degraded mode for affected names.
4. **Real-time licensed venue feed absent (relying on delayed web pages)**
   - Consequence: cannot make production tradable decisions safely.
5. **Cross-venue consolidated quotes unavailable**
   - Consequence: cannot claim true best execution optimization.

---

## 13) Test Plan

### 13.1 No-lookahead checks
- Unit tests for all feature joins enforcing `timestamp <= as_of_time`.
- Dataset-level invariant scans for leakage columns.
- Replay test using historical anchors and known late documents.

### 13.2 Schema validation
- Contract tests for all Pydantic models.
- LLM normalizer: malformed JSON, missing required fields, out-of-range values.

### 13.3 Degraded mode
- Simulate retrieval timeout, LLM outage, vectorizer failure, attention failure.
- Verify neutral semantics + missing flags + no forced BUY behavior.

### 13.4 Timestamp boundary enforcement
- Anchors near phase boundaries (e.g., 15:18, 15:29, 19:58).
- Verify +3 minute end-phase rollover rules.

### 13.5 Provisional-vs-confirmed flow leakage prevention
- For intraday anchors, assert only provisional snapshots are queryable.
- Negative tests ensure confirmed snapshots raise leakage exception.

### 13.6 NXT-vs-KRX session mapping correctness
- Parameterized tests for all session phases and holidays.
- Ensure NXT after-market execution stays on same trading day label.

### 13.7 Venue fallback correctness
- Stale eligibility snapshot -> disable NXT routing.
- NXT feed outage -> KRX-only if tradable.
- Venue clock mismatch -> fail closed and suppress unsafe entries.

---

## Orchestration Choice and Rationale
- **Temporal preferred** for long-running, stateful, retry-aware workflows with idempotency keys.
- Workflow per anchor:
  1) ingest/freshness checks
  2) optional semantic refresh
  3) vectorization + feature build
  4) predict/calibrate
  5) execution mapping + decision/risk/portfolio
  6) broker submit
  7) audit + metrics
- Circuit breaker per provider (LLM/search/vectorizer/broker).
- Dead-letter queue for irrecoverable records with full lineage.

---

## Korean Financial Language Validation Benchmark (Small but Mandatory)
- Build 4-slice benchmark set (200–400 items total):
  1) DART formal disclosure language
  2) IR/call style narratives
  3) mainstream Korean finance news
  4) Korean social/X rumor and jargon snippets
- Evaluate:
  - schema validity rate
  - extraction consistency across reruns
  - semantic drift/stability (same input repeated)
  - confidence calibration against human labels
- Policy:
  - If semantic confidence low or instability high -> downweight/disable semantic branch.

---

## Cost-Control Policy (LLM/Search Opex Target KRW 500~1000/day)
- Restrict full semantic refresh to anchors: 08:10, 09:35, 15:45, 20:05.
- Use event-triggered bursts only for high novelty/conflict.
- Symbol-level semantic calls only on top-ranked candidates and event-impacted names.
- Cache canonical event objects and embeddings with freshness TTL.
- Secondary cheap model for cross-check on low-confidence/high-impact items only.

---

## Non-Negotiable Operational Assertions
1. If real-time licensed market/venue feeds are missing, the system is **not production tradable**.
2. If provisional flow archives are missing, intraday flow alpha is **research-only**.
3. If NXT routing capability and eligibility state are stale, NXT must be disabled.
4. If LLM/schema/vector pipeline is degraded, system must continue in conservative numeric-only mode.
5. No lookahead breaches are release-blocking defects.
