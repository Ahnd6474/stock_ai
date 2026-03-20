from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Mapping, Protocol, Sequence, runtime_checkable

from .schemas import (
    EventMetadata,
    ExecutionPlan,
    ExecutionRequest,
    FlowSnapshot,
    FusedPrediction,
    LLMNormalizedEvent,
    TradeDecision,
    VectorPayload,
)


@runtime_checkable
class MarketDataIngestionService(Protocol):
    def ingest_session_bars(self, as_of_time: datetime) -> None: ...

    def ingest_quote_proxies(self, as_of_time: datetime) -> None: ...


@runtime_checkable
class FlowSnapshotStoreService(Protocol):
    def upsert_snapshot(self, snapshot: FlowSnapshot) -> None: ...

    def get_latest(self, symbol: str, as_of_time: datetime, intraday_anchor: bool) -> FlowSnapshot | None: ...


@runtime_checkable
class EventRetrievalService(Protocol):
    def retrieve(self, symbols: Sequence[str], as_of_time: datetime) -> Sequence[Mapping[str, object]]: ...


@runtime_checkable
class EventDedupAndClusteringService(Protocol):
    def normalize_and_cluster(
        self,
        docs: Sequence[Mapping[str, object]],
        as_of_time: datetime,
    ) -> Sequence[EventMetadata]: ...


@runtime_checkable
class LLMEventNormalizerService(Protocol):
    def normalize(self, payload: Mapping[str, object], retry_once: bool = True) -> LLMNormalizedEvent: ...


@runtime_checkable
class TextEncoderService(Protocol):
    def encode(self, text: str, dim: int) -> list[float]: ...

    def batch_encode(self, texts: Sequence[str], dim: int) -> Sequence[list[float]]: ...


@runtime_checkable
class AttentionAggregatorService(Protocol):
    def aggregate(self, items: Sequence[Mapping[str, object]], dim: int) -> list[float]: ...

    def aggregate_by_cluster(self, cluster_items: Mapping[str, Sequence[Mapping[str, object]]], dim: int) -> list[float]: ...


@runtime_checkable
class VectorizationPipelineService(Protocol):
    def build(
        self,
        summary: str,
        social: str = "",
        macro: str = "",
        source_doc_ids: list[str] | None = None,
        cluster_ids: list[str] | None = None,
        as_of_time: datetime | None = None,
        session_type: str = "OFF_MARKET",
    ) -> Mapping[str, object]: ...


@runtime_checkable
class FeatureStoreService(Protocol):
    def build_online_features(self, symbol: str, as_of_time: datetime, session_type: str) -> Mapping[str, object]: ...

    def build_offline_features(self, symbol: str, as_of_time: datetime, session_type: str) -> Mapping[str, object]: ...


@runtime_checkable
class PredictorTrainingService(Protocol):
    def train_walk_forward(
        self,
        rows: Sequence[Mapping[str, object]],
        feature_keys: Sequence[str],
        label_key: str,
        artifact_dir: str | Path,
        model_version: str = "linear_baseline_v1",
    ) -> tuple[Path, Path]: ...

    def train_multi_head(
        self,
        rows: Sequence[Mapping[str, object]],
        feature_keys: Sequence[str],
        artifact_dir: str | Path,
        label_keys: tuple[str, str, str] = ("er_20d", "dd_20d", "p_up_20d"),
        model_version: str = "linear_multihead_v1",
    ) -> Path: ...


@runtime_checkable
class CalibrationPipelineService(Protocol):
    def transform(self, value: float) -> float: ...


@runtime_checkable
class LiveInferenceServiceContract(Protocol):
    def run_for_symbol(
        self,
        symbol: str,
        as_of_time: datetime,
        raw_event_payload: Mapping[str, object],
        features: Mapping[str, object],
        venue_eligibility: str,
    ) -> TradeDecision: ...


@runtime_checkable
class VenueRouterExecutionMapperService(Protocol):
    def map_execution(self, req: ExecutionRequest) -> ExecutionPlan: ...


@runtime_checkable
class DecisionEngineService(Protocol):
    def decide(
        self,
        pred: FusedPrediction,
        plan: ExecutionPlan,
        *,
        trend_120m_ok: bool,
        tech_extension_high: bool,
        market_risk_off: bool,
        no_position: bool,
    ) -> TradeDecision: ...


@runtime_checkable
class RiskEngineService(Protocol):
    def apply(self, decision: TradeDecision, state: object) -> TradeDecision: ...


@runtime_checkable
class PortfolioEngineService(Protocol):
    def apply(self, decisions: Sequence[TradeDecision], current_gross: float, current_turnover: float = 0.0) -> list[TradeDecision]: ...


@runtime_checkable
class BacktesterService(Protocol):
    def run_trade(self, req: ExecutionRequest, as_of_time: datetime, bars: Sequence[object], horizon_bars: int = 20) -> object: ...


@runtime_checkable
class BrokerGatewayService(Protocol):
    def submit(self, order: object, market_price: float, liquidity_score: float) -> object: ...

    def cancel(self, order_id: str, at: datetime) -> object: ...

    def replace(self, order_id: str, new_qty: int, at: datetime) -> object: ...

    def reconcile(self, order_id: str, at: datetime) -> object: ...


@runtime_checkable
class ExperimentTrackingService(Protocol):
    def register_model_package(self, metadata: Mapping[str, object]) -> str: ...


@runtime_checkable
class MonitoringAuditService(Protocol):
    def snapshot(self) -> object: ...

    def append(self, entry: object) -> None: ...


@runtime_checkable
class OrchestrationService(Protocol):
    def run_anchor(
        self,
        symbol: str,
        anchor_time: datetime,
        payload: Mapping[str, object],
        features: Mapping[str, object],
        venue_eligibility: str,
        retry: int = 1,
    ) -> object: ...
