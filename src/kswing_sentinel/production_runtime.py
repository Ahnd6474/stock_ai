from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Callable
from typing import Literal

from pydantic import BaseModel

from .audit_log import AuditLogStore, DecisionAuditEntry, RuntimeAuditEvent
from .broker_gateway import BrokerCapabilities, BrokerGateway, OrderRequest
from .live import LiveInferenceService
from .monitoring import Monitoring
from .schemas import TradeDecision


TradingMode = Literal["BLOCKED", "KRX_ONLY", "KRX_NXT"]


class ProductionRuntimeConfig(BaseModel):
    requested_trading_mode: Literal["KRX_ONLY", "KRX_NXT"] = "KRX_NXT"
    allow_krx_only_fallback: bool = True
    require_cross_venue_quotes_for_nxt: bool = False
    required_env_vars: list[str] = []
    kill_switch_path: str | None = None
    audit_log_path: str | None = None
    metrics_log_path: str | None = None

    @classmethod
    def from_toml(cls, path: str | Path) -> "ProductionRuntimeConfig":
        with Path(path).open("rb") as fh:
            payload = tomllib.load(fh)
        return cls(**payload)


class ModelRuntimeRequirements(BaseModel):
    model_version: str = "v1"
    requires_realtime_market_data: bool = True
    requires_intraday_provisional_flow_archive: bool = True
    requires_intraday_provisional_flow_live: bool = True
    requires_semantic_live: bool = False
    requires_vectorizer_live: bool = False
    requires_nxt_for_strategy: bool = False


class RuntimeDependencyState(BaseModel):
    realtime_krx_feed_available: bool
    realtime_krx_feed_licensed: bool
    broker_api_available: bool
    broker_live_order_enabled: bool
    broker_krx_routing_available: bool
    provisional_flow_archive_available: bool
    provisional_flow_live_available: bool
    model_artifact_available: bool = True
    calibrator_available: bool = True
    audit_sink_writable: bool = True
    nxt_feed_available: bool = False
    nxt_feed_licensed: bool = False
    broker_nxt_routing_available: bool = False
    nxt_eligibility_snapshot_fresh: bool = False
    cross_venue_quotes_available: bool = False
    semantic_provider_available: bool = True
    vectorizer_available: bool = True


class ProductionReadinessReport(BaseModel):
    checked_at: datetime
    ready: bool
    trading_mode: TradingMode
    blocking_issues: list[str]
    warnings: list[str]
    degraded_flags: list[str]


class LiveTradingBlockedError(RuntimeError):
    def __init__(self, report: ProductionReadinessReport) -> None:
        self.report = report
        message = "; ".join(report.blocking_issues) or "live trading blocked"
        super().__init__(message)


class LiveOrderResult(BaseModel):
    symbol: str
    action: str
    selected_venue: str
    order_submitted: bool
    order_id: str | None = None
    execution_status: str | None = None
    filled_qty: int = 0
    rationale_codes: list[str]
    vetoes_triggered: list[str]


@dataclass(frozen=True)
class AnchorBatchDeadLetterRecord:
    anchor_time: datetime
    symbols: list[str]
    payload_by_symbol: dict[str, dict]
    features_by_symbol: dict[str, dict]
    venue_eligibility_by_symbol: dict[str, str]
    error_message: str
    attempts: int
    failed_at: datetime


class ProductionCircuitBreakerOpen(RuntimeError):
    pass


def _model_dump(model: BaseModel) -> dict:
    return model.model_dump() if hasattr(model, "model_dump") else model.dict()


class ProductionReadinessGate:
    def evaluate(
        self,
        config: ProductionRuntimeConfig,
        deps: RuntimeDependencyState,
        broker_capabilities: BrokerCapabilities,
        model_requirements: ModelRuntimeRequirements,
        as_of_time: datetime,
    ) -> ProductionReadinessReport:
        blocking: list[str] = []
        warnings: list[str] = []
        degraded: list[str] = []
        trading_mode: TradingMode = config.requested_trading_mode

        missing_env = [key for key in config.required_env_vars if not os.getenv(key)]
        if missing_env:
            blocking.append(f"MISSING_ENV:{','.join(sorted(missing_env))}")

        if self._kill_switch_active(config.kill_switch_path):
            blocking.append("KILL_SWITCH_ACTIVE")

        if model_requirements.requires_realtime_market_data:
            if not deps.realtime_krx_feed_available:
                blocking.append("KRX_REALTIME_FEED_UNAVAILABLE")
            if not deps.realtime_krx_feed_licensed:
                blocking.append("KRX_REALTIME_FEED_UNLICENSED")

        if not deps.broker_api_available:
            blocking.append("BROKER_API_UNAVAILABLE")
        if not deps.broker_live_order_enabled:
            blocking.append("BROKER_LIVE_ORDERS_DISABLED")
        if not deps.broker_krx_routing_available:
            blocking.append("BROKER_KRX_ROUTING_UNAVAILABLE")
        if not broker_capabilities.supports_live_trading:
            blocking.append("BROKER_CAPABILITY_LIVE_DISABLED")
        if broker_capabilities.dry_run_only:
            blocking.append("BROKER_DRY_RUN_ONLY")
        if not broker_capabilities.supports_krx:
            blocking.append("BROKER_CAPABILITY_KRX_DISABLED")

        if model_requirements.requires_intraday_provisional_flow_archive and not deps.provisional_flow_archive_available:
            blocking.append("PROVISIONAL_FLOW_ARCHIVE_MISSING")
        if model_requirements.requires_intraday_provisional_flow_live and not deps.provisional_flow_live_available:
            blocking.append("PROVISIONAL_FLOW_LIVE_MISSING")

        if not deps.model_artifact_available:
            blocking.append("MODEL_ARTIFACT_MISSING")
        if not deps.calibrator_available:
            blocking.append("CALIBRATOR_MISSING")
        if not deps.audit_sink_writable:
            blocking.append("AUDIT_SINK_NOT_WRITABLE")

        if not deps.semantic_provider_available:
            warnings.append("SEMANTIC_PROVIDER_DEGRADED_TO_NUMERIC_ONLY")
            degraded.append("SEMANTIC_DISABLED")
        if model_requirements.requires_semantic_live and not deps.semantic_provider_available:
            blocking.append("SEMANTIC_PROVIDER_REQUIRED_BUT_UNAVAILABLE")

        if not deps.vectorizer_available:
            warnings.append("TEXT_VECTORIZER_DEGRADED")
            degraded.append("TEXT_BRANCH_DISABLED")
        if model_requirements.requires_vectorizer_live and not deps.vectorizer_available:
            blocking.append("VECTORIZER_REQUIRED_BUT_UNAVAILABLE")

        if trading_mode == "KRX_NXT":
            nxt_issues: list[str] = []
            if not broker_capabilities.supports_nxt:
                nxt_issues.append("BROKER_CAPABILITY_NXT_DISABLED")
            if not deps.broker_nxt_routing_available:
                nxt_issues.append("BROKER_NXT_ROUTING_UNAVAILABLE")
            if not deps.nxt_feed_available:
                nxt_issues.append("NXT_FEED_UNAVAILABLE")
            if not deps.nxt_feed_licensed:
                nxt_issues.append("NXT_FEED_UNLICENSED")
            if not deps.nxt_eligibility_snapshot_fresh:
                nxt_issues.append("NXT_ELIGIBILITY_STALE")
            if config.require_cross_venue_quotes_for_nxt and not deps.cross_venue_quotes_available:
                nxt_issues.append("CROSS_VENUE_QUOTES_MISSING")

            if nxt_issues:
                if model_requirements.requires_nxt_for_strategy or not config.allow_krx_only_fallback:
                    blocking.extend(nxt_issues)
                else:
                    warnings.extend(nxt_issues)
                    degraded.append("NXT_DISABLED_KRX_ONLY_FALLBACK")
                    trading_mode = "KRX_ONLY"

        ready = not blocking
        if not ready:
            trading_mode = "BLOCKED"

        return ProductionReadinessReport(
            checked_at=as_of_time,
            ready=ready,
            trading_mode=trading_mode,
            blocking_issues=blocking,
            warnings=warnings,
            degraded_flags=degraded,
        )

    @staticmethod
    def _kill_switch_active(path: str | None) -> bool:
        if not path:
            return False
        file_path = Path(path)
        if not file_path.exists():
            return False
        content = file_path.read_text(encoding="utf-8").strip().lower()
        return content in {"1", "true", "on", "stop", "halt"}


class ProductionTradingEngine:
    def __init__(
        self,
        *,
        live_service: LiveInferenceService | None = None,
        broker_gateway: BrokerGateway,
        audit_store: AuditLogStore | None = None,
        monitoring: Monitoring | None = None,
        readiness_gate: ProductionReadinessGate | None = None,
    ) -> None:
        self.live = live_service or LiveInferenceService()
        self.broker_gateway = broker_gateway
        self.audit_store = audit_store or AuditLogStore()
        self.monitoring = monitoring or Monitoring()
        self.readiness_gate = readiness_gate or ProductionReadinessGate()
        self._symbol_run_cache: dict[str, LiveOrderResult] = {}

    @classmethod
    def from_runtime_config(
        cls,
        *,
        runtime_config: ProductionRuntimeConfig,
        broker_gateway: BrokerGateway,
        live_service: LiveInferenceService | None = None,
        readiness_gate: ProductionReadinessGate | None = None,
    ) -> "ProductionTradingEngine":
        return cls(
            live_service=live_service,
            broker_gateway=broker_gateway,
            audit_store=AuditLogStore(runtime_config.audit_log_path),
            monitoring=Monitoring(runtime_config.metrics_log_path),
            readiness_gate=readiness_gate,
        )

    def run_live_anchor_batch(
        self,
        *,
        symbols: list[str],
        anchor_time: datetime,
        payload_by_symbol: dict[str, dict],
        features_by_symbol: dict[str, dict],
        venue_eligibility_by_symbol: dict[str, str],
        last_price_by_symbol: dict[str, float],
        dependency_state: RuntimeDependencyState,
        runtime_config: ProductionRuntimeConfig,
        model_requirements: ModelRuntimeRequirements,
        equity_krw: float,
        position_qty_by_symbol: dict[str, int] | None = None,
        liquidity_score_by_symbol: dict[str, float] | None = None,
    ) -> tuple[ProductionReadinessReport, dict[str, LiveOrderResult]]:
        report = self.readiness_gate.evaluate(
            runtime_config,
            dependency_state,
            self.broker_gateway.capabilities,
            model_requirements,
            anchor_time,
        )
        self.audit_store.append_runtime_event(
            RuntimeAuditEvent(
                event_type="readiness_check",
                event_time=anchor_time,
                payload=_model_dump(report),
            )
        )
        if report.degraded_flags:
            self.monitoring.record_degraded_mode()
        if not report.ready:
            self.monitoring.emit_snapshot(
                anchor_time,
                {"trading_mode": report.trading_mode, "blocking_issues": report.blocking_issues},
            )
            raise LiveTradingBlockedError(report)

        results: dict[str, LiveOrderResult] = {}
        position_qty_by_symbol = position_qty_by_symbol or {}
        liquidity_score_by_symbol = liquidity_score_by_symbol or {}
        broker_supports_nxt = report.trading_mode == "KRX_NXT" and self.broker_gateway.capabilities.supports_nxt

        for symbol in symbols:
            cache_key = self._anchor_symbol_key(symbol, anchor_time)
            cached_result = self._symbol_run_cache.get(cache_key)
            if cached_result is not None:
                results[symbol] = cached_result
                continue

            venue_eligibility = venue_eligibility_by_symbol.get(symbol, "KRX_ONLY")
            if report.trading_mode == "KRX_ONLY":
                venue_eligibility = "KRX_ONLY"

            features = dict(features_by_symbol.get(symbol, {}))
            features.setdefault("text_branch_enabled", dependency_state.vectorizer_available)

            decision = self.live.run_for_symbol(
                symbol=symbol,
                as_of_time=anchor_time,
                raw_event_payload=payload_by_symbol.get(symbol, {}),
                features=features,
                venue_eligibility=venue_eligibility,
                broker_supports_nxt=broker_supports_nxt,
                venue_freshness_ok=dependency_state.nxt_feed_available if broker_supports_nxt else dependency_state.realtime_krx_feed_available,
                session_liquidity_ok=bool(features.get("session_liquidity_ok", True)),
                no_position=position_qty_by_symbol.get(symbol, 0) <= 0,
            )

            order_request = self._decision_to_order(
                decision,
                last_price=last_price_by_symbol[symbol],
                equity_krw=equity_krw,
                position_qty=position_qty_by_symbol.get(symbol, 0),
            )
            report_obj = None
            if order_request is not None:
                report_obj = self.broker_gateway.submit(
                    order_request,
                    market_price=last_price_by_symbol[symbol],
                    liquidity_score=liquidity_score_by_symbol.get(symbol, 1.0),
                )
                self.monitoring.record_slippage_gap(
                    actual_bps=decision.expected_slippage_bps,
                    expected_bps=decision.expected_slippage_bps,
                )

            # Cache completed symbol work before audit writes so same-anchor retries
            # do not resubmit orders if a later monitoring or audit step fails.
            result = LiveOrderResult(
                symbol=symbol,
                action=decision.action,
                selected_venue=decision.selected_venue,
                order_submitted=report_obj is not None,
                order_id=report_obj.order_id if report_obj is not None else None,
                execution_status=report_obj.status if report_obj is not None else None,
                filled_qty=report_obj.filled_qty if report_obj is not None else 0,
                rationale_codes=list(decision.rationale_codes),
                vetoes_triggered=list(decision.vetoes_triggered or []),
            )
            self._symbol_run_cache[cache_key] = result

            self.audit_store.append(
                DecisionAuditEntry(
                    symbol=symbol,
                    decision_time=anchor_time,
                    model_version=self.live.predictor.artifact.model_version,
                    prompt_version=getattr(self.live, "audit_prompt_version", self.live.normalizer.prompt_version),
                    vectorizer_version=self.live.vectorizer.encoder_version,
                    source_doc_ids=list(payload_by_symbol.get(symbol, {}).get("source_doc_ids", [])),
                    cluster_ids=list(payload_by_symbol.get(symbol, {}).get("cluster_ids", [])),
                    selected_venue=decision.selected_venue,
                    rationale_codes=list(decision.rationale_codes),
                )
            )
            if report_obj is not None:
                self.audit_store.append_runtime_event(
                    RuntimeAuditEvent(
                        event_type="order_submission",
                        event_time=anchor_time,
                        payload={
                            "symbol": symbol,
                            "order_id": report_obj.order_id,
                            "status": report_obj.status,
                            "filled_qty": report_obj.filled_qty,
                            "venue": report_obj.venue,
                        },
                    )
                )

            results[symbol] = result

        self.monitoring.emit_snapshot(
            anchor_time,
            {
                "trading_mode": report.trading_mode,
                "orders_submitted": sum(1 for x in results.values() if x.order_submitted),
                "symbols_processed": len(symbols),
            },
        )
        return report, results

    @staticmethod
    def _anchor_symbol_key(symbol: str, anchor_time: datetime) -> str:
        return f"{anchor_time.isoformat()}:{symbol}"

    def _decision_to_order(
        self,
        decision: TradeDecision,
        *,
        last_price: float,
        equity_krw: float,
        position_qty: int,
    ) -> OrderRequest | None:
        if decision.action in {"NO_TRADE", "WAIT_PULLBACK"}:
            return None

        if decision.action in {"BUY", "BUY_HALF"}:
            target_notional = max(0.0, equity_krw * float(decision.target_weight))
            qty = int(target_notional // max(last_price, 1e-6))
            side = "BUY"
        elif decision.action == "SELL":
            qty = max(0, int(position_qty))
            side = "SELL"
        elif decision.action == "REDUCE":
            qty = max(0, int(position_qty // 2))
            side = "SELL"
        else:
            return None

        if qty <= 0:
            return None

        tif = "DAY" if decision.session_type == "CORE_DAY" else "IOC"
        return OrderRequest(
            symbol=decision.symbol,
            side=side,
            qty=qty,
            venue=decision.selected_venue,
            limit_price=None,
            submitted_at=decision.as_of_time,
            tif=tif,
            session_type=decision.session_type,
        )


class ProductionOrchestrator:
    def __init__(
        self,
        engine: ProductionTradingEngine,
        *,
        backoff_schedule_sec: tuple[float, ...] = (0.0, 0.25, 1.0),
        circuit_breaker_threshold: int = 3,
        circuit_breaker_cooldown_sec: float = 300.0,
        sleep_fn: Callable[[float], None] | None = None,
        now_fn: Callable[[], datetime] | None = None,
    ) -> None:
        self.engine = engine
        self._idempotency_cache: dict[str, tuple[ProductionReadinessReport, dict[str, LiveOrderResult]]] = {}
        self.backoff_schedule_sec = backoff_schedule_sec
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.circuit_breaker_cooldown = timedelta(seconds=circuit_breaker_cooldown_sec)
        self.sleep_fn = sleep_fn or (lambda _: None)
        self.now_fn = now_fn or datetime.utcnow
        self.dead_letter_queue: list[AnchorBatchDeadLetterRecord] = []
        self._failure_streak = 0
        self._circuit_open_until: datetime | None = None

    def _assert_circuit_closed(self, anchor_time: datetime) -> None:
        if self._circuit_open_until is None:
            return
        if anchor_time >= self._circuit_open_until:
            self._circuit_open_until = None
            self._failure_streak = 0
            return
        raise ProductionCircuitBreakerOpen(f"circuit open until {self._circuit_open_until.isoformat()}")

    def _record_failure(
        self,
        *,
        anchor_time: datetime,
        symbols: list[str],
        payload_by_symbol: dict[str, dict],
        features_by_symbol: dict[str, dict],
        venue_eligibility_by_symbol: dict[str, str],
        attempts: int,
        error: Exception,
    ) -> None:
        self._failure_streak += 1
        if self._failure_streak >= self.circuit_breaker_threshold:
            self._circuit_open_until = anchor_time + self.circuit_breaker_cooldown
        self.dead_letter_queue.append(
            AnchorBatchDeadLetterRecord(
                anchor_time=anchor_time,
                symbols=list(symbols),
                payload_by_symbol={symbol: dict(payload) for symbol, payload in payload_by_symbol.items()},
                features_by_symbol={symbol: dict(features) for symbol, features in features_by_symbol.items()},
                venue_eligibility_by_symbol=dict(venue_eligibility_by_symbol),
                error_message=str(error),
                attempts=attempts,
                failed_at=self.now_fn(),
            )
        )
        self.engine.audit_store.append_runtime_event(
            RuntimeAuditEvent(
                event_type="anchor_batch_failure",
                event_time=anchor_time,
                payload={
                    "symbols": list(symbols),
                    "attempts": attempts,
                    "error": str(error),
                },
            )
        )
        self.engine.monitoring.record_degraded_mode()
        self.engine.monitoring.emit_snapshot(
            anchor_time,
            {
                "batch_failure": str(error),
                "batch_attempts": attempts,
                "symbols_processed": len(symbols),
            },
        )

    def drain_dead_letters(self) -> list[AnchorBatchDeadLetterRecord]:
        drained = list(self.dead_letter_queue)
        self.dead_letter_queue.clear()
        return drained

    def run_anchor(
        self,
        *,
        symbols: list[str],
        anchor_time: datetime,
        payload_by_symbol: dict[str, dict],
        features_by_symbol: dict[str, dict],
        venue_eligibility_by_symbol: dict[str, str],
        last_price_by_symbol: dict[str, float],
        dependency_state: RuntimeDependencyState,
        runtime_config: ProductionRuntimeConfig,
        model_requirements: ModelRuntimeRequirements,
        equity_krw: float,
        position_qty_by_symbol: dict[str, int] | None = None,
        liquidity_score_by_symbol: dict[str, float] | None = None,
        retry: int = 1,
    ) -> tuple[ProductionReadinessReport, dict[str, LiveOrderResult]]:
        idem_key = f"{anchor_time.isoformat()}:{','.join(sorted(symbols))}"
        if idem_key in self._idempotency_cache:
            return self._idempotency_cache[idem_key]
        self._assert_circuit_closed(anchor_time)

        last_exc: Exception | None = None
        attempts = max(1, retry + 1)
        for attempt in range(attempts):
            try:
                result = self.engine.run_live_anchor_batch(
                    symbols=symbols,
                    anchor_time=anchor_time,
                    payload_by_symbol=payload_by_symbol,
                    features_by_symbol=features_by_symbol,
                    venue_eligibility_by_symbol=venue_eligibility_by_symbol,
                    last_price_by_symbol=last_price_by_symbol,
                    dependency_state=dependency_state,
                    runtime_config=runtime_config,
                    model_requirements=model_requirements,
                    equity_krw=equity_krw,
                    position_qty_by_symbol=position_qty_by_symbol,
                    liquidity_score_by_symbol=liquidity_score_by_symbol,
                )
                self._failure_streak = 0
                self._idempotency_cache[idem_key] = result
                return result
            except LiveTradingBlockedError:
                raise
            except Exception as exc:
                last_exc = exc
                if attempt < attempts - 1:
                    backoff = self.backoff_schedule_sec[min(attempt, len(self.backoff_schedule_sec) - 1)]
                    if backoff > 0:
                        self.sleep_fn(backoff)

        if last_exc is not None:
            self._record_failure(
                anchor_time=anchor_time,
                symbols=symbols,
                payload_by_symbol=payload_by_symbol,
                features_by_symbol=features_by_symbol,
                venue_eligibility_by_symbol=venue_eligibility_by_symbol,
                attempts=attempts,
                error=last_exc,
            )
            raise last_exc
        raise RuntimeError("anchor batch failed without explicit exception")
