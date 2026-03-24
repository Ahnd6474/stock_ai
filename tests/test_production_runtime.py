from datetime import datetime, timezone
import json

import pytest

from kswing_sentinel.audit_log import AuditLogStore
from kswing_sentinel.broker_gateway import BrokerCapabilities, BrokerGateway
from kswing_sentinel.monitoring import Monitoring
from kswing_sentinel.production_runtime import (
    LiveOrderResult,
    LiveTradingBlockedError,
    ModelRuntimeRequirements,
    ProductionCircuitBreakerOpen,
    ProductionOrchestrator,
    ProductionReadinessGate,
    ProductionReadinessReport,
    ProductionRuntimeConfig,
    ProductionTradingEngine,
    RuntimeDependencyState,
)
from kswing_sentinel.schemas import TradeDecision


def _deps(**overrides):
    base = {
        "realtime_krx_feed_available": True,
        "realtime_krx_feed_licensed": True,
        "broker_api_available": True,
        "broker_live_order_enabled": True,
        "broker_krx_routing_available": True,
        "provisional_flow_archive_available": True,
        "provisional_flow_live_available": True,
        "model_artifact_available": True,
        "calibrator_available": True,
        "audit_sink_writable": True,
        "nxt_feed_available": True,
        "nxt_feed_licensed": True,
        "broker_nxt_routing_available": True,
        "nxt_eligibility_snapshot_fresh": True,
        "cross_venue_quotes_available": False,
        "semantic_provider_available": True,
        "vectorizer_available": True,
    }
    base.update(overrides)
    return RuntimeDependencyState(**base)


def test_readiness_blocks_without_licensed_realtime_feed(monkeypatch):
    monkeypatch.setenv("KRX_FEED_KEY", "x")
    cfg = ProductionRuntimeConfig(required_env_vars=["KRX_FEED_KEY"])
    broker = BrokerCapabilities(supports_nxt=True, supports_after_market=True)
    report = ProductionReadinessGate().evaluate(
        cfg,
        _deps(realtime_krx_feed_licensed=False),
        broker,
        ModelRuntimeRequirements(),
        datetime(2026, 3, 20, tzinfo=timezone.utc),
    )
    assert report.ready is False
    assert report.trading_mode == "BLOCKED"
    assert "KRX_REALTIME_FEED_UNLICENSED" in report.blocking_issues


def test_readiness_degrades_to_krx_only_when_nxt_is_not_ready():
    cfg = ProductionRuntimeConfig(requested_trading_mode="KRX_NXT", allow_krx_only_fallback=True)
    broker = BrokerCapabilities(supports_nxt=True, supports_after_market=True)
    report = ProductionReadinessGate().evaluate(
        cfg,
        _deps(nxt_feed_available=False, broker_nxt_routing_available=False, nxt_eligibility_snapshot_fresh=False),
        broker,
        ModelRuntimeRequirements(requires_nxt_for_strategy=False),
        datetime(2026, 3, 20, tzinfo=timezone.utc),
    )
    assert report.ready is True
    assert report.trading_mode == "KRX_ONLY"
    assert "NXT_DISABLED_KRX_ONLY_FALLBACK" in report.degraded_flags


def test_engine_blocks_live_orders_when_readiness_fails(tmp_path):
    cfg = ProductionRuntimeConfig(
        requested_trading_mode="KRX_ONLY",
        audit_log_path=str(tmp_path / "audit.jsonl"),
        metrics_log_path=str(tmp_path / "metrics.jsonl"),
    )
    engine = ProductionTradingEngine.from_runtime_config(
        runtime_config=cfg,
        broker_gateway=BrokerGateway(BrokerCapabilities(supports_nxt=True, supports_after_market=True)),
    )
    with pytest.raises(LiveTradingBlockedError):
        engine.run_live_anchor_batch(
            symbols=["005930"],
            anchor_time=datetime(2026, 3, 20, 9, 35, tzinfo=timezone.utc),
            payload_by_symbol={"005930": {}},
            features_by_symbol={"005930": {"flow_strength": 0.8, "trend_120m": 0.8, "extension_60m": 0.1}},
            venue_eligibility_by_symbol={"005930": "KRX_ONLY"},
            last_price_by_symbol={"005930": 100000.0},
            dependency_state=_deps(realtime_krx_feed_available=False),
            runtime_config=cfg,
            model_requirements=ModelRuntimeRequirements(),
            equity_krw=100_000_000,
        )


def test_engine_submits_krx_order_and_persists_audit(tmp_path):
    cfg = ProductionRuntimeConfig(
        requested_trading_mode="KRX_ONLY",
        audit_log_path=str(tmp_path / "audit.jsonl"),
        metrics_log_path=str(tmp_path / "metrics.jsonl"),
    )
    engine = ProductionTradingEngine.from_runtime_config(
        runtime_config=cfg,
        broker_gateway=BrokerGateway(
            BrokerCapabilities(
                supports_nxt=True,
                supports_after_market=True,
                supports_live_trading=True,
                supports_krx=True,
                dry_run_only=False,
            )
        ),
    )
    report, results = engine.run_live_anchor_batch(
        symbols=["005930"],
        anchor_time=datetime(2026, 3, 20, 9, 35, tzinfo=timezone.utc),
        payload_by_symbol={
            "005930": {
                "canonical_summary": "Order backlog and guidance improved.",
                "event_score": 0.7,
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
                "source_doc_ids": ["doc-1"],
                "cluster_ids": ["cluster-1"],
            }
        },
        features_by_symbol={"005930": {"flow_strength": 0.8, "trend_120m": 0.8, "extension_60m": 0.1}},
        venue_eligibility_by_symbol={"005930": "KRX_ONLY"},
        last_price_by_symbol={"005930": 100000.0},
        dependency_state=_deps(),
        runtime_config=cfg,
        model_requirements=ModelRuntimeRequirements(),
        equity_krw=100_000_000,
    )
    assert report.ready is True
    assert results["005930"].order_submitted is True
    assert results["005930"].selected_venue == "KRX"

    audit_lines = (tmp_path / "audit.jsonl").read_text(encoding="utf-8").strip().splitlines()
    metrics_lines = (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
    audit_records = [json.loads(line) for line in audit_lines]
    assert any(record["record_type"] == "runtime_event" for record in audit_records)
    assert any(
        record["record_type"] == "runtime_event" and record["event_type"] == "semantic_refresh_requested"
        for record in audit_records
    )
    decision_records = [record for record in audit_records if record["record_type"] == "decision"]
    assert decision_records
    assert decision_records[0]["prompt_version"] == "disabled"
    assert len(metrics_lines) >= 1


class FlakyBatchLiveService:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self._remaining_failures = {"000660": 1}
        self.normalizer = type("NormalizerStub", (), {"prompt_version": "disabled"})()
        self.vectorizer = type("VectorizerStub", (), {"encoder_version": "hashing_bow_v1"})()
        artifact = type("ArtifactStub", (), {"model_version": "test_model_v1"})()
        self.predictor = type("PredictorStub", (), {"artifact": artifact})()

    def run_for_symbol(self, symbol, as_of_time, raw_event_payload, features, venue_eligibility, **kwargs):
        self.calls.append(symbol)
        remaining = self._remaining_failures.get(symbol, 0)
        if remaining > 0:
            self._remaining_failures[symbol] = remaining - 1
            raise RuntimeError("transient live failure")
        return TradeDecision(
            symbol=symbol,
            action="BUY",
            target_weight=0.02,
            tranche_ratio=0.4,
            session_type="CORE_DAY",
            selected_venue="KRX",
            rationale_codes=["TEST_BUY"],
            as_of_time=as_of_time,
            execution_time=as_of_time,
            vetoes_triggered=[],
            risk_budget_used=0.02,
            expected_slippage_bps=8.0,
            degraded_mode_flags=[],
            exit_policy_hint="test",
        )


def test_production_orchestrator_retries_without_duplicate_orders(tmp_path):
    cfg = ProductionRuntimeConfig(
        requested_trading_mode="KRX_ONLY",
        audit_log_path=str(tmp_path / "audit.jsonl"),
        metrics_log_path=str(tmp_path / "metrics.jsonl"),
    )
    live_service = FlakyBatchLiveService()
    engine = ProductionTradingEngine.from_runtime_config(
        runtime_config=cfg,
        live_service=live_service,
        broker_gateway=BrokerGateway(
            BrokerCapabilities(
                supports_nxt=True,
                supports_after_market=True,
                supports_live_trading=True,
                supports_krx=True,
                dry_run_only=False,
            )
        ),
    )
    waits: list[float] = []
    orchestrator = ProductionOrchestrator(
        engine,
        backoff_schedule_sec=(0.1, 0.2),
        sleep_fn=waits.append,
    )

    report, results = orchestrator.run_anchor(
        symbols=["005930", "000660"],
        anchor_time=datetime(2026, 3, 20, 9, 35, tzinfo=timezone.utc),
        payload_by_symbol={"005930": {}, "000660": {}},
        features_by_symbol={
            "005930": {"flow_strength": 0.8, "trend_120m": 0.8, "extension_60m": 0.1},
            "000660": {"flow_strength": 0.7, "trend_120m": 0.8, "extension_60m": 0.2},
        },
        venue_eligibility_by_symbol={"005930": "KRX_ONLY", "000660": "KRX_ONLY"},
        last_price_by_symbol={"005930": 100000.0, "000660": 200000.0},
        dependency_state=_deps(),
        runtime_config=cfg,
        model_requirements=ModelRuntimeRequirements(),
        equity_krw=100_000_000,
        retry=1,
    )

    assert report.ready is True
    assert waits == [0.1]
    assert live_service.calls == ["005930", "000660", "000660"]
    assert engine.broker_gateway._counter == 2
    assert set(results) == {"005930", "000660"}
    assert all(result.order_submitted for result in results.values())

    audit_lines = (tmp_path / "audit.jsonl").read_text(encoding="utf-8").strip().splitlines()
    audit_records = [json.loads(line) for line in audit_lines]
    decision_records = [record for record in audit_records if record["record_type"] == "decision"]
    assert len(decision_records) == 2


class AlwaysFailingBatchEngine:
    def __init__(self) -> None:
        self.calls = 0
        self.audit_store = AuditLogStore()
        self.monitoring = Monitoring()

    def run_live_anchor_batch(self, **kwargs):
        self.calls += 1
        raise RuntimeError("persistent batch failure")


class RecordingBatchEngine:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.audit_store = AuditLogStore()
        self.monitoring = Monitoring()

    def run_live_anchor_batch(self, **kwargs):
        self.calls.append(dict(kwargs))
        return (
            ProductionReadinessReport(
                checked_at=kwargs["anchor_time"],
                ready=True,
                trading_mode="KRX_ONLY",
                blocking_issues=[],
                warnings=[],
                degraded_flags=[],
            ),
            {
                symbol: LiveOrderResult(
                    symbol=symbol,
                    action="BUY",
                    selected_venue="KRX",
                    order_submitted=True,
                    order_id=f"ORD-{index + 1}",
                    execution_status="FILLED",
                    filled_qty=1,
                    rationale_codes=["REDRIVE_OK"],
                    vetoes_triggered=[],
                )
                for index, symbol in enumerate(kwargs["symbols"])
            },
        )


def test_production_orchestrator_opens_circuit_and_drains_dead_letters():
    engine = AlwaysFailingBatchEngine()
    now = datetime(2026, 3, 20, 0, 0, tzinfo=timezone.utc)
    orchestrator = ProductionOrchestrator(
        engine,
        circuit_breaker_threshold=1,
        circuit_breaker_cooldown_sec=300,
        now_fn=lambda: now,
    )

    with pytest.raises(RuntimeError):
        orchestrator.run_anchor(
            symbols=["005930"],
            anchor_time=now,
            payload_by_symbol={"005930": {}},
            features_by_symbol={"005930": {"flow_strength": 0.1}},
            venue_eligibility_by_symbol={"005930": "KRX_ONLY"},
            last_price_by_symbol={"005930": 100000.0},
            dependency_state=_deps(),
            runtime_config=ProductionRuntimeConfig(requested_trading_mode="KRX_ONLY"),
            model_requirements=ModelRuntimeRequirements(),
            equity_krw=100_000_000,
            retry=0,
        )

    assert len(orchestrator.dead_letter_queue) == 1

    with pytest.raises(ProductionCircuitBreakerOpen):
        orchestrator.run_anchor(
            symbols=["005930"],
            anchor_time=now.replace(minute=1),
            payload_by_symbol={"005930": {}},
            features_by_symbol={"005930": {"flow_strength": 0.1}},
            venue_eligibility_by_symbol={"005930": "KRX_ONLY"},
            last_price_by_symbol={"005930": 100000.0},
            dependency_state=_deps(),
            runtime_config=ProductionRuntimeConfig(requested_trading_mode="KRX_ONLY"),
            model_requirements=ModelRuntimeRequirements(),
            equity_krw=100_000_000,
            retry=0,
        )

    drained = orchestrator.drain_dead_letters()
    assert drained[0].attempts == 1
    assert drained[0].error_message == "persistent batch failure"
    assert orchestrator.dead_letter_queue == []


def test_production_orchestrator_persists_dead_letters_when_configured(tmp_path):
    engine = AlwaysFailingBatchEngine()
    anchor_time = datetime(2026, 3, 20, 0, 0, tzinfo=timezone.utc)
    dead_letter_path = tmp_path / "dead_letters.jsonl"
    orchestrator = ProductionOrchestrator(engine, now_fn=lambda: anchor_time)

    with pytest.raises(RuntimeError):
        orchestrator.run_anchor(
            symbols=["005930"],
            anchor_time=anchor_time,
            payload_by_symbol={"005930": {"headline": "failure case"}},
            features_by_symbol={"005930": {"flow_strength": 0.1}},
            venue_eligibility_by_symbol={"005930": "KRX_ONLY"},
            last_price_by_symbol={"005930": 100000.0},
            dependency_state=_deps(),
            runtime_config=ProductionRuntimeConfig(
                requested_trading_mode="KRX_ONLY",
                dead_letter_log_path=str(dead_letter_path),
            ),
            model_requirements=ModelRuntimeRequirements(),
            equity_krw=100_000_000,
            retry=0,
        )

    records = [json.loads(line) for line in dead_letter_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert records[0]["record_type"] == "dead_letter"
    assert records[0]["symbols"] == ["005930"]
    assert records[0]["error_message"] == "persistent batch failure"
    assert records[0]["payload_by_symbol"]["005930"]["headline"] == "failure case"


def test_production_orchestrator_can_redrive_persisted_dead_letters(tmp_path):
    anchor_time = datetime(2026, 3, 20, 0, 0, tzinfo=timezone.utc)
    dead_letter_path = tmp_path / "dead_letters.jsonl"
    cfg = ProductionRuntimeConfig(
        requested_trading_mode="KRX_ONLY",
        dead_letter_log_path=str(dead_letter_path),
    )

    failing_orchestrator = ProductionOrchestrator(AlwaysFailingBatchEngine(), now_fn=lambda: anchor_time)
    with pytest.raises(RuntimeError):
        failing_orchestrator.run_anchor(
            symbols=["005930"],
            anchor_time=anchor_time,
            payload_by_symbol={"005930": {"headline": "redrive me"}},
            features_by_symbol={"005930": {"flow_strength": 0.1}},
            venue_eligibility_by_symbol={"005930": "KRX_ONLY"},
            last_price_by_symbol={"005930": 100000.0},
            dependency_state=_deps(),
            runtime_config=cfg,
            model_requirements=ModelRuntimeRequirements(),
            equity_krw=100_000_000,
            retry=0,
        )

    recording_engine = RecordingBatchEngine()
    orchestrator = ProductionOrchestrator(recording_engine)
    outcomes = orchestrator.redrive_persisted_dead_letters(
        last_price_by_symbol={"005930": 100000.0},
        dependency_state=_deps(),
        runtime_config=cfg,
        model_requirements=ModelRuntimeRequirements(),
        equity_krw=100_000_000,
    )

    assert len(outcomes) == 1
    assert outcomes[0].success is True
    assert recording_engine.calls[0]["payload_by_symbol"]["005930"]["headline"] == "redrive me"
    assert recording_engine.calls[0]["features_by_symbol"]["005930"]["flow_strength"] == 0.1
