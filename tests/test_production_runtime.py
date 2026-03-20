from datetime import datetime, timezone
import json

import pytest

from kswing_sentinel.broker_gateway import BrokerCapabilities, BrokerGateway
from kswing_sentinel.production_runtime import (
    LiveTradingBlockedError,
    ModelRuntimeRequirements,
    ProductionReadinessGate,
    ProductionRuntimeConfig,
    ProductionTradingEngine,
    RuntimeDependencyState,
)


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
    assert any(json.loads(line)["record_type"] == "runtime_event" for line in audit_lines)
    assert any(json.loads(line)["record_type"] == "decision" for line in audit_lines)
    assert len(metrics_lines) >= 1
