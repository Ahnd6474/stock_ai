from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Metrics:
    retrieval_success_rate: float = 1.0
    llm_schema_violation_rate: float = 0.0
    degraded_mode_rate: float = 0.0
    venue_selection_error_rate: float = 0.0
    feature_freshness_lag_sec: float = 0.0
    semantic_coverage_rate: float = 1.0
    slippage_vs_expected_bps: float = 0.0
    realized_dd_vs_predicted_dd_gap: float = 0.0


class Monitoring:
    def __init__(self) -> None:
        self.metrics = Metrics()

    def record_llm_violation(self) -> None:
        self.metrics.llm_schema_violation_rate += 1

    def record_degraded_mode(self) -> None:
        self.metrics.degraded_mode_rate += 1

    def record_feature_lag(self, lag_sec: float) -> None:
        self.metrics.feature_freshness_lag_sec = max(self.metrics.feature_freshness_lag_sec, lag_sec)

    def record_slippage_gap(self, actual_bps: float, expected_bps: float) -> None:
        self.metrics.slippage_vs_expected_bps = actual_bps - expected_bps

    def snapshot(self) -> Metrics:
        return self.metrics
