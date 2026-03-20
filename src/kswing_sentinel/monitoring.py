from __future__ import annotations

import json
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


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
    def __init__(self, jsonl_path: str | Path | None = None) -> None:
        self.metrics = Metrics()
        self.jsonl_path = Path(jsonl_path) if jsonl_path else None
        if self.jsonl_path is not None:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)

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

    def emit_snapshot(self, emitted_at: datetime, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = {
            **asdict(self.metrics),
            "emitted_at": emitted_at.isoformat(),
        }
        if extra:
            payload.update(extra)
        if self.jsonl_path is not None:
            with self.jsonl_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return payload
