from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Metrics:
    retrieval_success_rate: float = 1.0
    llm_schema_violation_rate: float = 0.0
    degraded_mode_rate: float = 0.0
    venue_selection_error_rate: float = 0.0


class Monitoring:
    def __init__(self) -> None:
        self.metrics = Metrics()

    def record_llm_violation(self) -> None:
        self.metrics.llm_schema_violation_rate += 1

    def snapshot(self) -> Metrics:
        return self.metrics
