from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class NumericFeatureRow:
    symbol: str
    timestamp: datetime
    features: dict


class FeatureStore:
    def __init__(self) -> None:
        self.rows: list[NumericFeatureRow] = []

    def put(self, row: NumericFeatureRow) -> None:
        self.rows.append(row)

    def get_latest(self, symbol: str, as_of_time: datetime) -> dict:
        candidates = [r for r in self.rows if r.symbol == symbol and r.timestamp <= as_of_time]
        if not candidates:
            return {}
        return max(candidates, key=lambda r: r.timestamp).features
