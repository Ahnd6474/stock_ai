from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class NumericFeatureRow:
    symbol: str
    timestamp: datetime
    features: dict
    session_type: str = "OFF_MARKET"
    freshness_flags: dict | None = None
    missingness_flags: dict | None = None


class FeatureStore:
    def __init__(self) -> None:
        self.rows: list[NumericFeatureRow] = []
        self.market_context: dict[str, float] = {}

    def put(self, row: NumericFeatureRow) -> None:
        self.rows.append(row)

    def get_latest(self, symbol: str, as_of_time: datetime) -> dict:
        candidates = [r for r in self.rows if r.symbol == symbol and r.timestamp <= as_of_time]
        if not candidates:
            return {}
        return max(candidates, key=lambda r: r.timestamp).features

    def build_online_features(self, symbol: str, as_of_time: datetime, session_type: str) -> dict:
        base = self.get_latest(symbol, as_of_time)
        out = dict(base)
        out.setdefault("session_type", session_type)
        out.setdefault("feature_fresh", bool(base))
        out.setdefault("missing_core_numeric", not bool(base))
        out.update(self.market_context)
        return out

    def build_offline_features(self, symbol: str, as_of_time: datetime, session_type: str) -> dict:
        out = self.build_online_features(symbol, as_of_time, session_type)
        out["offline_row"] = True
        return out

    def set_market_context(self, **kwargs: float) -> None:
        self.market_context.update(kwargs)
