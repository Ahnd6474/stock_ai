from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class FeatureRow:
    symbol: str
    timestamp: datetime
    as_of_time: datetime


class NoLookaheadError(RuntimeError):
    pass


class Backtester:
    def validate_no_lookahead(self, rows: list[FeatureRow]) -> None:
        bad = [r for r in rows if r.timestamp > r.as_of_time]
        if bad:
            raise NoLookaheadError(f"Found {len(bad)} leaking rows")
