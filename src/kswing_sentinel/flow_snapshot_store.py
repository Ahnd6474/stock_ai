from __future__ import annotations

from datetime import datetime

from .schemas import FlowSnapshot


class LeakageError(RuntimeError):
    pass


class FlowSnapshotStore:
    def __init__(self) -> None:
        self._rows: list[FlowSnapshot] = []

    def upsert_snapshot(self, row: FlowSnapshot) -> None:
        self._rows.append(row)

    def get_latest(self, symbol: str, as_of_time: datetime, intraday_anchor: bool) -> FlowSnapshot | None:
        rows = [r for r in self._rows if r.symbol == symbol and r.snapshot_at <= as_of_time]
        if intraday_anchor:
            leaked = [r for r in rows if r.preliminary_or_final == "CONFIRMED"]
            if leaked:
                raise LeakageError("Confirmed flow cannot be used for intraday anchors")
            rows = [r for r in rows if r.preliminary_or_final == "PROVISIONAL"]
        if not rows:
            return None
        return max(rows, key=lambda x: x.snapshot_at)
