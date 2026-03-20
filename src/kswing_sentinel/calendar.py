from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

KST = ZoneInfo("Asia/Seoul")


@dataclass(frozen=True)
class SessionWindow:
    name: str
    start: time
    end: time


class TradingCalendar:
    """KRX/NXT cash-equity trading day helper with pause-window awareness."""

    def __init__(
        self,
        holidays: set[date] | None = None,
        half_days: set[date] | None = None,
        session_calendar_version: str = "v1",
    ) -> None:
        self.holidays = holidays or set()
        self.half_days = half_days or set()
        self.session_calendar_version = session_calendar_version

        self.pause_windows: tuple[SessionWindow, ...] = (
            SessionWindow("PRE_OPEN_PAUSE", time(8, 50), time(9, 0)),
            SessionWindow("CLOSE_AUCTION_PAUSE", time(15, 20), time(15, 30)),
        )

    def is_trading_day(self, d: date) -> bool:
        return d.weekday() < 5 and d not in self.holidays

    def next_trading_day(self, d: date) -> date:
        cur = d + timedelta(days=1)
        while not self.is_trading_day(cur):
            cur += timedelta(days=1)
        return cur

    def add_trading_days(self, d: date, n: int) -> date:
        cur = d
        for _ in range(n):
            cur = self.next_trading_day(cur)
        return cur

    def normalize_ts(self, ts: datetime) -> datetime:
        if ts.tzinfo is None:
            raise ValueError("timestamp must be timezone aware")
        return ts.astimezone(KST)

    def is_half_day(self, d: date) -> bool:
        return d in self.half_days

    def is_in_pause_window(self, ts: datetime) -> bool:
        local = self.normalize_ts(ts)
        t = local.time()
        return any(w.start <= t < w.end for w in self.pause_windows)
