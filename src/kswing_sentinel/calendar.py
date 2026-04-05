from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

try:
    KST = ZoneInfo("Asia/Seoul")
except ZoneInfoNotFoundError:
    # Some minimal Python runtimes (notably on Windows) ship without IANA tzdata.
    # Fall back to a fixed-offset timezone so imports and tests can run without
    # requiring external `tzdata` installation.
    KST = timezone(timedelta(hours=9))


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
        session_calendar_version: str = "v2",
    ) -> None:
        default_holidays = {
            date(2026, 1, 1),
            date(2026, 2, 16),
            date(2026, 2, 17),
            date(2026, 2, 18),
            date(2026, 3, 1),
            date(2026, 5, 5),
            date(2026, 5, 24),
            date(2026, 6, 6),
            date(2026, 8, 15),
            date(2026, 9, 24),
            date(2026, 9, 25),
            date(2026, 9, 26),
            date(2026, 10, 3),
            date(2026, 10, 9),
            date(2026, 12, 25),
        }
        self.holidays = holidays or default_holidays
        self.half_days = half_days or set()
        self.session_calendar_version = session_calendar_version

        self.pause_windows: tuple[SessionWindow, ...] = (
            SessionWindow("PRE_OPEN_PAUSE", time(8, 50), time(9, 0)),
            SessionWindow("CLOSE_AUCTION_PAUSE", time(15, 20), time(15, 30)),
        )
        self.early_close_time = time(12, 0)

    def is_trading_day(self, d: date) -> bool:
        return d.weekday() < 5 and d not in self.holidays

    def next_trading_day(self, d: date) -> date:
        cur = d + timedelta(days=1)
        while not self.is_trading_day(cur):
            cur += timedelta(days=1)
        return cur

    def add_trading_days(self, d: date, n: int) -> date:
        cur = d
        step = 1 if n >= 0 else -1
        for _ in range(abs(n)):
            cur = cur + timedelta(days=step)
            while not self.is_trading_day(cur):
                cur = cur + timedelta(days=step)
        return cur

    def normalize_ts(self, ts: datetime) -> datetime:
        if ts.tzinfo is None:
            raise ValueError("timestamp must be timezone aware")
        return ts.astimezone(KST)

    def is_half_day(self, d: date) -> bool:
        return d in self.half_days

    def is_in_pause_window(self, ts: datetime) -> bool:
        local = self.normalize_ts(ts)
        if self.is_half_day(local.date()) and local.time() >= self.early_close_time:
            return True
        t = local.time()
        return any(w.start <= t < w.end for w in self.pause_windows)

    def is_tradable_minute(self, ts: datetime) -> bool:
        local = self.normalize_ts(ts)
        if not self.is_trading_day(local.date()):
            return False
        if self.is_in_pause_window(local):
            return False
        t = local.time()
        if self.is_half_day(local.date()):
            return time(8, 0) <= t < self.early_close_time
        return (time(8, 0) <= t < time(8, 50)) or (time(9, 0) <= t < time(15, 20)) or (time(15, 30) <= t < time(20, 0))
