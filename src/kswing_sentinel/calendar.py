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
    """KRX/NXT cash-equity trading day helper (weekend + explicit holiday set)."""

    def __init__(self, holidays: set[date] | None = None) -> None:
        self.holidays = holidays or set()

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
