from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from .schemas import SessionType

KST = ZoneInfo("Asia/Seoul")


def classify_session(ts: datetime) -> SessionType:
    if ts.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    t = ts.astimezone(KST).time()
    hm = t.hour * 60 + t.minute
    if 8 * 60 <= hm < 8 * 60 + 50:
        return "NXT_PRE"
    if 9 * 60 <= hm < 15 * 60 + 20:
        return "CORE_DAY"
    if 15 * 60 + 30 <= hm < 15 * 60 + 40:
        return "CLOSE_PRICE"
    if 15 * 60 + 40 <= hm < 20 * 60:
        return "NXT_AFTER"
    return "OFF_MARKET"


def round_to_next_5m(ts: datetime) -> datetime:
    dt = ts.replace(second=0, microsecond=0)
    add = (5 - dt.minute % 5) % 5
    if add == 0:
        add = 5
    return dt + timedelta(minutes=add)
