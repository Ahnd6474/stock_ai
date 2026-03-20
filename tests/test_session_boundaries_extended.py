from datetime import datetime
from zoneinfo import ZoneInfo

from kswing_sentinel.session_rules import classify_session

KST = ZoneInfo("Asia/Seoul")


def test_session_boundaries_are_mutually_exclusive():
    cases = [
        (datetime(2026, 3, 20, 8, 49, tzinfo=KST), "NXT_PRE"),
        (datetime(2026, 3, 20, 8, 50, tzinfo=KST), "OFF_MARKET"),
        (datetime(2026, 3, 20, 9, 0, tzinfo=KST), "CORE_DAY"),
        (datetime(2026, 3, 20, 15, 19, tzinfo=KST), "CORE_DAY"),
        (datetime(2026, 3, 20, 15, 20, tzinfo=KST), "OFF_MARKET"),
        (datetime(2026, 3, 20, 15, 29, tzinfo=KST), "OFF_MARKET"),
        (datetime(2026, 3, 20, 15, 30, tzinfo=KST), "CLOSE_PRICE"),
        (datetime(2026, 3, 20, 15, 39, tzinfo=KST), "CLOSE_PRICE"),
        (datetime(2026, 3, 20, 15, 40, tzinfo=KST), "NXT_AFTER"),
        (datetime(2026, 3, 20, 19, 59, tzinfo=KST), "NXT_AFTER"),
        (datetime(2026, 3, 20, 20, 0, tzinfo=KST), "OFF_MARKET"),
    ]
    for ts, expected in cases:
        assert classify_session(ts) == expected
