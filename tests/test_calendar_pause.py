from datetime import datetime
from zoneinfo import ZoneInfo

from kswing_sentinel.calendar import TradingCalendar
from kswing_sentinel.session_rules import classify_session

KST = ZoneInfo("Asia/Seoul")


def test_pause_window_is_off_market():
    cal = TradingCalendar()
    assert cal.is_in_pause_window(datetime(2026, 3, 20, 8, 55, tzinfo=KST))
    assert classify_session(datetime(2026, 3, 20, 8, 55, tzinfo=KST), cal) == "OFF_MARKET"
