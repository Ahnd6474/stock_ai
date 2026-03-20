from datetime import datetime
from zoneinfo import ZoneInfo

from kswing_sentinel.session_rules import classify_session

KST = ZoneInfo("Asia/Seoul")


def test_session_mapping_pre():
    assert classify_session(datetime(2026,3,20,8,10,tzinfo=KST)) == "NXT_PRE"


def test_session_mapping_after():
    assert classify_session(datetime(2026,3,20,18,30,tzinfo=KST)) == "NXT_AFTER"
