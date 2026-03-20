from datetime import datetime, timezone

import pytest

from kswing_sentinel.backtester import Backtester, FeatureRow, NoLookaheadError


def test_no_lookahead_ok():
    bt = Backtester()
    rows = [FeatureRow(symbol="005930", timestamp=datetime(2026,3,20,0,0,tzinfo=timezone.utc), as_of_time=datetime(2026,3,20,0,0,tzinfo=timezone.utc))]
    bt.validate_no_lookahead(rows)


def test_no_lookahead_fail():
    bt = Backtester()
    rows = [FeatureRow(symbol="005930", timestamp=datetime(2026,3,20,1,0,tzinfo=timezone.utc), as_of_time=datetime(2026,3,20,0,0,tzinfo=timezone.utc))]
    with pytest.raises(NoLookaheadError):
        bt.validate_no_lookahead(rows)
