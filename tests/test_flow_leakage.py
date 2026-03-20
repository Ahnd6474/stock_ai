from datetime import datetime, timezone

import pytest

from kswing_sentinel.flow_snapshot_store import FlowSnapshotStore, LeakageError
from kswing_sentinel.schemas import FlowSnapshot


def test_intraday_confirmed_flow_rejected():
    s = FlowSnapshotStore()
    s.upsert_snapshot(FlowSnapshot(
        symbol="005930", window="INTRADAY", foreign_net=1, institutional_net=1, program_net=1,
        preliminary_or_final="CONFIRMED", snapshot_at=datetime(2026,3,20,1,tzinfo=timezone.utc), as_of_time=datetime(2026,3,20,1,tzinfo=timezone.utc)
    ))
    with pytest.raises(LeakageError):
        s.get_latest("005930", datetime(2026,3,20,2,tzinfo=timezone.utc), intraday_anchor=True)
