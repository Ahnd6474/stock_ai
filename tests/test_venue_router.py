from datetime import date

from kswing_sentinel.cost_model import SessionCostModel
from kswing_sentinel.nxt_eligibility_store import EligibilitySnapshot, NXTEligibilityStore
from kswing_sentinel.venue_router import VenueContext, VenueRouter


def test_venue_router_respects_eligibility_and_freshness():
    st = NXTEligibilityStore()
    st.upsert(
        EligibilitySnapshot(
            date(2026, 3, 20),
            "v1",
            {"005930": "KRX_PLUS_NXT"},
            {"005930": True},
        )
    )
    r = VenueRouter(st, SessionCostModel())

    v = r.choose(VenueContext("005930", "NXT_AFTER", "v1", True, True, True))
    assert v in {"KRX", "NXT"}

    v2 = r.choose(VenueContext("005930", "NXT_AFTER", "v1", True, False, True))
    assert v2 == "KRX"


def test_venue_router_fail_closed_when_snapshot_stale():
    st = NXTEligibilityStore()
    st.upsert(
        EligibilitySnapshot(
            date(2026, 3, 18),
            "v2",
            {"005930": "KRX_PLUS_NXT"},
            {"005930": True},
        )
    )
    r = VenueRouter(st, SessionCostModel())
    venue, rationale = r.choose_with_rationale(
        VenueContext(
            "005930",
            "CORE_DAY",
            "v2",
            True,
            True,
            True,
            snapshot_date=date(2026, 3, 18),
            as_of_date=date(2026, 3, 20),
        )
    )
    assert venue == "KRX"
    assert "SNAPSHOT_STALE_FAIL_CLOSED" in rationale
