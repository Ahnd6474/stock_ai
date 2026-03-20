from datetime import datetime, timezone, timedelta

from kswing_sentinel.audit_log import AuditLogStore, DecisionAuditEntry


def test_audit_log_latest_for_symbol():
    store = AuditLogStore()
    t0 = datetime(2026, 3, 20, tzinfo=timezone.utc)
    store.append(DecisionAuditEntry("005930", t0, "m1", "p1", "v1", [], [], "KRX", ["A"]))
    store.append(DecisionAuditEntry("005930", t0 + timedelta(minutes=5), "m2", "p1", "v1", [], [], "NXT", ["B"]))

    latest = store.latest_for_symbol("005930")
    assert latest is not None
    assert latest.model_version == "m2"
