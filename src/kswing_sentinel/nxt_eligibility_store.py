from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

VenueEligibility = Literal["KRX_ONLY", "KRX_PLUS_NXT"]


@dataclass(frozen=True)
class EligibilitySnapshot:
    snapshot_date: date
    version: str
    symbol_to_eligibility: dict[str, VenueEligibility]
    symbol_to_broker_routable: dict[str, bool] | None = None


class NXTEligibilityStore:
    def __init__(self) -> None:
        self._snapshots: dict[str, EligibilitySnapshot] = {}
        self._by_date_version: dict[tuple[date, str], EligibilitySnapshot] = {}

    def upsert(self, snapshot: EligibilitySnapshot) -> None:
        self._snapshots[snapshot.version] = snapshot
        self._by_date_version[(snapshot.snapshot_date, snapshot.version)] = snapshot

    def get(self, version: str, symbol: str) -> VenueEligibility:
        snap = self._snapshots.get(version)
        if not snap:
            return "KRX_ONLY"
        return snap.symbol_to_eligibility.get(symbol, "KRX_ONLY")

    def get_with_date(self, snapshot_date: date, version: str, symbol: str) -> VenueEligibility:
        snap = self._by_date_version.get((snapshot_date, version))
        if not snap:
            return "KRX_ONLY"
        return snap.symbol_to_eligibility.get(symbol, "KRX_ONLY")

    def is_stale(self, snapshot_date: date, as_of_date: date, max_age_days: int = 1) -> bool:
        return (as_of_date - snapshot_date).days > max_age_days

    def is_broker_routable(self, version: str, symbol: str) -> bool:
        snap = self._snapshots.get(version)
        if not snap or not snap.symbol_to_broker_routable:
            return False
        return bool(snap.symbol_to_broker_routable.get(symbol, False))

    def resolve(
        self,
        version: str,
        symbol: str,
        *,
        snapshot_date: date | None = None,
        as_of_date: date | None = None,
        max_age_days: int = 1,
    ) -> tuple[VenueEligibility, bool, str | None]:
        snap = self._snapshots.get(version)
        if not snap:
            return "KRX_ONLY", False, "SNAPSHOT_NOT_FOUND"
        if snapshot_date and as_of_date and self.is_stale(snapshot_date, as_of_date, max_age_days=max_age_days):
            return "KRX_ONLY", False, "SNAPSHOT_STALE_FAIL_CLOSED"
        eligibility = snap.symbol_to_eligibility.get(symbol, "KRX_ONLY")
        routable = bool((snap.symbol_to_broker_routable or {}).get(symbol, False))
        return eligibility, routable, None
