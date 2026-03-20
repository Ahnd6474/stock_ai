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
