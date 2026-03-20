from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class NumericFeatureRow:
    symbol: str
    timestamp: datetime
    features: dict
    session_type: str = "OFF_MARKET"
    freshness_flags: dict | None = None
    missingness_flags: dict | None = None


@dataclass(frozen=True)
class MarketContextRow:
    timestamp: datetime
    context: dict
    context_version: str = "v1"


class FeatureStore:
    def __init__(self, freshness_threshold_sec: float = 60.0 * 20.0) -> None:
        self.rows: list[NumericFeatureRow] = []
        self.market_context_rows: list[MarketContextRow] = []
        self.market_context: dict[str, float] = {}
        self.freshness_threshold_sec = freshness_threshold_sec

    def put(self, row: NumericFeatureRow) -> None:
        self.rows.append(row)

    def put_market_context(self, row: MarketContextRow) -> None:
        self.market_context_rows.append(row)
        self.market_context = dict(row.context)

    def get_latest_row(self, symbol: str, as_of_time: datetime) -> NumericFeatureRow | None:
        candidates = [r for r in self.rows if r.symbol == symbol and r.timestamp <= as_of_time]
        if not candidates:
            return None
        return max(candidates, key=lambda r: r.timestamp)

    def get_latest(self, symbol: str, as_of_time: datetime) -> dict:
        latest = self.get_latest_row(symbol, as_of_time)
        if latest is None:
            return {}
        return latest.features

    def get_latest_market_context(self, as_of_time: datetime) -> MarketContextRow | None:
        candidates = [r for r in self.market_context_rows if r.timestamp <= as_of_time]
        if not candidates:
            return None
        return max(candidates, key=lambda r: r.timestamp)

    @staticmethod
    def _core_missing_flags(features: dict) -> dict[str, bool]:
        return {
            "flow_strength_missing": "flow_strength" not in features,
            "trend_120m_missing": "trend_120m" not in features,
            "extension_60m_missing": "extension_60m" not in features,
        }

    @staticmethod
    def _coerce_float(value: object, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _derive_market_risk_off(self, context: dict) -> bool:
        if bool(context.get("market_risk_off", False)):
            return True
        breadth = self._coerce_float(context.get("breadth_ratio"), 1.0)
        kospi_return = self._coerce_float(context.get("kospi_return_1d"), 0.0)
        usdkrw_return = self._coerce_float(context.get("usdkrw_return_1d"), 0.0)
        futures_basis = self._coerce_float(context.get("kospi200_futures_basis"), 0.0)
        return (breadth < 0.45 and kospi_return < -0.015) or usdkrw_return > 0.01 or futures_basis < -0.5

    def build_online_features(self, symbol: str, as_of_time: datetime, session_type: str) -> dict:
        row = self.get_latest_row(symbol, as_of_time)
        base = dict(row.features) if row is not None else {}
        context_row = self.get_latest_market_context(as_of_time)
        context = dict(context_row.context) if context_row is not None else {}
        freshness_flags = dict((row.freshness_flags or {}) if row is not None else {})
        missingness_flags = dict((row.missingness_flags or {}) if row is not None else {})
        missingness_flags.update(self._core_missing_flags(base))
        missing_core_numeric = any(missingness_flags.values())

        feature_age_sec = (as_of_time - row.timestamp).total_seconds() if row is not None else None
        context_age_sec = (as_of_time - context_row.timestamp).total_seconds() if context_row is not None else None
        freshness_flags["feature_fresh"] = feature_age_sec is not None and feature_age_sec <= self.freshness_threshold_sec
        freshness_flags["market_context_fresh"] = context_age_sec is not None and context_age_sec <= self.freshness_threshold_sec
        freshness_flags["market_context_present"] = bool(context_row)

        out = dict(base)
        out.update(context)
        out.setdefault("session_type", session_type)
        out["feature_fresh"] = freshness_flags["feature_fresh"]
        out["missing_core_numeric"] = missing_core_numeric
        out["feature_age_sec"] = feature_age_sec
        out["market_context_age_sec"] = context_age_sec
        out["freshness_flags"] = freshness_flags
        out["missingness_flags"] = missingness_flags
        out.setdefault("missing_flags", dict(missingness_flags))
        out.setdefault("stale_flags", dict(freshness_flags))
        out["market_context_version"] = context_row.context_version if context_row is not None else None
        out["regime_context_present"] = bool(context_row)
        out["market_risk_off"] = self._derive_market_risk_off(context)
        out["venue_session_flag"] = f"{out.get('selected_venue', 'NA')}:{session_type}"
        return out

    def build_offline_features(self, symbol: str, as_of_time: datetime, session_type: str) -> dict:
        out = self.build_online_features(symbol, as_of_time, session_type)
        out["offline_row"] = True
        out["offline_as_of_time"] = as_of_time.isoformat()
        return out
