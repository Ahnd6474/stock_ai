from datetime import datetime, timedelta, timezone

from kswing_sentinel.feature_store import FeatureStore, MarketContextRow, NumericFeatureRow


def test_feature_store_merges_market_context_and_flags():
    store = FeatureStore(freshness_threshold_sec=1800)
    as_of = datetime(2026, 3, 20, 1, 0, tzinfo=timezone.utc)
    store.put(
        NumericFeatureRow(
            symbol="005930",
            timestamp=as_of - timedelta(minutes=5),
            features={"flow_strength": 0.4, "trend_120m": 0.2},
            freshness_flags={"quote": "fresh"},
            missingness_flags={"event_score_missing": True},
        )
    )
    store.put_market_context(
        MarketContextRow(
            timestamp=as_of - timedelta(minutes=2),
            context={
                "kospi_return_1d": -0.02,
                "breadth_ratio": 0.4,
                "usdkrw_return_1d": 0.012,
            },
            context_version="macro_v1",
        )
    )

    online = store.build_online_features("005930", as_of, "CORE_DAY")
    assert online["feature_fresh"] is True
    assert online["market_context_version"] == "macro_v1"
    assert online["market_risk_off"] is True
    assert online["missing_core_numeric"] is True
    assert online["freshness_flags"]["market_context_fresh"] is True
    assert online["missingness_flags"]["extension_60m_missing"] is True


def test_feature_store_build_offline_features_preserves_online_shape():
    store = FeatureStore(freshness_threshold_sec=60)
    as_of = datetime(2026, 3, 20, 1, 0, tzinfo=timezone.utc)
    store.put(
        NumericFeatureRow(
            symbol="000660",
            timestamp=as_of - timedelta(minutes=1),
            features={"flow_strength": 0.1, "trend_120m": 0.3, "extension_60m": 0.05},
        )
    )

    offline = store.build_offline_features("000660", as_of, "CORE_DAY")
    assert offline["offline_row"] is True
    assert offline["session_type"] == "CORE_DAY"
    assert offline["missing_core_numeric"] is False
    assert offline["offline_as_of_time"] == as_of.isoformat()
