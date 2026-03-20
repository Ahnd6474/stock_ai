from datetime import datetime, timezone

import pytest

from kswing_sentinel.predictor import NumericFirstPredictor


def test_predictor_uses_artifact_feature_schema():
    predictor = NumericFirstPredictor(
        model_blob={
            "model_version": "linear_baseline_v1",
            "schema_version": "v1",
            "feature_keys": ["f1", "f2"],
            "weights": {"f1": 0.2, "f2": 0.1},
            "bias": 0.0,
        }
    )
    with pytest.raises(ValueError):
        predictor.predict("005930", "CORE_DAY", datetime(2026, 3, 20, tzinfo=timezone.utc), {"f1": 1.0})


def test_predictor_supports_tree_ensemble_artifact():
    predictor = NumericFirstPredictor(
        model_blob={
            "model_version": "gbdt_like_v1",
            "schema_version": "v2",
            "model_type": "tree_ensemble_v1",
            "feature_keys": ["flow_strength", "trend_120m", "extension_60m"],
            "heads": {
                "er_20d": {
                    "base_score": 0.01,
                    "trees": [
                        {"feature": "flow_strength", "threshold": 0.2, "left": -0.005, "right": 0.015},
                        {"feature": "trend_120m", "threshold": 0.0, "left": -0.003, "right": 0.01},
                    ],
                },
                "dd_20d": {
                    "base_score": 0.04,
                    "trees": [{"feature": "extension_60m", "threshold": 0.3, "left": 0.0, "right": 0.02}],
                },
                "p_up_20d": {
                    "base_score": 0.5,
                    "trees": [{"feature": "flow_strength", "threshold": 0.1, "left": -0.05, "right": 0.08}],
                },
            },
        }
    )
    pred = predictor.predict(
        "005930",
        "CORE_DAY",
        datetime(2026, 3, 20, tzinfo=timezone.utc),
        {"flow_strength": 0.35, "trend_120m": 0.2, "extension_60m": 0.1},
    )
    assert pred.er_20d > 0.01
    assert 0.0 <= pred.p_up_20d <= 1.0
