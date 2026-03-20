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
