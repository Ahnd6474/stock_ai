from datetime import datetime, timezone

from kswing_sentinel.calibration import ProbabilityCalibrator, QuantileAdjuster
from kswing_sentinel.predictor import NumericFirstPredictor


def test_calibrators_apply_bounds_and_adjustment():
    p = ProbabilityCalibrator(slope=1.2, intercept=-0.1)
    q = QuantileAdjuster(scale=1.5, bias=0.01)
    assert 0.0 <= p.transform(0.8) <= 1.0
    assert q.transform(0.02) > 0.02


def test_predictor_accepts_artifact_blob_and_calibration():
    predictor = NumericFirstPredictor(
        model_blob={"weights": {"flow_strength": 0.1, "trend_120m": 0.2, "extension_60m": -0.05}, "bias": 0.01},
        p_up_calibrator=ProbabilityCalibrator(slope=0.9, intercept=0.02),
        dd_adjuster=QuantileAdjuster(scale=1.1, bias=0.0),
    )
    pred = predictor.predict(
        "005930",
        "CORE_DAY",
        datetime(2026, 3, 20, tzinfo=timezone.utc),
        {"flow_strength": 0.3, "trend_120m": 0.4, "extension_60m": 0.2},
    )
    assert 0 <= pred.p_up_20d <= 1
    assert pred.dd_20d >= 0
