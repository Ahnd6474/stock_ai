from datetime import datetime, timezone

from kswing_sentinel.calibration import (
    ProbabilityCalibrator,
    QuantileAdjuster,
    calibration_report_binary,
    calibration_report_dd,
    summarize_uncertainty_buckets,
)
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


def test_calibrator_fit_and_report_improves_or_keeps_brier():
    preds = [0.2, 0.3, 0.7, 0.8]
    labels = [0.0, 0.0, 1.0, 1.0]
    c = ProbabilityCalibrator.fit(preds, labels)
    report = calibration_report_binary(preds, labels, c)
    assert report.metric_name == "brier"
    assert report.post <= report.pre + 1e-9


def test_dd_calibration_report_and_uncertainty_buckets():
    pred_dd = [0.02, 0.03, 0.05, 0.08]
    real_dd = [0.03, 0.04, 0.06, 0.07]
    q = QuantileAdjuster.fit(pred_dd, real_dd)
    report = calibration_report_dd(pred_dd, real_dd, q)
    assert report.metric_name == "mae_dd"
    assert report.post <= report.pre + 1e-9

    buckets = summarize_uncertainty_buckets([0.1, 0.4, 0.8, 0.9], [1.0, 0.0, 1.0, 0.0])
    assert len(buckets) == 3
    assert sum(b.count for b in buckets) == 4
