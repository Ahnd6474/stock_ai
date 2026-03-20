from datetime import date, datetime, timedelta, timezone
import json

from kswing_sentinel.predictor import NumericFirstPredictor
from kswing_sentinel.training import TrainingPipeline


def test_multihead_artifact_can_drive_predictor(tmp_path):
    rows = []
    d0 = date(2025, 1, 1)
    for i in range(200):
        d = d0 + timedelta(days=i)
        fs = i / 200.0
        tr = (i % 10) / 10.0
        ex = (i % 5) / 20.0
        rows.append(
            {
                "date": d,
                "flow_strength": fs,
                "trend_120m": tr,
                "extension_60m": ex,
                "er_20d": 0.01 + fs * 0.02,
                "dd_20d": 0.03 + ex * 0.02,
                "p_up_20d": min(0.95, max(0.05, 0.5 + tr * 0.1)),
            }
        )
    artifact_path = TrainingPipeline().train_multi_head(
        rows=rows,
        feature_keys=["flow_strength", "trend_120m", "extension_60m"],
        artifact_dir=tmp_path,
    )
    blob = json.loads(artifact_path.read_text(encoding="utf-8"))
    predictor = NumericFirstPredictor(model_blob=blob)
    pred = predictor.predict(
        "005930",
        "CORE_DAY",
        datetime(2026, 3, 20, tzinfo=timezone.utc),
        {"flow_strength": 0.3, "trend_120m": 0.2, "extension_60m": 0.1},
    )
    assert pred.er_20d != 0.0
    assert 0.0 <= pred.uncertainty <= 1.0
