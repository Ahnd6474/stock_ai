from datetime import date, timedelta
import json

from kswing_sentinel.training import TrainingPipeline


def test_training_pipeline_writes_artifact_and_fold_metrics(tmp_path):
    tp = TrainingPipeline()
    rows = []
    d0 = date(2025, 1, 1)
    for i in range(180):
        d = d0 + timedelta(days=i)
        flow = i / 180.0
        trend = (i % 20) / 20.0
        ext = (i % 7) / 10.0
        y = 0.02 * flow + 0.01 * trend - 0.005 * ext
        rows.append({"date": d, "flow_strength": flow, "trend_120m": trend, "extension_60m": ext, "er_20d": y})

    artifact_path, metrics_path = tp.train_walk_forward(
        rows=rows,
        feature_keys=["flow_strength", "trend_120m", "extension_60m"],
        label_key="er_20d",
        artifact_dir=tmp_path,
    )
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert artifact["label_key"] == "er_20d"
    assert "weights" in artifact
    assert isinstance(metrics, list)
