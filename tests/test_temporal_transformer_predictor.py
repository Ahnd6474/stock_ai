from datetime import date, datetime, timedelta, timezone
import json

import pytest

torch = pytest.importorskip("torch")

from kswing_sentinel.predictor import NumericFirstPredictor, build_temporal_state_matrix
from kswing_sentinel.training import TrainingPipeline


def test_build_temporal_state_matrix_uses_top_level_vector_for_latest_step_only():
    features = {
        "state_sequence": [
            {"numeric_features": {"flow_strength": 0.1}},
            {"numeric_features": {"flow_strength": 0.2}},
        ],
        "vector_payload": {
            "z_event": [1.0, 2.0],
            "z_social": [3.0],
            "z_macro": [4.0],
        },
    }

    matrix = build_temporal_state_matrix(
        features,
        numeric_feature_keys=["flow_strength"],
        vector_feature_dims={"z_event": 2, "z_social": 1, "z_macro": 1},
    )

    assert matrix[0] == [0.1, 0.0, 0.0, 0.0, 0.0]
    assert matrix[1] == [0.2, 1.0, 2.0, 3.0, 4.0]


def _row_with_state_sequence(index: int) -> dict:
    sequence: list[dict] = []
    for step in range(4):
        base = index + step + 1
        trend = 0.08 if base % 2 == 0 else -0.04
        event_score = 0.4 if step == 3 and index % 3 == 0 else 0.05
        extension = 0.03 * (base % 3)
        sequence.append(
            {
                "numeric_features": {
                    "close_return_1d": 0.01 * ((base % 5) - 2),
                    "volume_z": 0.2 + 0.02 * base,
                    "flow_strength": 0.15 + 0.01 * base,
                    "trend_120m": trend,
                    "extension_60m": extension,
                    "event_score": event_score,
                },
                "vector_payload": {
                    "z_event": [0.01 * base, 0.02 * base],
                    "z_social": [0.03 * base],
                    "z_macro": [0.04 * base],
                },
            }
        )

    latest = sequence[-1]["numeric_features"]
    regime = "event" if latest["event_score"] >= 0.35 else ("trend" if latest["trend_120m"] > 0 else "chop")
    return {
        "date": date(2025, 1, 1) + timedelta(days=index),
        "state_sequence": sequence,
        "er_20d": 0.01 + latest["flow_strength"] * 0.03 + latest["event_score"] * 0.02,
        "dd_20d": min(0.95, 0.03 + latest["extension_60m"] * 0.8),
        "p_up_20d": min(0.95, max(0.05, 0.5 + latest["trend_120m"] * 0.6 + latest["event_score"] * 0.1)),
        "flow_persist": min(1.0, max(0.0, 0.5 + latest["flow_strength"] * 0.4)),
        "uncertainty": min(1.0, 0.2 + latest["extension_60m"] * 0.5),
        "regime_final": regime,
    }


def test_training_pipeline_trains_temporal_transformer_and_predicts(tmp_path):
    rows = [_row_with_state_sequence(index) for index in range(24)]
    weights_path, artifact_path, metrics_path = TrainingPipeline().train_temporal_transformer(
        rows,
        numeric_feature_keys=[
            "close_return_1d",
            "volume_z",
            "flow_strength",
            "trend_120m",
            "extension_60m",
            "event_score",
        ],
        artifact_dir=tmp_path,
        vector_feature_dims={"z_event": 2, "z_social": 1, "z_macro": 1},
        max_seq_len=4,
        embedding_hidden_dim=16,
        d_model=16,
        context_num_layers=1,
        num_heads=4,
        num_layers=2,
        dropout=0.0,
        batch_size=4,
        epochs=2,
        lr=5e-3,
        device="cpu",
    )

    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    predictor = NumericFirstPredictor(artifact_path=artifact_path)
    prediction = predictor.predict(
        "005930",
        "CORE_DAY",
        datetime(2026, 3, 20, tzinfo=timezone.utc),
        {
            "state_sequence": rows[-1]["state_sequence"],
            "semantic_branch_enabled": False,
            "text_branch_enabled": True,
        },
    )
    flat_prediction = predictor.predict(
        "005930",
        "CORE_DAY",
        datetime(2026, 3, 20, tzinfo=timezone.utc),
        {
            "close_return_1d": 0.01,
            "volume_z": 0.5,
            "flow_strength": 0.4,
            "trend_120m": 0.1,
            "extension_60m": 0.05,
            "event_score": 0.2,
            "vector_payload": {
                "z_event": [0.2, 0.1],
                "z_social": [0.05],
                "z_macro": [0.03],
            },
            "semantic_branch_enabled": False,
            "text_branch_enabled": True,
        },
    )

    assert weights_path.exists()
    assert metrics_path.exists()
    assert artifact["model_type"] == "temporal_transformer_v1"
    assert artifact["weights_path"] == weights_path.name
    assert artifact["context_num_layers"] == 1
    assert prediction.model_version == artifact["model_version"]
    assert 0.0 <= prediction.p_up_20d <= 1.0
    assert 0.0 <= prediction.dd_20d <= 1.0
    assert prediction.regime_final in {"trend", "event", "chop", "risk_off"}
    assert 0.0 <= flat_prediction.p_up_20d <= 1.0
    assert 0.0 <= flat_prediction.uncertainty <= 1.0
