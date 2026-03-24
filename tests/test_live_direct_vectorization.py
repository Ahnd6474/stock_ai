from datetime import datetime, timezone

from kswing_sentinel.live import LiveInferenceService
from kswing_sentinel.schemas import FusedPrediction


class ExplodingNormalizer:
    prompt_version = "should_not_run"

    def normalize(self, payload, retry_once=True):
        raise AssertionError("LLM normalizer should not be called")


class RecordingVectorizer:
    encoder_version = "stub_vectorizer_v1"

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def build(
        self,
        summary,
        social="",
        macro="",
        source_doc_ids=None,
        cluster_ids=None,
        as_of_time=None,
        session_type="OFF_MARKET",
    ):
        self.calls.append(
            {
                "summary": summary,
                "source_doc_ids": list(source_doc_ids or []),
                "cluster_ids": list(cluster_ids or []),
                "as_of_time": as_of_time,
                "session_type": session_type,
            }
        )
        return {
            "z_event": [0.0] * 64,
            "z_social": [0.0] * 32,
            "z_macro": [0.0] * 16,
            "metadata": {"embedding_backend": "stub"},
        }


class RecordingPredictor:
    def __init__(self) -> None:
        self.calls: list[dict] = []

        class Artifact:
            model_version = "stub_model_v1"

        self.artifact = Artifact()

    def predict(self, symbol, session_type, as_of_time, features):
        self.calls.append(dict(features))
        return FusedPrediction(
            symbol=symbol,
            as_of_time=as_of_time,
            session_type=session_type,
            er_5d=0.01,
            er_20d=0.02,
            dd_20d=0.1,
            p_up_20d=0.6,
            flow_persist=0.8,
            uncertainty=0.1,
            regime_final="event",
            event_score=float(features.get("event_score", 0.0)),
            semantic_branch_enabled=bool(features.get("semantic_branch_enabled", False)),
            text_branch_enabled=bool(features.get("text_branch_enabled", False)),
            model_version=self.artifact.model_version,
            calibrator_version="stub_cal_v1",
        )


def test_live_service_bypasses_llm_and_vectorizes_raw_event_text():
    vectorizer = RecordingVectorizer()
    predictor = RecordingPredictor()
    live = LiveInferenceService(
        normalizer=ExplodingNormalizer(),
        vectorizer=vectorizer,
        predictor=predictor,
    )

    decision = live.run_for_symbol(
        symbol="005930",
        as_of_time=datetime(2026, 3, 20, 9, 35, tzinfo=timezone.utc),
        raw_event_payload={
            "headline": "Memory pricing improved",
            "body": "Management also pointed to stronger order visibility.",
            "source_doc_ids": ["doc-1"],
            "cluster_ids": ["cluster-1"],
            "event_score": 0.33,
        },
        features={
            "flow_strength": 0.8,
            "trend_120m": 0.8,
            "extension_60m": 0.1,
            "state_sequence": [
                {"numeric_features": {"flow_strength": 0.4, "trend_120m": 0.2, "extension_60m": 0.1}},
                {"numeric_features": {"flow_strength": 0.8, "trend_120m": 0.8, "extension_60m": 0.1}},
            ],
        },
        venue_eligibility="KRX_ONLY",
    )

    assert vectorizer.calls[0]["summary"] == (
        "Memory pricing improved\n\nManagement also pointed to stronger order visibility."
    )
    assert vectorizer.calls[0]["source_doc_ids"] == ["doc-1"]
    assert vectorizer.calls[0]["cluster_ids"] == ["cluster-1"]
    assert predictor.calls[0]["vector_payload"]["metadata"]["embedding_backend"] == "stub"
    assert "vector_payload" not in predictor.calls[0]["state_sequence"][0]
    assert predictor.calls[0]["state_sequence"][-1]["vector_payload"]["metadata"]["embedding_backend"] == "stub"
    assert predictor.calls[0]["semantic_branch_enabled"] is False
    assert predictor.calls[0]["text_branch_enabled"] is True
    assert predictor.calls[0]["event_score"] == 0.33
    assert decision.action == "BUY"


def test_live_service_disables_text_branch_when_no_raw_text_exists():
    vectorizer = RecordingVectorizer()
    predictor = RecordingPredictor()
    live = LiveInferenceService(
        normalizer=ExplodingNormalizer(),
        vectorizer=vectorizer,
        predictor=predictor,
    )

    live.run_for_symbol(
        symbol="005930",
        as_of_time=datetime(2026, 3, 20, 9, 35, tzinfo=timezone.utc),
        raw_event_payload={},
        features={"flow_strength": 0.1, "trend_120m": 0.8, "extension_60m": 0.1},
        venue_eligibility="KRX_ONLY",
    )

    assert vectorizer.calls == []
    assert predictor.calls[0]["semantic_branch_enabled"] is False
    assert predictor.calls[0]["text_branch_enabled"] is False
    assert predictor.calls[0]["event_score"] == 0.0
