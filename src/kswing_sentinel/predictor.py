from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any

from .calibration import ProbabilityCalibrator, QuantileAdjuster
from .schemas import FusedPrediction


@dataclass(frozen=True)
class ModelArtifact:
    model_version: str
    schema_version: str = "v1"


class NumericFirstPredictor:
    def __init__(
        self,
        artifact: ModelArtifact | None = None,
        model_blob: dict[str, Any] | None = None,
        artifact_path: str | Path | None = None,
        p_up_calibrator: ProbabilityCalibrator | None = None,
        dd_adjuster: QuantileAdjuster | None = None,
    ) -> None:
        self.artifact = artifact or ModelArtifact(model_version="numeric_baseline_v1")
        self.model_blob = model_blob or self._load_model_blob(artifact_path)
        if self.model_blob.get("model_version"):
            self.artifact = ModelArtifact(
                model_version=str(self.model_blob.get("model_version")),
                schema_version=str(self.model_blob.get("schema_version", "v1")),
            )
        self.p_up_calibrator = p_up_calibrator or ProbabilityCalibrator()
        self.dd_adjuster = dd_adjuster or QuantileAdjuster()

    def _load_model_blob(self, artifact_path: str | Path | None) -> dict[str, Any]:
        if artifact_path is None:
            return {}
        path = Path(artifact_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("artifact payload must be a JSON object")
        return payload

    def _model_type(self) -> str:
        return str(self.model_blob.get("model_type", "linear_head"))

    def validate_schema(self, features: dict) -> None:
        required = set(self.model_blob.get("feature_keys", ["flow_strength", "trend_120m", "extension_60m"]))
        missing = [k for k in required if k not in features]
        if missing:
            raise ValueError(f"missing required features: {missing}")

    def _predict_linear_head(self, head_blob: dict[str, Any], features: dict) -> float:
        weights = head_blob.get("weights", {})
        bias = float(head_blob.get("bias", 0.0))
        return bias + sum(float(weights.get(k, 0.0)) * float(features.get(k, 0.0)) for k in weights)

    def _predict_tree_ensemble_head(self, head_blob: dict[str, Any], features: dict) -> float:
        # 학습 산출물(JSON dump) 기반 GBDT 유사 추론기.
        score = float(head_blob.get("base_score", 0.0))
        trees = head_blob.get("trees", [])
        if not isinstance(trees, list):
            raise ValueError("trees must be a list")
        for tree in trees:
            feature = str(tree.get("feature", ""))
            threshold = float(tree.get("threshold", 0.0))
            left = float(tree.get("left", 0.0))
            right = float(tree.get("right", 0.0))
            value = float(features.get(feature, 0.0))
            score += left if value <= threshold else right
        return score

    def _predict_head(self, head_blob: dict[str, Any], features: dict) -> float:
        model_type = self._model_type()
        if model_type == "tree_ensemble_v1":
            return self._predict_tree_ensemble_head(head_blob, features)
        return self._predict_linear_head(head_blob, features)

    def predict(self, symbol: str, session_type: str, as_of_time: datetime, features: dict) -> FusedPrediction:
        self.validate_schema(features)
        flow_strength = float(features.get("flow_strength", 0.0))
        trend_120m = float(features.get("trend_120m", 0.0))
        extension = float(features.get("extension_60m", 0.0))
        event_score = float(features.get("event_score", 0.0))
        uncertainty = min(
            1.0,
            max(
                0.0,
                float(self.model_blob.get("uncertainty_base", 0.35))
                + float(self.model_blob.get("uncertainty_extension_mult", 0.25)) * abs(extension),
            ),
        )
        heads = self.model_blob.get("heads", {})
        if heads:
            er20 = self._predict_head(heads.get("er_20d", {}), features)
            dd20 = self._predict_head(heads.get("dd_20d", {}), features)
            pup_raw = self._predict_head(heads.get("p_up_20d", {}), features)
            pup = min(0.95, max(0.05, pup_raw))
        elif self.model_blob.get("weights"):
            er20 = self._predict_linear_head(self.model_blob, features)
            dd20 = max(0.01, 0.05 - 0.02 * trend_120m + 0.015 * max(extension, 0))
            pup = min(0.95, max(0.05, 0.5 + er20 * 2.0))
        else:
            er20 = 0.01 + 0.02 * flow_strength + 0.015 * trend_120m + 0.01 * event_score - 0.01 * max(extension, 0)
            dd20 = max(0.01, 0.05 - 0.02 * trend_120m + 0.015 * max(extension, 0))
            pup = min(0.95, max(0.05, 0.5 + er20 * 2.0))
        er5 = er20 * 0.45
        pup = self.p_up_calibrator.transform(pup)
        dd20 = self.dd_adjuster.transform(dd20)
        flow_persist = min(1.0, max(0.0, 0.5 + flow_strength * 0.4))
        regime = "risk_off" if features.get("market_risk_off") else ("trend" if trend_120m > 0 else "chop")
        return FusedPrediction(
            symbol=symbol,
            as_of_time=as_of_time,
            session_type=session_type,
            er_5d=er5,
            er_20d=er20,
            dd_20d=dd20,
            p_up_20d=pup,
            flow_persist=flow_persist,
            uncertainty=uncertainty,
            regime_final=regime,
            event_score=event_score,
        )
