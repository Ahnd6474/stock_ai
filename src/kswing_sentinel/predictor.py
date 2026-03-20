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

    def validate_schema(self, features: dict) -> None:
        required = {"flow_strength", "trend_120m", "extension_60m"}
        missing = [k for k in required if k not in features]
        if missing:
            raise ValueError(f"missing required features: {missing}")

    def predict(self, symbol: str, session_type: str, as_of_time: datetime, features: dict) -> FusedPrediction:
        self.validate_schema(features)
        flow_strength = float(features.get("flow_strength", 0.0))
        trend_120m = float(features.get("trend_120m", 0.0))
        extension = float(features.get("extension_60m", 0.0))
        event_score = float(features.get("event_score", 0.0))
        uncertainty = min(1.0, max(0.0, 0.35 + 0.25 * abs(extension)))
        if self.model_blob.get("weights"):
            weights = self.model_blob.get("weights", {})
            bias = float(self.model_blob.get("bias", 0.0))
            er20 = bias + sum(float(weights.get(k, 0.0)) * float(features.get(k, 0.0)) for k in weights)
        else:
            er20 = 0.01 + 0.02 * flow_strength + 0.015 * trend_120m + 0.01 * event_score - 0.01 * max(extension, 0)
        er5 = er20 * 0.45
        dd20 = max(0.01, 0.05 - 0.02 * trend_120m + 0.015 * max(extension, 0))
        pup = min(0.95, max(0.05, 0.5 + er20 * 2.0))
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
