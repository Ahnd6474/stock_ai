from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path


@dataclass(frozen=True)
class Fold:
    train_start: date
    train_end: date
    valid_start: date
    valid_end: date


class WalkForwardSplitter:
    def build(self, days: list[date], train_size: int = 120, valid_size: int = 20, step: int = 20) -> list[Fold]:
        folds: list[Fold] = []
        i = 0
        while i + train_size + valid_size <= len(days):
            tr = days[i:i+train_size]
            va = days[i+train_size:i+train_size+valid_size]
            folds.append(Fold(tr[0], tr[-1], va[0], va[-1]))
            i += step
        return folds


class TrainingPipeline:
    """Lightweight scaffold for walk-forward training orchestration."""

    def __init__(self) -> None:
        self.splitter = WalkForwardSplitter()

    def build_dataset(self, rows: list[dict], feature_keys: list[str], label_key: str) -> list[dict]:
        dataset: list[dict] = []
        for r in rows:
            if label_key not in r:
                continue
            dataset.append(
                {
                    "x": {k: float(r.get(k, 0.0)) for k in feature_keys},
                    "y": float(r[label_key]),
                    "date": r["date"],
                }
            )
        return dataset

    def train_baseline(self, dataset: list[dict], feature_keys: list[str]) -> dict:
        """Fit a tiny mean-target linear baseline for scaffolding.

        Returns serializable artifact.
        """
        if not dataset:
            return {"model_type": "linear_stub", "bias": 0.0, "weights": {k: 0.0 for k in feature_keys}}

        y_mean = sum(r["y"] for r in dataset) / len(dataset)
        # very light heuristic weights from feature covariance sign
        weights: dict[str, float] = {}
        for k in feature_keys:
            x = [r["x"][k] for r in dataset]
            xm = sum(x) / len(x)
            cov = sum((xi - xm) * (r["y"] - y_mean) for xi, r in zip(x, dataset)) / max(1, len(dataset))
            weights[k] = cov
        return {"model_type": "linear_stub", "bias": y_mean, "weights": weights}

    def save_artifact(self, artifact: dict, path: str) -> None:
        Path(path).write_text(json.dumps(artifact, ensure_ascii=False, indent=2))
