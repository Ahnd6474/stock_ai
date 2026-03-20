from __future__ import annotations

from dataclasses import dataclass
from datetime import date


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
