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
