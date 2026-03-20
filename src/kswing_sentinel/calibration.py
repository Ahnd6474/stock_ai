from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProbabilityCalibrator:
    """Simple affine + clip calibrator for probability heads."""

    slope: float = 1.0
    intercept: float = 0.0

    def transform(self, p: float) -> float:
        v = self.slope * p + self.intercept
        return min(1.0, max(0.0, v))


@dataclass(frozen=True)
class QuantileAdjuster:
    """Lightweight drawdown quantile adjustment."""

    scale: float = 1.0
    bias: float = 0.0

    def transform(self, value: float) -> float:
        return max(0.0, self.scale * value + self.bias)
