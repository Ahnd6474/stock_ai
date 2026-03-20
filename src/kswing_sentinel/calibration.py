from __future__ import annotations

from dataclasses import dataclass
from statistics import mean


@dataclass(frozen=True)
class CalibrationReport:
    metric_name: str
    pre: float
    post: float


@dataclass(frozen=True)
class BucketStat:
    bucket: str
    count: int
    realized_mean: float


@dataclass(frozen=True)
class ProbabilityCalibrator:
    """Simple affine + clip calibrator for probability heads."""

    slope: float = 1.0
    intercept: float = 0.0

    def transform(self, p: float) -> float:
        v = self.slope * p + self.intercept
        return min(1.0, max(0.0, v))

    @staticmethod
    def fit(preds: list[float], labels: list[float]) -> "ProbabilityCalibrator":
        if not preds or len(preds) != len(labels):
            return ProbabilityCalibrator()
        mp = mean(preds)
        my = mean(labels)
        var = sum((p - mp) ** 2 for p in preds)
        if var <= 1e-12:
            return ProbabilityCalibrator(1.0, my - mp)
        cov = sum((p - mp) * (y - my) for p, y in zip(preds, labels))
        slope = cov / var
        intercept = my - slope * mp
        return ProbabilityCalibrator(slope=slope, intercept=intercept)


@dataclass(frozen=True)
class QuantileAdjuster:
    """Lightweight drawdown quantile adjustment."""

    scale: float = 1.0
    bias: float = 0.0

    def transform(self, value: float) -> float:
        return max(0.0, self.scale * value + self.bias)

    @staticmethod
    def fit(pred_values: list[float], realized_values: list[float]) -> "QuantileAdjuster":
        if not pred_values or len(pred_values) != len(realized_values):
            return QuantileAdjuster()
        mp = mean(pred_values)
        my = mean(realized_values)
        var = sum((p - mp) ** 2 for p in pred_values)
        if var <= 1e-12:
            return QuantileAdjuster(1.0, max(0.0, my - mp))
        cov = sum((p - mp) * (y - my) for p, y in zip(pred_values, realized_values))
        scale = max(0.1, cov / var)
        bias = max(0.0, my - scale * mp)
        return QuantileAdjuster(scale=scale, bias=bias)


def calibration_report_binary(preds: list[float], labels: list[float], calibrator: ProbabilityCalibrator) -> CalibrationReport:
    if not preds or len(preds) != len(labels):
        return CalibrationReport("brier", 0.0, 0.0)
    pre = mean([(p - y) ** 2 for p, y in zip(preds, labels)])
    post_probs = [calibrator.transform(p) for p in preds]
    post = mean([(p - y) ** 2 for p, y in zip(post_probs, labels)])
    return CalibrationReport("brier", pre, post)


def calibration_report_dd(preds: list[float], realized: list[float], adjuster: QuantileAdjuster) -> CalibrationReport:
    if not preds or len(preds) != len(realized):
        return CalibrationReport("mae_dd", 0.0, 0.0)
    pre = mean([abs(p - y) for p, y in zip(preds, realized)])
    post_preds = [adjuster.transform(v) for v in preds]
    post = mean([abs(p - y) for p, y in zip(post_preds, realized)])
    return CalibrationReport("mae_dd", pre, post)


def summarize_uncertainty_buckets(uncertainties: list[float], outcomes: list[float]) -> list[BucketStat]:
    if not uncertainties or len(uncertainties) != len(outcomes):
        return []
    bins = [
        ("LOW", 0.0, 0.33),
        ("MID", 0.33, 0.66),
        ("HIGH", 0.66, 1.01),
    ]
    out: list[BucketStat] = []
    for name, lo, hi in bins:
        ys = [y for u, y in zip(uncertainties, outcomes) if lo <= u < hi]
        out.append(BucketStat(name, len(ys), mean(ys) if ys else 0.0))
    return out
