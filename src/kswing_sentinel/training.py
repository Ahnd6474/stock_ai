from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
from statistics import mean


@dataclass(frozen=True)
class FoldMetric:
    fold_index: int
    label_key: str
    mae: float


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

    def leakage_safe_join(self, feature_rows: list[dict], label_rows: list[dict], label_key: str) -> list[dict]:
        labels = {(r["symbol"], r["date"]): r for r in label_rows}
        joined: list[dict] = []
        for fr in feature_rows:
            if fr.get("as_of_date", fr["date"]) > fr["date"]:
                continue
            lab = labels.get((fr["symbol"], fr["date"]))
            if not lab or label_key not in lab:
                continue
            row = dict(fr)
            row[label_key] = lab[label_key]
            joined.append(row)
        return joined

    def fit_linear_model(self, dataset: list[dict], feature_keys: list[str], *, lr: float = 0.03, epochs: int = 300) -> dict:
        weights = {k: 0.0 for k in feature_keys}
        bias = 0.0
        if not dataset:
            return {"weights": weights, "bias": bias}
        n = float(len(dataset))
        for _ in range(epochs):
            grad_w = {k: 0.0 for k in feature_keys}
            grad_b = 0.0
            for row in dataset:
                y = float(row["y"])
                pred = bias + sum(weights[k] * float(row["x"].get(k, 0.0)) for k in feature_keys)
                err = pred - y
                grad_b += err
                for k in feature_keys:
                    grad_w[k] += err * float(row["x"].get(k, 0.0))
            bias -= lr * (grad_b / n)
            for k in feature_keys:
                weights[k] -= lr * (grad_w[k] / n)
        return {"weights": weights, "bias": bias}

    def _mae(self, model: dict, rows: list[dict], feature_keys: list[str]) -> float:
        if not rows:
            return 0.0
        errs = []
        for r in rows:
            pred = float(model["bias"]) + sum(float(model["weights"].get(k, 0.0)) * float(r["x"].get(k, 0.0)) for k in feature_keys)
            errs.append(abs(pred - float(r["y"])))
        return mean(errs)

    def train_walk_forward(
        self,
        rows: list[dict],
        feature_keys: list[str],
        label_key: str,
        artifact_dir: str | Path,
        model_version: str = "linear_baseline_v1",
    ) -> tuple[Path, Path]:
        dataset = self.build_dataset(rows, feature_keys, label_key)
        days = sorted({r["date"] for r in dataset})
        folds = self.splitter.build(days, train_size=120, valid_size=20, step=20)

        metrics: list[FoldMetric] = []
        final_model = {"weights": {k: 0.0 for k in feature_keys}, "bias": 0.0}
        for i, f in enumerate(folds):
            train_rows = [r for r in dataset if f.train_start <= r["date"] <= f.train_end]
            valid_rows = [r for r in dataset if f.valid_start <= r["date"] <= f.valid_end]
            if not train_rows:
                continue
            model = self.fit_linear_model(train_rows, feature_keys)
            final_model = model
            metrics.append(FoldMetric(i, label_key, self._mae(model, valid_rows, feature_keys)))

        out_dir = Path(artifact_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = out_dir / f"{label_key}_artifact.json"
        metrics_path = out_dir / f"{label_key}_fold_metrics.json"
        artifact_payload = {
            "model_version": model_version,
            "schema_version": "v1",
            "feature_keys": feature_keys,
            "label_key": label_key,
            "weights": final_model["weights"],
            "bias": final_model["bias"],
        }
        artifact_path.write_text(json.dumps(artifact_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        metrics_path.write_text(
            json.dumps([m.__dict__ for m in metrics], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return artifact_path, metrics_path

    def train_multi_head(
        self,
        rows: list[dict],
        feature_keys: list[str],
        artifact_dir: str | Path,
        label_keys: tuple[str, str, str] = ("er_20d", "dd_20d", "p_up_20d"),
        model_version: str = "linear_multihead_v1",
    ) -> Path:
        head_payload: dict[str, dict] = {}
        for label_key in label_keys:
            dataset = self.build_dataset(rows, feature_keys, label_key)
            model = self.fit_linear_model(dataset, feature_keys)
            head_payload[label_key] = model

        artifact = {
            "model_version": model_version,
            "schema_version": "v1",
            "feature_keys": feature_keys,
            "heads": head_payload,
            "uncertainty_base": 0.30,
            "uncertainty_extension_mult": 0.30,
        }
        out = Path(artifact_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "multi_head_artifact.json"
        path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
        return path
