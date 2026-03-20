from __future__ import annotations


def _weighted_mean(vectors: list[list[float]], weights: list[float]) -> list[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    wsum = sum(weights) if sum(weights) != 0 else 1.0
    out = [0.0] * dim
    for vec, w in zip(vectors, weights):
        for i in range(dim):
            out[i] += vec[i] * (w / wsum)
    return out


class HierarchicalAttentionAggregator:
    def aggregate(self, items: list[dict], dim: int) -> list[float]:
        if not items:
            return [0.0] * dim
        vectors = [x["vec"] for x in items]
        weights = [
            0.4 * x.get("freshness_score", 0.5)
            + 0.3 * x.get("source_quality_score", 0.5)
            + 0.2 * x.get("novelty_score", 0.5)
            + 0.1 * x.get("semantic_confidence", 0.5)
            for x in items
        ]
        return _weighted_mean(vectors, weights)
