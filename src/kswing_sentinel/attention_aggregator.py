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
    def _score(self, item: dict) -> float:
        return (
            0.4 * item.get("freshness_score", 0.5)
            + 0.3 * item.get("source_quality_score", 0.5)
            + 0.2 * item.get("novelty_score", 0.5)
            + 0.1 * item.get("semantic_confidence", 0.5)
        )

    def aggregate(self, items: list[dict], dim: int) -> list[float]:
        if not items:
            return [0.0] * dim
        vectors = [x["vec"] for x in items]
        weights = [self._score(x) for x in items]
        return _weighted_mean(vectors, weights)

    def aggregate_by_cluster(self, cluster_items: dict[str, list[dict]], dim: int) -> list[float]:
        if not cluster_items:
            return [0.0] * dim
        cluster_vectors: list[list[float]] = []
        cluster_weights: list[float] = []
        for items in cluster_items.values():
            cvec = self.aggregate(items, dim)
            cweight = sum(max(0.01, self._score(i)) for i in items)
            cluster_vectors.append(cvec)
            cluster_weights.append(cweight)
        return _weighted_mean(cluster_vectors, cluster_weights)
