from __future__ import annotations

import math


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


def _layer_norm(vector: list[float], eps: float = 1e-6) -> list[float]:
    if not vector:
        return []
    mean = sum(vector) / len(vector)
    var = sum((value - mean) ** 2 for value in vector) / len(vector)
    denom = math.sqrt(var + eps)
    return [(value - mean) / denom for value in vector]


def _softmax(values: list[float]) -> list[float]:
    if not values:
        return []
    max_value = max(values)
    exps = [math.exp(value - max_value) for value in values]
    total = sum(exps) or 1.0
    return [value / total for value in exps]


def _sinusoidal_position(index: int, dim: int) -> list[float]:
    if dim <= 0:
        return []
    out = [0.0] * dim
    for offset in range(0, dim, 2):
        scale = math.exp(-(offset / max(dim, 1)) * math.log(10000.0))
        angle = (index + 1) * scale
        out[offset] = math.sin(angle)
        if offset + 1 < dim:
            out[offset + 1] = math.cos(angle)
    return out


class HierarchicalAttentionAggregator:
    version = "hier_transformer_v3"

    def __init__(
        self,
        *,
        num_heads: int = 4,
        num_layers: int = 2,
        position_scale: float = 0.05,
        residual_mix: float = 0.65,
    ) -> None:
        self.num_heads = max(1, num_heads)
        self.num_layers = max(1, num_layers)
        self.position_scale = position_scale
        self.residual_mix = residual_mix

    def _score(self, item: dict) -> float:
        return (
            0.4 * item.get("freshness_score", 0.5)
            + 0.3 * item.get("source_quality_score", 0.5)
            + 0.2 * item.get("novelty_score", 0.5)
            + 0.1 * item.get("semantic_confidence", 0.5)
        )

    def _head_slices(self, dim: int) -> list[tuple[int, int]]:
        head_count = max(1, min(self.num_heads, dim))
        base = dim // head_count
        remainder = dim % head_count
        slices: list[tuple[int, int]] = []
        start = 0
        for head in range(head_count):
            width = base + (1 if head < remainder else 0)
            end = min(dim, start + max(1, width))
            slices.append((start, end))
            start = end
        return [item for item in slices if item[0] < item[1]]

    def _contextualize(self, vectors: list[list[float]], priors: list[float]) -> list[list[float]]:
        if not vectors:
            return []
        dim = len(vectors[0])
        encoded = []
        for index, vector in enumerate(vectors):
            position = _sinusoidal_position(index, dim)
            encoded.append(
                _layer_norm(
                    [vector[offset] + self.position_scale * position[offset] for offset in range(dim)]
                )
            )

        slices = self._head_slices(dim)
        biases = [math.log(max(weight, 1e-6)) for weight in priors]
        for _ in range(self.num_layers):
            updated: list[list[float]] = []
            for query_index, query in enumerate(encoded):
                head_context = [0.0] * dim
                for start, end in slices:
                    query_slice = query[start:end]
                    logits: list[float] = []
                    for key_index, key in enumerate(encoded):
                        key_slice = key[start:end]
                        score = sum(q * k for q, k in zip(query_slice, key_slice)) / math.sqrt(end - start)
                        score += biases[key_index]
                        score -= 0.03 * abs(query_index - key_index)
                        logits.append(score)
                    weights = _softmax(logits)
                    for offset, weight in enumerate(weights):
                        value_slice = encoded[offset][start:end]
                        for inner, value in enumerate(value_slice, start=start):
                            head_context[inner] += weight * value

                mixed = [
                    (1.0 - self.residual_mix) * query[offset] + self.residual_mix * head_context[offset]
                    for offset in range(dim)
                ]
                mixed = _layer_norm(mixed)
                feed_forward = [math.tanh(value) * 0.5 + value for value in mixed]
                updated.append(_layer_norm([mixed[offset] + 0.25 * feed_forward[offset] for offset in range(dim)]))
            encoded = updated
        return encoded

    def aggregate(self, items: list[dict], dim: int) -> list[float]:
        if not items:
            return [0.0] * dim
        vectors = [x["vec"] for x in items]
        weights = [max(0.01, self._score(x)) for x in items]
        contextualized = self._contextualize(vectors, weights)
        return _weighted_mean(contextualized, weights)

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
        contextualized = self._contextualize(cluster_vectors, cluster_weights)
        return _weighted_mean(contextualized, cluster_weights)
