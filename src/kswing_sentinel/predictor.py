from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timezone
import json
from math import copysign, log1p, pi, sin, cos
from pathlib import Path
from typing import Any

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - optional dependency
    torch = None
    nn = None

from .calibration import ProbabilityCalibrator, QuantileAdjuster
from .schemas import FusedPrediction


REGIME_LABELS = ("trend", "event", "chop", "risk_off")
DEFAULT_TEMPORAL_SEQUENCE_KEY = "state_sequence"
DEFAULT_TEMPORAL_NUMERIC_FEATURE_KEYS = ["flow_strength", "trend_120m", "extension_60m"]
DEFAULT_VECTOR_FEATURE_DIMS = {"z_event": 64, "z_social": 32, "z_macro": 16}
DEFAULT_TEMPORAL_TIME_FEATURE_DIM = 20
DEFAULT_TEMPORAL_DELTA_FEATURE_DIM = 7


@dataclass(frozen=True)
class ModelArtifact:
    model_version: str
    schema_version: str = "v1"


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _resolve_vector_feature_dims(payload: object) -> dict[str, int]:
    if not isinstance(payload, dict):
        return dict(DEFAULT_VECTOR_FEATURE_DIMS)
    out: dict[str, int] = {}
    for key, value in payload.items():
        try:
            dim = int(value)
        except (TypeError, ValueError):
            continue
        if dim > 0:
            out[str(key)] = dim
    return out or dict(DEFAULT_VECTOR_FEATURE_DIMS)


def _flatten_vector(raw_value: object, dim: int) -> list[float]:
    if isinstance(raw_value, list):
        values = [_safe_float(item, 0.0) for item in raw_value[:dim]]
        if len(values) < dim:
            values.extend([0.0] * (dim - len(values)))
        return values
    return [0.0] * dim


def _vector_source_for_state(state: dict, key: str) -> object:
    if key in state:
        return state.get(key)
    vector_payload = state.get("vector_payload")
    if isinstance(vector_payload, dict) and key in vector_payload:
        return vector_payload.get(key)
    vectors = state.get("vectors")
    if isinstance(vectors, dict) and key in vectors:
        return vectors.get(key)
    return None


def _top_level_vector_source(features: dict | None, key: str) -> object:
    if not isinstance(features, dict):
        return None
    if key in features:
        return features.get(key)
    top_vector_payload = features.get("vector_payload")
    if isinstance(top_vector_payload, dict) and key in top_vector_payload:
        return top_vector_payload.get(key)
    top_vectors = features.get("vectors")
    if isinstance(top_vectors, dict) and key in top_vectors:
        return top_vectors.get(key)
    return None


def _vector_metadata(payload: object) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        return metadata
    return None


def _parse_datetime(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def _extract_datetime_from_mapping(payload: object, keys: tuple[str, ...]) -> datetime | None:
    if not isinstance(payload, dict):
        return None
    for key in keys:
        parsed = _parse_datetime(payload.get(key))
        if parsed is not None:
            return parsed
    return None


def _resolve_temporal_states(
    features: dict,
    *,
    sequence_key: str,
    max_seq_len: int | None = None,
) -> tuple[list[dict], bool]:
    raw_states = features.get(sequence_key)
    using_explicit_sequence = isinstance(raw_states, list) and bool(raw_states)
    if isinstance(raw_states, list) and raw_states:
        states = [state for state in raw_states if isinstance(state, dict)]
    else:
        states = [features]
    if max_seq_len is not None and max_seq_len > 0 and len(states) > max_seq_len:
        states = states[-max_seq_len:]
    return states, using_explicit_sequence


def _state_timestamp(
    state: dict,
    *,
    features: dict | None = None,
    allow_top_level_fallback: bool = False,
) -> datetime | None:
    keys = (
        "as_of_time",
        "timestamp",
        "observed_at",
        "collected_at",
        "available_at",
        "retrieved_at",
        "published_at",
        "generated_at",
    )
    numeric = state.get("numeric_features")
    for source in (
        state,
        numeric,
        _vector_metadata(state.get("vector_payload")),
        _vector_metadata(state.get("vectors")),
        state.get("metadata"),
    ):
        parsed = _extract_datetime_from_mapping(source, keys)
        if parsed is not None:
            return parsed
    if allow_top_level_fallback and isinstance(features, dict):
        for source in (
            features,
            _vector_metadata(features.get("vector_payload")),
            _vector_metadata(features.get("vectors")),
            features.get("metadata"),
        ):
            parsed = _extract_datetime_from_mapping(source, keys)
            if parsed is not None:
                return parsed
    return None


def _resolve_target_time(
    features: dict,
    *,
    states: list[dict],
    explicit_as_of_time: datetime | None = None,
) -> datetime | None:
    if explicit_as_of_time is not None:
        return explicit_as_of_time if explicit_as_of_time.tzinfo is not None else explicit_as_of_time.replace(tzinfo=timezone.utc)
    for source in (
        features,
        _vector_metadata(features.get("vector_payload")),
        _vector_metadata(features.get("vectors")),
        features.get("metadata"),
    ):
        parsed = _extract_datetime_from_mapping(
            source,
            ("as_of_time", "timestamp", "observed_at", "collected_at", "available_at", "generated_at"),
        )
        if parsed is not None:
            return parsed
    for state in reversed(states):
        parsed = _state_timestamp(state, features=features, allow_top_level_fallback=False)
        if parsed is not None:
            return parsed
    raw_date = features.get("date")
    if isinstance(raw_date, date) and not isinstance(raw_date, datetime):
        return datetime.combine(raw_date, time.min, tzinfo=timezone.utc)
    return None


def _delta_source_for_state(
    state: dict,
    *,
    features: dict | None = None,
    allow_top_level_fallback: bool = False,
) -> dict[str, Any]:
    keys = (
        "delta_doc_count",
        "new_doc_count",
        "updated_doc_count",
        "delta_novelty_mean",
        "delta_source_quality_mean",
        "delta_freshness_mean",
        "time_since_last_collection_sec",
    )
    for source in (
        state.get("delta_features"),
        state.get("social_delta"),
        state.get("semantic_delta"),
        _vector_metadata(state.get("vector_payload")),
        _vector_metadata(state.get("vectors")),
        state.get("metadata"),
    ):
        if isinstance(source, dict) and any(key in source for key in keys):
            return source
    if allow_top_level_fallback and isinstance(features, dict):
        for source in (
            features.get("delta_features"),
            features.get("social_delta"),
            features.get("semantic_delta"),
            _vector_metadata(features.get("vector_payload")),
            _vector_metadata(features.get("vectors")),
            features.get("metadata"),
        ):
            if isinstance(source, dict) and any(key in source for key in keys):
                return source
    return {}


def _cyclical_pair(value: float, period: float) -> tuple[float, float]:
    if period <= 0:
        return (0.0, 0.0)
    angle = 2.0 * pi * (value / period)
    return (sin(angle), cos(angle))


def _time_feature_row(
    *,
    anchor_time: datetime | None,
    target_time: datetime | None,
    relative_step: int,
    step_count: int,
    since_prev_seconds: float,
) -> list[float]:
    step_denominator = max(1, step_count - 1)
    relative_step_norm = relative_step / float(step_denominator)
    relative_seconds = (anchor_time - target_time).total_seconds() if anchor_time is not None and target_time is not None else float(relative_step)
    relative_seconds_norm = _clamp(relative_seconds / 86400.0, -1.0, 1.0)
    relative_log_norm = copysign(
        min(1.0, log1p(abs(relative_seconds)) / log1p(30.0 * 86400.0)),
        relative_seconds,
    )
    since_prev_norm = min(1.0, log1p(max(0.0, since_prev_seconds)) / log1p(7.0 * 86400.0))
    if anchor_time is None:
        absolute = [0.0] * 16
    else:
        month_sin, month_cos = _cyclical_pair(float(anchor_time.month - 1), 12.0)
        day_sin, day_cos = _cyclical_pair(float(anchor_time.day - 1), 31.0)
        hour_sin, hour_cos = _cyclical_pair(float(anchor_time.hour), 24.0)
        minute_sin, minute_cos = _cyclical_pair(float(anchor_time.minute), 60.0)
        second_sin, second_cos = _cyclical_pair(float(anchor_time.second), 60.0)
        absolute = [
            (anchor_time.year - 2000.0) / 50.0,
            (anchor_time.month - 1.0) / 11.0 if anchor_time.month > 0 else 0.0,
            (anchor_time.day - 1.0) / 30.0 if anchor_time.day > 0 else 0.0,
            anchor_time.hour / 23.0 if anchor_time.hour >= 0 else 0.0,
            anchor_time.minute / 59.0 if anchor_time.minute >= 0 else 0.0,
            anchor_time.second / 59.0 if anchor_time.second >= 0 else 0.0,
            month_sin,
            month_cos,
            day_sin,
            day_cos,
            hour_sin,
            hour_cos,
            minute_sin,
            minute_cos,
            second_sin,
            second_cos,
        ]
    return [relative_step_norm, relative_seconds_norm, relative_log_norm, since_prev_norm, *absolute]


def _delta_feature_row(source: dict[str, Any], *, since_prev_seconds: float) -> list[float]:
    delta_doc_count = _safe_float(source.get("delta_doc_count", source.get("doc_delta_count", 0.0)), 0.0)
    new_doc_count = _safe_float(source.get("new_doc_count", 0.0), 0.0)
    updated_doc_count = _safe_float(source.get("updated_doc_count", source.get("modified_doc_count", 0.0)), 0.0)
    if delta_doc_count <= 0.0:
        delta_doc_count = max(0.0, new_doc_count + updated_doc_count)
    novelty_mean = _clamp(
        _safe_float(source.get("delta_novelty_mean", source.get("novelty_mean", 0.0)), 0.0),
        0.0,
        1.0,
    )
    quality_mean = _clamp(
        _safe_float(source.get("delta_source_quality_mean", source.get("source_quality_mean", 0.0)), 0.0),
        0.0,
        1.0,
    )
    freshness_mean = _clamp(
        _safe_float(source.get("delta_freshness_mean", source.get("freshness_mean", 0.0)), 0.0),
        0.0,
        1.0,
    )
    time_since_last_collection = max(
        0.0,
        _safe_float(source.get("time_since_last_collection_sec", since_prev_seconds), since_prev_seconds),
    )
    return [
        min(1.0, delta_doc_count / 8.0),
        min(1.0, new_doc_count / 8.0),
        min(1.0, updated_doc_count / 8.0),
        novelty_mean,
        quality_mean,
        freshness_mean,
        min(1.0, log1p(time_since_last_collection) / log1p(7.0 * 86400.0)),
    ]


def infer_temporal_event_score(features: dict, sequence_key: str = DEFAULT_TEMPORAL_SEQUENCE_KEY) -> float:
    if "event_score" in features:
        return _safe_float(features.get("event_score"), 0.0)
    sequence = features.get(sequence_key)
    if isinstance(sequence, list) and sequence:
        last = sequence[-1]
        if isinstance(last, dict):
            numeric = last.get("numeric_features")
            if isinstance(numeric, dict) and "event_score" in numeric:
                return _safe_float(numeric.get("event_score"), 0.0)
            if "event_score" in last:
                return _safe_float(last.get("event_score"), 0.0)
    return 0.0


def build_temporal_state_matrix(
    features: dict,
    *,
    numeric_feature_keys: list[str],
    vector_feature_dims: dict[str, int],
    sequence_key: str = DEFAULT_TEMPORAL_SEQUENCE_KEY,
    max_seq_len: int | None = None,
) -> list[list[float]]:
    states, using_explicit_sequence = _resolve_temporal_states(
        features,
        sequence_key=sequence_key,
        max_seq_len=max_seq_len,
    )
    matrix: list[list[float]] = []
    for index, state in enumerate(states):
        numeric = state.get("numeric_features")
        numeric_source = numeric if isinstance(numeric, dict) else state
        row = [_safe_float(numeric_source.get(key), 0.0) for key in numeric_feature_keys]
        allow_top_level_fallback = not using_explicit_sequence or index == len(states) - 1
        for key, dim in vector_feature_dims.items():
            vector_source = _vector_source_for_state(state, key)
            if vector_source is None and allow_top_level_fallback:
                vector_source = _top_level_vector_source(features, key)
            row.extend(_flatten_vector(vector_source, dim))
        matrix.append(row)

    if not matrix:
        input_dim = len(numeric_feature_keys) + sum(vector_feature_dims.values())
        matrix = [[0.0] * input_dim]
    return matrix


def build_temporal_time_matrix(
    features: dict,
    *,
    as_of_time: datetime | None = None,
    sequence_key: str = DEFAULT_TEMPORAL_SEQUENCE_KEY,
    max_seq_len: int | None = None,
) -> list[list[float]]:
    states, using_explicit_sequence = _resolve_temporal_states(
        features,
        sequence_key=sequence_key,
        max_seq_len=max_seq_len,
    )
    target_time = _resolve_target_time(features, states=states, explicit_as_of_time=as_of_time)
    matrix: list[list[float]] = []
    previous_anchor: datetime | None = None
    last_index = len(states) - 1
    for index, state in enumerate(states):
        allow_top_level_fallback = not using_explicit_sequence or index == last_index
        state_time = _state_timestamp(
            state,
            features=features,
            allow_top_level_fallback=allow_top_level_fallback,
        )
        since_prev_seconds = 0.0
        if previous_anchor is not None and state_time is not None:
            since_prev_seconds = max(0.0, (state_time - previous_anchor).total_seconds())
        matrix.append(
            _time_feature_row(
                anchor_time=state_time or target_time,
                target_time=target_time,
                relative_step=index - last_index,
                step_count=len(states),
                since_prev_seconds=since_prev_seconds,
            )
        )
        previous_anchor = state_time or previous_anchor or target_time
    if not matrix:
        matrix = [[0.0] * DEFAULT_TEMPORAL_TIME_FEATURE_DIM]
    return matrix


def build_temporal_social_matrix(
    features: dict,
    *,
    social_dim: int,
    sequence_key: str = DEFAULT_TEMPORAL_SEQUENCE_KEY,
    max_seq_len: int | None = None,
) -> list[list[float]]:
    if social_dim <= 0:
        return []
    states, using_explicit_sequence = _resolve_temporal_states(
        features,
        sequence_key=sequence_key,
        max_seq_len=max_seq_len,
    )
    matrix: list[list[float]] = []
    for index, state in enumerate(states):
        allow_top_level_fallback = not using_explicit_sequence or index == len(states) - 1
        social_source = _vector_source_for_state(state, "z_social")
        if social_source is None and allow_top_level_fallback:
            social_source = _top_level_vector_source(features, "z_social")
        matrix.append(_flatten_vector(social_source, social_dim))
    if not matrix:
        matrix = [[0.0] * social_dim]
    return matrix


def build_temporal_delta_matrix(
    features: dict,
    *,
    as_of_time: datetime | None = None,
    sequence_key: str = DEFAULT_TEMPORAL_SEQUENCE_KEY,
    max_seq_len: int | None = None,
) -> list[list[float]]:
    states, using_explicit_sequence = _resolve_temporal_states(
        features,
        sequence_key=sequence_key,
        max_seq_len=max_seq_len,
    )
    target_time = _resolve_target_time(features, states=states, explicit_as_of_time=as_of_time)
    matrix: list[list[float]] = []
    previous_anchor: datetime | None = None
    last_index = len(states) - 1
    for index, state in enumerate(states):
        allow_top_level_fallback = not using_explicit_sequence or index == last_index
        state_time = _state_timestamp(
            state,
            features=features,
            allow_top_level_fallback=allow_top_level_fallback,
        )
        since_prev_seconds = 0.0
        if previous_anchor is not None and state_time is not None:
            since_prev_seconds = max(0.0, (state_time - previous_anchor).total_seconds())
        elif previous_anchor is not None and target_time is not None and state_time is None:
            since_prev_seconds = max(0.0, (target_time - previous_anchor).total_seconds())
        delta_source = _delta_source_for_state(
            state,
            features=features,
            allow_top_level_fallback=allow_top_level_fallback,
        )
        matrix.append(_delta_feature_row(delta_source, since_prev_seconds=since_prev_seconds))
        previous_anchor = state_time or previous_anchor or target_time
    if not matrix:
        matrix = [[0.0] * DEFAULT_TEMPORAL_DELTA_FEATURE_DIM]
    return matrix


def build_temporal_model_inputs(
    features: dict,
    *,
    numeric_feature_keys: list[str],
    vector_feature_dims: dict[str, int],
    as_of_time: datetime | None = None,
    sequence_key: str = DEFAULT_TEMPORAL_SEQUENCE_KEY,
    max_seq_len: int | None = None,
    include_time_features: bool = True,
    include_delta_features: bool = True,
) -> dict[str, list[list[float]]]:
    return {
        "state_sequence": build_temporal_state_matrix(
            features,
            numeric_feature_keys=numeric_feature_keys,
            vector_feature_dims=vector_feature_dims,
            sequence_key=sequence_key,
            max_seq_len=max_seq_len,
        ),
        "time_sequence": (
            build_temporal_time_matrix(
                features,
                as_of_time=as_of_time,
                sequence_key=sequence_key,
                max_seq_len=max_seq_len,
            )
            if include_time_features
            else []
        ),
        "social_sequence": build_temporal_social_matrix(
            features,
            social_dim=int(vector_feature_dims.get("z_social", 0)),
            sequence_key=sequence_key,
            max_seq_len=max_seq_len,
        ),
        "delta_sequence": (
            build_temporal_delta_matrix(
                features,
                as_of_time=as_of_time,
                sequence_key=sequence_key,
                max_seq_len=max_seq_len,
            )
            if include_delta_features
            else []
        ),
    }


class _GPTBlock(nn.Module if nn is not None else object):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        if nn is None:
            raise RuntimeError("torch is required for temporal transformer predictors")
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln_2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Any, attn_mask: Any, padding_mask: Any | None = None) -> Any:
        h = self.ln_1(x)
        attn_out, _ = self.attn(
            h,
            h,
            h,
            attn_mask=attn_mask,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        x = x + attn_out
        x = x + self.ff(self.ln_2(x))
        return x


class _GatedDeltaNetwork(nn.Module if nn is not None else object):
    def __init__(self, *, social_dim: int, d_model: int, delta_feature_dim: int, dropout: float) -> None:
        if nn is None:
            raise RuntimeError("torch is required for temporal transformer predictors")
        super().__init__()
        gate_hidden_dim = max(d_model, social_dim * 2, 32)
        candidate_hidden_dim = max(d_model, social_dim, 32)
        gate_input_dim = (social_dim * 3) + d_model + delta_feature_dim
        candidate_input_dim = (social_dim * 2) + d_model + delta_feature_dim
        self.gate = nn.Sequential(
            nn.Linear(gate_input_dim, gate_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_dim, d_model),
            nn.Sigmoid(),
        )
        self.candidate = nn.Sequential(
            nn.Linear(candidate_input_dim, candidate_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(candidate_hidden_dim, d_model),
            nn.Tanh(),
        )
        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        social_sequence: Any,
        time_context: Any,
        delta_features: Any | None = None,
        padding_mask: Any | None = None,
    ) -> Any:
        prev_social = torch.zeros_like(social_sequence)
        prev_social[:, 1:, :] = social_sequence[:, :-1, :]
        social_delta = social_sequence - prev_social
        delta_abs = social_delta.abs()
        gate_parts = [social_sequence, prev_social, delta_abs, time_context]
        candidate_parts = [social_delta, delta_abs, time_context]
        if delta_features is not None:
            gate_parts.append(delta_features)
            candidate_parts.append(delta_features)
        gate = self.gate(torch.cat(gate_parts, dim=-1))
        candidate = self.candidate(torch.cat(candidate_parts, dim=-1))
        out = self.output_norm(gate * candidate)
        if padding_mask is not None:
            out = out.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return out


class TemporalStateAttentionModel(nn.Module if nn is not None else object):
    def __init__(
        self,
        *,
        input_dim: int,
        embedding_hidden_dim: int = 128,
        d_model: int = 128,
        context_num_layers: int = 1,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 32,
        time_feature_dim: int = 0,
        social_feature_dim: int = 0,
        delta_feature_dim: int = 0,
    ) -> None:
        if nn is None or torch is None:
            raise RuntimeError("torch is required for temporal transformer predictors")
        super().__init__()
        self.max_seq_len = max_seq_len
        self.time_feature_dim = max(0, int(time_feature_dim))
        self.social_feature_dim = max(0, int(social_feature_dim))
        self.delta_feature_dim = max(0, int(delta_feature_dim))
        self.state_embed = nn.Sequential(
            nn.Linear(input_dim, embedding_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_hidden_dim, d_model),
            nn.Dropout(dropout),
        )
        self.time_embed: nn.Module | None = None
        if self.time_feature_dim > 0:
            time_hidden_dim = max(32, embedding_hidden_dim // 2)
            self.time_embed = nn.Sequential(
                nn.Linear(self.time_feature_dim, time_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(time_hidden_dim, d_model),
                nn.Dropout(dropout),
            )
        self.delta_network: nn.Module | None = None
        if self.social_feature_dim > 0 and (self.time_feature_dim > 0 or self.delta_feature_dim > 0):
            self.delta_network = _GatedDeltaNetwork(
                social_dim=self.social_feature_dim,
                d_model=d_model,
                delta_feature_dim=self.delta_feature_dim,
                dropout=dropout,
            )
        self.position_embedding = nn.Parameter(torch.zeros(max_seq_len, d_model))
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        self.context_blocks = nn.ModuleList(
            [_GPTBlock(d_model=d_model, num_heads=num_heads, dropout=dropout) for _ in range(context_num_layers)]
        )
        self.blocks = nn.ModuleList([_GPTBlock(d_model=d_model, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.er_head = nn.Linear(d_model, 1)
        self.dd_head = nn.Linear(d_model, 1)
        self.p_up_head = nn.Linear(d_model, 1)
        self.uncertainty_head = nn.Linear(d_model, 1)
        self.flow_head = nn.Linear(d_model, 1)
        self.regime_head = nn.Linear(d_model, len(REGIME_LABELS))

    def _causal_mask(self, seq_len: int, device: Any) -> Any:
        return torch.ones((seq_len, seq_len), dtype=torch.bool, device=device).triu(1)

    def _normalize_sequence_input(self, seq: Any | None, *, expected_seq_len: int) -> Any | None:
        if seq is None:
            return None
        if seq.ndim == 2:
            seq = seq.unsqueeze(0)
        if seq.size(1) > expected_seq_len:
            seq = seq[:, -expected_seq_len:, :]
        return seq

    def forward(
        self,
        state_sequence: Any,
        padding_mask: Any | None = None,
        *,
        time_features: Any | None = None,
        social_sequence: Any | None = None,
        delta_features: Any | None = None,
    ) -> dict[str, Any]:
        if state_sequence.ndim == 2:
            state_sequence = state_sequence.unsqueeze(0)
        if padding_mask is not None and padding_mask.ndim == 1:
            padding_mask = padding_mask.unsqueeze(0)
        if state_sequence.size(1) > self.max_seq_len:
            state_sequence = state_sequence[:, -self.max_seq_len :, :]
            if padding_mask is not None:
                padding_mask = padding_mask[:, -self.max_seq_len :]
        seq_len = int(state_sequence.size(1))
        time_features = self._normalize_sequence_input(time_features, expected_seq_len=seq_len)
        social_sequence = self._normalize_sequence_input(social_sequence, expected_seq_len=seq_len)
        delta_features = self._normalize_sequence_input(delta_features, expected_seq_len=seq_len)
        x = self.state_embed(state_sequence)
        if padding_mask is not None:
            padding_mask = padding_mask.to(x.device)
        if self.time_embed is not None:
            if time_features is None:
                time_features = torch.zeros(
                    (x.size(0), seq_len, self.time_feature_dim),
                    dtype=x.dtype,
                    device=x.device,
                )
            else:
                time_features = time_features.to(x.device)
            time_context = self.time_embed(time_features)
            x = x + time_context
        else:
            time_context = torch.zeros_like(x)
        if self.delta_network is not None:
            if social_sequence is None:
                social_sequence = torch.zeros(
                    (x.size(0), seq_len, self.social_feature_dim),
                    dtype=x.dtype,
                    device=x.device,
                )
            else:
                social_sequence = social_sequence.to(x.device)
            if self.delta_feature_dim > 0:
                if delta_features is None:
                    delta_features = torch.zeros(
                        (x.size(0), seq_len, self.delta_feature_dim),
                        dtype=x.dtype,
                        device=x.device,
                    )
                else:
                    delta_features = delta_features.to(x.device)
            x = x + self.delta_network(
                social_sequence,
                time_context,
                delta_features=delta_features,
                padding_mask=padding_mask,
            )
        x = x + self.position_embedding[:seq_len].unsqueeze(0).to(x.device)
        attn_mask = self._causal_mask(seq_len, x.device)
        for block in self.context_blocks:
            x = block(x, None, padding_mask)
        for block in self.blocks:
            x = block(x, attn_mask, padding_mask)
        final = self.ln_f(x[:, -1, :])
        return {
            "er_20d": self.er_head(final).squeeze(-1),
            "dd_20d": torch.sigmoid(self.dd_head(final).squeeze(-1)),
            "p_up_20d": torch.sigmoid(self.p_up_head(final).squeeze(-1)),
            "uncertainty": torch.sigmoid(self.uncertainty_head(final).squeeze(-1)),
            "flow_persist": torch.sigmoid(self.flow_head(final).squeeze(-1)),
            "regime_logits": self.regime_head(final),
        }


class NumericFirstPredictor:
    def __init__(
        self,
        artifact: ModelArtifact | None = None,
        model_blob: dict[str, Any] | None = None,
        artifact_path: str | Path | None = None,
        p_up_calibrator: ProbabilityCalibrator | None = None,
        dd_adjuster: QuantileAdjuster | None = None,
    ) -> None:
        self._artifact_path = Path(artifact_path) if artifact_path is not None else None
        self._artifact_base_dir = self._artifact_path.parent if self._artifact_path is not None else None
        self.artifact = artifact or ModelArtifact(model_version="numeric_baseline_v1")
        self.model_blob = model_blob or self._load_model_blob(self._artifact_path)
        if self.model_blob.get("model_version"):
            self.artifact = ModelArtifact(
                model_version=str(self.model_blob.get("model_version")),
                schema_version=str(self.model_blob.get("schema_version", "v1")),
            )
        self.p_up_calibrator = p_up_calibrator or ProbabilityCalibrator()
        self.dd_adjuster = dd_adjuster or QuantileAdjuster()
        self._temporal_model: TemporalStateAttentionModel | None = None

    def _load_model_blob(self, artifact_path: Path | None) -> dict[str, Any]:
        if artifact_path is None:
            return {}
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("artifact payload must be a JSON object")
        return payload

    def _model_type(self) -> str:
        return str(self.model_blob.get("model_type", "linear_head"))

    def _temporal_sequence_key(self) -> str:
        return str(self.model_blob.get("sequence_key", DEFAULT_TEMPORAL_SEQUENCE_KEY))

    def _temporal_numeric_feature_keys(self) -> list[str]:
        keys = self.model_blob.get("numeric_feature_keys") or self.model_blob.get("feature_keys")
        if isinstance(keys, list) and keys:
            return [str(key) for key in keys]
        return list(DEFAULT_TEMPORAL_NUMERIC_FEATURE_KEYS)

    def _temporal_vector_feature_dims(self) -> dict[str, int]:
        return _resolve_vector_feature_dims(self.model_blob.get("vector_feature_dims"))

    def _temporal_max_seq_len(self) -> int:
        return int(self.model_blob.get("max_seq_len", 32))

    def _temporal_time_feature_dim(self) -> int:
        return max(0, int(self.model_blob.get("time_feature_dim", 0)))

    def _temporal_delta_feature_dim(self) -> int:
        return max(0, int(self.model_blob.get("delta_feature_dim", 0)))

    def _resolve_sidecar_path(self, raw_path: object) -> Path:
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError("temporal transformer artifact requires weights_path")
        candidate = Path(raw_path)
        if candidate.is_absolute():
            return candidate
        if self._artifact_base_dir is not None:
            return self._artifact_base_dir / candidate
        return candidate

    def _validate_temporal_schema(self, features: dict) -> None:
        required = self._temporal_numeric_feature_keys()
        sequence_key = self._temporal_sequence_key()
        sequence = features.get(sequence_key)
        if isinstance(sequence, list) and sequence:
            for index, state in enumerate(sequence):
                if not isinstance(state, dict):
                    raise ValueError(f"state_sequence[{index}] must be a dict")
                numeric = state.get("numeric_features")
                numeric_source = numeric if isinstance(numeric, dict) else state
                missing = [key for key in required if key not in numeric_source]
                if missing:
                    raise ValueError(f"missing required temporal features in step {index}: {missing}")
            return

        missing = [key for key in required if key not in features]
        if missing:
            raise ValueError(f"missing required temporal features: {missing}")

    def validate_schema(self, features: dict) -> None:
        if self._model_type() == "temporal_transformer_v1":
            self._validate_temporal_schema(features)
            return
        required = set(self.model_blob.get("feature_keys", ["flow_strength", "trend_120m", "extension_60m"]))
        missing = [key for key in required if key not in features]
        if missing:
            raise ValueError(f"missing required features: {missing}")

    def _predict_linear_head(self, head_blob: dict[str, Any], features: dict) -> float:
        weights = head_blob.get("weights", {})
        bias = float(head_blob.get("bias", 0.0))
        return bias + sum(float(weights.get(key, 0.0)) * float(features.get(key, 0.0)) for key in weights)

    def _predict_tree_ensemble_head(self, head_blob: dict[str, Any], features: dict) -> float:
        score = float(head_blob.get("base_score", 0.0))
        trees = head_blob.get("trees", [])
        if not isinstance(trees, list):
            raise ValueError("trees must be a list")
        for tree in trees:
            feature = str(tree.get("feature", ""))
            threshold = float(tree.get("threshold", 0.0))
            left = float(tree.get("left", 0.0))
            right = float(tree.get("right", 0.0))
            value = float(features.get(feature, 0.0))
            score += left if value <= threshold else right
        return score

    def _predict_head(self, head_blob: dict[str, Any], features: dict) -> float:
        if self._model_type() == "tree_ensemble_v1":
            return self._predict_tree_ensemble_head(head_blob, features)
        return self._predict_linear_head(head_blob, features)

    def _load_temporal_model(self) -> TemporalStateAttentionModel:
        if self._temporal_model is not None:
            return self._temporal_model
        if nn is None or torch is None:
            raise RuntimeError("torch is required for temporal transformer predictors")

        numeric_feature_keys = self._temporal_numeric_feature_keys()
        vector_feature_dims = self._temporal_vector_feature_dims()
        input_dim = len(numeric_feature_keys) + sum(vector_feature_dims.values())
        model = TemporalStateAttentionModel(
            input_dim=input_dim,
            embedding_hidden_dim=int(self.model_blob.get("embedding_hidden_dim", 128)),
            d_model=int(self.model_blob.get("d_model", 128)),
            context_num_layers=int(self.model_blob.get("context_num_layers", 0)),
            num_heads=int(self.model_blob.get("num_heads", 4)),
            num_layers=int(self.model_blob.get("num_layers", 2)),
            dropout=float(self.model_blob.get("dropout", 0.1)),
            max_seq_len=self._temporal_max_seq_len(),
            time_feature_dim=self._temporal_time_feature_dim(),
            social_feature_dim=int(vector_feature_dims.get("z_social", 0)),
            delta_feature_dim=self._temporal_delta_feature_dim(),
        )
        with self._resolve_sidecar_path(self.model_blob.get("weights_path")).open("rb") as fh:
            state_dict = torch.load(fh, map_location="cpu")
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        model.load_state_dict(state_dict)
        model.eval()
        self._temporal_model = model
        return model

    def _predict_temporal_transformer(
        self,
        symbol: str,
        session_type: str,
        as_of_time: datetime,
        features: dict,
    ) -> FusedPrediction:
        model = self._load_temporal_model()
        model_inputs = build_temporal_model_inputs(
            features,
            numeric_feature_keys=self._temporal_numeric_feature_keys(),
            vector_feature_dims=self._temporal_vector_feature_dims(),
            as_of_time=as_of_time,
            sequence_key=self._temporal_sequence_key(),
            max_seq_len=self._temporal_max_seq_len(),
            include_time_features=self._temporal_time_feature_dim() > 0,
            include_delta_features=self._temporal_delta_feature_dim() > 0,
        )
        tensor = torch.tensor(model_inputs["state_sequence"], dtype=torch.float32).unsqueeze(0)
        time_tensor = None
        if self._temporal_time_feature_dim() > 0 and model_inputs["time_sequence"]:
            time_tensor = torch.tensor(model_inputs["time_sequence"], dtype=torch.float32).unsqueeze(0)
        social_tensor = None
        social_dim = int(self._temporal_vector_feature_dims().get("z_social", 0))
        if social_dim > 0 and model_inputs["social_sequence"]:
            social_tensor = torch.tensor(model_inputs["social_sequence"], dtype=torch.float32).unsqueeze(0)
        delta_tensor = None
        if self._temporal_delta_feature_dim() > 0 and model_inputs["delta_sequence"]:
            delta_tensor = torch.tensor(model_inputs["delta_sequence"], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            outputs = model(
                tensor,
                time_features=time_tensor,
                social_sequence=social_tensor,
                delta_features=delta_tensor,
            )

        er20 = float(outputs["er_20d"][0].detach().cpu())
        dd20 = self.dd_adjuster.transform(float(outputs["dd_20d"][0].detach().cpu()))
        pup = self.p_up_calibrator.transform(float(outputs["p_up_20d"][0].detach().cpu()))
        uncertainty = _clamp(float(outputs["uncertainty"][0].detach().cpu()), 0.0, 1.0)
        flow_persist = _clamp(float(outputs["flow_persist"][0].detach().cpu()), 0.0, 1.0)
        regime_logits = outputs["regime_logits"][0].detach().cpu().tolist()
        regime_index = max(range(len(regime_logits)), key=lambda index: regime_logits[index]) if regime_logits else 2
        regime = REGIME_LABELS[regime_index]
        if features.get("market_risk_off"):
            regime = "risk_off"
        event_score = infer_temporal_event_score(features, sequence_key=self._temporal_sequence_key())

        return FusedPrediction(
            symbol=symbol,
            as_of_time=as_of_time,
            session_type=session_type,
            er_5d=er20 * 0.45,
            er_20d=er20,
            dd_20d=dd20,
            p_up_20d=_clamp(pup, 0.0, 1.0),
            flow_persist=flow_persist,
            uncertainty=uncertainty,
            regime_final=regime,
            event_score=event_score,
            semantic_branch_enabled=bool(features.get("semantic_branch_enabled", True)),
            text_branch_enabled=bool(features.get("text_branch_enabled", True)),
            model_version=self.artifact.model_version,
            calibrator_version=str(self.model_blob.get("calibrator_version", "affine_v1")),
            missing_flags=features.get("missing_flags"),
            stale_flags=features.get("stale_flags"),
        )

    def predict(self, symbol: str, session_type: str, as_of_time: datetime, features: dict) -> FusedPrediction:
        self.validate_schema(features)
        if self._model_type() == "temporal_transformer_v1":
            return self._predict_temporal_transformer(symbol, session_type, as_of_time, features)

        flow_strength = float(features.get("flow_strength", 0.0))
        trend_120m = float(features.get("trend_120m", 0.0))
        extension = float(features.get("extension_60m", 0.0))
        event_score = float(features.get("event_score", 0.0))
        uncertainty = min(
            1.0,
            max(
                0.0,
                float(self.model_blob.get("uncertainty_base", 0.35))
                + float(self.model_blob.get("uncertainty_extension_mult", 0.25)) * abs(extension),
            ),
        )
        heads = self.model_blob.get("heads", {})
        if heads:
            er20 = self._predict_head(heads.get("er_20d", {}), features)
            dd20 = self._predict_head(heads.get("dd_20d", {}), features)
            pup_raw = self._predict_head(heads.get("p_up_20d", {}), features)
            pup = min(0.95, max(0.05, pup_raw))
        elif self.model_blob.get("weights"):
            er20 = self._predict_linear_head(self.model_blob, features)
            dd20 = max(0.01, 0.05 - 0.02 * trend_120m + 0.015 * max(extension, 0))
            pup = min(0.95, max(0.05, 0.5 + er20 * 2.0))
        else:
            er20 = 0.01 + 0.02 * flow_strength + 0.015 * trend_120m + 0.01 * event_score - 0.01 * max(extension, 0)
            dd20 = max(0.01, 0.05 - 0.02 * trend_120m + 0.015 * max(extension, 0))
            pup = min(0.95, max(0.05, 0.5 + er20 * 2.0))
        er5 = er20 * 0.45
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
            semantic_branch_enabled=bool(features.get("semantic_branch_enabled", True)),
            text_branch_enabled=bool(features.get("text_branch_enabled", True)),
            model_version=self.artifact.model_version,
            calibrator_version="affine_v1",
            missing_flags=features.get("missing_flags"),
            stale_flags=features.get("stale_flags"),
        )
