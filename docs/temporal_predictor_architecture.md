# Temporal Predictor Architecture

## 목적

이 문서는 현재 저장소의 텍스트 벡터화 경로와 temporal predictor 경로를 실제 구현 기준으로 정리한다.
특히 "각 시점 상태를 벡터로 인코딩하고 attention으로 예측한다"는 현재 구조를 문서화하는 데 초점을 둔다.

## 현재 live 추론 흐름

기본 live 추론은 아래 순서로 동작한다.

1. raw event payload에서 `headline`, `body`, `text`, `summary`, `canonical_summary`를 합쳐 텍스트를 만든다.
2. 텍스트를 `VectorizationPipeline`에 넣어 `z_event`, `z_social`, `z_macro`를 만든다.
3. predictor가 temporal artifact를 로드한 경우, `state_sequence`를 읽어 시점별 상태 행렬로 바꾼다.
4. 각 step의 numeric 피처와 vector 피처를 합쳐 timestep embedding을 만든다.
5. self-attention context block과 causal attention block을 거쳐 마지막 step representation으로 여러 target을 동시에 예측한다.

관련 코드:

- [`../src/kswing_sentinel/live.py`](../src/kswing_sentinel/live.py)
- [`../src/kswing_sentinel/vectorization.py`](../src/kswing_sentinel/vectorization.py)
- [`../src/kswing_sentinel/predictor.py`](../src/kswing_sentinel/predictor.py)

## 텍스트 벡터화 구조

현재 텍스트 경로는 `LLM -> JSON -> summary embedding`이 아니라 `raw text -> direct vectorization`이다.

### 세부 단계

1. 문장 분리
2. 각 문장을 RoBERTa encoder로 인코딩
3. sentence vectors를 hierarchical transformer aggregator로 집계
4. 최종적으로 아래 세 개의 벡터를 생성

- `z_event`
- `z_social`
- `z_macro`

기본 차원은 다음과 같다.

- `z_event`: 64
- `z_social`: 32
- `z_macro`: 16

## predictor 입력 형태

temporal predictor가 가장 잘 쓰는 입력은 `state_sequence`다.

예시:

```json
{
  "state_sequence": [
    {
      "numeric_features": {
        "close_return_1d": 0.01,
        "volume_z": 0.42,
        "flow_strength": 0.31,
        "trend_120m": 0.14,
        "extension_60m": 0.08,
        "event_score": 0.20
      },
      "vector_payload": {
        "z_event": [0.1, 0.2],
        "z_social": [0.3],
        "z_macro": [0.4]
      }
    }
  ]
}
```

### step 구성 원칙

- numeric 피처는 `numeric_features` 아래에 넣는 것이 가장 명확하다.
- vector는 `vector_payload` 또는 `vectors`로 넣을 수 있다.
- 최신 live 경로는 raw text에서 만든 `vector_payload`를 마지막 step에 자동 주입한다.
- `state_sequence`가 없으면 flat feature dict를 단일 step sequence처럼 사용한다.

## timestep embedding

각 step은 아래 순서로 embedding 된다.

1. numeric feature key 순서대로 값을 읽는다.
2. `z_event`, `z_social`, `z_macro`를 펼쳐 뒤에 붙인다.
3. 연결된 벡터를 2-layer MLP에 통과시킨다.

구조:

```text
concatenated_step_vector
-> Linear(input_dim, embedding_hidden_dim)
-> GELU
-> Dropout
-> Linear(embedding_hidden_dim, d_model)
-> Dropout
```

이 단계는 "각 시점의 숫자 상태와 텍스트 상태를 같은 latent space로 묶는 역할"을 한다.

## attention 구조

현재 temporal predictor의 attention은 두 단계다.

### 1. self-attention context block

- causal mask 없이 시계열 전체를 본다
- 각 시점이 다른 과거/현재 시점과 어떻게 연결되는지 먼저 contextualize 한다

### 2. causal attention block

- upper-triangular causal mask를 사용한다
- 마지막 step representation을 만들 때 미래를 보지 않게 제한한다

즉 구조는 다음과 같다.

```text
timestep embeddings
-> self-attention context blocks
-> causal attention blocks
-> final step representation
-> task heads
```

## 출력 head

현재 head는 아래 target을 함께 예측한다.

- `er_20d`
- `dd_20d`
- `p_up_20d`
- `uncertainty`
- `flow_persist`
- `regime_logits`

최종 `FusedPrediction`으로 변환될 때는:

- `er_5d = er_20d * 0.45`
- `regime_logits`의 argmax로 `regime_final` 결정
- `p_up_20d`, `dd_20d`는 calibrator를 거친다

## 학습 artifact

`TrainingPipeline.train_temporal_transformer()`는 temporal predictor용 artifact와 weights를 저장한다.

artifact에는 보통 아래 정보가 들어간다.

- `model_type`
- `model_version`
- `schema_version`
- `sequence_key`
- `numeric_feature_keys`
- `vector_feature_dims`
- `max_seq_len`
- `embedding_hidden_dim`
- `d_model`
- `context_num_layers`
- `num_heads`
- `num_layers`
- `dropout`
- `weights_path`

이 artifact를 `NumericFirstPredictor(artifact_path=...)`에 넘기면 temporal predictor가 로드된다.

관련 코드:

- [`../src/kswing_sentinel/training.py`](../src/kswing_sentinel/training.py)
- [`../src/kswing_sentinel/predictor.py`](../src/kswing_sentinel/predictor.py)

## 현재 제한 사항

- 저장소 안에서 historical `state_sequence`를 자동 생성해 주는 온라인 feature store는 아직 완성돼 있지 않다.
- 기본 live 경로는 latest text vector를 마지막 step에 붙여 주지만, 과거 step의 vector history는 호출자가 준비해야 한다.
- temporal predictor는 baseline artifact 경로까지는 연결됐지만, 대규모 학습 운영과 서빙 최적화는 아직 포함하지 않는다.

## 요약

현재 구조를 한 줄로 줄이면 다음과 같다.

`raw text -> sentence-level RoBERTa -> hierarchical transformer -> vector payload -> timestep embedding MLP -> self-attention context -> causal attention -> multi-head stock prediction`
