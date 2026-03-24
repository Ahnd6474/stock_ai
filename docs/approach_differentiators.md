# 기존 기법과 현재 접근의 차이

## 문서 목적

이 문서는 국내 스윙 트레이딩 시스템에서 흔히 보이는 접근과, 현재 `stock_ai` 코드베이스가 실제로 구현하고 있는 접근을 비교해 정리한 메모다.
과장된 로드맵이 아니라, 지금 저장소에서 확인 가능한 코드 경로를 기준으로 쓴다.

## 한눈에 보는 차이

| 항목 | 일반적인 접근 | 현재 저장소의 접근 | 차이점 |
| --- | --- | --- | --- |
| 텍스트 처리 | 뉴스 요약을 사람이 해석하거나 간단한 감성 점수로 축약 | raw text를 바로 벡터화 | 기본 live path에서 LLM JSON 정규화를 거치지 않는다 |
| 텍스트 인코딩 | 문서 전체를 한 번에 임베딩하거나 규칙 기반으로 자름 | 문장 단위 RoBERTa 후 계층적 트랜스포머 집계 | 긴 문서를 문장 수준 정보로 분해한 뒤 다시 모은다 |
| 예측기 입력 | 단일 시점 숫자 피처 위주 | 시점별 상태 벡터 시퀀스 | 숫자 피처와 텍스트 벡터를 같은 timestep embedding으로 묶는다 |
| 시계열 모델 | 선형/트리/단순 RNN | self-attention context 후 causal attention | 과거 문맥을 먼저 압축한 뒤, 예측 시점에서는 causal mask를 유지한다 |
| 실행 규칙 | 학습용 라벨, 백테스트, 라이브가 다르게 흘러가기 쉬움 | execution mapper를 공용 사용 | 라벨과 실행 규칙의 분리를 줄인다 |
| 장애 대응 | 일부 provider가 죽어도 묵시적으로 계속 진행 | readiness gate와 degraded flags | fail-closed와 보수적 fallback을 선호한다 |

## 핵심 차별점

### 1. 기본 live path는 LLM 정규화가 아니라 direct vectorization이다

현재 기본 live 경로에서는 `LLMEventNormalizer`를 거치지 않고, raw payload의 `headline`, `body`, `text`, `summary`를 합친 문자열을 바로 벡터화한다.

- `LiveInferenceService`는 기본적으로 `semantic_branch_enabled=False`로 두고 텍스트를 `VectorizationPipeline`에 직접 넣는다.
- optional LLM normalizer 코드는 저장소에 남아 있지만, 현재 기본 live 추론의 필수 단계는 아니다.
- 이 구조는 provider 장애나 schema 파싱 실패를 기본 의사결정 경로에서 떼어내는 쪽에 가깝다.

관련 코드:

- [`../src/kswing_sentinel/live.py`](../src/kswing_sentinel/live.py)
- [`../src/kswing_sentinel/llm_event_normalizer.py`](../src/kswing_sentinel/llm_event_normalizer.py)

### 2. 텍스트는 문장 단위 RoBERTa 후 계층적으로 다시 모은다

현재 벡터화 파이프라인은 문서를 한 덩어리로 처리하지 않는다.

- 텍스트를 문장 단위로 나눈다.
- 각 문장을 RoBERTa encoder로 개별 인코딩한다.
- sentence vectors를 hierarchical transformer aggregator로 다시 모아서 `z_event`, `z_social`, `z_macro`를 만든다.

즉, 구조는 `raw text -> sentence split -> RoBERTa -> hierarchical transformer aggregation`이다.

관련 코드:

- [`../src/kswing_sentinel/vectorization.py`](../src/kswing_sentinel/vectorization.py)
- [`../src/kswing_sentinel/text_encoder.py`](../src/kswing_sentinel/text_encoder.py)
- [`../src/kswing_sentinel/attention_aggregator.py`](../src/kswing_sentinel/attention_aggregator.py)

### 3. 예측기는 단일 시점 계산기보다 시계열 상태 인코더에 가깝다

기존의 흔한 구조는 `현재 시점 feature dict -> 선형/트리 모델`이다.
현재 저장소는 그 호환 경로를 유지하면서도, 새 artifact에서는 시계열 predictor를 받을 수 있게 바뀌었다.

- 입력은 `state_sequence` 기준이다.
- 각 step은 `numeric_features`와 optional `vector_payload`를 가진다.
- numeric 피처와 `z_event/z_social/z_macro`를 펼쳐 붙인 뒤, 2-layer MLP로 timestep embedding을 만든다.

즉 predictor는 이제 "flat feature scorer"만이 아니라 "상태 시퀀스 인코더" 역할을 한다.

관련 코드:

- [`../src/kswing_sentinel/predictor.py`](../src/kswing_sentinel/predictor.py)

### 4. attention도 바로 causal만 쓰지 않고 context를 먼저 만든다

현재 temporal predictor는 GPT 스타일 causal attention만 바로 쓰지 않는다.

- 먼저 self-attention context block으로 시계열의 상호작용을 한 번 contextualize 한다.
- 그 다음 causal attention block으로 예측 시점 이전 정보만 보게 한다.
- 마지막 step representation으로 `er_20d`, `dd_20d`, `p_up_20d`, `uncertainty`, `flow_persist`, `regime`을 예측한다.

이 구조는 "과거 문맥 압축"과 "미래 차단"을 분리하려는 선택이다.

관련 코드:

- [`../src/kswing_sentinel/predictor.py`](../src/kswing_sentinel/predictor.py)
- [`../src/kswing_sentinel/training.py`](../src/kswing_sentinel/training.py)

### 5. KRX/NXT와 session-aware execution은 여전히 핵심이다

모델 구조가 바뀌어도 이 저장소의 실전적인 차별점은 여전히 execution layer에 있다.

- `NXT_PRE`, `CORE_DAY`, `CLOSE_PRICE`, `NXT_AFTER`, `OFF_MARKET`를 구분한다.
- venue freshness, broker cutoff, NXT 가능 여부를 보고 execution plan을 만든다.
- 라벨 생성, 백테스트, 라이브 추론이 같은 execution mapper를 재사용한다.

텍스트/모델을 강화해도 실행 규칙을 느슨하게 두지 않는다는 점이 이 저장소의 방향이다.

관련 코드:

- [`../src/kswing_sentinel/execution_mapper.py`](../src/kswing_sentinel/execution_mapper.py)
- [`../src/kswing_sentinel/backtester.py`](../src/kswing_sentinel/backtester.py)
- [`../src/kswing_sentinel/label_builder.py`](../src/kswing_sentinel/label_builder.py)

### 6. 장애 시에는 공격적으로 축소하고, 결정 경로는 남긴다

현재 구현은 "가능하면 계속 돌리자"보다 "안전한 범위까지만 돌리자"에 더 가깝다.

- `ProductionReadinessGate`가 feed, broker, artifact, audit sink 상태를 먼저 평가한다.
- `ProductionOrchestrator`는 anchor batch에서 transient failure가 나면 backoff 재시도, circuit breaker, dead-letter queue, optional JSONL persistence로 복원 가능성을 남긴다.
- production runtime은 문서에 적힌 semantic refresh anchor(08:10, 09:35, 15:45, 20:05)와 event-burst payload를 기준으로 refresh 요청을 표시하고 audit event로 남긴다.
- text/vectorizer가 실패하면 branch를 끄고 보수적으로 후퇴한다.
- audit에는 model version, prompt version, vectorizer version, source doc id, cluster id, rationale code가 남는다.

이 방향은 모델 성능보다 운영 복구 가능성과 사후 추적성을 우선한다.

관련 코드:

- [`../src/kswing_sentinel/production_runtime.py`](../src/kswing_sentinel/production_runtime.py)
- [`../src/kswing_sentinel/audit_log.py`](../src/kswing_sentinel/audit_log.py)

## 아직 과장하면 안 되는 부분

아래는 방향성은 맞지만 아직 완성형이라고 부르기 어려운 부분이다.

- temporal predictor는 들어갔지만, 안정적인 historical `state_sequence` 생성과 저장은 외부에서 공급해야 한다.
- 현재 학습 파이프라인은 temporal transformer artifact를 만들 수 있지만, 대규모 데이터 엔지니어링과 운영용 serving stack은 포함하지 않는다.
- optional LLM normalizer는 저장소에 남아 있지만, 기본 live path의 중심은 아니다.
- backtester는 execution realism을 꽤 반영하지만, 포트폴리오 단위의 완전한 event-driven 엔진은 아니다.

## 요약

현재 저장소의 차별점은 "LLM을 더 많이 쓴다"가 아니다.
오히려 `raw text direct vectorization`, `sentence-level RoBERTa + hierarchical transformer`, `state-sequence temporal predictor`, `session-aware execution`, `fail-closed runtime`를 하나의 일관된 방향으로 묶고 있다는 점에 있다.
