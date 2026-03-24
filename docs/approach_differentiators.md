# 기존 기법과 차별점 정리

## 문서 목적

이 문서는 국내 주식 스윙 트레이딩 시스템에서 흔히 보이는 접근과, 현재 `stock_ai` 코드베이스가 취하는 접근을 비교해 정리한 메모다.
기준은 "이미 코드와 문서로 확인되는 내용"이며, 아직 구현되지 않은 로드맵 항목은 차별점으로 과장하지 않는다.

여기서 말하는 "기존 기법"은 대체로 다음 범주를 뜻한다.

- 일봉 또는 정규장 중심의 단순 지표형 스윙 전략
- 뉴스/공시를 사람이 읽어 반영하거나, LLM에 직접 매수/매도 판단을 맡기는 방식
- 백테스트와 실거래 실행 규칙이 분리된 연구용 파이프라인
- 장애 대응, 감사 로그, degraded mode가 약한 모델 중심 시스템

## 한눈에 보는 비교

| 항목 | 일반적인 접근 | 이 저장소의 접근 | 차별점 |
| --- | --- | --- | --- |
| 시그널 구조 | 기술적 지표 단독 또는 뉴스 감성 점수 보조 | `NumericFirstPredictor` 중심 + 이벤트/텍스트는 보조 branch | LLM이나 텍스트를 주 엔진이 아니라 보조 정보로 제한 |
| LLM 사용 방식 | 자유 서술 응답이나 요약 결과를 사람이 해석 | `LLMEventNormalizer`가 strict JSON schema로 정규화 | 의미 해석을 구조화하고 실패 시 중립값으로 강등 |
| 시장 구조 가정 | 정규장 단일 세션, 단일 거래소 가정 | KRX/NXT와 `NXT_PRE`, `CORE_DAY`, `CLOSE_PRICE`, `NXT_AFTER`를 구분 | 국내 시장 미시구조를 세션 단위로 반영 |
| 체결 가정 | 시그널 시점 종가 체결, same-bar fill, 고정 슬리피지 | 다음 tradable 5분 구간 기준 실행, 세션 종료 임박 시 rollover | 백테스트와 실거래 괴리를 줄이는 쪽으로 설계 |
| 연구-운영 정합성 | 라벨 생성, 백테스트, 라이브 로직이 분리 | `ExecutionMapper`를 live/backtest/label에서 공용 사용 | 실행 규칙이 한 곳에서 정의됨 |
| leakage 통제 | `published_at` 기준 정렬 또는 사후 보정 | `as_of_time` 기준, provisional/confirmed flow 분리 | 데이터 정합성을 시스템 레벨에서 강제 |
| 운영 안정성 | 모델만 뜨면 실행, 장애는 수동 대응 | `ProductionReadinessGate`, kill switch, degraded flags | 실행 가능 여부를 먼저 판정하고 fail-closed 지향 |
| 감사 가능성 | 예측값 또는 주문 결과만 저장 | 모델/프롬프트/벡터화 버전, source doc, rationale까지 기록 | 사후 분석과 재현성 확보에 유리 |

## 핵심 차별점

### 1. LLM을 "트레이더"가 아니라 "정규화 계층"으로 사용

일반적인 LLM 결합 전략은 뉴스 요약이나 감성 분석을 그대로 매매 판단에 연결하는 경우가 많다.
이 저장소는 반대로 숫자 모델을 중심에 두고, LLM은 이벤트 정규화와 보조 semantic factor에만 제한한다.

- `LiveInferenceService`는 `LLMEventNormalizer`와 `VectorizationPipeline`을 호출한 뒤에도 최종 예측은 `NumericFirstPredictor`를 거친다.
- `llm_event_normalizer.py`는 strict JSON schema를 강제하고, 실패하면 `NEUTRAL_EVENT`로 내려간다.
- 이 구조는 "LLM이 장애를 내면 전체 트레이딩도 같이 망가지는" 결합을 피하려는 설계다.

관련 코드:

- [`../src/kswing_sentinel/live.py`](../src/kswing_sentinel/live.py)
- [`../src/kswing_sentinel/llm_event_normalizer.py`](../src/kswing_sentinel/llm_event_normalizer.py)
- [`../src/kswing_sentinel/predictor.py`](../src/kswing_sentinel/predictor.py)

### 2. KRX/NXT를 별도 세션과 별도 venue로 다룬다

기존 스윙 시스템은 "한국 시장 = 정규장"처럼 단순화하는 경우가 많다.
이 저장소는 KRX/NXT 이원 구조와 장전, 본장, 종가 단일가, 시간외를 구분해 실행 정책을 바꾼다.

- `ExecutionMapper`는 세션별 종료 시각, broker cutoff, venue freshness, NXT 가능 여부를 함께 본다.
- 장 종료 임박이면 다음 세션으로 넘기고, off-core 구간에서 NXT가 불가능하면 다음 KRX 본장으로 롤오버한다.
- 비용 모델도 venue와 session에 따라 달라진다.

관련 코드:

- [`../src/kswing_sentinel/execution_mapper.py`](../src/kswing_sentinel/execution_mapper.py)
- [`../src/kswing_sentinel/cost_model.py`](../src/kswing_sentinel/cost_model.py)
- [`./k_swing_sentinel_v1_2.md`](./k_swing_sentinel_v1_2.md)

### 3. 라벨, 백테스트, 라이브가 같은 실행 규칙을 공유한다

연구 단계에서는 성과가 좋아도 실거래에서 무너지는 시스템의 흔한 원인 중 하나는, 라벨 생성과 실제 체결 로직이 서로 다르기 때문이다.
이 저장소는 실행 규칙을 `ExecutionMapper` 한 군데에 모으고, 라벨 생성과 백테스트가 이를 재사용한다.

- `LabelBuilder`는 진입 시점과 세션, 비용 계산에 `ExecutionMapper`를 사용한다.
- `Backtester`도 동일한 mapper를 통해 entry 시점과 비용을 계산한다.
- 결과적으로 "학습 데이터의 체결 규칙"과 "실거래 체결 규칙"이 크게 벌어지는 것을 줄이는 방향이다.

관련 코드:

- [`../src/kswing_sentinel/label_builder.py`](../src/kswing_sentinel/label_builder.py)
- [`../src/kswing_sentinel/backtester.py`](../src/kswing_sentinel/backtester.py)
- [`../src/kswing_sentinel/execution_mapper.py`](../src/kswing_sentinel/execution_mapper.py)

### 4. leakage 방지를 명시적인 계약으로 다룬다

일반적인 데이터 파이프라인은 "대충 시계열 순서를 지킨다" 수준에서 끝나는 경우가 많다.
이 저장소는 `as_of_time`과 provisional/confirmed 구분을 통해 leakage를 명시적으로 막으려 한다.

- 아키텍처 문서는 dataset, feature, document, vector, decision 모두에 `as_of_time`을 들고 가는 것을 기본 계약으로 둔다.
- `FlowSnapshotStore`는 intraday anchor에서 confirmed flow가 섞이면 `LeakageError`를 발생시킨다.
- `Backtester`는 `timestamp > as_of_time`인 row를 금지한다.

관련 코드:

- [`../src/kswing_sentinel/flow_snapshot_store.py`](../src/kswing_sentinel/flow_snapshot_store.py)
- [`../src/kswing_sentinel/backtester.py`](../src/kswing_sentinel/backtester.py)
- [`./k_swing_sentinel_v1_2.md`](./k_swing_sentinel_v1_2.md)

### 5. 장애 시 "그냥 계속 돌리는" 대신 fail-closed와 degraded mode를 택한다

많은 시스템이 데이터 피드나 LLM, broker 상태가 애매할 때도 억지로 실행을 이어간다.
이 저장소는 오히려 막아야 할 때 막고, 살릴 수 있을 때만 축소 운용하는 쪽에 가깝다.

- `ProductionReadinessGate`는 env, broker, feed, artifact, audit sink 상태를 먼저 검사한다.
- NXT 조건이 불충분하면 전체 중단 대신 `KRX_ONLY`로 강등할 수 있다.
- semantic provider나 vectorizer가 죽으면 경고와 degraded flag를 남기고 numeric-only에 가깝게 후퇴한다.

관련 코드:

- [`../src/kswing_sentinel/production_runtime.py`](../src/kswing_sentinel/production_runtime.py)
- [`../src/kswing_sentinel/llm_event_normalizer.py`](../src/kswing_sentinel/llm_event_normalizer.py)

### 6. 예측값보다 decision lineage를 더 중요하게 남긴다

보통은 주문 결과만 남기고 "왜 그 결정을 했는지"는 로그에 충분히 남지 않는다.
이 저장소는 최소한의 decision lineage를 남기는 쪽으로 설계돼 있다.

- `AuditLogStore`는 model version, prompt version, vectorizer version, source doc id, cluster id, rationale code를 기록한다.
- `VectorizationPipeline`은 encoder/tokenizer/backend와 `as_of_time`, `session_type` 메타데이터를 함께 남긴다.
- 이런 형태는 사후 디버깅, 리서치 재현, 품질 회고에 유리하다.

관련 코드:

- [`../src/kswing_sentinel/audit_log.py`](../src/kswing_sentinel/audit_log.py)
- [`../src/kswing_sentinel/vectorization.py`](../src/kswing_sentinel/vectorization.py)

## 현재 기준에서 "차별점"이라고만 말하기 어려운 부분

아래 항목은 방향성은 맞지만, 아직 완성된 우위라고 부르기에는 이르다.

- 텍스트 벡터는 생성되고 검증되지만, 기본 predictor 입력에 완전히 융합되어 있지는 않다.
- `NumericFirstPredictor`는 baseline scaffold에 가깝고, production-grade 모델 스택으로 보긴 어렵다.
- 백테스터는 session-aware cost와 execution parity를 갖지만, 완전한 event-driven portfolio simulator는 아니다.
- 실제 broker/live feed 연동은 추상화와 테스트가 중심이며, 검증된 운영 연결이 내장된 상태는 아니다.

## 한 줄 정리

이 저장소의 핵심 차별점은 "새로운 알파 공식"보다 "국내 시장 세션 구조, 실행 정합성, leakage 통제, 장애 대응, 감사 가능성"을 한 시스템 안에서 같이 다루려는 점에 있다.
