# Roadmap: Scaffold → Production (GitHub Issue Drafts)

아래 항목은 현재 코드베이스의 scaffold를 실제 운영 가능한 시스템으로 전환하기 위한 우선순위 이슈 초안입니다.
각 섹션은 그대로 복붙하여 GitHub Issue로 사용할 수 있습니다.

---

## 0) Epic: Scaffold를 실제 모델·임베딩·백테스트 정합성으로 전환

**Title**
`[EPIC] Replace scaffold components with production-grade model, embedding, and backtesting consistency`

**Description**
현재 시스템은 인터페이스/계약/테스트 뼈대는 갖췄지만, 핵심 로직 일부가 pseudo/stub 수준입니다.
본 Epic은 아래 하위 이슈를 통해 다음을 달성합니다:

- 가짜/선형 stub 제거
- 학습-추론-백테스트 정합성 확보
- 운영 가능한 캘린더/세션/브로커 상태머신 구축
- 온라인/오프라인 패리티와 lineage 강화

**Definition of Done**
- 하위 이슈 1~15 모두 완료
- E2E dry-run + backtest 재현성 검증 통과
- 운영 모드에서 decision lineage 및 degraded mode 관측 가능

---

## 1) `vectorization.py` pseudo-vector 제거, 실제 임베딩 연결

**Title**
`[P0] Replace pseudo vectors in vectorization.py with real embedding model`

**Context**
현재 metadata는 있으나 벡터 생성은 SHA-256 기반 pseudo-vector.

**Scope**
- Korean BERT/sentence embedding 모델 선택 및 로딩 경로 추가
- 배치 인코딩 + 캐시 + 버전 메타데이터 저장
- 임베딩 차원/모델버전 불일치 시 hard fail

**Acceptance Criteria**
- 동일 입력에 대해 deterministic embedding 재현
- 모델/토크나이저 버전이 산출물 metadata에 기록
- pseudo-vector 코드 경로 제거 또는 feature flag로 완전 차단

---

## 2) `predictor.py` 선형 baseline → 학습 artifact 추론기

**Title**
`[P0] Replace linear baseline in predictor.py with trained model artifact inference`

**Context**
artifact/calibrator 연결은 있으나 본질은 JSON weight 합산 선형 계산기.

**Scope**
- LightGBM/CatBoost 등 실제 학습 artifact 로더 추가
- feature schema fingerprint 검증
- 추론 시 calibrator 포함한 output contract 고정

**Acceptance Criteria**
- 학습 산출 artifact로만 추론 가능(수동 weight 주입 금지)
- schema mismatch 시 명시적 오류
- inference latency/throughput 기준값 기록

---

## 3) `backtester.py` single-trade 평가기 → portfolio 시뮬레이터

**Title**
`[P0] Expand backtester.py from single-trade evaluator to portfolio simulator`

**Scope**
- 다종목/다포지션 portfolio simulation
- halt/suspension, partial fill, slippage path 반영
- turnover, drawdown path, exposure time-series 산출

**Acceptance Criteria**
- 일자별 equity curve 및 drawdown 시계열 산출
- 체결/미체결/부분체결 상태 재현 가능
- 기존 단일거래 테스트 + 신규 포트폴리오 테스트 통과

---

## 4) `execution_mapper.py` / `label_builder.py` / `backtester.py` 정합성 고정

**Title**
`[P0] Enforce single source of truth for execution, labels, and backtest fills`

**Scope**
- 체결 규칙을 공통 모듈로 추출
- label 생성과 backtest fill이 동일 규칙/파라미터 사용
- drift detection test 추가

**Acceptance Criteria**
- 세 모듈 간 규칙 해시 동일
- 한쪽 규칙 변경 시 다른 모듈 테스트 자동 실패

---

## 5) 공식 거래 캘린더/세션 관리 외부화

**Title**
`[P0] Externalize official market calendar and session definitions`

**Scope**
- 연도별 하드코딩 날짜 제거
- 공식 캘린더 소스/버전 파일 기반 로딩
- 업데이트 검증 스크립트 및 롤백 전략

**Acceptance Criteria**
- 캘린더 데이터가 코드와 분리되어 버전관리됨
- 신규 연도 반영이 코드 수정 없이 가능

---

## 6) `training.py` 실학습 파이프라인 완성

**Title**
`[P1] Complete training.py with walk-forward training pipeline`

**Scope**
- dataset join, walk-forward split, fold별 학습/검증
- artifact 저장, fold metric 저장, manifest 생성

**Acceptance Criteria**
- 단일 커맨드로 학습~artifact 생성 완료
- fold별 metric과 최종 선택 모델 기록

---

## 7) Calibration 평가 체계 강화

**Title**
`[P1] Strengthen calibration with measurable pre/post metrics`

**Scope**
- pre/post calibration metric 수집(ECE/Brier 등)
- uncertainty bucket 성능 비교
- drawdown 보정 영향 검증

**Acceptance Criteria**
- calibration 전후 성능 비교 리포트 자동 생성
- 악화 시 배포 차단 기준 존재

---

## 8) `venue_router.py` 정책+비용 통합 라우터 고도화

**Title**
`[P1] Upgrade venue_router.py to policy + cost integrated routing`

**Scope**
- session policy + cost + stale state + liquidity deterioration 통합
- fallback/override 정책 명시

**Acceptance Criteria**
- 라우팅 결정에 정책 근거 로그 남김
- stale/liquidity 악화 상황에서 보수적 경로 선택 검증

---

## 9) `broker_gateway.py` 주문 상태머신 강화

**Title**
`[P1] Implement production-grade order state machine in broker_gateway.py`

**Scope**
- NEW/PARTIAL/FILLED/CANCELED/REJECTED 전이 고정
- replace/reconcile, cutoff, TIF 정책 반영

**Acceptance Criteria**
- 상태전이 불가능 케이스 차단
- replay 가능한 주문 이벤트 로그 제공

---

## 10) `feature_store.py` online/offline parity 보장

**Title**
`[P1] Guarantee online/offline parity in feature_store.py`

**Scope**
- 동일 feature definition registry 사용
- point-in-time join 보장 및 leakage guard 강화
- parity test suite 추가

**Acceptance Criteria**
- 동일 시점 기준 online/offline feature 값 일치
- 누수 탐지 테스트 상시 통과

---

## 11) `event_store.py` lineage/dedup 완성

**Title**
`[P1] Complete lineage and dedup in event_store.py`

**Scope**
- canonical event, cluster, source lineage, novelty score 운영화
- dedup 기준 및 TTL 정책 명시

**Acceptance Criteria**
- 동일 이벤트 중복 입력 시 canonical merge 동작
- lineage trace로 출처 역추적 가능

---

## 12) `llm_event_normalizer.py` production path 분리

**Title**
`[P1] Split production path in llm_event_normalizer.py with robust fallback`

**Scope**
- provider adapter 계층화
- schema violation/fallback/low-confidence 로깅 강화
- low-confidence downweight 정책 적용

**Acceptance Criteria**
- provider 장애 시 deterministic fallback 동작
- 모든 비정상 케이스가 구조화 로그로 남음

---

## 13) `text_encoder.py` / `attention_aggregator.py` 실제화

**Title**
`[P1] Implement real text encoder and attention aggregator`

**Scope**
- 실제 인코더/집계 모델 연결
- masking, padding, batching, max-length 정책 명시

**Acceptance Criteria**
- 더미 encoder/aggregator 경로 제거
- 최소 성능 기준(오프라인 metric) 충족

---

## 14) `risk_engine.py` / `portfolio_engine.py` 상태기반 확장

**Title**
`[P1] Expand risk_engine.py and portfolio_engine.py with stateful constraints`

**Scope**
- beta/sector/turnover/liquidity cap
- staged entries 및 상태 기반 리밸런싱 반영

**Acceptance Criteria**
- 제약 위반 주문 자동 차단/축소
- 포지션 변화가 제약 로그와 함께 설명 가능

---

## 15) `live.py` / `orchestration.py` / `monitoring.py` 운영 통합

**Title**
`[P1] Integrate live orchestration and monitoring for production operations`

**Scope**
- anchor run, retry/backoff, degraded-mode logging
- decision lineage, slippage tracking, alert 연결

**Acceptance Criteria**
- 장애/지연 상황에서 재시도 정책 검증
- 실거래 의사결정에 대한 lineage 추적 가능

---

## Suggested Labels

- Priority: `P0`, `P1`
- Type: `model`, `infra`, `backtest`, `execution`, `risk`, `data-quality`
- Status: `roadmap`, `needs-design`, `ready-for-implementation`

## Suggested Milestones

1. **M1 (P0 Hardening)**: #1~#5
2. **M2 (Training + Execution)**: #6~#10
3. **M3 (Semantic + Ops)**: #11~#15
