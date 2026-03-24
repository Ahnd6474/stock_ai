# Stock AI Master Execution TODO (All Items Must Be Executed)

목표: 이 문서는 단순 아이디어 목록이 아니라, **실제로 전부 실행**하기 위한 마스터 백로그입니다.
원칙: 아래 20개 TODO는 **누락 없이 전부 완료**를 목표로 하며, 선후관계를 반드시 지킵니다.

---

## 2026-03-20 재평가 스냅샷 (Baseline 인정 + 실전 미달)

- 저장소 구조/모듈 분리: **7/10**
- 실행·세션·라우팅 골격: **5/10**
- 백테스트/라벨 정합성: **4/10**
- 예측 모델 실제성: **2/10**
- 텍스트/LLM/벡터 실질성: **2/10**
- 리스크/포트폴리오: **4/10**
- 운영/감사/모니터링: **3/10**
- 실거래 가능성: **1/10**

**총평:** "좋은 설계문서를 따라 만든 구현 스캐폴드 + 일부 toy implementation".

### 우선순위 고정 원칙 (순서 변경 금지)
1. `calendar.py`
2. `session_rules.py`
3. `execution_mapper.py`
4. `label_builder.py`
5. `backtester.py`
6. `cost_model.py`
7. `nxt_eligibility_store.py`
8. `venue_router.py`
9. `broker_gateway.py`
10. `training.py`
11. `predictor.py`
12. calibration
13. `feature_store.py`
14. `risk_engine.py`
15. `portfolio_engine.py`
16. `event_store.py`
17. `llm_event_normalizer.py`
18. `text_encoder.py`
19. `attention_aggregator.py`
20. `vectorization.py`
21. `live.py`
22. `orchestration.py`
23. `monitoring.py`

### 즉시 반영된 공통 요구사항
- 모든 항목 완료 조건에 테스트 증적을 강제한다.
- `execution_mapper` / `label_builder` / `backtester`는 동일한 실행 규칙 소스를 공유해야 한다.
- `predictor.py`는 하드코딩 baseline 제거를 목표로 artifact + calibrator 경로를 기본 지원한다.

### 최근 반영 진행 (2026-03-20)
- A1 `calendar.py`: 기본 휴장일 세트 + half-day 조기 종료 처리 + `is_tradable_minute()` 추가 (**IN_PROGRESS**)
- A2 `session_rules.py`: 캘린더 tradable minute 기반 경계 배타화 및 확장 경계 테스트 추가 (**IN_PROGRESS**)
- A3 `execution_mapper.py`: broker cutoff/신선도/유동성/브로커 미지원 별 rollover reason 세분화 (**IN_PROGRESS**)
- A4 `label_builder.py`: entry/exit side별 비용 반영 + corporate action/halt censoring hook 추가 (**IN_PROGRESS**)
- A5 `backtester.py`: same-bar 진입 금지 + partial-fill ratio + portfolio summary 경로 추가 (**IN_PROGRESS**)
- B1 `cost_model.py`: liquidity bucket + 비선형 participation impact + `cost_model_version=v2` 반영 (**IN_PROGRESS**)
- B2 `nxt_eligibility_store.py`: snapshot stale fail-closed resolve + broker routability 분리 저장 (**IN_PROGRESS**)
- B3 `venue_router.py`: stale snapshot 차단/정책 기반 rationale 강화 (**IN_PROGRESS**)
- B4 `broker_gateway.py`: replace/reconcile/TIF 검증을 포함한 상태 확장 (**IN_PROGRESS**)
- C1 `training.py`: walk-forward 학습 + fold metric/artifact 저장 파이프라인 추가 (**IN_PROGRESS**)
- C2 `predictor.py`: artifact feature schema 기반 검증 및 model_version 반영 (**IN_PROGRESS**)
- C3 `calibration.py`: calibrator fitting + pre/post(Brier) 비교 리포트 추가 (**IN_PROGRESS**)
- C1-EXT `training.py`: multi-head artifact(`er_20d`,`dd_20d`,`p_up_20d`) 생성 경로 추가 (**IN_PROGRESS**)
- C2-EXT `predictor.py`: multi-head artifact 추론 및 uncertainty 파라미터 적용 (**IN_PROGRESS**)
- C3-EXT `calibration.py`: DD calibration(MAE) + uncertainty bucket 요약 리포트 추가 (**IN_PROGRESS**)
- D1 `feature_store.py`: freshness/missingness/session flags + market context merge + online/offline parity 보강 (**DONE**)
- D2 시장/거시 컨텍스트: KOSPI/KOSDAQ/futures/USDKRW/breadth 기반 regime context 병합 경로 추가 (**DONE**)
- E1 `event_store.py`: cluster lineage + novelty score + canonical dedup/cluster 복원 경로 추가 (**DONE**)
- F1 `orchestration.py` / `production_runtime.py`: retry/backoff + circuit breaker + dead-letter queue + anchor-batch idempotent retry 경로 추가 (**IN_PROGRESS**)

## 2026-03-20 구현 감사 상태

- `pytest -q` 기준 전체 테스트: **61 passed**
- 코드/테스트 기준 사실상 완료 또는 강하게 반영된 영역
  - A1, A2, A3
  - B1, B2, B3, B4
  - C2, C3, C1-EXT, C2-EXT, C3-EXT
  - D1, D2, D3, D4
  - E1, E2, E3
  - F1 (batch retry/backoff/circuit breaker/DLQ + production runtime anchor re-entry)
- 아직 todo 원문 기준으로 추가 구현이 필요한 영역
  - A4: corporate action / halt / suspension censoring hook
  - A5: portfolio mode 관점의 백테스트 확장
  - C1: LightGBM baseline 및 artifact/metric 표준화
  - F1: semantic refresh 정책 고도화 + dead-letter 영속화 + 운영 복구 런북 정리

## 이번 변경 증적

- Test command: `python -m pytest -q tests/test_production_runtime.py tests/test_orchestration_resilience.py tests/test_label_builder_extensions.py tests/test_backtester_portfolio.py`
- Test result: `11 passed`
- Regression command: `python -m pytest -q`
- Regression result: `78 passed`
- Rollback strategy:
  - `src/kswing_sentinel/production_runtime.py`
  - `tests/test_production_runtime.py`
  - `README.md`
  - `docs/implementation_todo.md`
  - `docs/approach_differentiators.md`
  - `docs/architecture_block_diagram.svg`
  - `docs/architecture_flow.svg`

---

## Execution Rules (필수)

- 상태값은 `NOT_STARTED -> IN_PROGRESS -> BLOCKED -> DONE`만 사용.
- **Phase 게이트 방식**: 한 Phase의 `DONE`율 100%가 되기 전 다음 Phase 신규 착수 금지.
- 각 TODO는 아래 4개를 반드시 남김:
  1) 코드 PR 링크
  2) 테스트 증적(명령어 + 결과)
  3) 버전 메타(`*_version`)
  4) 롤백 전략
- `execution_mapper`, `label_builder`, `backtester`는 반드시 동일 규칙 소스 공유(정합성 최우선).

---

## Phase A (P0) — Execution / Calendar / Label / Backtest

### [A1] TODO-01 `calendar.py` 실전화
- **Status:** NOT_STARTED
- **Depends on:** 없음
- **Implement**
  - KRX 공식 거래일/휴장일/조기폐장 반영
  - NXT 유효 세션 시간 + pause 구간 명시
  - `next_trading_day`, `add_trading_days`를 공식 캘린더로 통일
  - `session_calendar_version` 도입
- **Definition of Done**
  - 휴장/조기폐장/pause 테스트 통과
  - anchor 분류 정확도 테스트 통과
- **Test commands**
  - `pytest -q tests/test_session_mapping.py`
  - `pytest -q tests/test_timestamp_boundaries.py`

### [A2] TODO-02 `session_rules.py` 경계 재정의
- **Status:** NOT_STARTED
- **Depends on:** A1
- **Implement**
  - `NXT_PRE`, `CORE_DAY`, `CLOSE_PRICE`, `NXT_AFTER`, `OFF_MARKET` 상호 배타화
  - overlap은 세션 메타데이터로 별도 기록
  - `classify_session()`에서 캘린더 의존
- **Definition of Done**
  - 경계값(08:49, 08:50, 09:00, 15:19, 15:20, 15:29, 15:30, 15:39, 15:40, 19:59, 20:00) 테스트 통과

### [A3] TODO-03 `execution_mapper.py` 고도화
- **Status:** NOT_STARTED
- **Depends on:** A1, A2
- **Implement**
  - tradable phase 계산을 캘린더 기반으로 변경
  - broker cutoff, NXT eligibility snapshot, venue/feed/clock 상태 반영
  - rollover reason 세분화
  - `expected_cost_bps` 계산을 `SessionCostModel`로 통일
- **Definition of Done**
  - live/label/backtest가 같은 mapper 사용
  - execution parity 테스트 통과
- **Test commands**
  - `pytest -q tests/test_session_mapping.py tests/test_venue_fallback.py`

### [A4] TODO-04 `label_builder.py` 확장
- **Status:** NOT_STARTED
- **Depends on:** A3
- **Implement**
  - `er_5d`, `er_20d`, `dd_20d`, `p_up_20d`
  - interrupted horizon, session/venue/execution timestamp, entry/exit cost 기록
  - corporate action hook + halt/suspension censoring
- **Definition of Done**
  - 라벨 스키마/값 검증 테스트 통과
- **Test commands**
  - `pytest -q tests/test_label_builder.py tests/test_no_lookahead.py`

### [A5] TODO-05 `backtester.py` 엔진화
- **Status:** NOT_STARTED
- **Depends on:** A3, A4
- **Implement**
  - executable bars, partial fill, same-bar fill 금지
  - 세션/venue별 fill + 비용/세금/슬리피지/임팩트 반영
  - single-trade + portfolio 모드 지원
- **Definition of Done**
  - gross/net 수익률 및 세션별 분해 산출
- **Test commands**
  - `pytest -q tests/test_backtester_realism.py tests/test_no_lookahead.py`

---

## Phase B (P1) — Cost / Routing / Broker

### [B1] TODO-06 `cost_model.py` 실전화
- **Status:** NOT_STARTED
- **Depends on:** A5
- **Implement**
  - 세션별 curve + participation 비선형 impact
  - liquidity bucket, spread proxy, buy/sell tax 반영
  - `cost_model_version` 관리
- **Definition of Done**
  - participation 증가 시 total bps 증가 테스트 통과

### [B2] TODO-07 `nxt_eligibility_store.py` 강화
- **Status:** NOT_STARTED
- **Depends on:** B1
- **Implement**
  - snapshot date + version 관리
  - stale/미존재 fail-closed
  - symbol routable, broker support 분리
- **Definition of Done**
  - stale snapshot이면 NXT 자동 disable

### [B3] TODO-08 `venue_router.py` 정책형 라우팅
- **Status:** NOT_STARTED
- **Depends on:** B1, B2
- **Implement**
  - implementation shortfall 기반 라우팅
  - 세션별 보수 정책, uncertain state 차단
  - routing rationale 로깅
- **Definition of Done**
  - 정책 + 비용 동시 반영 테스트 통과
- **Test commands**
  - `pytest -q tests/test_venue_router.py tests/test_venue_fallback.py`

### [B4] TODO-09 `broker_gateway.py` 상태머신 확장
- **Status:** NOT_STARTED
- **Depends on:** B3
- **Implement**
  - NEW/PARTIAL/FILLED/REJECTED/CANCELED
  - cancel/replace, TIF, cutoff, reconciliation
- **Definition of Done**
  - paper-trading 시뮬레이터 기준 충족
- **Test commands**
  - `pytest -q tests/test_broker_gateway.py`

---

## Phase C (P2) — Training / Prediction / Calibration

### [C1] TODO-10 `training.py` 학습 파이프라인화
- **Status:** NOT_STARTED
- **Depends on:** A5, B1
- **Implement**
  - dataset builder + walk-forward split
  - leakage-safe feature/label join
  - LightGBM baseline + artifact/metrics 저장
- **Definition of Done**
  - `er_20d`, `dd_20d`, `p_up_20d` 학습 가능
- **Test commands**
  - `pytest -q tests/test_training_splitter.py tests/test_flow_leakage.py`

### [C2] TODO-11 `predictor.py` 모델 추론 교체
- **Status:** NOT_STARTED
- **Depends on:** C1
- **Implement**
  - artifact load + schema validation + `FusedPrediction` 변환
  - uncertainty 및 calibrator 연결
- **Definition of Done**
  - 하드코딩 predictor 제거

### [C3] TODO-12 Calibration 모듈 추가
- **Status:** NOT_STARTED
- **Depends on:** C1, C2
- **Implement**
  - `p_up_20d` calibration, `dd_20d` quantile 조정
  - pre/post metric 산출
- **Definition of Done**
  - calibration artifact 저장 + 비교 리포트 생성

---

## Phase D (P3/P4 일부) — Feature / Risk / Portfolio

### [D1] TODO-13 `feature_store.py` 오프라인/온라인 정합화
- **Status:** NOT_STARTED
- **Depends on:** C1
- **Implement**
  - session-aware windows + freshness/missingness/venue/session flags
  - `build_online_features()`, `build_offline_features()` 구현
- **Definition of Done**
  - parity 테스트 통과

### [D2] TODO-14 시장/거시 컨텍스트 피처
- **Status:** NOT_STARTED
- **Depends on:** D1
- **Implement**
  - KOSPI, KOSDAQ, KOSPI200 futures, USDKRW, breadth/RS 지표
- **Definition of Done**
  - 심볼 피처 row에 regime context 포함

### [D3] TODO-15 `risk_engine.py` 확장
- **Status:** NOT_STARTED
- **Depends on:** D1, D2
- **Implement**
  - hard stop + veto + uncertainty/dd 기반 size shrink
  - session aggressiveness + regime flip exit hook
- **Definition of Done**
  - entry/exit risk rationale 로깅 가능
- **Test commands**
  - `pytest -q tests/test_risk_portfolio.py`

### [D4] TODO-16 `portfolio_engine.py` 확장
- **Status:** NOT_STARTED
- **Depends on:** D3
- **Implement**
  - sector/correlation/turnover/liquidity cap
  - staged entry + rebalance + priority
- **Definition of Done**
  - multi-name decision -> executable order set 변환

---

## Phase E (P5) — Event / LLM / Vector

### [E1] TODO-17 `event_store.py` lineage/dedup
- **Status:** NOT_STARTED
- **Depends on:** D4
- **Implement**
  - canonical_event_id, cluster_id, source_lineage
  - novelty/dedup/source quality/source type
- **Definition of Done**
  - 이벤트 계보 복원 가능

### [E2] TODO-18 `llm_event_normalizer.py` 프로덕션화
- **Status:** NOT_STARTED
- **Depends on:** E1
- **Implement**
  - provider adapter, prompt version, retry/fallback, schema metric
  - low-confidence downweight + evidence span 검증
- **Definition of Done**
  - 감사 가능한 semantic output + degraded metrics

### [E3] TODO-19 `text_encoder.py`/`attention_aggregator.py`/`vectorization.py`
- **Status:** NOT_STARTED
- **Depends on:** E2
- **Implement**
  - BERT encoder, hierarchical attention, structured vector payload
  - metadata version/traceability 강화
- **Definition of Done**
  - placeholder 제거 + deterministic artifact 관리
- **Test commands**
  - `pytest -q tests/test_vectorization_metadata.py tests/test_schema_validation.py tests/test_degraded_mode.py`

---

## Phase F (P6) — Live / Orchestration / Monitoring / Audit

### [F1] TODO-20 운영 파이프라인 통합
- **Status:** NOT_STARTED
- **Depends on:** A~E 전부 DONE
- **Implement**
  - anchor-based batch inference + semantic refresh
  - idempotency/retry/backoff/circuit breaker/DLQ
  - 운영 메트릭 + decision lineage audit log
- **Definition of Done**
  - 지정 anchor run(08:10~20:05) 동작
  - 주문 단위 결정 복원 가능

---

## 20개 GitHub Issue 템플릿 (복붙용)

아래 형식으로 20개 이슈 생성:

```text
[TODO-XX] <모듈명>: <핵심 목표>

## Why
- (비즈니스/정합성 이유)

## Scope
- [ ] 구현 항목 1
- [ ] 구현 항목 2
- [ ] 구현 항목 3

## DoD
- [ ] 단위/통합 테스트 통과
- [ ] 버전 메타 추가
- [ ] 롤백 전략 문서화

## Evidence
- PR:
- Test command:
- Test result:
- Runtime sample/log:
```

---

## 최종 체크리스트 (Release Gate)

- [ ] TODO-01 ~ TODO-20 모두 DONE
- [ ] Backtest/Label/Execution parity 검증 리포트 완료
- [ ] Model/Prompt/Vector/Cost 버전 고정 및 기록 완료
- [ ] 운영 대시보드 + 감사 로그 + 장애 복구 절차 완료
- [ ] 프로덕션 런북 업데이트 완료

> 한 줄 결론: **“전부 실행”은 순서와 게이트를 지킬 때만 달성됩니다.**
