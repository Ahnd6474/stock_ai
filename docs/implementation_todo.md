# Stock AI Master Execution TODO (All Items Must Be Executed)

목표: 이 문서는 단순 아이디어 목록이 아니라, **실제로 전부 실행**하기 위한 마스터 백로그입니다.
원칙: 아래 20개 TODO는 **누락 없이 전부 완료**를 목표로 하며, 선후관계를 반드시 지킵니다.

---

## Progress Snapshot (2026-03-20)

- DONE: TODO-03(기본 cost 연동/cutoff), TODO-05(기본 run_trade), TODO-07(버전+날짜 저장), TODO-08(라우팅 rationale), TODO-13(online/offline build), TODO-20 일부(idempotency/retry/circuit-breaker 스캐폴드).
- IN_PROGRESS: TODO-01, TODO-02, TODO-04, TODO-06, TODO-09, TODO-10~12, TODO-14~19 (일부 모듈은 스캐폴드 구현 완료).
- NOT_STARTED: 외부 실데이터/실거래 연동, 실모델 학습/캘리브레이션 아티팩트, 실제 Korean BERT 연동.

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
