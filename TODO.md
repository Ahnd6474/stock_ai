# TODO

현재 저장소는 production-oriented scaffold 단계이므로, 아래 항목을 우선순위 기준으로 정리한다.
상세 Epic/Issue 초안은 `docs/ROADMAP_GITHUB_ISSUES.md`를 따른다.

## P0 — 현재 실행 흐름에서 바로 메울 것

- [ ] `ProductionTradingEngine` 앞단에 `EventRetrieval -> Dedup/Cluster -> LLM normalization` 경로를 실제 서비스 객체로 연결하고, 지금처럼 외부에서 정제 payload를 직접 넣는 구조를 줄이기
- [ ] `VectorizationPipeline.build()`의 `z_event`, `z_social`, `z_macro`를 `NumericFirstPredictor` 입력 feature로 실제 반영하기
- [ ] `FeatureStore`, `FlowSnapshotStore`, `NXT eligibility` 조회를 런타임에 연결해서 `features_by_symbol`, `last_price_by_symbol`를 수동 주입하지 않도록 바꾸기
- [ ] `DecisionEngine` 뒤에 `RiskEngine`, `PortfolioEngine`를 끼워 넣고 주문 전 최종 포지션/노출 한도를 적용하기
- [ ] 주문 제출 뒤 `BrokerGateway.reconcile()` 기반 체결 확인, 포지션 갱신, 후속 audit/monitoring 갱신 경로를 추가하기
- [ ] `ProductionOrchestrator`와 `TemporalLikeOrchestrator`의 idempotency, retry, dead-letter 책임을 하나의 운영 경로로 정리하기

## P0 — scaffold에서 실제 동작으로 바꿀 것

- [ ] `predictor.py` 선형 baseline 의존을 줄이고, 학습 artifact 기반 multi-head 추론을 기본 경로로 전환
- [ ] `execution_mapper.py`, `label_builder.py`, `backtester.py`의 체결 규칙과 fill 가정을 단일 규칙으로 맞추기
- [ ] 공식 거래 캘린더/세션 정의를 외부 데이터 파일 또는 관리 가능한 설정으로 분리하기
- [ ] `backtester.py`를 portfolio simulator로 확장해서 주문 단위가 아니라 상태 전이 단위로 검증하기
- [ ] `enc/Model.py` 프로토타입을 학습/벡터화 경로 중 하나에 연결할지, 실험 코드로 유지할지 역할을 확정하기

## P1 — 모델/데이터 품질 고도화

- [ ] walk-forward 학습 파이프라인 완성
- [ ] calibration 전/후 metric 자동 산출
- [ ] online/offline feature parity 테스트 강화
- [ ] provisional vs confirmed flow 히스토리 저장 정책 명확화
- [ ] event lineage / dedup / novelty score 운영 경로 고정

## P1 — 실거래 운영성 강화

- [ ] venue router에 비용/유동성/stale-state 통합
- [ ] broker gateway 주문 상태머신 강화
- [ ] degraded mode 및 fallback 로그 표준화
- [ ] runtime readiness 체크 결과를 dependency probe와 자동 연결
- [ ] 체결 이후 재시도/취소/정정 정책을 브로커 어댑터 단위로 분리

## P2 — 문서화 및 운영 준비

- [ ] README에 백테스트 한계와 production data gaps 명시
- [ ] 예제 config와 실제 runtime contract 예시 확장
- [ ] 의사결정 lineage 예시(JSON) 추가
- [ ] 운영 체크리스트(runbook) 작성
- [ ] 배포 전 검증 체크리스트 작성

## Done 기준

다음을 만족하면 scaffold 단계를 벗어난 것으로 본다.

- 실제 학습 artifact 기반 추론
- 텍스트/수치 피처 융합이 실시간 경로에서 동작
- label / execution / backtest fill 정합성 확보
- no-lookahead / session boundary / fallback 테스트 통과
- online/offline parity 검증 가능
- README와 실제 구현 상태가 일치
