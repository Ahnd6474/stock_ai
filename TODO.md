# TODO

현재 저장소는 production-oriented scaffold를 넘어서, `direct vectorization + temporal attention predictor` 단계까지 들어와 있다.
남은 작업은 "모델 구조를 더 만드는 것"보다, 이 경로를 실제 데이터와 운영 경로에 끝까지 연결하는 쪽에 있다.

상세 Epic/Issue 초안은 `docs/ROADMAP_GITHUB_ISSUES.md`를 따른다.

## P0 — temporal predictor 경로를 실제로 완성할 것

- [ ] `FeatureStore`, `FlowSnapshotStore`, 텍스트 벡터 히스토리를 이용해 symbol별 historical `state_sequence` 생성 및 저장 경로를 추가하기
- [ ] temporal predictor가 기대하는 `state_sequence` runtime contract를 고정하고, `LiveInferenceService`가 flat feature fallback보다 sequence-first 경로를 우선 사용하게 정리하기
- [ ] 시간 임베딩 개선: `minute_of_session`, `day_of_week`, `session_type`, `time_gap_from_prev_step`, `time_since_event`를 predictor 입력에 반영하고, learned `time_embedding`으로 넣을지 timestep MLP 입력으로 합칠지 확정하기
- [ ] temporal predictor용 walk-forward 학습, validation report, artifact selection 기준을 추가해서 `train_temporal_transformer()`를 실사용 가능한 학습 경로로 올리기
- [ ] temporal predictor의 calibration, uncertainty, regime 품질을 실데이터 기준으로 다시 맞추기

## P0 — runtime 통합에서 바로 메울 것

- [ ] `ProductionTradingEngine` 앞단에 `EventRetrieval -> Dedup/Cluster -> raw payload assembly` 경로를 실제 서비스 객체로 연결하고, optional LLM normalization branch는 필요할 때만 켜지게 정리하기
- [ ] `FeatureStore`, `FlowSnapshotStore`, `NXT eligibility` 조회를 런타임에 연결해서 `features_by_symbol`, `last_price_by_symbol`를 수동 주입하지 않도록 바꾸기
- [ ] `DecisionEngine` 뒤에 `RiskEngine`, `PortfolioEngine`를 끼워 넣고 주문 전 최종 포지션/노출 한도를 적용하기
- [ ] 주문 제출 뒤 `BrokerGateway.reconcile()` 기반 체결 확인, 포지션 갱신, 후속 audit/monitoring 갱신 경로를 추가하기
- [ ] `ProductionOrchestrator`와 `TemporalLikeOrchestrator`의 idempotency, retry, dead-letter 책임을 하나의 운영 경로로 정리하기

## P1 — 모델/데이터 품질 고도화

- [ ] online/offline feature parity 테스트를 temporal `state_sequence` 기준으로 확장하기
- [ ] provisional vs confirmed flow 히스토리 저장 정책을 temporal feature 생성 경로까지 포함해 명확화하기
- [ ] event lineage / dedup / novelty score 운영 경로를 고정하고, vector history와 같이 묶어 재현 가능하게 만들기
- [ ] 텍스트 벡터 drift와 temporal artifact 성능 저하를 같이 볼 수 있는 평가 지표를 추가하기
- [ ] `enc/Model.py` 프로토타입을 temporal 학습/벡터화 경로 중 하나에 연결할지, 실험 코드로 유지할지 역할을 확정하기

## P1 — 실거래 운영성 강화

- [ ] venue router에 비용/유동성/stale-state를 통합하기
- [ ] broker gateway 주문 상태머신을 강화하기
- [ ] degraded mode 및 fallback 로그를 표준화하기
- [ ] runtime readiness 체크 결과를 dependency probe와 자동 연결하기
- [ ] 체결 이후 재시도/취소/정정 정책을 브로커 어댑터 단위로 분리하기
- [ ] 공식 거래 캘린더/세션 정의를 외부 데이터 파일 또는 관리 가능한 설정으로 분리하기

## P2 — 문서화 및 운영 준비

- [ ] `docs/k_swing_sentinel_v1_2.md`를 현재 구현 상태에 맞게 업데이트하기
- [ ] 예제 config와 실제 runtime contract 예시를 확장하기
- [ ] 의사결정 lineage 예시(JSON)를 추가하기
- [ ] 운영 체크리스트(runbook)를 작성하기
- [ ] 배포 전 검증 체크리스트를 작성하기

## 최근 반영

- [x] 기본 live path에서 `LLMEventNormalizer`를 필수 단계로 두지 않고 raw text direct vectorization 경로로 전환
- [x] sentence-level RoBERTa + hierarchical transformer로 `z_event`, `z_social`, `z_macro` 생성
- [x] `vector_payload`를 마지막 step에 연결해 temporal predictor가 텍스트 벡터를 실제 입력으로 사용
- [x] `state_sequence` 기반 `temporal_transformer_v1` artifact load/train 경로 추가
- [x] timestep embedding 뒤에 self-attention context block과 causal attention block을 연속 배치
- [x] `ProductionOrchestrator`에 persisted dead-letter JSONL 로드 및 `redrive_persisted_dead_letters()` 재처리 helper 추가
- [x] 문서에 적은 semantic refresh anchor와 event-burst 신호를 runtime audit event로 남기도록 반영
- [x] README, docs, SVG 다이어그램을 현재 runtime 상태와 한글 표기 기준으로 동기화

## Done 기준

다음을 만족하면 scaffold 단계를 벗어난 것으로 본다.

- historical `state_sequence`가 live/backtest/training 경로에서 같은 계약으로 생성된다
- 시간 임베딩이 predictor에 통합되고, 성능 및 안정성 기준이 검증된다
- 실제 학습 artifact 기반 temporal 추론이 기본 live 경로에서 동작한다
- label / execution / backtest fill 정합성이 확보된다
- no-lookahead / session boundary / fallback / temporal sequence 테스트가 통과한다
- online/offline parity와 문서 상태가 실제 구현과 일치한다
