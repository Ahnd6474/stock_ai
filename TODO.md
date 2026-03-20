# TODO

현재 저장소는 production-oriented scaffold 단계이므로, 아래 항목을 우선순위 기준으로 정리한다.
상세 Epic/Issue 초안은 `docs/ROADMAP_GITHUB_ISSUES.md`를 따른다.

## P0 — 지금 바로 정리할 것

- [ ] `README.md` 머지 충돌 흔적 제거 (`<<<<<<<`, `=======`, `>>>>>>>`)
- [ ] README에 `implemented / stub / planned` 구분 명시
- [ ] `.idea/` 및 `src/k_swing_sentinel.egg-info/` 추적 여부 재검토
- [ ] marketdata extra 의존성(`requests`)와 문서 내용 동기화
- [ ] 예제 실행 경로를 실제 현재 코드 기준으로 검증

## P0 — scaffold에서 실제 동작으로 바꿀 것

- [ ] `vectorization.py` pseudo-vector 제거, 실제 임베딩 연결
- [ ] `predictor.py` 선형 baseline 제거, 학습 artifact 추론기로 교체
- [ ] `execution_mapper.py`, `label_builder.py`, `backtester.py` 체결 규칙 단일화
- [ ] 공식 거래 캘린더/세션 정의를 외부 데이터 파일로 분리
- [ ] `backtester.py`를 portfolio simulator로 확장

## P1 — 모델/데이터 품질 고도화

- [ ] walk-forward 학습 파이프라인 완성
- [ ] calibration 전/후 metric 자동 산출
- [ ] online/offline feature parity 테스트 강화
- [ ] provisional vs confirmed flow 히스토리 저장 정책 명확화
- [ ] event lineage / dedup / novelty score 운영 경로 고정

## P1 — 실거래 운영성 강화

- [ ] venue router에 비용/유동성/stale-state 통합
- [ ] broker gateway 주문 상태머신 강화
- [ ] risk / portfolio engine 상태 기반 제약 확장
- [ ] live/orchestration/monitoring 연동 강화
- [ ] degraded mode 및 fallback 로그 표준화

## P2 — 문서화 및 운영 준비

- [ ] README에 백테스트 한계와 production data gaps 명시
- [ ] 예제 config와 실제 runtime contract 예시 확장
- [ ] 의사결정 lineage 예시(JSON) 추가
- [ ] 운영 체크리스트(runbook) 작성
- [ ] 배포 전 검증 체크리스트 작성

## Done 기준

다음을 만족하면 scaffold 단계를 벗어난 것으로 본다.

- 실제 학습 artifact 기반 추론
- label / execution / backtest fill 정합성 확보
- no-lookahead / session boundary / fallback 테스트 통과
- online/offline parity 검증 가능
- README와 실제 구현 상태가 일치
