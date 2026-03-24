# Dead-Letter Recovery Runbook

## 목적

`ProductionOrchestrator`가 anchor batch를 끝까지 처리하지 못하면 dead letter를 메모리 큐와 선택적 JSONL 파일에 남긴다.
이 문서는 `dead_letter_log_path`를 켠 환경에서 기본적인 확인과 재처리 순서를 정리한다.

## 전제 조건

- `configs/production_runtime.example.toml` 또는 실제 runtime config에 `dead_letter_log_path`가 설정돼 있어야 한다.
- dead letter는 원본 예외를 대체하지 않는다. 먼저 실패 원인을 제거한 뒤 재처리해야 한다.
- 같은 anchor 재실행 시에는 기존 audit/metrics/dead-letter 로그와 함께 검토한다.

## 로그 형식

각 줄은 JSONL 한 레코드이며 아래 필드를 포함한다.

- `record_type`
- `anchor_time`
- `symbols`
- `payload_by_symbol`
- `features_by_symbol`
- `venue_eligibility_by_symbol`
- `error_message`
- `attempts`
- `failed_at`

## 현재 제공되는 helper

현재 구현에는 아래 보조 메서드가 포함된다.

- `ProductionOrchestrator.load_persisted_dead_letters(dead_letter_log_path)`
- `ProductionOrchestrator.redrive_persisted_dead_letters(...)`

즉, dead letter 파일을 사람이 직접 읽어 입력을 복원할 수도 있고, 실패 원인 제거 후 helper를 호출해 기록된 anchor batch를 다시 흘려보낼 수도 있다.

## 운영 절차

1. `error_message`와 같은 시각의 audit/metrics 로그를 같이 확인한다.
2. feed, broker, artifact, vectorizer, kill-switch 등 즉시 차단 원인이 제거됐는지 확인한다.
3. dead letter의 `anchor_time`, `symbols`, payload/features를 기준으로 재실행 입력을 복원하거나 `redrive_persisted_dead_letters(...)` 호출 준비를 한다.
4. 같은 anchor를 다시 돌릴 때는 원인 제거 여부를 먼저 검증하고, 중복 주문 방지 규칙을 유지한 채 helper 또는 수동 입력으로 재실행한다.
5. 재실행 성공 후에는 dead letter 원본은 보존하되, 운영 기록에 처리 완료 여부를 남긴다.

## 주의 사항

- dead letter 파일은 영속 로그이며, 백그라운드 daemon형 자동 redrive 큐는 아니다. 다만 `redrive_persisted_dead_letters()`로 명시적 재처리는 가능하다.
- 주문이 이미 일부 제출된 경우 broker/audit 로그로 실제 체결 상태를 먼저 대조한다.
- 반복 실패가 이어지면 circuit breaker가 열릴 수 있으므로, 원인 제거 없이 재시도만 반복하지 않는다.
