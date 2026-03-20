# K-Swing Sentinel v1.2 (MVP Scaffold)

K-Swing Sentinel is a production-oriented Python scaffold for a KRX/NXT swing-trading workflow.  
The codebase focuses on **strict contracts**, **predictable runtime behavior**, and **safe fallbacks** so teams can iterate on strategy logic without losing operational guardrails.

This repository currently includes:

- Typed contracts and schemas via Pydantic.
- Session-aware KRX/NXT classification.
- Venue-aware execution mapping with conservative rollover.
- Numeric-first predictor stub plus decision engine.
- LLM normalizer with strict schema validation and degraded fallback behavior.
- Flow snapshot store with provisional/confirmed leakage guard.
- Production runtime gate for readiness checks and startup validation.
- Unit tests that cover contracts, runtime boundaries, routing, and fallback logic.

## Key files

- Architecture and operating rules: `docs/k_swing_sentinel_v1_2.md`
- Typed schemas: `src/kswing_sentinel/schemas.py`
- Service contracts: `src/kswing_sentinel/contracts.py`
- Example JSON payloads: `src/kswing_sentinel/example_payloads.py`
- Production readiness and live gate: `src/kswing_sentinel/production_runtime.py`
- Example live config: `configs/production_runtime.example.toml`
- Example semantic stack config: `configs/semantic_stack.example.toml`

## Quick start

### 1) Create a local environment and install dev dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

### 2) Run tests

```bash
pytest -q
```

### 3) (Optional) Install semantic-stack extras

If you plan to wire LLM and ML components locally:

```bash
pip install -e .[dev,llm,ml]
```

Default semantic stack example:

- Search: internal context only
- Reasoning: `x-ai/grok-4.1-fast`

API keys are intentionally not stored in the repository.
Set `OPENROUTER_API_KEY` in your runtime environment instead.

## Common test targets

Run a specific test module while iterating:

```bash
pytest -q tests/test_production_runtime.py
pytest -q tests/test_venue_router.py
pytest -q tests/test_llm_provider_path.py
```

## Repository layout

```text
configs/                  Example runtime and semantic-stack configuration
docs/                     Architecture docs and roadmap notes
src/kswing_sentinel/      Core implementation (schemas, contracts, runtime, routing)
tests/                    Unit tests for runtime behavior and guardrails
```

## 주식 예측 AI 에이전트화 전체 계획

아래는 제안된 end-to-end 파이프라인 초안입니다.

1. `grok-4.1-fast + grok search` 기반으로 **회사 사회적 맥락 텍스트**를 수집합니다.
   - 정형화된 프롬프트 템플릿을 사용해 입력 품질을 고정합니다.
2. 1의 요약 결과를 **BERT 문장 단위 인코딩**으로 변환합니다.
3. BERT 출력 벡터들을 **attention 가중합**해 사회 임베딩을 생성합니다.
4. **시계열 임베딩**을 구성합니다.
   - 날짜(date)와 시간(time)을 분리해 각각 임베딩합니다.
5. 주식 차트(OHLCV 등)를 수집하고, `pandas-ta-classic`으로 지표를 계산한 뒤 선형층을 통과시켜 **상태 임베딩**을 만듭니다.
6. 시계열 임베딩과 `(상태 + 사회)` 임베딩을 concat하여 통합 표현을 구성합니다.
7. 다음 시점의 **리스크**와 **로그 증가량(log return increment)**을 예측합니다.
   - 구체 예측 헤드/학습 방식은 추가 설계가 필요합니다.
8. 예측 결과를 기반으로 에이전트가 주문을 실행합니다.
   - 리스크에 반비례하게 자금 투입량을 조절합니다.
   - 로그 증가량 기반으로 매수/매도 수량을 산출합니다(실수값 회귀).
   - 음수면 매도, 양수면 매수합니다.
   - 매도 시 보유량을 초과해 매도하지 않습니다.

## Roadmap

- GitHub issue 초안: `docs/ROADMAP_GITHUB_ISSUES.md`
