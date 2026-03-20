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

<<<<<<< HEAD
Default semantic stack example:

- Search: internal context only
- Reasoning: `x-ai/grok-4.1-fast`

API keys are intentionally not stored in the repository.
Set `OPENROUTER_API_KEY` in your runtime environment instead.
=======
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
>>>>>>> f7ce4571a6b77ae8d443c14be5bbc74981deb04e

## Roadmap

- GitHub issue 초안: `docs/ROADMAP_GITHUB_ISSUES.md`
