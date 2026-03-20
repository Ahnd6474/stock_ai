# K-Swing Sentinel v1.2 (MVP Scaffold)

This repository now contains a working Python scaffold implementing the production design baseline:

- Typed contracts via Pydantic.
- Session-aware KRX/NXT classification.
- Venue-aware execution mapping with conservative rollover.
- Numeric-first predictor stub + decision engine.
- LLM normalizer with strict schema and degraded fallback.
- Flow snapshot store with provisional/confirmed leakage guard.
- Basic orchestration and unit tests.
- Service contracts for the production module boundaries.
- Machine-validated example payloads for audit and integration tests.

## Key files

- Architecture and operating rules: `docs/k_swing_sentinel_v1_2.md`
- Typed schemas: `src/kswing_sentinel/schemas.py`
- Service contracts: `src/kswing_sentinel/contracts.py`
- Example JSON payloads: `src/kswing_sentinel/example_payloads.py`
- Production readiness and live gate: `src/kswing_sentinel/production_runtime.py`
- Example live config: `configs/production_runtime.example.toml`
- Example semantic stack config: `configs/semantic_stack.example.toml`

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest -q
```

For actual semantic stack wiring:

```bash
pip install -e .[dev,llm,ml]
```

## Roadmap

- GitHub issue 초안: `docs/ROADMAP_GITHUB_ISSUES.md`
