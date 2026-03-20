# K-Swing Sentinel v1.2 (MVP Scaffold)

This repository now contains a working Python scaffold implementing the production design baseline:

- Typed contracts via Pydantic.
- Session-aware KRX/NXT classification.
- Venue-aware execution mapping with conservative rollover.
- Numeric-first predictor stub + decision engine.
- LLM normalizer with strict schema and degraded fallback.
- Flow snapshot store with provisional/confirmed leakage guard.
- Basic orchestration and unit tests.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest -q
```
