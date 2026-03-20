# K-Swing Sentinel v1.2 (Build-out in progress)

This repo now contains a **functional implementation baseline** (not final production) for the Korean long-only swing system:

- Typed contracts for trading/event/flow/execution payloads (real Pydantic when installed, compatibility fallback otherwise).
- KRX/NXT session-aware mapper with conservative rollovers.
- Versionable NXT eligibility snapshot store.
- Session/venue-aware cost model.
- Label builder tied to the same execution mapper logic used in live simulation.
- Feature/event stores with as-of-time semantics.
- Numeric-first predictor + decision/risk/portfolio engines.
- LLM normalizer fallback/degraded mode.
- Temporal-like orchestration with idempotency key.
- Unit tests for leakage, rollover, fallback, session mapping, labels, risk/portfolio, and walk-forward splitting.

## Run tests

```bash
pytest -q
```

## Important

This repository still needs real integrations for:

1. Real-time licensed KRX/NXT data feeds.
2. Broker routing + execution report plumbing.
3. Korean BERT encoder and hierarchical attention stack.
4. Calibrated LightGBM/CatBoost training + model registry.
5. Full monitoring/drift/circuit breaker stack.

Until those are connected, this is a production-oriented implementation baseline, not live-capital-ready.
