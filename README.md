# K-Swing Sentinel v0.1.0

K-Swing Sentinel is a production-oriented Python scaffold for a KRX/NXT swing-trading workflow.
The codebase is strongest in typed contracts, session and venue handling, readiness gates, safe fallbacks, and testable runtime behavior.
It is not yet a turnkey live-trading system with bundled market data and broker connectivity.

## At a Glance

- Good fit if you want to extend a trading-system scaffold with strict schemas, deterministic execution rules, and safety guardrails.
- Less complete if you need a finished live stack, production broker integration, or a full portfolio simulator out of the box.
- Most modules are library-style building blocks. The main developer workflows today are tests, training helpers, data-collection scripts, and runtime integration work.

## What Is Already Implemented

- Pydantic-based contracts and schema validation across the runtime surface
- KRX/NXT session classification, venue-aware routing rules, and conservative execution mapping
- Decision logic, risk-aware trade action selection, and degraded fallback handling
- Cost-aware backtesting utilities with no-lookahead validation
- Walk-forward training scaffolding with artifact export
- Numeric-first predictor loading local JSON artifacts
- LLM event normalization with structured-output validation and numeric-only fallback
- Text vectorization with a transformer-backed path and a hashing fallback
- Audit logging, monitoring hooks, orchestration helpers, and production readiness checks
- Data-collection scripts and sample training datasets under `data/training/`

## What Is Still Partial

- Live inference exists, but external payloads, numeric features, venue eligibility, last prices, and dependency state still have to be supplied by the caller
- LLM integration supports OpenRouter-style providers, but credentials and provider operations are outside the repository
- The text branch is validated and tracked, but its vectors are not yet fused into the default numeric predictor path
- The predictor and training pipeline are baseline scaffolds, not a production-grade LightGBM or CatBoost stack
- The backtester enforces execution realism better than a toy simulator, but it is not yet a full event-driven portfolio engine

## What Is Not Included Yet

- Licensed real-time KRX or NXT feeds
- Confirmed live brokerage integration
- Deployment automation, shadow-trading runbooks, and operational procedures
- Full portfolio-level execution, reconciliation, and post-trade lifecycle coverage

## Quick Start

### Prerequisites

- Python `>=3.11`
- Optional for LLM smoke tests: `OPENROUTER_API_KEY`
- Optional for live-runtime experiments: `KRX_FEED_KEY`, `BROKER_API_KEY`, and `BROKER_ACCOUNT_ID`

### Full local environment

Bash:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`requirements.txt` installs the editable package with `dev`, `llm`, `ml`, and `marketdata` extras.

### Minimal development environment

If you only want the test and core runtime surface:

```bash
pip install -e '.[dev]'
```

PowerShell:

```powershell
pip install -e ".[dev]"
```

### Run tests

```bash
python -m pytest -q
```

On restricted environments, you may need to point pytest at a writable temp directory:

```bash
python -m pytest -q --basetemp .tmp/pytest -p no:cacheprovider
```

## Common Commands

Run a focused test module:

```bash
python -m pytest -q tests/test_production_runtime.py
python -m pytest -q tests/test_venue_router.py
python -m pytest -q tests/test_training_pipeline_artifacts.py
```

Collect a small FinanceDataReader dataset:

```bash
python scripts/collect_fdr_training_data.py --symbols 005930 000660 035420
```

Collect multi-timeframe Yahoo-based sample datasets:

```bash
python scripts/collect_intraday_training_data.py --symbols 005930 000660 --prefix krx_sample
```

Smoke-test the LLM normalizer path:

```bash
python scripts/smoke_test_grok.py
```

## Runtime Entry Points

There is no polished end-user CLI yet. The main integration points are Python classes:

- [`src/kswing_sentinel/production_runtime.py`](src/kswing_sentinel/production_runtime.py)
  - `ProductionRuntimeConfig`
  - `ProductionReadinessGate`
  - `ProductionTradingEngine`
  - `ProductionOrchestrator`
- [`src/kswing_sentinel/live.py`](src/kswing_sentinel/live.py)
  - `LiveInferenceService`

The current live path expects the caller to provide:

- symbols
- raw event payloads
- numeric features
- venue eligibility flags
- last prices
- runtime dependency state
- runtime config
- model requirements

That design keeps the core logic testable, but it also means this repository does not yet include the surrounding production services needed for live deployment.

## Configuration and Data

- [`configs/production_runtime.example.toml`](configs/production_runtime.example.toml)
  - Trading mode selection
  - Required environment variables
  - Kill switch path
  - Audit and metrics log paths
- [`configs/semantic_stack.example.toml`](configs/semantic_stack.example.toml)
  - Search provider settings
  - LLM provider and model settings
  - Encoder backend and model settings
- [`data/training/`](data/training/)
  - Sample daily, 60-minute, and 15-minute feature datasets

## Repository Layout

```text
configs/                  Example runtime and semantic-stack configuration
data/training/            Sample training datasets
docs/                     Architecture notes and roadmap documents
enc/                      Experimental encoder prototype
scripts/                  Data collection and smoke-test helpers
src/kswing_sentinel/      Core implementation
tests/                    Unit tests for contracts, runtime behavior, and guardrails
TODO.md                   Working implementation checklist
```

## Key Documents

- Architecture and operating rules: [`docs/k_swing_sentinel_v1_2.md`](docs/k_swing_sentinel_v1_2.md)
- Approach and differentiators: [`docs/approach_differentiators.md`](docs/approach_differentiators.md)
- Roadmap draft: [`docs/ROADMAP_GITHUB_ISSUES.md`](docs/ROADMAP_GITHUB_ISSUES.md)
- Detailed implementation notes: [`docs/implementation_todo.md`](docs/implementation_todo.md)
- Current task checklist: [`TODO.md`](TODO.md)

## Current Priorities

- Replace remaining scaffold-level predictor pieces with stronger trained artifacts
- Connect live runtime inputs to real feature, event, and venue-eligibility sources
- Expand execution realism and portfolio simulation
- Keep README, TODOs, and architecture documents aligned with actual implementation status
