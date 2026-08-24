# arthiq-ai

The Arthiq AI service. It receives a structured market snapshot from
`arthiq-backend`, asks a configured AI provider for a reading, and returns a
validated, provider-neutral `TradingDecision`.

```
arthiq-backend  ──authenticated REST/JSON──▶  arthiq-ai
                                                 │
                                    context ─▶ prompt ─▶ provider
                                                 │        (ollama | gemini | fake)
                                                 ▼
                                        output validation
                                                 │
arthiq-backend  ◀──── TradingDecision ───────────┘
```

## What this service cannot do

The AI has no authority to trade, and that is enforced structurally rather than
by policy. This repository contains no broker client, no database driver, no
Firebase SDK and no order-placement code — `app/` imports none of them, and CI
fails the build if that changes. The backend remains the system of record and
the only authority for trading.

A `TradingDecision` is an analytical opinion. It guarantees nothing, and
`confidence` is the model's self-reported conviction, **not** a calibrated
probability of profit — see [docs/architecture.md](docs/architecture.md).

## Quick start

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"     # Windows: .venv\Scripts\pip
cp .env.example .env
```

Set two things in `.env`:

```bash
AI_SERVICE_API_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
OLLAMA_MODEL=<a model you have pulled>
```

Run it:

```bash
uvicorn app.main:app --reload --port 8100
curl http://localhost:8100/health
```

Call it (the service key authenticates the *calling service*, not a user):

```bash
curl -X POST http://localhost:8100/internal/v1/analyze \
  -H "X-API-Key: $AI_SERVICE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
        "symbol": "RELIANCE",
        "exchange": "NSE",
        "interval": "ONE_DAY",
        "as_of": "2026-08-21T10:00:00Z",
        "trading_date": "2026-08-21",
        "price": "1316.0000",
        "technical": {"rsi_14": "54.20", "ema_20": "1308.40", "ema_50": "1297.80"}
      }'
```

No AI installed? `AI_PROVIDER=fake` runs the whole path offline with a
deterministic provider.

## Local AI with Ollama

Ollama is the default development provider. It is an **external** service — it
is not in the Docker image and is not required in production.

```bash
ollama serve
ollama pull qwen3:8b                   # any instruct model you prefer
export OLLAMA_MODEL=qwen3:8b
export OLLAMA_THINK=false              # reasoning models only (qwen3, deepseek-r1)
python scripts/verify_ollama.py        # one call, prints the decision
```

No model name is hard-coded anywhere in the source; `OLLAMA_MODEL` must be set.

## Tests

```bash
ruff format --check .
ruff check .
pytest
```

The default run needs no Ollama, no Gemini key, no API keys and no internet.
Live-provider tests are marked `integration` and are excluded unless asked for
— see [docs/testing.md](docs/testing.md).

## Docker

```bash
docker compose up --build     # http://localhost:8100/health
```

The image runs as a non-root user, contains no secrets and no Ollama, and its
health check performs no AI call.

## Documentation

| Document | Contents |
| --- | --- |
| [docs/architecture.md](docs/architecture.md) | Layering, domain model, context builder, prompt versioning, cost control, security |
| [docs/providers.md](docs/providers.md) | The `AIProvider` interface, Ollama, Gemini, the fake provider, adding a provider |
| [docs/api.md](docs/api.md) | The exact backend integration contract, authentication, errors |
| [docs/testing.md](docs/testing.md) | Unit tests, and how to run the Ollama and Gemini integration tests |

## Layout

```
app/
  core/        configuration, error envelope, logging, redaction, service auth
  domain/      AnalysisRequest, TradingDecision, enums  (provider-neutral)
  context/     bounded, deterministic context builder
  prompts/     versioned prompts (technical_analysis_v1)
  providers/   AIProvider interface + fake / ollama / gemini adapters
  analysis/    output validation, service pipeline, internal router
  health/      GET /health
tests/         unit tests (offline)  +  tests/integration (opt-in)
scripts/       container entrypoint, Ollama verification
```
