# Testing

## The default run

```bash
ruff format --check .
ruff check .
pytest
```

The default suite needs **no Ollama, no Gemini key, no API keys and no
internet**. Everything runs against `FakeAIProvider`, and the two HTTP adapters
are exercised through `httpx.MockTransport` — real adapter code, no sockets.

Three rules hold for every test outside `tests/integration/`:

* **No network.** No test opens a connection to anything.
* **No secrets.** The service key is a literal test value; vendor keys are never
  read from the environment. An autouse fixture clears every settings variable,
  so a developer's real `.env` cannot change a result.
* **Deterministic.** No clock or randomness in an assertion. `FakeAIProvider`
  derives its answer from a hash of the prompt, so the same input gives the same
  output on every machine and every run.

## What is covered

| Area | File |
| --- | --- |
| Settings, model pinning, key rotation, secret handling | `tests/test_config.py` |
| `AnalysisRequest` validation and input bounds | `tests/test_domain_request.py` |
| `TradingDecision` / `DecisionDraft`: BUY, SELL, HOLD, confidence bounds, risk levels, invalid decisions, forged metadata | `tests/test_domain_decision.py` |
| Context builder: bounds, trimming, determinism, content | `tests/test_context_builder.py` |
| Prompt versioning, registry, rendering, prompt content | `tests/test_prompts.py` |
| `FakeAIProvider`, the provider abstraction, provider selection | `tests/test_providers.py` |
| Ollama and Gemini adapters through a mock transport | `tests/test_provider_http.py` |
| Output validation: fences, prose, malformed JSON, invalid values | `tests/test_output_validation.py` |
| Provider-error translation and credential redaction | `tests/test_error_translation.py` |
| The analysis pipeline: provenance, prompts, cost controls, failures, timeouts | `tests/test_analysis_service.py` |
| Health, authentication, `/analyze`, error envelope, secret exposure | `tests/test_api.py` |

Useful invocations:

```bash
pytest -v                          # verbose
pytest tests/test_api.py           # one file
pytest -k "timeout or redaction"   # by name
```

## Live-provider tests

Live tests carry the `integration` marker and are excluded by the default
pytest options (`-m 'not integration'` in `pyproject.toml`). They are never part
of CI.

### Ollama

Requires a running Ollama and a pulled model. **Does not require internet.**

```bash
ollama serve
ollama pull qwen3:8b

export OLLAMA_MODEL=qwen3:8b
export OLLAMA_THINK=false          # qwen3 is a reasoning model
export AI_SERVICE_API_KEY=any-value-for-this-test
pytest -m integration tests/integration/test_ollama_live.py -v
```

On Windows PowerShell:

```powershell
$env:OLLAMA_MODEL = "qwen3:8b"
pytest -m integration tests/integration/test_ollama_live.py -v
```

`OLLAMA_MODEL` may equally be set in `.env`; the tests read resolved settings,
not just the process environment.

Optional: `OLLAMA_BASE_URL` (default `http://localhost:11434`),
`OLLAMA_TIMEOUT_SECONDS` (overrides the configured budget for the run) and
`OLLAMA_THINK=false` for a reasoning model. On CPU, expect roughly two minutes
per analysis — the whole file takes a few minutes.

What it checks:

1. it connects to the configured `OLLAMA_BASE_URL`;
2. it uses the configured `OLLAMA_MODEL`;
3. it sends one small, bounded analysis request;
4. it receives a response; and
5. that response validates into a `TradingDecision`, with provider, model and
   prompt version stamped, and a non-empty invalidating condition for any
   `BUY`/`SELL`.

The tests **skip** — they do not fail — when `OLLAMA_MODEL` is unset, Ollama is
not running, or the model is not pulled. A developer without Ollama sees no red.

There is also a one-shot script that does the same thing outside pytest and
prints the decision:

```bash
OLLAMA_MODEL=<model> python scripts/verify_ollama.py
```

It exits 0 on success and 1 on failure, so it works as a pre-flight check.

### Gemini

Costs money. It is one request with a bounded output, and it is deliberately
minimal.

```bash
export AI_PROVIDER=gemini
export GEMINI_API_KEY=<secret>
export GEMINI_MODEL=gemini-2.5-flash-lite
pytest -m integration tests/integration/test_gemini_live.py -v
```

It skips unless all three are set. Exactly one billed call is made; the
readiness test in the same file makes none.

Never run the Gemini live test in CI, in a loop, or as a health check.

## Adding tests

* Use `FakeAIProvider` for anything about the pipeline, the API or a decision.
* Use `httpx.MockTransport` for anything about a vendor's wire format.
* Mark anything that touches a real provider with `@pytest.mark.integration`,
  and make it `pytest.skip` — not fail — when its configuration is absent.
* Assert on error **codes** rather than messages; messages are for humans and
  may be reworded.
