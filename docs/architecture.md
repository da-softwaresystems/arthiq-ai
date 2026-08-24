# Architecture

## What this service is

One process with one job: turn a structured market snapshot into a validated
trading opinion. It is a standalone service, not a module of the backend, and
not a monolith of its own — there are no internal services, no queues, no
workers and no database.

```
arthiq-backend                          arthiq-ai
──────────────                          ─────────
 user auth (Firebase)      HTTPS         service auth (X-API-Key)
 market data              ─────▶         context builder
 technical analysis                      prompt registry
 paper trading                           AI provider  ──▶ Ollama / Gemini
 system of record         ◀─────         output validation
                        TradingDecision
```

## What it deliberately cannot do

The constraint "AI has no authority to execute trades" is enforced by what this
repository contains, not by a runtime check that could be bypassed:

| Not present | Consequence |
| --- | --- |
| No broker client, no Angel One code | The service cannot place an order or read a position |
| No database driver, no ORM, no migrations | The service cannot read or write the backend's data |
| No Firebase SDK, no user model | The service has no notion of an end user to act on behalf of |
| No `EXECUTE` in the `Decision` enum | The output vocabulary has no instruction to act in it |

CI asserts the first three by failing the build if `app/` grows an import of a
broker, database or user-auth package.

## Layering

```
app/domain      the vocabulary          (knows nothing about anything else)
app/context     request  -> context     (depends on domain)
app/prompts     context  -> prompt text (depends on nothing but itself)
app/providers   prompt   -> raw text    (depends on domain for neutral types)
app/analysis    orchestration + output validation + HTTP router
app/core        configuration, errors, logging, redaction, service auth
```

Dependencies point one way. `app/analysis` knows about providers; no provider
knows about analysis; nothing outside `app/providers` imports `httpx` for a
vendor call or names a vendor at all.

## Domain model

### AnalysisRequest

The service performs no lookups: if a fact is not on the request, it does not
inform the decision. The request carries the instrument (`symbol`, `exchange`,
`interval`), when it applies (`as_of`, `trading_date`), the `price`, the
computed `technical` indicators, optional `market` context, an optional short
series of `recent_closes`, optional structured `observations`, and the
`depth` the caller is willing to pay for.

Four rules make the contract hold up over time:

* **No vendor fields.** Nothing mentions Angel One or a symbol token. How the
  backend stores a candle is not this service's business.
* **No raw history.** There is no candle list. `recent_closes` is capped at 200
  by the schema and trimmed further by configuration.
* **Strict.** Every model sets `extra="forbid"`. An unrecognised key is contract
  drift, and a 422 is a cheaper way to find it than a silently ignored
  indicator. Adding a field is therefore a deliberate, versioned change.
* **Exact numbers.** Prices and indicators are `Decimal`, serialised by the
  backend as JSON strings (`"1316.0000"`), because a JSON float would round a
  price on the way in.

### TradingDecision

The response separates *what the model claimed* from *what the service knows*:

```
DecisionDraft      decision, confidence, risk_level, reasoning,
(model output)     key_factors, invalidating_conditions

DecisionMetadata   provider, model, prompt_name, prompt_version, depth,
(service-owned)    generated_at, request_id, latency_ms, usage
```

`TradingDecision` is the two together. The split is a security boundary: the
draft forbids unknown fields, so a model that emits `"provider": "gemini"` in
its JSON is rejected rather than believed. Provenance is stamped from the
objects that actually did the work.

Two validation rules are controls rather than formalities:

* the decision vocabulary is closed — `BUY`, `SELL`, `HOLD`, and nothing else
  is mapped onto them; and
* a `BUY` or a `SELL` must carry at least one invalidating condition. An
  actionable call with nothing to monitor is refused.

### Confidence is not a probability

`confidence` is the model's self-reported conviction on a 0–1 scale. Nothing
measures it against outcomes. 0.80 does **not** mean four in five such calls
will be profitable, and it must not be used as one input to a probability
calculation. It is an ordering hint between decisions from the same prompt
version and model, and nothing more. A value outside 0–1 (a model answering
`78`) is rejected, never rescaled — guessing at intent would invent a
confidence.

## Context builder

`app/context/builder.py` turns a validated request into the exact text the
model sees. Two properties, both tested independently of any provider:

* **Bounded.** The builder trims `recent_closes` and `observations` to
  configured limits (`CONTEXT_MAX_RECENT_CLOSES`, `CONTEXT_MAX_OBSERVATIONS`),
  keeping the newest, and records how many entries it dropped. A caller cannot
  grow the prompt by sending more history.
* **Deterministic.** No clock, no environment, no randomness, and a stable
  ordering for indicators and observations. The same request renders the same
  bytes, which is what makes a decision reproducible for a given prompt version
  and model.

Absent indicators are omitted, never zeroed: `null` means "not warmed up yet",
the same meaning the backend gives it.

## Prompt versioning

A prompt is an artefact with a `name`, a `version`, a `purpose` and content —
not a string literal inside a service function. `technical_analysis_v1` is the
only prompt in this milestone.

Prompts are immutable. A change in wording is a new version, because a decision
recorded against `technical_analysis_v1` must mean the same thing next month.
Every `TradingDecision` carries the version that produced it, and a caller may
pin one with `prompt_version`; an unknown version is a 422 before any provider
call is made.

The registry (`app/prompts/registry.py`) is the only way to resolve a prompt, so
the set of prompts that can run is enumerable — and is reported by
`GET /internal/v1/provider/readiness`.

## Output validation

No provider text becomes a decision without passing `app/analysis/output.py`.
Two structural accommodations are made — a fenced or prose-wrapped JSON object
is extracted, and `decision`/`risk_level` are upper-cased — and nothing else.
No semantic repair: `"STRONG BUY"` is not mapped onto `BUY`, and a confidence of
`78` is not divided by 100. Anything that does not validate becomes a
`PROVIDER_INVALID_RESPONSE`, which the backend can distinguish from an outage.

## Cost control

The service is designed to be cheap by default and to make expensive behaviour
an explicit choice.

| Control | Where |
| --- | --- |
| Bounded context | `CONTEXT_MAX_*`, enforced by the context builder |
| Bounded output | `AI_MAX_OUTPUT_TOKENS`, sent on every provider call |
| JSON-only responses | Ollama `format=json`, Gemini `responseMimeType` — no tokens spent on prose |
| One call per request | The service has no retry loop, no self-critique pass, no second opinion |
| Fail before spending | Unknown prompt version, invalid request and failed auth all short-circuit before the provider is called |
| Free health checks | `GET /health` touches no provider; Gemini readiness is configuration-only |
| Cheap model by default | `depth=ROUTINE` (the default) uses `GEMINI_MODEL`; `depth=DEEP` uses `GEMINI_DEEP_MODEL` |
| Timeouts | `AI_REQUEST_TIMEOUT_SECONDS`, applied by the adapter and again as a hard backstop |

Depth-to-model is provider configuration, not routing logic: no business rule
anywhere names a model, and there is no automatic escalation. The caller asks
for `DEEP` or it does not.

## Observability

Structured logging (JSON outside development), with a request id that the
backend can supply as `X-Request-ID` so one id spans both services and is
echoed back in the decision's metadata.

A completed analysis logs provider, model, symbol, interval, depth, prompt
version, latency, decision, confidence and token usage. A failure logs the
neutral error code and a scrubbed reason.

Never logged: API keys, authorization headers, and — by default — the prompt or
the provider's response. `LOG_PROMPTS` and `LOG_PROVIDER_RESPONSES` turn those
on at DEBUG for local debugging only, and both are truncated even then.

## Security

* **Service authentication, not user authentication.** The backend
  authenticates the end user; this service authenticates the backend with a
  shared key. See [docs/api.md](api.md).
* **Fails closed.** No `AI_SERVICE_API_KEY` means every internal call is
  refused, not that every caller is trusted.
* **No client-supplied authority.** A user id in a request body would be a
  rejected unknown field, and would grant nothing if it were accepted.
* **Provider credentials are ours alone.** They come from the environment and
  are never accepted from a client, echoed in a response, or written to a log.
* **Redaction by shape and by value.** Every provider error message is scrubbed
  of credentials as it is constructed (by pattern) and again when logged (by
  configured value).
* **Generic client errors.** A client learns a stable error code and whether to
  retry. It never receives a stack trace, a vendor payload or a model id.

## Deliberately not here

Kubernetes, Kafka, Celery, a database, a vector store, RAG, LangGraph, CrewAI,
agent frameworks, and AI routing. M5.1 is the foundation: one prompt, one call,
one validated answer. Anything above may be justified later, on evidence.
