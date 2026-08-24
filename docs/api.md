# API

Base URL in development: `http://localhost:8100`. Interactive docs at `/docs`
when `ENABLE_DOCS=true`.

| Method | Path | Auth | Purpose |
| --- | --- | --- | --- |
| `GET` | `/health` | none | Liveness. No provider call. |
| `POST` | `/internal/v1/analyze` | `X-API-Key` | One analysis → one `TradingDecision` |
| `GET` | `/internal/v1/provider/readiness` | `X-API-Key` | Provider status and available prompts. No inference. |

## Authentication

`/internal/v1/*` is **internal**. There is no anonymous route, no user session,
no cookie and no browser flow. The only intended caller is `arthiq-backend`.

```
X-API-Key: <AI_SERVICE_API_KEY>
```

* The key authenticates the **calling service**. The backend authenticates the
  end user with Firebase and never forwards that identity here; this service has
  no notion of a user, and a user id in a request body is a rejected unknown
  field.
* Comparison is constant-time. A rejected call learns only that it was rejected
   — the presented value is never logged or echoed.
* Rotation: `AI_SERVICE_API_KEY` accepts a comma-separated list, so both the old
  and the new key are valid during a rollout.
  `AI_SERVICE_API_KEY=new-key,old-key`
* **Fails closed.** With no key configured, every internal call is refused with
  `503 CONFIGURATION_ERROR`.

Generate a key with:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

## `GET /health`

Liveness only, and deliberately says nothing about any dependency: a container
must not be restarted because Ollama is restarting, and a health probe must
never trigger a billable inference.

```json
{ "status": "ok", "service": "arthiq-ai", "version": "0.1.0", "environment": "development" }
```

## `POST /internal/v1/analyze`

### Request

```json
{
  "symbol": "RELIANCE",
  "exchange": "NSE",
  "interval": "ONE_DAY",
  "as_of": "2026-08-21T10:00:00Z",
  "trading_date": "2026-08-21",
  "price": "1316.0000",
  "technical": {
    "rsi_14": "54.20",
    "ema_20": "1308.40",
    "ema_50": "1297.80",
    "macd": { "line": "4.21", "signal": "3.10", "histogram": "1.11" },
    "atr_14": "18.40"
  }
}
```

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `symbol` | string ≤32 | yes | Upper-cased; `A-Z 0-9 - . & _` only |
| `exchange` | `NSE` \| `BSE` | yes | |
| `interval` | `FIVE_MINUTE` \| `FIFTEEN_MINUTE` \| `ONE_HOUR` \| `ONE_DAY` | yes | Matches the backend's `CandleInterval` |
| `as_of` | datetime | yes | **Must be timezone-aware**; normalised to UTC |
| `trading_date` | date | yes | The exchange session the bar belongs to |
| `price` | decimal string | yes | > 0. Send as a string to keep precision |
| `technical` | object | no | All fields optional; `null` = not warmed up |
| `market` | object | no | `trend`, `index_symbol`, `index_change_percent`, `sector`, `notes` (≤500) |
| `recent_closes` | decimal string[] | no | Oldest first, ≤200; trimmed further by config |
| `observations` | object[] | no | ≤50 of `{observed_at, kind ≤40, summary ≤240}` |
| `depth` | `ROUTINE` \| `DEEP` | no | Default `ROUTINE` (cheaper model) |
| `prompt_version` | string | no | Pin a prompt, e.g. `technical_analysis_v1` |

`technical` accepts `rsi_14`, `sma_20`, `sma_50`, `sma_200`, `ema_20`,
`ema_50`, `macd{line,signal,histogram}`, `atr_14`,
`bollinger{upper,middle,lower}`, `volume_sma_20`.

Two contract rules worth stating plainly:

* **Unknown fields are rejected** (`422`). Every model forbids extras, so adding
  a field is a deliberate change on both sides rather than a silent no-op.
* **Numbers cross the wire as strings.** The service parses them as `Decimal`;
  a JSON float would round a price.

Optional header: `X-Request-ID`. When the backend supplies one, it is used as
the correlation id, returned in `metadata.request_id`, echoed in the response
header, and present on every log line for that request.

### Response — `200`

```json
{
  "decision": "BUY",
  "confidence": 0.78,
  "risk_level": "MEDIUM",
  "reasoning": "Price is holding above both EMA20 and EMA50 and MACD is positive; RSI-14 at 54.20 leaves room before overbought.",
  "key_factors": ["price above EMA50", "MACD line above signal"],
  "invalidating_conditions": ["Daily close below EMA50", "RSI-14 above 75"],
  "metadata": {
    "provider": "gemini",
    "model": "gemini-2.5-flash-lite",
    "prompt_name": "technical_analysis",
    "prompt_version": "technical_analysis_v1",
    "depth": "ROUTINE",
    "generated_at": "2026-08-21T10:00:03.481920Z",
    "request_id": "b1f0c2",
    "latency_ms": 812,
    "usage": { "prompt_tokens": 612, "completion_tokens": 82, "total_tokens": 694 }
  }
}
```

| Field | Meaning |
| --- | --- |
| `decision` | `BUY`, `SELL` or `HOLD`. There is no `EXECUTE`; this service authorises nothing |
| `confidence` | 0–1, self-reported by the model. **Not a calibrated probability** — see below |
| `risk_level` | `LOW`, `MEDIUM` or `HIGH` — the risk of acting on this reading now |
| `reasoning` | ≤2000 characters citing the values that drove the call |
| `key_factors` | ≤6 short phrases |
| `invalidating_conditions` | ≤6 checkable conditions. **Guaranteed non-empty for `BUY`/`SELL`** |
| `metadata` | Stamped by the service, never by the model |

`usage` fields are `null` when the provider does not report them, and
`request_id` is `null` when the caller supplied none.

**On `confidence`:** it is the model's conviction, not a probability of profit.
Nothing calibrates it against outcomes. Use it to order decisions from the same
prompt version and model; do not multiply it into an expected value.

**On `metadata`:** a model cannot forge it. Model output is validated into a
draft that forbids unknown fields, so `"provider": "…"` inside the model's JSON
is rejected rather than believed; the service fills the metadata from the
provider and prompt objects that actually ran.

### Errors

One envelope, always:

```json
{ "error": { "code": "PROVIDER_TIMEOUT", "message": "The AI provider did not respond in time", "details": { "provider": "ollama" } } }
```

| Status | Code | When | Retry? |
| --- | --- | --- | --- |
| 401 | `UNAUTHORIZED` | Missing or wrong `X-API-Key` | No — fix the key |
| 422 | `VALIDATION_ERROR` | Invalid request, unknown field, unknown `prompt_version` | No — fix the request |
| 429 | `PROVIDER_RATE_LIMITED` | Provider throttled us (may send `Retry-After`) | Yes, after backoff |
| 502 | `PROVIDER_INVALID_RESPONSE` | Model output failed validation or was truncated | Rarely — same prompt, same result |
| 502 | `PROVIDER_AUTH_FAILED` | *Our* provider credentials were rejected | No — operator action |
| 503 | `PROVIDER_UNAVAILABLE` | Provider unreachable or 5xx | Yes |
| 503 | `PROVIDER_MODEL_UNAVAILABLE` | Model missing or not pulled | No — operator action |
| 503 | `PROVIDER_NOT_CONFIGURED` / `PROVIDER_UNSUPPORTED` | Misconfigured provider | No — operator action |
| 503 | `CONFIGURATION_ERROR` | No service key configured | No — operator action |
| 504 | `PROVIDER_TIMEOUT` | No answer within `AI_REQUEST_TIMEOUT_SECONDS` | Yes |

A `422` costs nothing: request validation, authentication and prompt resolution
all happen before any provider call.

Error responses never contain a stack trace, a vendor payload, a prompt, a model
id or any credential.

## `GET /internal/v1/provider/readiness`

Reports whether the configured provider could serve a decision, and which prompt
versions this build can run. It never runs an inference, so it is free to poll.

```json
{
  "provider": "ollama",
  "model": "qwen3:8b",
  "ready": true,
  "detail": "model available",
  "default_prompt_version": "technical_analysis_v1",
  "prompt_versions": ["technical_analysis_v1"]
}
```

Useful to the backend at startup: if `ready` is `false`, AI features can be
disabled up front instead of failing per request.

## Backend integration contract

What `arthiq-backend` must do:

1. Authenticate the end user (Firebase) — this service never sees that identity.
2. Build the snapshot from its own market data and technical analysis and send
   it to `POST /internal/v1/analyze` with `X-API-Key` and, ideally, its own
   `X-Request-ID`.
3. Serialise every number as a JSON string, and send `as_of` timezone-aware.
4. Treat the response as advisory. Nothing here executes; the backend decides
   what, if anything, to do — and may ignore any decision.
5. Store `metadata.prompt_version`, `metadata.provider` and `metadata.model`
   alongside any persisted decision. Without them a stored decision cannot be
   explained later.
6. Branch on the error `code`, not on the message. Messages are for humans.
7. Never forward a client-supplied provider key, model name or prompt to this
   service. Provider configuration is deployment configuration.

What this service guarantees in return:

* The response is provider-neutral: switching Ollama to Gemini changes
  `metadata`, and nothing else in the payload shape.
* `decision` is one of exactly three values, and a `BUY`/`SELL` always carries
  at least one invalidating condition.
* One backend request causes at most one provider call.
