# Providers

## The boundary

Everything inside `app/providers/` may know that Ollama speaks `/api/chat` and
that Gemini wants an `x-goog-api-key` header. Nothing outside it does.

```
                    AIProvider  (abstract)
                         │
      ┌──────────────────┼──────────────────┐
FakeAIProvider     OllamaProvider     GeminiProvider
 (offline)          (local)             (cloud)
```

The rest of the service depends on the interface, never on a vendor. No vendor
response object, vendor exception or SDK type crosses this line — which is what
makes adding Anthropic or OpenAI later one new file plus one registry entry.

## The interface

```python
class AIProvider(ABC):
    name: str

    def model_for(self, depth: AnalysisDepth) -> str: ...
    async def complete(self, request: CompletionRequest) -> CompletionResult: ...
    async def check_readiness(self) -> ProviderReadiness: ...
    async def aclose(self) -> None: ...
```

A provider does one narrow thing: it turns a rendered prompt into text and
reports what that cost. It does not parse decisions, does not know what a
`TradingDecision` is, and does not decide what a failure means over HTTP.

The neutral types:

| Type | Carries |
| --- | --- |
| `CompletionRequest` | `system`, `user`, `max_output_tokens`, `temperature`, `timeout_seconds`, `depth` |
| `CompletionResult` | `text` (raw, untrusted), `provider`, `model`, `latency_ms`, `usage` |
| `ProviderReadiness` | `provider`, `model`, `ready`, `detail` — never runs an inference |
| `TokenUsage` | `prompt_tokens`, `completion_tokens`, `total_tokens`, all optional |

`model_for(depth)` is how cost tiers stay out of the domain: the provider holds
the depth-to-model mapping, so no business rule anywhere names a model.

## Error translation

Adapters never raise a vendor exception or an HTTP concept. They raise one of
these, and `app/analysis/errors.py` decides what each means to a client:

| Provider error | Meaning | HTTP | Code |
| --- | --- | --- | --- |
| `ProviderTimeoutError` | No answer within the budget | 504 | `PROVIDER_TIMEOUT` |
| `ProviderUnavailableError` | Unreachable or 5xx | 503 | `PROVIDER_UNAVAILABLE` |
| `ProviderRateLimitError` | Throttled (may carry `Retry-After`) | 429 | `PROVIDER_RATE_LIMITED` |
| `ProviderAuthError` | *Our* credentials were rejected | 502 | `PROVIDER_AUTH_FAILED` |
| `ProviderResponseError` | Answered, but unusable or invalid | 502 | `PROVIDER_INVALID_RESPONSE` |
| `ProviderModelUnavailableError` | Model missing or not pulled | 503 | `PROVIDER_MODEL_UNAVAILABLE` |
| `ProviderNotConfiguredError` | No key or no model configured | 503 | `PROVIDER_NOT_CONFIGURED` |
| `UnsupportedProviderError` | `AI_PROVIDER` names an unknown provider | 503 | `PROVIDER_UNSUPPORTED` |

503 and 504 mean *try again later*. 502 means *the provider answered and the
answer was unusable* — retrying the identical prompt is unlikely to help.

Every message is scrubbed of credentials when the exception is constructed,
because vendors do echo request URLs back in error bodies and a URL can carry a
key. The message the client sees is generic; the specific reason goes to the
logs, scrubbed again by configured value.

## Selection

`AI_PROVIDER` picks the provider; `app/providers/registry.py` is the only module
that imports an adapter. Selection fails closed — an unknown provider, a missing
key or an unconfigured model raises rather than returning something that quietly
answers with a stub.

```python
PROVIDERS = {"fake": ..., "ollama": OllamaProvider, "gemini": GeminiProvider}
```

## Ollama — the local default

Ollama is the default development provider and needs no internet, no account
and no key. It is an **external** service: it is not in the Docker image, and
production does not run it.

```bash
AI_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:8b                # any instruct model you have pulled
OLLAMA_DEEP_MODEL=                   # optional; falls back to OLLAMA_MODEL
OLLAMA_THINK=false                   # reasoning models only; see below
```

No model name appears in the source. A model baked into code is a model that
silently differs between machines, so `OLLAMA_MODEL` must be set or the provider
refuses to build.

Details:

* The adapter posts to `/api/chat` with `stream=false` and `format="json"`, and
  passes `temperature` and `num_predict` (the output cap) on every call.
* `check_readiness()` calls `/api/tags` — it lists models and runs no
  inference, so it is free to poll. A bare configured name matches the
  `name:latest` that Ollama reports.
* **Reasoning models.** qwen3 and deepseek-r1 emit a thinking chain that this
  service discards but that still spends the output budget — a thinking run can
  exhaust `num_predict` before it writes any JSON, which surfaces as a timeout
  or a truncated response. `OLLAMA_THINK=false` turns it off. The field is sent
  only when configured, because a model without thinking support rejects it.
* HTTP 404 becomes `ProviderModelUnavailableError` ("pull it first"), a
  connection failure becomes `ProviderUnavailableError`, and
  `done_reason=length` becomes `ProviderResponseError` — a truncated JSON object
  is not a decision.

Verify a local install end to end:

```bash
OLLAMA_MODEL=<model> python scripts/verify_ollama.py
```

## Gemini — the cloud provider

Called over its REST API with `httpx`, not the vendor SDK: one fewer dependency
to track, no vendor object can escape the module, and the adapter is fully
testable through `httpx.MockTransport` with no key and no network.

```bash
AI_PROVIDER=gemini
GEMINI_API_KEY=<secret>
GEMINI_MODEL=gemini-2.5-flash-lite     # routine, low-cost analysis
GEMINI_DEEP_MODEL=gemini-2.5-flash     # deeper analysis (depth=DEEP only)
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta
```

**Pinned ids only.** A floating alias (`…-latest`) is rejected at settings load.
A model that moves underneath a prompt version makes a recorded decision
irreproducible, which defeats the point of versioning the prompt.

Details:

* The key travels in the `x-goog-api-key` **header**, never in the URL, so it
  cannot land in a proxy log or an echoed request line.
* `responseMimeType: application/json` and a bounded `maxOutputTokens` keep the
  answer to the payload, not prose around it.
* `check_readiness()` makes **no API call**. A readiness probe that reached
  Gemini would bill the account on every poll of a health dashboard.
* 401/403 becomes `ProviderAuthError` with the body discarded — a rejected-key
  response can quote the key back. `finishReason=MAX_TOKENS`, a safety stop and
  a blocked prompt all become `ProviderResponseError`.

## FakeAIProvider — the test provider

Deterministic, in-process, offline. It opens no socket, needs no key and no
model, and returns the same text for the same prompt on every machine and every
run. Unit tests use it wherever possible, and `AI_PROVIDER=fake` runs the whole
request path with no AI installed.

```python
FakeAIProvider()  # deterministic synthesised answer
FakeAIProvider(responses=[buy_json, sell_json])  # canned; the last one repeats
FakeAIProvider(error=ProviderTimeoutError(...))  # error-translation tests
FakeAIProvider(delay_seconds=0.5)  # timeout tests
provider.calls  # every CompletionRequest it saw
```

The built-in heuristic reads RSI out of the rendered context so that BUY, SELL
and HOLD are all reachable and the output looks like something a model might
have said. It is a stub, not a strategy, and reflects no market view.

## Adding a provider

1. Write `app/providers/<vendor>.py` implementing `AIProvider`, translating
   vendor failures into the neutral errors above and letting no vendor type
   escape.
2. Add it to `PROVIDERS` in `app/providers/registry.py`.
3. Add its settings (key as `SecretStr`, explicit pinned model ids) to
   `app/core/config.py` and `.env.example`.
4. Test it through `httpx.MockTransport`; put anything live behind the
   `integration` marker.

Nothing in `app/domain`, `app/context`, `app/prompts` or `app/analysis` changes.
