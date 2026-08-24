# ---------------------------------------------------------------------------
# Arthiq AI service
#
# Deliberately NOT in this image: Ollama, any model weights, any API key, and
# any broker or database client. Ollama is an external service in local
# development; in production the provider is reached over HTTPS with a key
# supplied by the environment at run time.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first so application edits do not invalidate the layer.
COPY pyproject.toml README.md ./
COPY app ./app
# setuptools leaves build/ and *.egg-info behind in the source directory; the
# package is already installed into site-packages by then.
RUN pip install --upgrade pip && pip install . && rm -rf build ./*.egg-info

COPY scripts ./scripts

RUN chmod +x scripts/*.sh \
    && useradd --create-home --uid 1000 arthiq \
    && chown -R arthiq:arthiq /app
USER arthiq

EXPOSE 8100

# Liveness only: no provider call, so a health probe never bills anyone and an
# unavailable model never restarts the container.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS http://localhost:8100/health || exit 1

CMD ["./scripts/start.sh"]
