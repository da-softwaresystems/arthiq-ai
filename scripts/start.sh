#!/usr/bin/env sh
# Container entrypoint.
#
# No migrations, no database, no broker session: this service starts a web
# server and nothing else. UVICORN_ARGS lets a local compose run add --reload
# without a second image.
set -eu

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8100}"

echo "Starting arthiq-ai on ${HOST}:${PORT} (provider: ${AI_PROVIDER:-ollama})"

# shellcheck disable=SC2086
exec uvicorn app.main:app --host "${HOST}" --port "${PORT}" ${UVICORN_ARGS:-}
