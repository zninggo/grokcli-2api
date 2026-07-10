#!/usr/bin/env bash
# Start grokcli-2api on Linux / macOS
set -euo pipefail
cd "$(dirname "$0")"

# Optional overrides:
#   export GROK2API_PORT=3000
#   export GROK2API_HOST=0.0.0.0          # listen all interfaces (server)
#   export GROK2API_OPEN_BROWSER=0        # headless: no browser
#   export GROK2API_ADMIN_PASSWORD='...'  # skip first-run UI password
#   export GROK2API_DEFAULT_MODEL=grok-4.5
#   export GROK2API_DATA_DIR=./data

if ! command -v python3 >/dev/null 2>&1 && ! command -v python >/dev/null 2>&1; then
  echo "ERROR: python3 not found. Install Python 3.10+ first." >&2
  exit 1
fi

PY=python3
command -v python3 >/dev/null 2>&1 || PY=python

if ! $PY -c "import fastapi, uvicorn, httpx" 2>/dev/null; then
  echo "Installing dependencies..."
  $PY -m pip install -r requirements.txt
fi

if ! $PY -c "import curl_cffi, requests" 2>/dev/null; then
  echo "Installing remaining dependencies..."
  $PY -m pip install -r requirements.txt
fi

# Vendored registration package path
export PYTHONPATH="$(pwd)/grok-build-auth${PYTHONPATH:+:$PYTHONPATH}"

# Sensible defaults for servers (multi-account pool)
export GROK2API_OPEN_BROWSER="${GROK2API_OPEN_BROWSER:-0}"
export GROK2API_HOST="${GROK2API_HOST:-127.0.0.1}"
export GROK2API_PORT="${GROK2API_PORT:-3000}"
export GROK2API_ACCOUNT_MODE="${GROK2API_ACCOUNT_MODE:-round_robin}"
export GROK2API_TOKEN_MAINTAIN="${GROK2API_TOKEN_MAINTAIN:-1}"

PORT="$GROK2API_PORT"
echo "Starting grokcli-2api..."
echo "  Admin:  http://127.0.0.1:${PORT}/admin"
echo "  Health: http://127.0.0.1:${PORT}/health"
echo "  OpenAI: http://127.0.0.1:${PORT}/v1"
echo "  Account mode: ${GROK2API_ACCOUNT_MODE}"
echo ""
echo "  Auth (standalone, no local Grok CLI):"
echo "    1) Admin → 设备码登录 (native OIDC) — open URL on phone, enter code"
echo "    2) Import JWT / auth.json (merge on) → multi-account pool"
echo "    3) Mode round_robin rotates accounts; 401/429 auto-failover"
echo ""

exec $PY app.py
