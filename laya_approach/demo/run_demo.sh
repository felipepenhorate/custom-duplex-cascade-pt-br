#!/usr/bin/env bash
# Start the Onda Travel demo.
#
#   ./laya_approach/demo/run_demo.sh
#   ./laya_approach/demo/run_demo.sh --mock        # no models, text chat only
#   PORT=8080 ./laya_approach/demo/run_demo.sh     # another port
#   PYTHON=/path/to/venv/bin/python ./laya_approach/demo/run_demo.sh
#
# The launcher picks a known local environment when PYTHON is not set, so it
# does not silently fall back to a system Python without the GPU packages.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [[ -z "${PYTHON:-}" ]]; then
  for candidate in \
    "$ROOT/.venv-unsloth/bin/python" \
    "$ROOT/.venv-demo/bin/python" \
    "$HOME/unsloth_uv/bin/python"; do
    if [[ -x "$candidate" ]]; then
      PYTHON="$candidate"
      break
    fi
  done
fi

# If a venv exists and is not active, use it (or source it for tools that need
# its environment variables).
if [[ -z "${VIRTUAL_ENV:-}" && -z "${PYTHON:-}" && -f "$ROOT/.venv-unsloth/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv-unsloth/bin/activate"
fi

PYTHON="${PYTHON:-python3}"
PORT="${PORT:-7860}"

echo "[onda] python: $PYTHON" >&2
if ! "$PYTHON" -c "import fastapi, uvicorn, numpy, scipy, websockets" >/dev/null 2>&1; then
  echo "Missing demo dependencies for $PYTHON." >&2
  echo "Install them with:" >&2
  echo "  $PYTHON -m pip install -r laya_approach/demo/requirements.txt" >&2
  exit 1
fi

PORT_ARG=(--port "$PORT")
for arg in "$@"; do
  if [[ "$arg" == "--port" || "$arg" == --port=* ]]; then
    PORT_ARG=()
    break
  fi
done
exec "$PYTHON" -m laya_approach.demo.server "${PORT_ARG[@]}" "$@"
