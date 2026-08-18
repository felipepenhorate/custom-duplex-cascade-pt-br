#!/usr/bin/env bash
# M3 - launch the full DuplexCascade-PT stack:
#   1. llama-server   (llama.cpp)   - fine-tuned GGUF  (OpenAI-compatible :8080)
#   2. stt_service.py (faster-whisper)                 (ws :31607)
#   3. tts_service.py (pocket-tts)                     (ws :31608)
#   4. server.py      (bridge / web demo)              (http :31606)
#
# Usage:
#   ./run_services.sh                          # defaults
#   LLAMA_CPP_DIR=/path/to/llama.cpp ./run_services.sh
#   GGUF=.../DuplexCascade-PT-q4_k_m.gguf ./run_services.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-/home/penhfel/unsloth_uv/bin/python}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/llama.cpp}"
LLAMA_SERVER="${LLAMA_SERVER:-$LLAMA_CPP_DIR/build/bin/llama-server}"

# Runs live on the HDD (/mnt/f) to save SSD space - see DuplexCascade/README.md
# "Storage layout" for details. Override with RUNS_ROOT to relocate.
RUNS_ROOT="${RUNS_ROOT:-/mnt/f/duplex_cascade_runs}"

# default GGUF: serve the latest fine-tuned model (continue > full), falls back
# to any gguf under the runs root. Override with GGUF=/path/to/model.gguf
GGUF="${GGUF:-$RUNS_ROOT/continue/export/gguf/DuplexCascade-PT-q4_k_m.gguf}"
if [ ! -f "$GGUF" ]; then
  GGUF="${GGUF:-$RUNS_ROOT/full/export/gguf/DuplexCascade-PT-q4_k_m.gguf}"
fi
if [ ! -f "$GGUF" ]; then
  GGUF="$(ls "$RUNS_ROOT"/continue/export/gguf/*.gguf "$RUNS_ROOT"/full/export/gguf/*.gguf 2>/dev/null | head -1 || true)"
fi
if [ -z "$GGUF" ] || [ ! -f "$GGUF" ]; then
  echo "ERROR: no GGUF found under $RUNS_ROOT. Pass GGUF=/path/to/model.gguf" >&2
  exit 1
fi

LLM_PORT="${LLM_PORT:-8080}"
STT_PORT="${STT_PORT:-31607}"
TTS_PORT="${TTS_PORT:-31608}"
BRIDGE_PORT="${BRIDGE_PORT:-31606}"

echo "== DuplexCascade-PT =="
echo "  GGUF  : $GGUF"
echo "  LLM   : llama-server :$LLM_PORT"
echo "  STT   : :$STT_PORT"
echo "  TTS   : :$TTS_PORT"
echo "  Web   : http://localhost:$BRIDGE_PORT"

pkill -f "llama-server -m $GGUF" 2>/dev/null || true
sleep 1

# 1) llama.cpp server (OpenAI-compatible), serving the fine-tuned model.
#    -sp: emit duplex special tokens (<|user is speaking|>, ...) in completions
"$LLAMA_SERVER" \
  -m "$GGUF" \
  --port "$LLM_PORT" \
  --host 127.0.0.1 \
  -c 8192 \
  --parallel 1 \
  -sp \
  >"$ROOT/logs/llm.log" 2>&1 &
echo "llama-server pid $!  -> logs/llm.log"

# wait for llama-server to be ready
for i in $(seq 1 60); do
  if curl -sf "http://127.0.0.1:$LLM_PORT/v1/models" >/dev/null 2>&1; then
    echo "llama-server ready"
    break
  fi
  sleep 1
done

# 2) STT (faster-whisper, pt)
"$PY" "$ROOT/services/stt_service.py" --port "$STT_PORT" --model medium >"$ROOT/logs/stt.log" 2>&1 &
echo "stt pid $!  -> logs/stt.log"

# 3) TTS (pocket-tts, portuguese)
"$PY" "$ROOT/services/tts_service.py" --port "$TTS_PORT" --language portuguese \
  >"$ROOT/logs/tts.log" 2>&1 &
echo "tts pid $!  -> logs/tts.log"

# 4) bridge / web demo
"$PY" "$ROOT/DuplexCascade/server.py" \
  --port "$BRIDGE_PORT" \
  --stt-ws "ws://127.0.0.1:$STT_PORT" \
  --tts-ws "ws://127.0.0.1:$TTS_PORT" \
  --llm-api-base "http://127.0.0.1:$LLM_PORT/v1" \
  >"$ROOT/logs/bridge.log" 2>&1 &
echo "bridge pid $!  -> logs/bridge.log"

echo
echo "Open the browser demo:  http://localhost:$BRIDGE_PORT"
echo "Stop with: pkill -f llama-server; pkill -f stt_service; pkill -f tts_service; pkill -f DuplexCascade/server.py"