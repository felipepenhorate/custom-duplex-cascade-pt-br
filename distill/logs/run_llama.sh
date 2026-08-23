#!/usr/bin/env bash
# Serves the distilled DuplexCascade-Distill GGUF from the HDD runs dir.
# Override the model with: RUNS_ROOT=... LLAMA_MODEL=... ./logs/run_llama.sh
RUNS_ROOT="${RUNS_ROOT:-/mnt/f/duplex_cascade_runs}"
LLAMA_MODEL="${LLAMA_MODEL:-$RUNS_ROOT/distill/full/export/gguf/DuplexCascade-Distill-q4_k_m.gguf}"
if [ ! -f "$LLAMA_MODEL" ]; then
  LLAMA_MODEL="$(ls "$RUNS_ROOT"/distill/full/export/gguf/*.gguf 2>/dev/null | head -1)"
fi
exec /home/penhfel/llama.cpp/build/bin/llama-server \
  -m "$LLAMA_MODEL" \
  --port 8080 --host 127.0.0.1 -c 8192 --parallel 1 -sp \
  --repeat-penalty 1.15 \
  >/home/penhfel/github/duplex_cascade_distill/logs/llm.log 2>&1 </dev/null