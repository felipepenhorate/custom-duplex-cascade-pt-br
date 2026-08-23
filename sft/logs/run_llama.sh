#!/usr/bin/env bash
# Serves the latest fine-tuned DuplexCascade-PT GGUF from the HDD runs dir.
# Runs live on /mnt/f - see DuplexCascade/README.md "Storage layout".
# Override the model with: RUNS_ROOT=... LLAMA_MODEL=... ./logs/run_llama.sh
RUNS_ROOT="${RUNS_ROOT:-/mnt/f/duplex_cascade_runs}"
LLAMA_MODEL="${LLAMA_MODEL:-$RUNS_ROOT/continue/export/gguf/DuplexCascade-PT-q4_k_m.gguf}"
if [ ! -f "$LLAMA_MODEL" ]; then
  LLAMA_MODEL="$RUNS_ROOT/full/export/gguf/DuplexCascade-PT-q4_k_m.gguf"
fi
exec /home/penhfel/llama.cpp/build/bin/llama-server \
  -m "$LLAMA_MODEL" \
  --port 8080 --host 127.0.0.1 -c 8192 --parallel 1 -sp \
  >/home/penhfel/duplex_cascade/logs/llm.log 2>&1 </dev/null