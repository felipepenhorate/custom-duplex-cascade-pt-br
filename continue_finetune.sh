#!/usr/bin/env bash
# M3-continue - regenerate a longer-response training set and CONTINUE the
# fine-tune of DuplexCascade-PT so the model learns both short and long
# assistant utterances.
#
# Steps:
#   1. Serve Gemma 4 E4B on :8081 (logs/run_gemma.sh) and generate dialogues
#      with a mix of short + long assistant turns.
#   2. Merge the new long dialogues with the original M1 short ones.
#   3. Build the duplex micro-turn dataset (stage 2) with VARIABLE-length
#      assistant chunks (system-chunk-min=10, system-chunk-max=48). The
#      original chunker fixed every assistant micro-turn at 10 tokens, which
#      trained the model to always speak in ~10-token bursts; the variable
#      chunker lets ~16% of assistant turns be >15 tokens so the model learns
#      longer utterances while keeping the turn-taking pattern.
#   4. Tokenize + label it (stage 3).
#   5. Continue training from the previous merged model (full run), NOT from
#      the base - the special-token embeddings and turn-taking are already
#      learned and stay intact.
#   6. Export merged bf16 + GGUF (q4_k_m) so llama-server can serve it.
#
# Usage:
#   ./continue_finetune.sh                      # defaults below
#   FORCE_DATA=1 ./continue_finetune.sh         # rebuild data with new chunking
#   MAX_STEPS=2000 ./continue_finetune.sh       # more training steps
#
# Requires: logs/run_gemma.sh already serving Gemma on :8081
#           (./logs/run_gemma.sh &)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-/home/penhfel/unsloth_uv/bin/python}"
LLAMACPP="${LLAMACPP:-$HOME/llama.cpp}"

# Runs live on the HDD (/mnt/f) to save SSD space - see DuplexCascade/README.md
# "Storage layout" for details. Override with RUNS_ROOT to relocate.
RUNS_ROOT="${RUNS_ROOT:-/mnt/f/duplex_cascade_runs}"

N_LONG="${N_LONG:-5000}"          # new long/mixed dialogues to generate
MAX_STEPS="${MAX_STEPS:-1000}"    # continuation training steps
WORKERS="${WORKERS:-4}"
API_BASE="${API_BASE:-http://127.0.0.1:8081/v1}"
OUT_RUN="${OUT_RUN:-$RUNS_ROOT/continue}"
SKIP_DATA="${SKIP_DATA:-0}"       # 1 = reuse existing data (dialogue/prep done)

echo "== DuplexCascade-PT continuation (longer responses) =="
echo "  API base (Gemma): $API_BASE"
echo "  new dialogues : $N_LONG"
echo "  train steps   : $MAX_STEPS"
echo "  out run dir   : $OUT_RUN"
echo "  skip data gen : $SKIP_DATA"

if [ "${SKIP_DATA:-0}" != "1" ] && [ -f "$ROOT/data/prepped_continue/dataset_dict.json" ]; then
  echo "  prepped data already exists - set SKIP_DATA=1 to reuse it"
fi

# --- 1) generate longer dialogues with Gemma --------------------------------
if [ "${SKIP_DATA:-0}" = "1" ] && [ "${FORCE_DATA:-0}" != "1" ]; then
  echo "[1/6] SKIPPING generation (using existing data)"
else
  echo "[1/6] generating $N_LONG long/mixed dialogues via Gemma..."
  "$PY" "$ROOT/data/build_long_dialogues.py" \
    --api-base "$API_BASE" \
    --n-dialogues "$N_LONG" \
    --workers "$WORKERS" \
    --out "$ROOT/data/dialogues_long_pt.jsonl"
fi

# --- 2) combine with the original short dialogues ---------------------------
if [ "${SKIP_DATA:-0}" = "1" ] && [ "${FORCE_DATA:-0}" != "1" ]; then
  echo "[2/6] SKIPPING combine (using existing data)"
else
  echo "[2/6] combining old short + new long dialogues..."
  COMBINED="$ROOT/data/dialogues_combined.jsonl"
  : > "$COMBINED"
  if [ -f "$ROOT/data/dialogues_pt.jsonl" ]; then
    cat "$ROOT/data/dialogues_pt.jsonl" >> "$COMBINED"
  fi
  cat "$ROOT/data/dialogues_long_pt.jsonl" >> "$COMBINED"
  echo "  combined dialogues: $(wc -l < "$COMBINED")"
fi

# --- 3) build the duplex micro-turn dataset ---------------------------------
# Variable-length system chunks (default now) let the model learn to speak
# longer utterances. Set FORCE_DATA=1 to rebuild data with the new chunking.
if [ "${SKIP_DATA:-0}" = "1" ] && [ "${FORCE_DATA:-0}" != "1" ]; then
  echo "[3/6] SKIPPING duplex build (using existing data)"
else
  echo "[3/6] building duplex micro-turns (variable-length chunks)..."
  "$PY" "$ROOT/data/build_duplex_dataset.py" \
    --dialogues "$COMBINED" \
    --out "$ROOT/data/duplex_train_continue.jsonl" \
    --system-chunk-min "${SYSTEM_CHUNK_MIN:-10}" \
    --system-chunk-max "${SYSTEM_CHUNK_MAX:-48}"
fi

# --- 4) tokenize + labels ---------------------------------------------------
if [ "${SKIP_DATA:-0}" = "1" ] && [ "${FORCE_DATA:-0}" != "1" ]; then
  echo "[4/6] SKIPPING prep (using existing data)"
else
  echo "[4/6] preparing (tokenize + labels)..."
  "$PY" "$ROOT/training/prep_dataset.py" \
    --duplex "$ROOT/data/duplex_train_continue.jsonl" \
    --out "$ROOT/data/prepped_continue" \
    --max-seq "${MAX_SEQ:-2048}"
fi

# --- 5) continue the fine-tune from the merged model -------------------------
# Base = previous full run's merged bf16 model. n_added==0 (specials already
# in the tokenizer) so the trained special embeddings are preserved.
BASE_MERGED="$RUNS_ROOT/full/export/merged_bf16"

# Free VRAM: QLoRA training of a 4B model needs most of the 16GB card. The
# Gemma generator server and the DuplexCascade llama-server (if the demo is
# running) must be stopped, or training segfaults on OOM.
if [ "${SKIP_DATA:-0}" != "1" ]; then
  echo "[5/6] freeing VRAM (stopping llama-servers on :8080 and :8081)..."
  pkill -f "llama-server -m /home/penhfel/Models/Gemma_4_E4B" 2>/dev/null || true
  pkill -f "llama-server -m $RUNS_ROOT/full/export" 2>/dev/null || true
  pkill -f "llama-server -m $RUNS_ROOT/continue/export" 2>/dev/null || true
  sleep 3
fi
echo "[5/6] continuing fine-tune from $BASE_MERGED ($MAX_STEPS steps)..."
# expandable_segments reduces CUDA memory fragmentation over long runs
# (a 4B QLoRA run sits right at the 16GB ceiling - the OOM at step ~746 was
# a fragmented-allocation failure, not a single oversized example)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"$PY" "$ROOT/training/train_qlora.py" \
  --dataset "$ROOT/data/prepped_continue" \
  --model "$BASE_MERGED" \
  --out-dir "$OUT_RUN" \
  --max-steps "$MAX_STEPS" \
  --per-device-batch-size 1 \
  --grad-accum 16 \
  --max-seq-length 2048 \
  --logging-steps 50

# --- 5b) export: merge the NEW adapter onto the OLD merged model ------------
echo "[5b/6] merging new adapter onto previous merged model..."
"$PY" "$ROOT/training/export_model.py" \
  --adapter "$OUT_RUN/adapter" \
  --base-model "$BASE_MERGED" \
  --out "$OUT_RUN/export"

# --- 6) export GGUF (q4_k_m) for llama-server -------------------------------
# The llama.cpp build may not include llama-quantize (the cmake target list is
# sometimes trimmed). Build it on demand so conversion never fails silently.
if [ ! -x "$LLAMACPP/build/bin/llama-quantize" ]; then
  echo "[6/6] llama-quantize not built - building it now..."
  cmake --build "$LLAMACPP/build" --config Release -j 16 --target llama-quantize
fi
echo "[6/6] exporting GGUF..."
"$PY" "$ROOT/training/export_gguf.py" \
  --merged "$OUT_RUN/export/merged_bf16" \
  --llamacpp "$LLAMACPP" \
  --out "$OUT_RUN/export/gguf" \
  --quant q4_k_m

echo
echo "Done! New GGUF: $OUT_RUN/export/gguf/DuplexCascade-PT-q4_k_m.gguf"
echo "Serve it with:"
echo "  GGUF=$OUT_RUN/export/gguf/DuplexCascade-PT-q4_k_m.gguf ./run_services.sh"