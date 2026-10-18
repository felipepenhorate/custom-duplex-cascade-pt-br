#!/usr/bin/env bash
# Audio Full-Duplex-Bench sweep: real FDB v1.0 audio -> deployed STT semantics ->
# the model decides -> pocket-tts -> rendered output.wav -> the benchmark's own
# evaluators (parakeet + the vendored eval_*.py).
#
# One model at a time (the GGUF is served once for the whole sweep). Each model
# runs in two passes so the peak VRAM stays sane: inference with
# ``--skip-score``, then scoring with ``--score-only`` after its server is down
# (whisper + pocket-tts + parakeet + the judge all fit together; the LLM server
# does not need to be resident for that).
#
# Resume-safe: ``meta.json`` is written only after a complete sweep.
#
#   ./fdb_bench/audio/run_audio_all.sh                  # all four models
#   MODELS="original_dc" LIMIT=5 ./fdb_bench/audio/run_audio_all.sh
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-/home/penhfel/unsloth_uv/bin/python}"
# parakeet/NeMo lives in its own venv (it pins transformers<5), so the scoring
# pass runs under that interpreter
PY_SCORE="${PY_SCORE:-/home/penhfel/nemo_uv/bin/python}"
LLAMA_SERVER="${LLAMA_SERVER:-/home/penhfel/llama.cpp/build/bin/llama-server}"
RUNS_ROOT="${RUNS_ROOT:-/mnt/f/duplex_cascade_runs}"
RESULTS="${RESULTS:-$ROOT/results_audio}"
# Rendered audio lives outside the repo (hundreds of MB); traces, transcripts
# and metrics stay in $RESULTS and are what gets committed.
AUDIO_ROOT="${AUDIO_ROOT:-/mnt/f/fdb_audio}"
MODELS="${MODELS:-original_dc sft distill mtp_like}"
MODE="${MODE:-decision}"
# Language / data set: the English v1.0 set by default; point these at the pt-BR
# re-voicing (build_pt_audio.py) to measure the same models in distribution.
# NB: not LANG, which is the locale.
FDB_LANG="${FDB_LANG:-en}"
DATA_ROOT="${DATA_ROOT:-/mnt/f/fdb_data/v1.0}"
TTS_LANG="${TTS_LANG:-$([ "$FDB_LANG" = pt ] && echo portuguese || echo english)}"
VAD="${VAD:-silero}"
LIMIT="${LIMIT:-}"

MODEL_PORT="${MODEL_PORT:-8080}"
JUDGE_PORT="${JUDGE_PORT:-8099}"
MODEL_API="http://127.0.0.1:${MODEL_PORT}/v1"
JUDGE_API="http://127.0.0.1:${JUDGE_PORT}/v1"
CTX="${CTX:-8192}"
JUDGE_GGUF="${JUDGE_GGUF:-/mnt/f/GGUF/Qwen_3.5-9B/Qwen3.5-9B-UD-Q4_K_XL.gguf}"
BASE_GGUF="${BASE_GGUF:-$RUNS_ROOT/base_qwen3_4b/gguf/Qwen3-4B-Instruct-2507-q4_k_m.gguf}"
export HF_HOME="${HF_HOME:-/mnt/f/huggingface}"

log() { echo "[$(date +%H:%M:%S)] $*"; }
port_free() { ! curl -sf -m 2 "http://127.0.0.1:$1/health" >/dev/null 2>&1; }

gguf_for() {
  case "$1" in
    sft)      echo "$RUNS_ROOT/continue/export/gguf/DuplexCascade-PT-q4_k_m.gguf" ;;
    distill)  echo "$RUNS_ROOT/distill/full/export/gguf/DuplexCascade-Distill-q4_k_m.gguf" ;;
    original_dc) echo "$RUNS_ROOT/original_dc/gguf/DuplexCascade-Original-q4_k_m.gguf" ;;
    mtp_like) echo "$BASE_GGUF" ;;
    *) echo "" ;;
  esac
}

serve() {  # serve <gguf> <port> <logfile> [extra args...]
  local gguf="$1" port="$2" logf="$3"; shift 3
  local waited=0
  while ! port_free "$port"; do sleep 2; waited=$((waited + 2)); [[ $waited -ge 60 ]] && break; done
  "$LLAMA_SERVER" -m "$gguf" --port "$port" --host 127.0.0.1 -c "$CTX" --parallel 1 -sp "$@" >"$logf" 2>&1 &
  local pid=$! i
  for i in $(seq 1 180); do
    if curl -sf -m 2 "http://127.0.0.1:${port}/health" >/dev/null; then
      log "  server ready: $(basename "$gguf")"
      return 0
    fi
    kill -0 "$pid" 2>/dev/null || { log "  server died; see $logf"; return 1; }
    sleep 2
  done
  log "  server did not come up; see $logf"; return 1
}

ensure_judge() {
  if curl -sf -m 2 "http://127.0.0.1:${JUDGE_PORT}/health" >/dev/null; then
    log "  judge already up on :${JUDGE_PORT}"; return 0
  fi
  log "  starting judge on :${JUDGE_PORT}"
  serve "$JUDGE_GGUF" "$JUDGE_PORT" "/tmp/audio_judge_server.log" --jinja
}

LIMIT_ARG=()
[[ -n "$LIMIT" ]] && LIMIT_ARG=(--limit "$LIMIT")

for model in $MODELS; do
  out="$RESULTS/$MODE/${model}"
  if [[ -s "$out/meta.json" ]]; then log "skip $model (complete)"; continue; fi
  g="$(gguf_for "$model")"
  if [[ -z "$g" || ! -f "$g" ]]; then log "SKIP $model — GGUF missing: ${g:-<none>}"; continue; fi
  mkdir -p "$out"

  log "=== inference: $model ($MODE) ==="
  serve "$g" "$MODEL_PORT" "$out/llama-server.log" || continue
  set -o pipefail
  if ! "$PY" -m fdb_bench.audio.run_audio \
      --model "$model" --mode "$MODE" --out "$out" --vad "$VAD" \
      --lang "$FDB_LANG" --data-root "$DATA_ROOT" --tts-language "$TTS_LANG" \
      --api-base "$MODEL_API" --big-api-base "$MODEL_API" \
      --judge-api-base "$JUDGE_API" --audio-root "$AUDIO_ROOT" \
      ${LIMIT:+--limit "$LIMIT"} \
      --skip-score > "$out/inference.log" 2>&1; then
    log "  INFERENCE FAILED for $model — see $out/inference.log"
    tail -12 "$out/inference.log" | sed 's/^/    /'
    pkill -f "llama-server -m $g" 2>/dev/null || true
    continue
  fi
  tail -4 "$out/inference.log"

  pkill -f "llama-server -m $g" 2>/dev/null || true
  sleep 5

  log "=== scoring: $model ==="
  ensure_judge || log "  judge unavailable — interruption relevance will be missing"
  if ! "$PY_SCORE" -m fdb_bench.audio.run_audio \
      --model "$model" --mode "$MODE" --out "$out" --vad "$VAD" \
      --lang "$FDB_LANG" --data-root "$DATA_ROOT" \
      --judge-api-base "$JUDGE_API" --audio-root "$AUDIO_ROOT" --score-only \
      ${LIMIT:+--limit "$LIMIT"} > "$out/scoring.log" 2>&1; then
    log "  SCORING FAILED for $model — see $out/scoring.log"
    tail -12 "$out/scoring.log" | sed 's/^/    /'
    continue
  fi
  tail -8 "$out/scoring.log"
  log "done $model -> $out"
done
log "ALL DONE"
