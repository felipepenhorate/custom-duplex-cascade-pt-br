#!/usr/bin/env bash
# End-to-end Full-Duplex-Bench run for the three trained variants.
#
#   ./fdb_bench/run_all.sh                 # full sweep, pt-BR + en
#   LIMIT=3 ./fdb_bench/run_all.sh         # fast smoke (3 scenarios/subset)
#   MODELS="distill" LANGS="pt" ./fdb_bench/run_all.sh
#
# Phases, per model:
#   1. serve its GGUF with llama-server (-sp is mandatory: it is what makes the
#      duplex special tokens come back from /v1/completions)
#   2. run every scenario through the simulator  -> results/<lang>/<model>/
#   3. restart llama-server on the judge GGUF and score the interruptions
# Phase 3 (aggregation) runs once at the end over every model.
#
# Everything is sequential on purpose: one 4B model plus the 9B judge would not
# both fit in 16 GB of VRAM alongside their KV caches.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
PY="${PY:-/home/penhfel/unsloth_uv/bin/python}"
LLAMA_DIR="${LLAMA_DIR:-/home/penhfel/llama.cpp}"
LLAMA_SERVER="$LLAMA_DIR/build/bin/llama-server"

RUNS_ROOT="${RUNS_ROOT:-/mnt/f/duplex_cascade_runs}"
RESULTS="${RESULTS:-$ROOT/results}"
LIMIT="${LIMIT:-}"                       # empty = FDB v1.0 sizes (727/subset set)
LANGS="${LANGS:-pt en}"
MODELS="${MODELS:-sft distill mtp_like base}"
WPS="${WPS:-3.25}"                       # calibrated clock, see clock.py
PRIME_MODE="${PRIME_MODE:-decision}"
# Named system-prompt configuration per variant (see adapters.PROMPT_CANDIDATES
# and fdb_bench/PROMPTS.md). Leave empty to use each variant's calibrated
# default; set PROMPT_CANDIDATE to force one for all variants (used by
# calibrate_prompts.py).
PROMPT_CANDIDATE="${PROMPT_CANDIDATE:-}"
SEED="${SEED:-1234}"
CTX="${CTX:-8192}"
# judge
JUDGE_GGUF="${JUDGE_GGUF:-/mnt/f/GGUF/Qwen_3.5-9B/Qwen3.5-9B-UD-Q4_K_XL.gguf}"
JUDGE_PORT="${JUDGE_PORT:-8099}"
JUDGE_API="http://127.0.0.1:${JUDGE_PORT}/v1"
MODEL_PORT="${MODEL_PORT:-8098}"
MODEL_API="http://127.0.0.1:${MODEL_PORT}/v1"
BASE_GGUF="${BASE_GGUF:-$RUNS_ROOT/base_qwen3_4b/gguf/Qwen3-4B-Instruct-2507-q4_k_m.gguf}"

LIMIT_ARG=()
[[ -n "$LIMIT" ]] && LIMIT_ARG=(--limit "$LIMIT")

log() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

port_free() {  # port_free <port> — no listener once nothing is accepting
  ! curl -sf -m 2 "http://127.0.0.1:$1/health" > /dev/null 2>&1
}

serve() {  # serve <gguf> <port> <logfile> [extra args...]
  local gguf="$1" port="$2" logf="$3"; shift 3
  # the judge is a stock chat model: it has no duplex special tokens, so -sp
  # (and its required flag) would be meaningless there
  local ctx="${CTX}"
  local spflag="-sp"
  [[ "$gguf" == *Qwen3.5-9B* ]] && spflag=""

  local attempt
  for attempt in 1 2; do
    pkill -f "llama-server -m $gguf" 2>/dev/null || true
    # Wait for the previous server to actually release the port. Without this,
    # llama-server can exit with "couldn't bind HTTP server socket" and the
    # whole run is silently truncated (this happened to distill pt at 482/727).
    local waited=0
    while ! port_free "$port"; do
      sleep 2; waited=$((waited + 2))
      [[ $waited -ge 60 ]] && break
    done

    "$LLAMA_SERVER" -m "$gguf" --port "$port" --host 127.0.0.1 \
      -c "$ctx" --parallel 1 $spflag "$@" > "$logf" 2>&1 &
    local pid=$!
    local i
    for i in $(seq 1 120); do
      if curl -sf -m 2 "http://127.0.0.1:${port}/health" > /dev/null; then
        echo "  llama-server ready (pid $pid, port $port, try $attempt): $(basename "$gguf")"
        return 0
      fi
      if ! kill -0 "$pid" 2>/dev/null; then
        echo "  llama-server died during startup (try $attempt); see $logf" >&2
        break
      fi
      sleep 2
    done
    [[ $attempt -eq 1 ]] && echo "   retrying after a port/server failure ..."
  done
  echo "  ERROR: llama-server did not come up; see $logf" >&2
  tail -20 "$logf" >&2
  return 1
}

gguf_for() {  # gguf_for <model-key>
  case "$1" in
    sft)      echo "$RUNS_ROOT/continue/export/gguf/DuplexCascade-PT-q4_k_m.gguf" ;;
    sft_m2)   echo "$RUNS_ROOT/full/export/gguf/DuplexCascade-PT-q4_k_m.gguf" ;;
    distill)  echo "$RUNS_ROOT/distill/full/export/gguf/DuplexCascade-Distill-q4_k_m.gguf" ;;
    base)     echo "$BASE_GGUF" ;;
    mtp_like) echo "$BASE_GGUF" ;;   # content model; the companion runs in-process
    original_dc)  echo "$RUNS_ROOT/original_dc/gguf/DuplexCascade-Original-q4_k_m.gguf" ;;
    original_dc_hf) echo "" ;;       # in-process only; see original_dc.py
    *) echo "" ;;
  esac
}

for lang in $LANGS; do
  for model in $MODELS; do
    # primary (per-variant default) -> <model>; explicit override -> candidate
    if [[ -n "$PROMPT_CANDIDATE" ]]; then
      out="$RESULTS/$lang/${model}-${PROMPT_CANDIDATE}"
    else
      out="$RESULTS/$lang/${model}"
    fi
    # meta.json is written only after the LAST scenario, so it — not raw.jsonl —
    # is the completion marker. Testing raw.jsonl silently skipped runs that had
    # been interrupted midway.
    if [[ -s "$out/meta.json" ]]; then
      log "skip $model/$lang (complete: meta.json present)"
      continue
    fi
    mkdir -p "$out"
    g="$(gguf_for "$model")"
    if [[ -z "$g" || ! -f "$g" ]]; then
      log "SKIP $model/$lang — GGUF not found: ${g:-<none>}"
      continue
    fi

    log "inference: $model / $lang"
    serve "$g" "$MODEL_PORT" "$out/llama-server.log" || { rm -rf "$out"; continue; }
    # mtp_like's content model is the SAME stock GGUF as the `base` control.
    # One server is enough: the companion (a 0.8B tag classifier) runs
    # in-process and only the content LLM goes over HTTP. An earlier version
    # started a second server for it, which both doubled the VRAM and picked
    # port 8099 — the judge's port.
    big_api="$MODEL_API"

    "$PY" -m fdb_bench.run_inference \
      --model "$model" --lang "$lang" --out "$out" \
      --api-base "$MODEL_API" --big-api-base "$big_api" \
      --prime-mode "$PRIME_MODE" --seed "$SEED" \
      --words-per-second "$WPS" "${LIMIT_ARG[@]}" \
      ${PROMPT_CANDIDATE:+--prompt-candidate "$PROMPT_CANDIDATE"} \
      2>&1 | tee "$out/inference.log" | tail -20

    pkill -f "llama-server -m $g" 2>/dev/null || true
    sleep 3
    local waited=0
    while ! port_free "$MODEL_PORT"; do sleep 2; waited=$((waited + 2)); [[ $waited -ge 60 ]] && break; done

    # ---- judge phase: score the interruption responses -------------------
    log "judge: $model / $lang"
    serve "$JUDGE_GGUF" "$JUDGE_PORT" "$out/judge-server.log" --jinja || {
      echo "  judge failed to start; interruption rating will be missing" >&2
      continue
    }
    "$PY" -m fdb_bench.run_judge \
      --results "$out" --api-base "$JUDGE_API" --lang "$lang" \
      2>&1 | tee "$out/judge.log" | tail -8
    pkill -f "llama-server -m $JUDGE_GGUF" 2>/dev/null || true
    sleep 2
  done
done

for lang in $LANGS; do
  [[ -d "$RESULTS/$lang" ]] || continue
  log "metrics: $lang"
  "$PY" -m fdb_bench.run_metrics --results "$RESULTS/$lang" --words-per-second "$WPS"
done

# cross-language / cross-variant view (pt vs en transfer, tag diagnostics)
if [[ -f "$RESULTS/pt/summary.json" && -f "$RESULTS/en/summary.json" ]]; then
  log "comparison"
  "$PY" -m fdb_bench.compare --results "$RESULTS" --markdown --out "$RESULTS/comparison.md"
fi

log "done — see $RESULTS/<lang>/summary.md and $RESULTS/comparison.md"
