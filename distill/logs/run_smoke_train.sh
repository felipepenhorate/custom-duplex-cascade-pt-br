#!/usr/bin/env bash
# M2 smoke run of DuplexCascade-Distill training.
#
# Preps a small subset of the M1 output (first N duplex rows + optional anchors)
# then runs train_distill.py for a few steps to validate the two-term loss
# (weighted tag CE + content forward-KL vs the frozen base) on the 4080.
#
# VRAM tip: stop the llama.cpp teacher server (:8082) and any other GPU load
# BEFORE training — a 4B QLoRA run with the extra teacher forward sits at the
# 16 GB ceiling.
#
# Usage:
#   ./logs/run_smoke_train.sh              # N=300 duplex rows + anchors
#   N=500 KL=0.3 ./logs/run_smoke_train.sh # override
RUNS_ROOT="${RUNS_ROOT:-/mnt/f/duplex_cascade_runs}"
N="${N:-300}"
KL="${KL:-0.3}"
TEMP="${TEMP:-1.0}"
TOPK="${TOPK:-32}"

set -e
cd "$(dirname "$0")/.."

DUPLEX=data/teacher_duplex_train.jsonl
SMOKE_DUPLEX=/tmp/opencode/smoke_duplex.jsonl
head -"$N" "$DUPLEX" > "$SMOKE_DUPLEX"

echo "[smoke] prepping ${N} duplex rows (+ anchors if present)"
ANCHOR_ARGS=()
if [ -f data/anchor_prompts.jsonl ]; then
  ANCHOR_ARGS=(--anchors data/anchor_prompts.jsonl)
fi
/home/penhfel/unsloth_uv/bin/python training/prep_dataset.py \
  --duplex "$SMOKE_DUPLEX" "${ANCHOR_ARGS[@]}" \
  --out /tmp/opencode/prepped_smoke --max-seq 2048 --mask-first-tokens 2

echo "[smoke] training (steps=$STEPS, kl=$KL, temp=$TEMP, topk=$TOPK)"
/home/penhfel/unsloth_uv/bin/python training/train_distill.py \
  --dataset /tmp/opencode/prepped_smoke --max-steps "${STEPS:-30}" \
  --per-device-batch-size 1 --grad-accum 8 --max-seq-length 2048 \
  --lr 1e-5 --embedding-lr 2e-5 --warmup-steps 2 --logging-steps 2 \
  --kl-weight "$KL" --teacher-temperature "$TEMP" --top-k "$TOPK" \
  --out-dir "$RUNS_ROOT/distill/smoke"