#!/usr/bin/env bash
# M2 driver: prep the full 10k+anchors dataset, run the lambda mini-sweep,
# pick the best lambda, then run the full 2k-step training + merged export.
#
# Selection heuristic: score = tag_accuracy - 0.5*kl_mean over the final step of
# each sweep run; the best config must also have tag_accuracy >= 0.60 (protocol
# learned). Higher lambda = stronger anchoring (better retention) as long as the
# tags still learn. The chosen config is written to
# /mnt/f/duplex_cascade_runs/distill/chosen_lambda.json.
#
# GPU must be free (no llama-server, no other training) before launching.
# Usage:
#   ./logs/run_sweep_and_full.sh
# Overrides: LAMBDAS="0.1 0.3 1.0" SWEEP_STEPS=300 FULL_STEPS=2000

set -e
cd "$(dirname "$0")/.."

RUNS_ROOT="${RUNS_ROOT:-/mnt/f/duplex_cascade_runs}"
RUNS="$RUNS_ROOT/distill"
PREPPED="$RUNS/prepped"
LAMBDAS="${LAMBDAS:-0.1 0.3 1.0}"
SWEEP_STEPS="${SWEEP_STEPS:-300}"
FULL_STEPS="${FULL_STEPS:-2000}"
TEMP="${TEMP:-1.0}"
TOPK="${TOPK:-32}"
LR="${LR:-1e-4}"
EMBED_LR="${EMBED_LR:-2e-4}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== [sweep] prepping full dataset (10k duplex + anchors) ==="
mkdir -p "$RUNS"
/home/penhfel/unsloth_uv/bin/python training/prep_dataset.py \
  --duplex data/teacher_duplex_train.jsonl \
  --anchors data/anchor_prompts.jsonl \
  --out "$PREPPED" --max-seq 2048 --mask-first-tokens 2

echo "=== [sweep] lambda mini-sweep: $LAMBDAS ==="
declare -A TAG_ACC KL_MEAN
for L in $LAMBDAS; do
  OUTDIR="$RUNS/sweep/lambda_$L"
  LOG="$RUNS/sweep/lambda_$L.log"
  mkdir -p "$RUNS/sweep"
  echo "--- sweep lambda=$L (steps=$SWEEP_STEPS) -> $LOG"
  /home/penhfel/unsloth_uv/bin/python training/train_distill.py \
    --dataset "$PREPPED" --max-steps "$SWEEP_STEPS" \
    --per-device-batch-size 1 --grad-accum 8 --max-seq-length 2048 \
    --lr "$LR" --embedding-lr "$EMBED_LR" --warmup-steps 10 \
    --logging-steps 20 --save-steps 0 \
    --kl-weight "$L" --teacher-temperature "$TEMP" --top-k "$TOPK" \
    --out-dir "$OUTDIR" > "$LOG" 2>&1
  read TAG KL < <(/home/penhfel/unsloth_uv/bin/python - "$LOG" <<'PYEOF'
import ast, re, sys
path = sys.argv[1]
best = None
for line in open(path, encoding="utf-8"):
    if "tag_accuracy" not in line:
        continue
    m = re.search(r"\{.*'tag_accuracy'.*\}", line)
    if not m:
        continue
    try:
        d = ast.literal_eval(m.group(0))
        if "tag_accuracy" in d and d.get("tag_accuracy") is not None:
            best = (float(d["tag_accuracy"]), float(d.get("kl_mean", 1.0)))
    except Exception:
        pass
if best is None:
    print("0 1.0")
else:
    print(best[0], best[1])
PYEOF
)
  TAG_ACC[$L]="$TAG"
  KL_MEAN[$L]="$KL"
  echo "   lambda=$L: tag_accuracy=$TAG kl_mean=$KL"
done

echo "=== [sweep] picking best lambda ==="
CHOSEN=""
BEST_SCORE=-999
for L in $LAMBDAS; do
  TAG="${TAG_ACC[$L]}"
  KL="${KL_MEAN[$L]}"
  # skip configs that did not learn the protocol
  if awk "BEGIN{exit !($TAG >= 0.60)}"; then
    SCORE=$(awk "BEGIN{print $TAG - 0.5*$KL}")
    echo "   candidate lambda=$L: score=$SCORE"
    if awk "BEGIN{exit !($SCORE > $BEST_SCORE)}"; then
      BEST_SCORE="$SCORE"
      CHOSEN="$L"
    fi
  else
    echo "   candidate lambda=$L: SKIPPED (tag_accuracy=$TAG < 0.60)"
  fi
done
if [ -z "$CHOSEN" ]; then
  echo "!!! no sweep config passed the tag_accuracy>=0.60 gate; defaulting to 0.3"
  CHOSEN="0.3"
fi
echo "{\"lambda\": \"$CHOSEN\", \"score\": \"$BEST_SCORE\", \"tag_accuracy\": \"${TAG_ACC[$CHOSEN]}\", \"kl_mean\": \"${KL_MEAN[$CHOSEN]}\"}" \
  > "$RUNS/chosen_lambda.json"
echo "=== [full] chosen lambda=$CHOSEN; full run steps=$FULL_STEPS ==="
mkdir -p "$RUNS/full"

/home/penhfel/unsloth_uv/bin/python training/train_distill.py \
  --dataset "$PREPPED" --max-steps "$FULL_STEPS" \
  --per-device-batch-size 1 --grad-accum 16 --max-seq-length 2048 \
  --lr "$LR" --embedding-lr "$EMBED_LR" --warmup-steps 20 \
  --logging-steps 50 --save-steps 500 --export-merged \
  --kl-weight "$CHOSEN" --teacher-temperature "$TEMP" --top-k "$TOPK" \
  --out-dir "$RUNS/full" > "$RUNS/full/full.log" 2>&1

echo "=== [full] done: $RUNS/full (adapter + merged_16bit + train_cfg.json) ==="
echo "[full] log: $RUNS/full/full.log"