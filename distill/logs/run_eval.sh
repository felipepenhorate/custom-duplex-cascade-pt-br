#!/usr/bin/env bash
# M3 — run the capability + turn-taking evals on the three models (one at a
# time; each bf16 model needs the full 16 GB GPU). Results to logs/eval/.
set -e
cd "$(dirname "$0")/.."
mkdir -p logs/eval

BASE="Qwen/Qwen3-4B-Instruct-2507"
DISTILL="/mnt/f/duplex_cascade_runs/distill/full/export/merged_bf16"
SFT="/mnt/f/duplex_cascade_runs/continue/export/merged_bf16"
PY="/home/penhfel/unsloth_uv/bin/python"
PREPPED="/mnt/f/duplex_cascade_runs/distill/prepped_1024_cached/dataset"
CACHE="/mnt/f/duplex_cascade_runs/distill/prepped_1024_cached"

for spec in "base:$BASE" "distill:$DISTILL" "sft:$SFT"; do
  name="${spec%%:*}"; path="${spec#*:}"
  echo "================ $name ================"
  $PY eval/eval_capabilities.py --model "$path" --name "$name" --max-tokens 800 \
      > "logs/eval/capabilities_$name.json" 2> "logs/eval/capabilities_$name.err"
  $PY eval/eval_turntaking.py --model "$path" --name "$name" \
      --prepped "$PREPPED" --cache-dir "$CACHE" \
      > "logs/eval/turntaking_$name.json" 2> "logs/eval/turntaking_$name.err"
done

echo "================ SUMMARY ================"
for name in base distill sft; do
  echo "--- $name ---"
  cat "logs/eval/capabilities_$name.json" | $PY -c "import json,sys; d=json.load(sys.stdin); print('FC acc:', d['function_calling_accuracy'], '| GEN acc:', d['general_accuracy'])"
  cat "logs/eval/turntaking_$name.json" | $PY -c "import json,sys; d=json.load(sys.stdin); print('tag acc:', d['tag_accuracy'], '| KL drift:', d['kl_drift'])"
done