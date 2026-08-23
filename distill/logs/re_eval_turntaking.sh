#!/usr/bin/env bash
# Re-run the turn-taking eval (seeded teacher cache) after the cache finishes.
set -e
cd "$(dirname "$0")/.."

CACHE_PID="$1"
while kill -0 "$CACHE_PID" 2>/dev/null; do sleep 20; done
sleep 5

PY="/home/penhfel/unsloth_uv/bin/python"
PREPPED="/mnt/f/duplex_cascade_runs/distill/prepped_1024_cached/dataset"
CACHE="/mnt/f/duplex_cascade_runs/distill/prepped_1024_cached"
for spec in "base:Qwen/Qwen3-4B-Instruct-2507" \
            "distill:/mnt/f/duplex_cascade_runs/distill/full/export/merged_bf16" \
            "sft:/mnt/f/duplex_cascade_runs/continue/export/merged_bf16"; do
  name="${spec%%:*}"; path="${spec#*:}"
  $PY eval/eval_turntaking.py --model "$path" --name "$name" \
      --prepped "$PREPPED" --cache-dir "$CACHE" \
      > "logs/eval/turntaking_${name}.json" 2> "logs/eval/turntaking_${name}.err"
done
echo "================ TURNTALKING SUMMARY ================"
for name in base distill sft; do
  echo -n "$name: "
  cat "logs/eval/turntaking_$name.json" | $PY -c "import json,sys; d=json.load(sys.stdin); print('tag acc', d['tag_accuracy'], '| KL drift', d['kl_drift'])"
done