#!/usr/bin/env bash
# M3-extra — standard lm-eval benchmarks (no chat template; raw scoring, fair
# across base/distill/sft). Tasks: hellaswag, arc_challenge, mmlu (subset),
# humaneval (pass@1). Results -> logs/eval/bench_<name>/*.json + a summary.
set -e
cd "$(dirname "$0")/.."
mkdir -p logs/eval

PY="/home/penhfel/unsloth_uv/bin/python"
OUT="logs/eval"
# humaneval is excluded by default: it EXECUTES generated code (needs
# HF_ALLOW_CODE_EVAL=1) and chat models score poorly at raw code completion.
# ifeval is excluded by default: generation is ~11 s/prompt (~100 min/model),
# too slow for a 3-model comparison; instruction-following is covered by the
# custom general set (eval_capabilities.py).
TASKS="hellaswag,arc_challenge,mmlu"
ARGS="--batch_size 8 --seed 0 --limit 2000"

for spec in "base:Qwen/Qwen3-4B-Instruct-2507" \
            "distill:/mnt/f/duplex_cascade_runs/distill/full/export/merged_bf16" \
            "sft:/mnt/f/duplex_cascade_runs/continue/export/merged_bf16"; do
  name="${spec%%:*}"; path="${spec#*:}"
  echo "================ $name ================"
  $PY -m lm_eval run --model hf \
      --model_args "pretrained=$path,dtype=bfloat16" \
      --tasks "$TASKS" \
      --num_fewshot 0 \
      $ARGS \
      --output_path "$OUT/bench_$name" \
      > "$OUT/bench_$name.log" 2>&1
done

echo "================ BENCHMARK SUMMARY ================"
for name in base distill sft; do
  echo -n "$name: "
  f=$(find "$OUT/bench_$name" -name "results_*.json" 2>/dev/null | head -1)
  if [ -n "$f" ]; then
    $PY - "$f" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
res = d.get("results", {})
out = []
for t in ("hellaswag", "arc_challenge", "mmlu"):
    if t in res:
        r = res[t]
        acc = r.get("acc_norm,none", r.get("acc,none"))
        out.append(f"{t}={acc}")
print(" | ".join(out))
PYEOF
  else
    echo "no results"
  fi
done