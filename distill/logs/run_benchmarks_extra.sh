#!/usr/bin/env bash
# M3-extra — gsm8k (5-shot math), truthfulqa_mc2 (factuality), ifeval (sample).
# Each task run separately so limits/fewshot/batch differ. Results -> logs/eval/benchx_<name>/.
set -e
cd "$(dirname "$0")/.."
mkdir -p logs/eval

PY="/home/penhfel/unsloth_uv/bin/python"
OUT="logs/eval"

for spec in "base:Qwen/Qwen3-4B-Instruct-2507" \
            "distill:/mnt/f/duplex_cascade_runs/distill/full/export/merged_bf16" \
            "sft:/mnt/f/duplex_cascade_runs/continue/export/merged_bf16"; do
  name="${spec%%:*}"; path="${spec#*:}"
  echo "================ $name ================"
  # gsm8k: 5-shot CoT, subset 300, generation batch 2
  $PY -m lm_eval run --model hf \
      --model_args "pretrained=$path,dtype=bfloat16" \
      --tasks gsm8k --num_fewshot 5 --limit 300 --batch_size 2 --seed 0 \
      --output_path "$OUT/benchx_$name" \
      > "$OUT/benchx_${name}_gsm8k.log" 2>&1
  # truthfulqa_mc2: 0-shot, full set, loglikelihood
  $PY -m lm_eval run --model hf \
      --model_args "pretrained=$path,dtype=bfloat16" \
      --tasks truthfulqa_mc2 --num_fewshot 0 --batch_size 8 --seed 0 \
      --output_path "$OUT/benchx_$name" \
      > "$OUT/benchx_${name}_tqa.log" 2>&1
  # ifeval: 0-shot, sample 100, generation batch 1
  $PY -m lm_eval run --model hf \
      --model_args "pretrained=$path,dtype=bfloat16" \
      --tasks ifeval --num_fewshot 0 --limit 100 --batch_size 1 --seed 0 \
      --output_path "$OUT/benchx_$name" \
      > "$OUT/benchx_${name}_ifeval.log" 2>&1
done

echo "================ EXTRA SUMMARY ================"
for name in base distill sft; do
  echo -n "$name: "
  $PY - "$name" <<'PYEOF'
import glob, json, sys
name = sys.argv[1]
res = {}
for f in glob.glob(f"logs/eval/benchx_{name}/*/results_*.json"):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    res.update(d.get("results", {}))
out = []
for t, key in (("gsm8k", "exact_match"), ("truthfulqa_mc2", "mc2"), ("ifeval", "inst_level_inst_acc")):
    if t in res:
        r = res[t]
        v = r.get(key + ",none") or r.get(key)
        if t == "ifeval" and v is None:
            v = r.get("inst_level_strict_acc,none")
        out.append(f"{t}={v}")
print(" | ".join(out))
PYEOF
done