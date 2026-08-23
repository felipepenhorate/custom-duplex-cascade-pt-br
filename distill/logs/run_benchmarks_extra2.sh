#!/usr/bin/env bash
# M3-extra (part 2): IFEval sample for base (gsm8k+tqa already done), and the
# full extra suite for distill/sft. Results -> logs/eval/benchx_<name>/.
set -e
cd "$(dirname "$0")/.."
mkdir -p logs/eval
PY="/home/penhfel/unsloth_uv/bin/python"
OUT="logs/eval"
IFE_LIMIT="${IFE_LIMIT:-40}"

DISTILL="/mnt/f/duplex_cascade_runs/distill/full/export/merged_bf16"
SFT="/mnt/f/duplex_cascade_runs/continue/export/merged_bf16"

run_ifeval() {
  local name="$1" path="$2"
  $PY -m lm_eval run --model hf \
      --model_args "pretrained=$path,dtype=bfloat16" \
      --tasks ifeval --num_fewshot 0 --limit "$IFE_LIMIT" --batch_size 1 --seed 0 \
      --output_path "$OUT/benchx_$name" \
      > "$OUT/benchx_${name}_ifeval.log" 2>&1
}

echo "=== base: ifeval (sample) ==="
run_ifeval base "Qwen/Qwen3-4B-Instruct-2507"

for spec in "distill:$DISTILL" "sft:$SFT"; do
  name="${spec%%:*}"; path="${spec#*:}"
  echo "=== $name: gsm8k + truthfulqa + ifeval ==="
  $PY -m lm_eval run --model hf \
      --model_args "pretrained=$path,dtype=bfloat16" \
      --tasks gsm8k --num_fewshot 5 --limit 300 --batch_size 2 --seed 0 \
      --output_path "$OUT/benchx_$name" \
      > "$OUT/benchx_${name}_gsm8k.log" 2>&1
  $PY -m lm_eval run --model hf \
      --model_args "pretrained=$path,dtype=bfloat16" \
      --tasks truthfulqa_mc2 --num_fewshot 0 --batch_size 8 --seed 0 \
      --output_path "$OUT/benchx_$name" \
      > "$OUT/benchx_${name}_tqa.log" 2>&1
  run_ifeval "$name" "$path"
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
        res.update(json.load(open(f)).get("results", {}))
    except Exception:
        pass
def pick(r, prefixes):
    for k, v in r.items():
        for p in prefixes:
            if k.startswith(p):
                return v
    return None
out = []
g = res.get("gsm8k", {})
out.append(f"gsm8k={pick(g, ['exact_match,strict-match','exact_match,flexible-extract'])}")
t = res.get("truthfulqa_mc2", {})
out.append(f"tqa_mc2={pick(t, ['acc,'])}")
i = res.get("ifeval", {})
inst = pick(i, ["inst_level_inst_acc,"])
out.append(f"ifeval_inst={inst}")
print(" | ".join(out))
PYEOF
done