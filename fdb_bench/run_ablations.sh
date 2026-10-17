#!/usr/bin/env bash
# Ablation runs for the FDB section (see FINDINGS.md -> Ablations, and
# article/main.tex section IX). Writes results_ablation/, which is committed.
#
# Arms:
#   A1  chunker   : whole sentences instead of training-sized ASR pieces (SFT, pt)
#   A2  turn len  : multi-sentence turns instead of training-sized turns (SFT, pt)
#   A3  companion : v5 (7 tags, no promptable-controller augmentations) (pt)
# Each arm differs from the main run in exactly one factor.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
cd "$ROOT"

LS="/home/penhfel/llama.cpp/build/bin/llama-server"
PY="/home/penhfel/unsloth_uv/bin/python"
RUNS=/mnt/f/duplex_cascade_runs
PORT=8097
OUT=results_ablation

SFT_GGUF="$RUNS/continue/export/gguf/DuplexCascade-PT-q4_k_m.gguf"
MTP_V5="$RUNS/mtp_like/runs/final_v5/merged"

free_port() {
  local waited=0
  while curl -sf -m 2 "http://127.0.0.1:$1/health" >/dev/null 2>&1; do
    sleep 2; waited=$((waited + 2)); [ "$waited" -ge 60 ] && break
  done
}

serve_sft() {
  pkill -f "llama-server -m $SFT_GGUF" 2>/dev/null || true
  free_port "$PORT"
  $LS -m "$SFT_GGUF" --port "$PORT" --host 127.0.0.1 -c 8192 --parallel 1 -sp \
      > /tmp/opencode/abl_sft.log 2>&1 &
  for _ in $(seq 1 90); do sleep 5; curl -sf -m 2 "http://127.0.0.1:$PORT/health" >/dev/null && return 0; done
  echo "  !! sft server failed"; return 1
}

echo "############ A1: chunker = sentence (SFT, pt) ############"
serve_sft && $PY -u -m fdb_bench.run_inference --model sft --lang pt \
  --out "$OUT/pt/sft-sentencechunks" --api-base "http://127.0.0.1:$PORT/v1" \
  --chunker sentence --turn-style training 2>&1 | tail -6
pkill -f "llama-server -m $SFT_GGUF" 2>/dev/null || true; sleep 3

echo "############ A2: turn_style = long (SFT, pt) ############"
serve_sft && $PY -u -m fdb_bench.run_inference --model sft --lang pt \
  --out "$OUT/pt/sft-longturns" --api-base "http://127.0.0.1:$PORT/v1" \
  --chunker training --turn-style long 2>&1 | tail -6
pkill -f "llama-server -m $SFT_GGUF" 2>/dev/null || true; sleep 3

echo "############ A3: companion v5 (pt) ############"
# v5's content model is the same stock GGUF the main run uses; serve it here so
# the arm is self-contained.
BIG_GGUF="$RUNS/base_qwen3_4b/gguf/Qwen3-4B-Instruct-2507-q4_k_m.gguf"
pkill -f "llama-server -m $BIG_GGUF" 2>/dev/null || true
free_port 8098
$LS -m "$BIG_GGUF" --port 8098 --host 127.0.0.1 -c 8192 --parallel 1 \
    > /tmp/opencode/abl_big.log 2>&1 &
for _ in $(seq 1 90); do sleep 5; curl -sf -m 2 http://127.0.0.1:8098/health >/dev/null && break; done
# v5 runs in-process (it is a merged HF checkpoint, not a GGUF)
$PY -u - <<'PYEOF'
import json, sys
sys.path.insert(0, "/home/penhfel/github/custom-duplex-cascade-pt-br")
from pathlib import Path
from fdb_bench import protocol as P
from fdb_bench import scenarios as S
from fdb_bench.clock import SpeechClock
from fdb_bench.run_inference import build_chunker, _serialise
from fdb_bench.simulate import DuplexSimulator
from fdb_bench.adapters import LlamaClient, PolicyAdapter
from fdb_bench.policy_companion import load_policy

v5 = "/mnt/f/duplex_cascade_runs/mtp_like/runs/final_v5/merged"
out = Path("results_ablation/pt/mtp_like_v5"); out.mkdir(parents=True, exist_ok=True)
policy = load_policy(v5)
big = LlamaClient("http://127.0.0.1:8098/v1")
ad = PolicyAdapter(policy, big, name="mtp_like_v5",
                   system_prompt=P.MTP_COMPANION_SYSTEM,
                   big_system=P.SFT_SYSTEM, prompt_candidate="instruction")
sim = DuplexSimulator(ad, SpeechClock(3.25))
chunker = build_chunker()
n = 0
with (out / "raw.jsonl").open("w", encoding="utf-8") as fh:
    for task in S.TASKS:
        for sc in S.build_scenarios(task, lang="pt", chunker=chunker):
            r = sim.run(sc); r.compute_s = getattr(ad, "last_call_s", 0.0)
            fh.write(json.dumps(_serialise(r), ensure_ascii=False) + "\n"); fh.flush()
            n += 1
            if n % 50 == 0:
                print(f"  v5 {n}", flush=True)
meta = {"model": "mtp_like_v5", "label": "Companion v5 (no promptable controller) [instruction]",
        "kind": "policy_companion", "lang": "pt", "tasks": list(S.TASKS), "seed": 1234,
        "prime_mode": "decision", "prompt_candidate": "instruction",
        "system_prompt": P.MTP_COMPANION_SYSTEM, "words_per_second": 3.25,
        "n_scenarios": n, "n_errors": 0, "chunker": "TrainingChunker", "turn_style": "training",
        "ablation": "companion v5: 7 tags, no promptable-controller / augmentation set"}
(out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print(f"wrote {out} ({n} scenarios)")
PYEOF

pkill -f "llama-server -m $BIG_GGUF" 2>/dev/null || true
echo "ABLATIONS_DONE"
