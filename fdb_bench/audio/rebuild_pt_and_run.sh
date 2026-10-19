#!/usr/bin/env bash
# Rebuild the pt-BR audio set, verify it, and only then run the sweep.
#
# Built for an unattended overnight run, so every stage is resume-safe and the
# expensive one is gated on the cheap one:
#
#   1. rebuild (rafael voice, compression-only time fitting -- heavy expansion is
#      what droned the first two attempts);
#   2. verify -- re-recognise every built input.wav and count the scenarios whose
#      transcript still shows a held-vowel run;
#   3. GATE -- if more than GATE_PCT of scenarios are degraded, stop and say so.
#      A 10-hour sweep on bad audio is worse than no sweep;
#   4. sweep all four models (pt-BR prompts, Portuguese assistant voice, 16k
#      context for the upstream model's long Portuguese prompts), then print the
#      comparison table.
#
#   ./fdb_bench/audio/rebuild_pt_and_run.sh
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export HF_HOME="${HF_HOME:-/mnt/f/huggingface}"
PY="${PY:-/home/penhfel/unsloth_uv/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/f/fdb_data/pt_v1.0}"
GATE_PCT="${GATE_PCT:-15}"          # max % of scenarios allowed to still degrade
MAX_WORDS="${MAX_WORDS:-15}"

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "PT_REBUILD_START ($DATA_ROOT)"
# keep asr_en.json (an English-ASR pass we do not want to repeat), drop the audio
# and the audit trail so every scenario is rebuilt rather than skipped
find "$DATA_ROOT" -type f \( -name 'input.wav' -o -name 'context.wav' \
     -o -name 'interrupt.wav' -o -name 'translation.json' \) -delete
log "  cleared $(find "$DATA_ROOT" -name asr_en.json | wc -l) asr caches kept, audio cleared"

$PY -m fdb_bench.audio.build_pt_audio --stage build --max-words "$MAX_WORDS" \
    || { log "PT_REBUILD_FAILED"; exit 1; }
log "PT_REBUILD_DONE"

$PY -m fdb_bench.audio.build_pt_audio --stage verify \
    || { log "PT_VERIFY_FAILED"; exit 1; }

READ=$($PY - <<'PYEOF'
import json, sys
from pathlib import Path
rows = json.loads(Path("/mnt/f/fdb_data/pt_v1.0/verify.json").read_text())
bad = sum(1 for r in rows if r["degraded"])
print(f"{bad} {len(rows)} {100*bad/max(1,len(rows)):.1f}")
PYEOF
)
BAD=$(echo "$READ" | cut -d' ' -f1); N=$(echo "$READ" | cut -d' ' -f2); PCT=$(echo "$READ" | cut -d' ' -f3)
log "PT_VERIFY_RESULT ${BAD}/${N} degraded (${PCT}%), gate is ${GATE_PCT}%"
# the comparison runs in python: `bc` is not installed here, and a missing
# comparator would silently fail the gate closed and skip the sweep
GATE=$($PY -c "import sys; print(1 if float('$PCT') > float('$GATE_PCT') else 0)" 2>/dev/null || echo 1)
if [ "$GATE" = "1" ]; then
  log "PT_GATE_FAIL: too much degraded audio, NOT starting the sweep."
  log "  inspect /mnt/f/fdb_data/pt_v1.0/verify.json and translation.json flags"
  exit 2
fi
log "PT_GATE_PASS"

log "PT_SWEEP_START"
FDB_LANG=pt DATA_ROOT="$DATA_ROOT" TTS_LANG=portuguese \
  RESULTS="$ROOT/results_audio_pt" AUDIO_ROOT=/mnt/f/fdb_audio_pt \
  MODELS="original_dc sft distill mtp_like" CTX=16384 \
  ./fdb_bench/audio/run_audio_all.sh
log "PT_SWEEP_EXIT=$?"

log "PT_COMPARISON"
$PY -m fdb_bench.audio.compare_audio --results results_audio_pt 2>&1 | tail -40
log "PT_ALL_DONE"
