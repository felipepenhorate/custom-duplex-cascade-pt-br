#!/usr/bin/env bash
# Launch the MTP-like full-duplex demo stack:
#   STT (faster-whisper pt)  :31607
#   TTS (pocket-tts pt)      :31608
#   bridge (companion + big model + web UI) :31606
#
# Usage:
#   ./run_services.sh
#   # open http://localhost:31606
#
# Env overrides:
#   STT_MODEL      faster-whisper model (default: small)
#   COMPANION      merged policy model dir (default: /mnt/f/duplex_cascade_runs/mtp_like/runs/final_v5/merged)
#   BIG_MODEL      stock LLM served by the bridge (default: Qwen/Qwen3-4B-Instruct-2507)
#   PORTS          override bridge/stt/tts ports: "31606 31607 31608"
set -e
cd "$(dirname "$0")"

PY=${PY:-/home/penhfel/unsloth_uv/bin/python}
SFT_SERVICES=../sft/services
STT_MODEL=${STT_MODEL:-small}
COMPANION=${COMPANION:-/mnt/f/duplex_cascade_runs/mtp_like/runs/final_v6/merged}
BIG_MODEL=${BIG_MODEL:-Qwen/Qwen3-4B-Instruct-2507}
POLICY_RULE=${POLICY_RULE:-"No CPF, apenas números são permitidos. Se o cliente disser uma letra em vez de um número, interrompa e avise que apenas números são aceitos."}
read -r BRIDGE_PORT STT_PORT TTS_PORT <<< "${PORTS:-31606 31607 31608}"

echo "[run] STT:  faster-whisper '$STT_MODEL' (pt) on :$STT_PORT"
"$PY" "$SFT_SERVICES/stt_service.py" --port "$STT_PORT" --model "$STT_MODEL" \
    --partial-commit-s 0.7 --partial-interval-s 0.3 \
    > logs/stt.log 2>&1 &
STT_PID=$!
echo "[run] TTS:  pocket-tts (portuguese) on :$TTS_PORT"
"$PY" "$SFT_SERVICES/tts_service.py" --port "$TTS_PORT" \
    > logs/tts.log 2>&1 &
TTS_PID=$!
echo "[run] bridge: companion=$COMPANION big=$BIG_MODEL on :$BRIDGE_PORT"
BRIDGE_ARGS=(--port "$BRIDGE_PORT")
BRIDGE_ARGS+=(--stt-ws "ws://127.0.0.1:$STT_PORT" --tts-ws "ws://127.0.0.1:$TTS_PORT")
BRIDGE_ARGS+=(--companion "$COMPANION" --big-model "$BIG_MODEL")
if [ -n "${POLICY_RULE:-}" ]; then
    BRIDGE_ARGS+=(--policy-rule "$POLICY_RULE")
fi
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" demo/server.py "${BRIDGE_ARGS[@]}" \
    > logs/bridge.log 2>&1 &
BRIDGE_PID=$!

echo "[run] open http://localhost:$BRIDGE_PORT (logs in logs/)"
cleanup() {
    echo "[run] stopping..."
    kill "$BRIDGE_PID" "$STT_PID" "$TTS_PID" 2>/dev/null || true
}
trap cleanup INT TERM EXIT
wait "$BRIDGE_PID"