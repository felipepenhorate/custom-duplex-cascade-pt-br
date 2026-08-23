#!/usr/bin/env bash
# Watches the M1 generation; when it completes, stops the teacher llama-server
# (frees the 16 GB 4080) and launches the M2 smoke run.
#
# Launched detached (setsid) so it survives the shell that spawned it.
cd "$(dirname "$0")/.." || exit 1

TARGET="${TARGET:-10000}"
OUT=data/teacher_duplex_train.jsonl
M1_PID="${M1_PID:-36119}"

echo "[watch] waiting for M1 (pid $M1_PID) to finish or $OUT to reach $TARGET lines"
while :; do
  if ! kill -0 "$M1_PID" 2>/dev/null; then
    echo "[watch] M1 process finished"
    break
  fi
  n=$(wc -l < "$OUT" 2>/dev/null || echo 0)
  if [ "$n" -ge "$TARGET" ]; then
    echo "[watch] $OUT reached $n lines"
    break
  fi
  sleep 20
done
sleep 5

echo "[watch] stopping teacher llama-server :8082"
pkill -f "llama-server.*8082" || true
sleep 5

echo "[watch] launching M2 smoke (N=${N:-512}, STEPS=${STEPS:-100}, KL=${KL:-0.3})"
N="${N:-512}" STEPS="${STEPS:-100}" KL="${KL:-0.3}" ./logs/run_smoke_train.sh \
  > logs/m2_smoke.log 2>&1
echo "[watch] smoke finished rc=$? -> logs/m2_smoke.log"