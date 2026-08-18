#!/usr/bin/env bash
exec /home/penhfel/unsloth_uv/bin/python /home/penhfel/duplex_cascade/DuplexCascade/server.py \
  --port 31606 \
  --stt-ws ws://127.0.0.1:31607 \
  --tts-ws ws://127.0.0.1:31608 \
  --llm-api-base http://127.0.0.1:8080/v1 \
  >/home/penhfel/duplex_cascade/logs/bridge.log 2>&1 </dev/null