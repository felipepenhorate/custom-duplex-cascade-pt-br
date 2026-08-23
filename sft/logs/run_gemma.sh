#!/usr/bin/env bash
# Serve the Gemma 4 E4B GGUF on its own port (default 8081) for dialogue
# generation. It must NOT clash with the DuplexCascade llama-server on :8080,
# which the live demo depends on. --parallel 4 lets build_long_dialogues.py
# generate several dialogues concurrently.
exec /home/penhfel/llama.cpp/build/bin/llama-server \
  -m /home/penhfel/Models/Gemma_4_E4B/gemma-4-E4B-it-UD-Q4_K_XL.gguf \
  --port 8081 --host 127.0.0.1 -c 8192 --parallel 4 \
  >/home/penhfel/duplex_cascade/logs/gemma.log 2>&1 </dev/null