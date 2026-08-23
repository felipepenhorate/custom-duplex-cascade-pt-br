#!/usr/bin/env bash
# Serve the BASE Qwen3-4B-Instruct model via llama.cpp for parallel teacher
# generation (M1b / full run). The teacher must be the ORIGINAL base model, NOT
# the fine-tuned DuplexCascade GGUF.
#
# A Qwen3-4B-Instruct-2507 GGUF is not in /mnt/f/GGUF yet. Convert once with the
# llama.cpp toolchain (same recipe as duplex_cascade SPEC 6.5 export_gguf):
#   python ~/llama.cpp/convert_hf_to_gguf.py /mnt/f/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/<snap> \
#     --outfile /mnt/f/GGUF/Qwen3-4B-Instruct-2507/qwen3-4b-it.gguf --outtype q8_0
# then point LLAMA_MODEL at it.
#
# Port 8082 keeps it clear of the duplex llama-server (:8080) and the data
# generator (:8081). -sp so the duplex special tokens survive API output if the
# teacher ever emits them (--allow-tags mode); thinking stays on (Qwen3 base
# prompt); the API-side chat_template_kwargs handle enable_thinking=False.
LLAMA_MODEL="${LLAMA_MODEL:-/home/penhfel/Models/Qwen3-4B-Instruct-2507-Q8_0.gguf}"
if [ ! -f "$LLAMA_MODEL" ]; then
  echo "No base Qwen3-4B GGUF found. Convert it first (see header), or use --in-process." >&2
  exit 1
fi
exec /home/penhfel/llama.cpp/build/bin/llama-server \
  -m "$LLAMA_MODEL" \
  --port 8082 --host 127.0.0.1 -c 8192 --parallel 4 -sp \
  >/home/penhfel/github/duplex_cascade_distill/logs/teacher_api.log 2>&1 </dev/null