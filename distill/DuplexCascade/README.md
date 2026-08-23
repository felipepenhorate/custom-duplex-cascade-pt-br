# DuplexCascade-Distill: real-time demo for the self-distilled model

This folder is the DuplexCascade GUI adapted to test the **self-distilled**
DuplexCascade model (M4 of `duplex_cascade_distill`). It is a full-duplex speech
demo: streaming ASR (faster-whisper, pt), the distill LLM served by llama.cpp
(OpenAI-compatible), and streaming TTS (pocket-tts, pt), wired by the bridge
`server.py` to a web UI (mic in / audio out / live transcript).

The base is the **`custom-duplex-cascade-pt-br`** line of the GUI (the most
updated one), which adds robustness fixes over the original `duplex_cascade/`
server: a `stt_send_lock` serializing all writes to the shared STT websocket
(concurrent sends could hang the bridge), a 2 s timeout on the STT Eos send,
a 30 s timeout on the LLM call, in-place history truncation (the original rebind
made `history` local and the first answer crashed with `UnboundLocalError`), and
traceback logging on the tick task.

## What was adapted (vs. custom-duplex-cascade-pt-br)

* **System prompt** → the DuplexCascade trigger prompt used to generate the
  teacher content (`data/teacher_prompts.py`, content-only variant), plus an
  explicit turn-close instruction: *"Quando terminar de responder, emita
  <\|user is thinking\|> para indicar que seu turno terminou"*. Two things need
  this prompt:
  - The distill model's content is anchored to the frozen base under that
    prompt, so the same prompt at inference keeps it on-distribution (a short
    generic prompt drifts it, e.g. stray foreign tokens / markdown).
  - Unlike the SFT baseline, the distill model does NOT spontaneously emit the
    `<\|user is thinking\|>` turn-close tag in free generation (it keeps
    talking/repeating when the user is silent). The explicit instruction makes
    it close the turn reliably.
  - **v2 (2026-08)**: the trigger was bumped to `TRIGGER_VERSION="v2"` — the
    content instruction changed from *"Responda de forma curta"* to *"Responda
    de forma natural e completa ... Dê uma resposta coesa que responda de fato
    ao que o usuário perguntou, geralmente em uma a três frases curtas. Não se
    repita"*. The v1-trained model FOLLOWS this new prompt at inference (it
    retained instruction-following), fixing the repetition/rambling seen on
    substantive questions (e.g. *"como funciona hardware"*). No teacher
    regeneration or retraining was needed; the same v2 prompt is used at
    inference (`server.py:SYSTEM_PROMPT`) and for future teacher regeneration.
* **GGUF** → the distilled model, quantized q4_k_m:
  `/mnt/f/duplex_cascade_runs/distill/full/export/gguf/DuplexCascade-Distill-q4_k_m.gguf`.
* Labels/title → DuplexCascade-Distill.
* **Generation robustness (2026-08)**: fixes for degenerate output in live runs:
  - `--max-new-tokens` 96 → **256**. 96 truncated substantive answers mid-sentence,
    and the `<|no voice|>` continuation after a truncation is where the model
    collapsed into a degenerate token loop (`ő ő ő ő...`).
  - `--repeat-penalty 1.15` (llama-server + bridge payload): suppresses
    degenerate-repetition loops.
  - The micro-turn loop now ends the turn when the LLM finishes **naturally**
    (`finish_reason="stop"`, i.e. it emitted `<|im_end|>`), instead of always
    re-prompting with `<|no voice|>`. Only a `max_tokens` cut-off (`"length"`)
    continues the turn. Without this, a completed answer was prodded into
    rambling "vou esperar sua próxima palavra..." fillers that polluted the
    history and made later turns degenerate.
  - `_strip_degenerate_tail` removes trailing identical short-token runs
    (`ő ő ő`, `é é é`) before they reach TTS/history.
  - The root cause and the full debugging story are in the project
    `README.md` ("Known issue: degenerate repetition", § M4). If you want the
    complete-answer behavior *native* instead of prompt-following, regenerate
    the teacher data with the v2 trigger and retrain — see **`SPEC.md` §6 R0**.
* The duplex micro-turn loop, special-token handling, filler stripping and the
  web UI are otherwise **unchanged** from the custom line.

## Requirements

The unsloth venv has all deps; the three services need:

```bash
# inside the venv (e.g. /home/penhfel/unsloth_uv/bin/python)
pip install websockets faster-whisper pocket-tts
```

## Run

Four components: LLM (llama.cpp), ASR (faster-whisper), TTS (pocket-tts), and the
bridge. Use the orchestration script:

```bash
./run_services.sh
```

This launches (see `run_services.sh` for overrides via env vars):
1. **llama-server** (:8080) serving `DuplexCascade-Distill-q4_k_m.gguf`. The `-sp`
   flag is REQUIRED so the duplex special tokens are emitted in completions.
2. **STT** (services/stt_service.py, :31607) — faster-whisper `medium`, pt.
3. **TTS** (services/tts_service.py, :31608) — pocket-tts `portuguese`, voice
   `rafael`, 24 kHz streaming.
4. **Bridge** (DuplexCascade/server.py, :31606) — serves the web demo, wires
   browser ↔ STT ↔ llama.cpp ↔ TTS.

Then open: http://localhost:31606

To launch manually:

```bash
# 1) LLM (llama.cpp, OpenAI-compatible) — -sp for duplex special tokens
llama-server -m /mnt/f/duplex_cascade_runs/distill/full/export/gguf/DuplexCascade-Distill-q4_k_m.gguf \
  --port 8080 --host 127.0.0.1 -c 8192 --parallel 1 -sp

# 2) STT
python services/stt_service.py --port 31607 --model medium

# 3) TTS
python services/tts_service.py --port 31608 --language portuguese

# 4) bridge / web demo
python DuplexCascade/server.py --port 31606 \
  --stt-ws ws://127.0.0.1:31607 \
  --tts-ws ws://127.0.0.1:31608 \
  --llm-api-base http://127.0.0.1:8080/v1
```

The GGUF lives on the HDD (`/mnt/f`) to save SSD space; `RUNS_ROOT` relocates it,
and `GGUF=/path/to/model.gguf` overrides the model (e.g. to compare against
`DuplexCascade-PT-q4_k_m.gguf`).

## GGUF build (reproduce)

```bash
cd ~/llama.cpp
python convert_hf_to_gguf.py \
  /mnt/f/duplex_cascade_runs/distill/full/export/merged_bf16 \
  --outfile /mnt/f/duplex_cascade_runs/distill/full/export/gguf/DuplexCascade-Distill-f16.gguf \
  --outtype f16
build/bin/llama-quantize \
  /mnt/f/duplex_cascade_runs/distill/full/export/gguf/DuplexCascade-Distill-f16.gguf \
  /mnt/f/duplex_cascade_runs/distill/full/export/gguf/DuplexCascade-Distill-q4_k_m.gguf \
  q4_k_m
```

## License

[MIT](./DuplexCascade/LICENSE). Built on
[DuplexCascade](https://github.com/sbintuitions/DuplexCascade) (arXiv:2603.09180).