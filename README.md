# Custom DuplexCascade-PT-BR

A **fork** of [**DuplexCascade**](https://github.com/sbintuitions/DuplexCascade)
("Full-Duplex Speech-to-Speech Dialogue with VAD-Free Cascaded ASR–LLM–TTS
Pipeline and Micro-Turn Optimization", sbintuitions) that:

- **Retrains the conversational LLM with QLoRA (4-bit) via [Unsloth](https://github.com/unslothai/unsloth)**
  instead of the original LoRA+HF approach.
- **Targets Brazilian Portuguese (pt-BR)** instead of English.
- **Replaces the Kyutai STT/TTS backends** with open-source alternatives:
  - **STT** → [Whisper](https://github.com/SYSTRAN/faster-whisper) via `faster-whisper`
    (CTranslate2, `pt`).
  - **TTS** → [pocket-tts](https://github.com/Kyutai-Labs/pocket-tts) (Kyutai, 100M,
    streaming, `portuguese`).
- **Serves the fine-tuned model with [llama.cpp](https://github.com/ggml-org/llama.cpp)**
  (OpenAI-compatible, streaming) instead of the original in-process loader.

The `DuplexCascade/` subdirectory is the vendored inference base from the original
article (adapted for this fork — see [Fork provenance](#fork-provenance)). `SPEC.md`
is the working design/spec document and may lag the code.

---

## What this repo contains

```
duplex_cascade/
├── SPEC.md                      # working design/spec (may lag code)
├── README.md                    # this file
├── LEARNINGS.md                 # hard-won training learnings & pitfalls
├── run_services.sh              # launch the full stack (llama.cpp + STT + TTS + bridge)
├── continue_finetune.sh         # regenerate longer-response data + continue training
├── DuplexCascade/               # vendored fork of the original inference server + web UI
│   ├── server.py                # bridge / web demo (:31606)
│   └── web/                     # static demo UI
├── services/
│   ├── stt_service.py           # faster-whisper streaming (:31607)
│   └── tts_service.py           # pocket-tts streaming (:31608)
├── data/
│   ├── common.py                # special tokens, loss weights, pt-BR helpers
│   ├── build_dialogues.py       # M1 dialogue generator (short replies)
│   ├── build_long_dialogues.py  # M1-long generator (short + long replies)
│   └── build_duplex_dataset.py  # M1 duplex micro-turn builder
├── training/
│   ├── prep_dataset.py          # M1 tokenize + labels (stage 3)
│   ├── train_qlora.py           # M2 QLoRA fine-tune (weighted/masked loss)
│   ├── export_model.py          # merge adapter → bf16 + train_cfg.json
│   └── export_gguf.py           # convert bf16 → GGUF (q4_k_m)
└── logs/                        # runtime launch helpers (run_llama.sh, run_gemma.sh, ...)
```

---

## Quick start (run the demo)

The four components — **llama.cpp LLM** (:8080), **STT** (:31607), **TTS** (:31608)
and the **bridge/web demo** (:31606) — are orchestrated by `run_services.sh`:

```bash
./run_services.sh
```

Then open `http://localhost:31606`.

Prerequisites:

- **Python deps**: `pip install -r requirements.txt` (runtime) — or add
  `-r requirements-train.txt` to also install the training pipeline deps.
- A llama.cpp build with `llama-server` (and `llama-quantize` for GGUF export).
- The trained GGUF model (see below; defaults to `$RUNS_ROOT/continue/...`).

The script picks the newest fine-tuned GGUF automatically. Override with:

```bash
GGUF=/path/to/DuplexCascade-PT-q4_k_m.gguf ./run_services.sh
```

### Storage layout / model runs

Trained runs are large (~46 GB) and by default live on the HDD mounted at
`/mnt/f` to save SSD space:

```
/mnt/f/duplex_cascade_runs/
├── full/       # original QLoRA fine-tune (M2) - base for continuation
└── continue/   # continuation fine-tune with longer-response data (M3, default)
```

Every script honors the `RUNS_ROOT` env var to relocate the runs, e.g.
`RUNS_ROOT=/other/path ./run_services.sh`.

---

## Recipe: recreate this approach for a new language / dataset

This is the full pipeline used to build DuplexCascade-PT. It is designed to be
re-run for **any language** (or a different dataset/backbone) by swapping a few
constants. Milestones map to `SPEC.md` §11 (M1 data → M2 training → M3 services).

### 0. Decisions to make first

Set these before anything else (see `SPEC.md` §6.1, §12):

| Choice | This project | Notes |
|---|---|---|
| Base model | `Qwen/Qwen3-4B-Instruct-2507` | dense Qwen3; unsloth fast path; ~5–7 GB QLoRA on a 16 GB 4080 |
| Data generator | Gemma 4 E4B via llama.cpp (`:8081`) | any instruct LLM over an OpenAI-compatible API works |
| STT | faster-whisper `small`/`medium`, `<lang>` | `--model` + `--language` |
| TTS | pocket-tts `portuguese` | pocket-tts language code + default voice |
| Target language | pt-BR | stopwords/accents in `data/common.py` |

### 1. Generate dialogues (`data/build_dialogues.py`)

Serve an instruct LLM (Gemma 4 E4B here) on its own port:

```bash
./logs/run_gemma.sh &        # llama-server -m <gemma.gguf> --port 8081 --parallel 4
python data/build_dialogues.py \
  --api-base http://127.0.0.1:8081/v1 \
  --n-dialogues 5000 --workers 4 \
  --out data/dialogues_pt.jsonl
```

- Prompts ask for realistic multi-turn **chat-style** pt-BR dialogues (`Usuário:` /
  `Assistente:` lines), 6–14 turns.
- `build_long_dialogues.py` is the variant that mixes **short + long** assistant
  replies (this fixed the "model only ever answers in 1–2 short sentences" issue).
- Output is deduped (MinHash-style `sha256_dedup_key`), filtered by pt-BR language
  heuristics (`is_portuguese`), turn-length variance, and repetition checks.

For a **new language**: add stopwords + accent characters in `data/common.py`
(`_PT_STOPWORDS`, `_ACCENT_CHARS`), localize the generation prompts, and pass the
target language to STT/TTS.

### 2. Build the duplex micro-turn dataset (`data/build_duplex_dataset.py`)

This is the core trick from the paper (§3.3). It converts plain dialogues into
interleaved **micro-turns** supervised with special tokens:

- After each user micro-turn, insert a `<|user is speaking|>` system micro-turn.
- After each system micro-turn, insert `<|no voice|>` user micro-turns.
- Prepend `<|user finish speaking|>` to each system turn.
- Random micro-turn lengths: user 1–7 tokens, **system variable** 10–48 tokens
  (default `--system-chunk-min 10 --system-chunk-max 48` — variable length taught
  the model to speak in longer utterances).
- Probabilistic phenomena: natural pauses (extra `<|no voice|>`), user
  interruptions (`<|user interruption|>`), backchannels (`<|user backchannel|>`),
  thinking pauses (`<|user is thinking|>`).

```bash
python data/build_duplex_dataset.py \
  --dialogues data/dialogues_pt.jsonl \
  --out data/duplex_train.jsonl \
  --system-chunk-min 10 --system-chunk-max 48
```

### 3. Tokenize + labels (`training/prep_dataset.py`)

Builds ChatML sequences with `input_ids` / `labels` / `weights`:

- Loss is computed **only on system micro-turns** (user positions are `-100`).
- Each predicted special token carries its **loss weight** from `data/common.TOKEN_WEIGHT`
  (`<|user finish speaking|>=10`, `<|user interruption|>=5`, ...).

```bash
python training/prep_dataset.py \
  --duplex data/duplex_train.jsonl \
  --out data/prepped \
  --tokenizer Qwen/Qwen3-4B-Instruct-2507 \
  --max-seq 2048
```

### 4. Fine-tune with QLoRA (`training/train_qlora.py`)

```bash
# smoke run first
python training/train_qlora.py --dataset data/prepped --max-steps 100 \
  --per-device-batch-size 2 --grad-accum 8 --out-dir /mnt/f/duplex_cascade_runs/smoke

# full run (2k steps, ~12 h on a 16 GB 4080 with embed fine-tuning)
python training/train_qlora.py --dataset data/prepped --max-steps 2000 \
  --per-device-batch-size 1 --grad-accum 16 --max-seq-length 2048 \
  --out-dir /mnt/f/duplex_cascade_runs/full
```

- Uses a custom `DuplexSFTTrainer` (subclass of `UnslothTrainer`) implementing the
  paper's **weighted/masked CE** (§6.3) — Unsloth has no native per-token weighting.
- 4-bit NF4 QLoRA on all linear modules (`q,k,v,o,gate,up,down`), `r=16 α=16`,
  plus `modules_to_save=["embed_tokens","lm_head"]` so the new special-token
  embeddings are trained.
- **VRAM tip**: stop the llama-server data generator and any serving llama-server
  before training (a 4B QLoRA run sits right at the 16 GB ceiling). Set
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to avoid fragmented-allocation
  OOM on long runs.

### 5. Continue training for longer responses (`continue_finetune.sh`)

To teach the model both short **and** long answers without losing the learned
turn-taking behavior, continue from the previous **merged** model (not the base):

```bash
./logs/run_gemma.sh &
FORCE_DATA=1 MAX_STEPS=1000 ./continue_finetune.sh
```

`continue_finetune.sh` drives the whole loop: generate long dialogues → combine
with the originals → build duplex micro-turns (variable chunking) → prep → continue
training from the merged `full` run → export bf16 → GGUF.

### 6. Export for serving (`training/export_model.py` + `training/export_gguf.py`)

```bash
python training/export_model.py \
  --adapter /mnt/f/duplex_cascade_runs/full/adapter \
  --base-model Qwen/Qwen3-4B-Instruct-2507 \
  --out /mnt/f/duplex_cascade_runs/full/export

python training/export_gguf.py \
  --merged /mnt/f/duplex_cascade_runs/full/export/merged_bf16 \
  --llamacpp ~/llama.cpp --out /mnt/f/duplex_cascade_runs/full/export/gguf \
  --quant q4_k_m
```

- Base must be loaded in **bf16** (not 4-bit) before merge — merging onto the 4-bit
  base keeps quantized bnb weights and trips transformers' weight-conversion checks.
- The **tokenizer must be saved with the merged model**; without it llama.cpp fails
  with "cannot find tokenizer merges in model file".
- `model_state.safetensors` + `train_cfg.json` are also written for the original
  `server.py` load path (optional).

### 7. Serve + test

```bash
./run_services.sh
# open http://localhost:31606
```

The llama-server is launched with **`-sp`** (REQUIRED — emits the duplex special
tokens in completions; without it the API silently drops them).

---

## Fork provenance

This project is a **fork of [sbintuitions/DuplexCascade](https://github.com/sbintuitions/DuplexCascade)**.
The `DuplexCascade/` subdirectory (inference server + web UI) is vendored from that
repo and **heavily adapted** for this pt-BR QLoRA stack. The original project and
paper:

> DuplexCascade: Full-Duplex Speech-to-Speech Dialogue with VAD-Free Cascaded
> ASR-LLM-TTS Pipeline and Micro-Turn Optimization — Jianing Yang, Yusuke Fujita,
> Yui Sudo (sbintuitions). [Demo](https://sbintuitions.github.io/DuplexCascadeDemo/) ·
> [Paper](https://arxiv.org/abs/2603.09180).

See the original repo for the unmodified upstream code and its MIT license.

---

## License

The vendored `DuplexCascade/` base is MIT-licensed (see `DuplexCascade/LICENSE`).
Data generated by open models is fine to distribute; see `SPEC.md` §12 for the
licensing notes of each component (Whisper MIT, pocket-tts CC BY 4.0, Common Voice
pt CC0).
