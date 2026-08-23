# SPEC — DuplexCascade-v2-PT: Brazilian Portuguese Full-Duplex Speech Dialogue

Version: 1.0 (draft)
Date: 2026-08-11

## 1. Objective

Produce a new version of `duplex_cascade_article.pdf` ("DuplexCascade: Full-Duplex
Speech-to-Speech Dialogue with VAD-Free Cascaded ASR–LLM–TTS Pipeline and Micro-Turn
Optimization") that:

1. **Retrains the conversational LLM with QLoRA (4-bit) via Unsloth** instead of the
   original LoRA+HF-style adaptation (original: q/v LoRA r=16 α=32, per peft).
2. **Targets Brazilian Portuguese (pt-BR)** instead of English.
3. **Replaces the Kyutai STT/TTS backends** with open-source alternatives:
   - STT → **Whisper** (via `faster-whisper`, CTranslate2, `pt` language).
   - TTS → **pocket-tts** (Kyutai, 100M, streaming, `portuguese` model — officially
     supported). 24 kHz mono output, 1920-sample (80 ms) frames compatible with the
     existing browser protocol.
4. **Builds a pt-BR training dataset** for the multiplexed micro-turn training:
   synthetic multi-turn Portuguese dialogues + optional real speech from **Common
   Voice pt** and audio synthesis with **Qwen3-TTS** (Apache-2.0, supports
   Portuguese) for pipeline/eval purposes.

The final deliverable is a **new article (LaTeX → PDF)** describing the pt-BR model
(DuplexCascade-PT), its training recipe, data pipeline, the open-source speech stack
and evaluations. The repo also contains runnable code: training scripts, data-building
scripts, and the adapted inference server.

## 2. What exists today (baseline to build on)

- Paper: `duplex_cascade_article.pdf` — original English paper (LoRA, Qwen2-7B-Instruct,
  50k UltraChat dialogues, 5k steps, weighted token loss, DSM-ASR/DSM-TTS streaming backends).
- `DuplexCascade/`:
  - `model.py` — inference wrapper, `Model.enable_lora_adapter()` (peft, mode "qv").
  - `server.py` — `KyutaiBridgeServer` (websocket bridge), special tokens
    `<|no voice|>`, `<|user is talking|>`, `<|user finish talking|>`,
    `<|user is thinking|>`, `<|user interruption|>`, `<|user backchannel|>`;
    STT/TTS over msgpack websockets (24 kHz float32), LLM micro-turn ticking every
    `--overlap-window-s` (0.6 s), Qwen2 ChatML prompt template, streaming generation
    via `TokenQueueStreamer`, TTS buffering with period-stripping, backchannel /
    interruption handling.
  - `web/` — static demo UI (index.html, client.js, style.css), 24 kHz PCM up/down.
- Hardware available: 1× RTX 4080 (16 GB), CUDA 13.1, torch 2.10.0+cu130,
  **unsloth 2026.4.4** installed in `/home/penhfel/unsloth_uv` (python 3.13), 1 TB disk,
  31 GB RAM. (Original paper: 8× H100, 5 h.)

## 3. Known original-code defects to fix in v2

1. **Special token mismatch**: `server.py:612` checks `<|assistant_backchannel|>` but
   `SPECIAL_TOKENS` defines `<|user backchannel|>` — the system-backchannel branch is
   dead code. Adopt the exact paper token set in v2:
   `<|no voice|>`, `<|user is speaking|>`, `<|user finish speaking|>`,
   `<|user is thinking|>`, `<|user interruption|>`, `<|user backchannel|>`,
   `<|system backchannel|>` (keep `<|` `|>` delimiters; tokenizer adds them as
   special tokens with random Gaussian init during training).
2. `model.py` trains/loads via `peft` state dict with `modules_to_save=[embed_tokens,
   lm_head]`; v2 will use Unsloth artifacts (merged adapter state, see §6.5).
3. `stt_pre_silence_s` logic is Kyutai-specific; keep as optional config, default off.

## 4. Architecture (unchanged core, new backends)

```
Browser (24 kHz PCM)  ⇄  server.py (bridge, unchanged protocol)  ⇄  LLM (QLoRA)
        │                                                              │
        ▼                                                              ▼
  new stt_service.py                                         new tts_service.py
  Whisper (faster-whisper pt)                                pocket-tts (portuguese)
  16 kHz resample inside; VAD + chunked incremental          streams 24 kHz PCM, 80 ms frames
  transcription → per-word events (msgpack ws, :31607)       (msgpack ws, :31608)
```

- Messages on the STT/TTS websockets keep the Kyutai msgpack shape
  (`{"type":"Audio","pcm":[...]}`, `{"type":"Text",...}`, `{"type":"Eos"}`, outbound
  `{"type":"Word","text":...}`) so `server.py` integration is minimally invasive.
- `server.py` changes: rename backend wiring (no more `kyutai-api-key`), add
  `--stt-language pt`, `--tts-language portuguese` (or `portuguese_24l`) and a
  `--tts-voice` preset (pocket-tts built-in pt voice or a voice cloned from a short
  pt-BR speaker sample, e.g. from Common Voice).

## 5. Training data for pt-BR (three sources)

### 5.1 Primary: 50k synthetic multi-turn pt-BR dialogues (text)
- Generate conversational multi-turn dialogues directly in pt-BR with an instruct LLM
  served by a **llama.cpp instance via its OpenAI-compatible API** (`http://127.0.0.1:8080/v1`,
  currently serving `Gemma_4_E4B/gemma-4-E4B-it-UD-Q4_K_XL.gguf` — chosen over the previous
  `Qwen3.5-9B-UD` GGUF for better pt-BR output; see §12 decision points). No local model
  weights are loaded by the builder; every prompt is one `/v1/chat/completions` request
  with thinking mode disabled (`chat_template_kwargs: {"enable_thinking": false}`) so the
  whole generation budget goes into the dialogue text — this knob applies to both Qwen3
  and Gemma chat templates in llama.cpp. Fine-tuning base for the duplex model is
  `Qwen/Qwen3-4B-Instruct-2507` (§6.1); the generator serves only as a data source.
- Scenario-driven prompt template: chat topics (daily life, travel, tech, food,
  customer service, small talk), speaker personas, 6–14 turns per dialogue,
  instruction: "converse em português brasileiro coloquial".
- Scripts in `data/build_dialogues.py` with:
  - generation via `LlamaCppClient` (requests → `/v1/chat/completions`, retry/backoff
    on 503 "slot busy", `--workers` concurrent requests),
  - dedup (MinHash on normalized text), length/noise filters (reject dialogues with
    non-pt content via language ID check, e.g. `fasttext-langdetect` or
    `langdetect`),
  - quality scoring: keep top-k by heuristic (turn length variance, no repetition
    loops) — cheap and offline, no labeler needed.
- Fallback/mix-in if quality is insufficient: public pt-BR conversational data
  (e.g. OpenAssistant/portuguese splits, Opus100 pt dialogues) — decision point §8.

### 5.2 Duplex micro-turn construction (port of paper §3.3 to pt-BR)
Reuse the exact methodology on the pt-BR dialogues:
1. Split user/system utterances into micro-turns; after user micro-turns insert
   `<|user is speaking|>` system micro-turns; after system micro-turns insert
   `<|no voice|>` user micro-turns; prepend `<|user finish speaking|>` to each
   system turn.
2. Random micro-turn length: user 1–7 tokens, system fixed 10 tokens.
3. Natural pauses: 10% probability, 1–5 extra `<|no voice|>` micro-turns → model
   should emit `<|user is speaking|>`.
4. User interruption: 30% probability per system turn; first interrupting micro-turn
   supervised as `<|user interruption|>` then `<|user is speaking|>`.
5. User backchannels: 1% probability replace `<|no voice|>` with a short pt-BR
   backchannel ("aham", "tá", "sim", "ok", "entendi"...) → model emits
   `<|user backchannel|>`.
6. System backchains: post-process with an LLM (same generator as §5.1, or the
   Qwen3-TTS family if convenient) to insert `<BC/>` markers at pt-BR backchannel
   points → supervise `<|system backchannel|>` (DuplexCascade-PT-β variant).
7. User thinking: after system turns insert 1–20 `<|no voice|>` micro-turns →
   supervise `<|user is thinking|>`.
- Implemented in `data/build_duplex_dataset.py` with the same RNG seeds for
  reproducibility; output: streaming-compatible JSONL/parquet with labels.

### 5.3 Synthetic speech: Qwen3-TTS + Common Voice pt (for pipeline & eval, not LLM text training)
- **Qwen3-TTS** (`Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` or `-Base` with voice
  cloning; Apache-2.0; Portuguese supported): synthesize audio for a slice of the
  §5.1 dialogues (say 1k dialogues) to produce **DuplexCascade-PT speech eval set**:
  user turns spoken → Whisper pt transcript → LLM → pocket-tts. Used to measure the
  end-to-end chain and to validate whisper word-streaming behavior.
- **Common Voice pt** (Mozilla `mozilla-foundation/common_voice_*_*` pt subset,
  ~18k+ hours recorded pt, including pt-BR clips): used (a) as ASR eval for Whisper
  pt (CER/WER), (b) as candidate speaker reference audio for pocket-tts voice
  cloning, (c) optional ASR fine-tuning input if Whisper pt accuracy is poor
  (decision point §8 — default: no Whisper fine-tuning in v1).
- Qwen3-TTS may also be used to **synthesize the 6 turn-taking phenomena** of
  §5.2 into audio (pause handling, interruptions, backchannels) to build a pt-BR
  analogue of Full-Duplex-Bench — see §7.2.

Note: LLM fine-tuning itself stays **text-only** (faithful to the paper's core claim),
speech synthesis is used for testing/eval of the ASR-TTS chain only.

## 6. Training (QLoRA via Unsloth)

### 6.1 Base model — DECIDED: `Qwen/Qwen3-4B-Instruct-2507`
- User decision (2026-08-12): **`Qwen/Qwen3-4B-Instruct-2507`** — chosen over the
  previously planned `Qwen/Qwen3-8B`: best fit for the micro-turn task (instruction-tuned
  dense Qwen3, `Qwen3ForCausalLM`, no thinking-by-default breakage) at roughly half the
  VRAM (~5–7 GB QLoRA instead of ~8–11 GB), which is decisive on the single 16 GB 4080 —
  especially alongside the llama.cpp data-generation instance. `unsloth` fast path applies
  (dense Qwen3 architecture; `FastQwen3Model`-compatible).
- Verdict on candidates: dense `Qwen3` (4B/8B) accepted; rejected are
  `Qwen/Qwen3.5-4B` (multimodal hybrid GDN/SSM+attention,
  `Qwen3_5ForConditionalGeneration`; no unsloth fast path — generic fallback with forced
  fp32, no packing, thinking-mode-by-default breaks the hard-coded micro-turn prompt
  template and text streaming in `server.py`; would require a full inference rewrite) and
  `Qwen2.5-7B-Instruct` (fallback option, no thinking mode).
- **Thinking-mode handling**: Qwen3's chat template renders a `think` block when
  `enable_thinking=True`. The duplex prompt is built manually (never through
  `apply_chat_template`), and fine-tuning data never contains thinking markers, so the
  fine-tuned model learns the micro-turn format directly; verify during the smoke run
  that generation emits no thinking tokens (if any appear, add a system text such as
  "Responda direto, sem raciocínio." to the prompt builder — §7.3).
- Tokenizer: Qwen3 ChatML (`<|im_start|>`/`<|im_end|>`) matches the hard-coded template
  in `server.py:166`; add the 7 duplex special tokens via `add_special_tokens`. The
  duplex micro-turn ids (data/build_duplex_dataset.py) and prep/training tokenization
  must all use THIS tokenizer (`--tokenizer Qwen/Qwen3-4B-Instruct-2507`), since user
  chunk boundaries are stored as base-vocab token ids.

### 6.2 Recipe (mapped from original §4.1)
| Hyperparameter | Original (LoRA) | v2 (QLoRA) |
|---|---|---|
| Quantization | none (BF16) | **4-bit NF4** via `FastLanguageModel` (DoubleQuant off to save VRAM), `load_in_4bit` |
| Adapter | LoRA q,v r=16 α=32 | **QLoRA** on all linear: q,k,v,o,gate,up,down r=16 α=16 (unsloth defaults), LoRA dropout 0 |
| Embeddings |  fully fine-tuned | `lora_alpha` fine-tuned embed (`resize_token_embeddings` + training head) |
| Loss | weighted CE (special-token weights 1/10/5/2/1/3), masked to system micro-turns | same masking scheme **and** weighted-loss (custom UnslothTrainer loss wrapper, §6.3) |
| Optimizer | AdamW lr=1e-5, 500-step warmup | AdamW (unsloth default lr 2e-4, betas, weight decay 0) |
| Steps / batch | 5k × 32    | 2k × 32 (DECIDED 2026-08-12: keep embed/lm_head fine-tuning via `modules_to_save`, which costs ~21 s/step on the 4080 → 2k ≈ 12 h; 5k would be ~29 h. LoRA-only alternative is ~1 s/step but leaves embeddings frozen) |
| Seq len | 4096 | 4096 (unsloth `max_seq_length=4096`, enabled `use_gradient_checkpointing="unsloth"`) |
| Scheduling | linear warmup 500 | linear warmup (or unsloth cosine) |

Expected VRAM: ~5–7 GB for Qwen3-4B-Instruct-2507 4-bit + 4096-seq batches on 16 GB
4080 (roughly half of the 8B estimate; leaves headroom to run the llama.cpp data
generator concurrently if needed). Estimated wall time for 2k steps with embed
fine-tuning (single 4080, bf16 compute): **≈ 12 h** (measured ~21 s/step in the smoke
run). LoRA-only (no `modules_to_save`) measures ~1 s/step → 5k ≈ 2 h, at the cost of
frozen embeddings — fallback if runtime becomes a blocker.

### 6.3 Weighted special-token loss with Unsloth
- Unsloth's trainer does not natively support per-token loss weighting. Implement a
  small `CustomUnslothTrainer` (subclass `UnslothTrainer`) overriding the compute of
  CE with weights:
  `<|user is speaking|>`=1, `<|user finish speaking|>`=10, `<|user interruption|>`=5,
  `<|user backchannel|>`=2, `<|user is thinking|>`=1, `<|system backchannel|>`=3;
  and label masking: user micro-turn tokens and `<|no voice|>` user tokens get
  `label=-100` (train only on system micro-turn predictions) — port of
  `train.py` semantics from the original paper.
- Verify loss shape/labels on a 64-sample smoke run with a small lr before the full run.

### 6.4 Tokenizer & special tokens
- Load base tokenizer, `add_special_tokens` for the 7 tokens (§3.1), init their
  embeddings randomly (Gaussian, σ≈0.02) before training (not via resize only).

### 6.5 Artifacts & integration with inference
- **Vocab mismatch (verified in the smoke run)**: `Qwen3-4B-Instruct-2507` ships a model
  with `vocab_size=151936` but an HF tokenizer of only 151669 base tokens. Training adds
  7 specials → tokenizer 151676, and `resize_token_embeddings(151676)` truncates the
  unused tail rows (260 dead rows). The adapter is therefore only compatible with the
  **tokenizer saved inside the adapter dir** (151676) + `resize_token_embeddings` before
  `PeftModel.from_pretrained`. Load recipe that works (smoke-verified):
  `tok = AutoTokenizer.from_pretrained(<adapter_dir>); model.resize_token_embeddings(len(tok)); PeftModel.from_pretrained(model, <adapter_dir>)`.
- **Export pipeline (verified 2026-08-13)**: the trained adapter is exported to the
  server as a **bf16 merged model + GGUF** via `training/export_model.py` +
  `training/export_gguf.py`:
  1. Load base in **bf16** (NOT 4-bit) → `resize_token_embeddings(len(adapter_tokenizer))`
     → `PeftModel.from_pretrained` → `merge_and_unload()` → `save_pretrained`.
     (Loading the base 4-bit keeps bnb-quantized weights after merge and trips
     transformers' tied-weight/weight-conversion checks; unsloth's
     `unsloth_save_model(merged_4bit_forced)` also fails with `NotImplementedError`.
     The LoRA deltas are dtype-independent, so merging onto bf16 works cleanly.)
  2. `tok.save_pretrained(merged_dir)` is REQUIRED — without it the converter falls
     back to the base tokenizer (missing the 7 specials and BPE merges) and llama.cpp
     fails with "cannot find tokenizer merges in model file".
  3. GGUF: llama.cpp toolchain directly (`~/llama.cpp/convert_hf_to_gguf.py --outtype
     bf16` then `build/bin/llama-quantize ... q4_k_m`) — unsloth's `save_to_gguf`
     failed downloading its converter script. The Qwen3-4B-Instruct-2507 tokenizer
     hash (`4f53cda1...`) must be registered as `qwen2` in `conversion/base.py`
     `get_vocab_base_pre`.
  4. Outputs: `merged_bf16/` (+tokenizer), `model_state.safetensors`, `train_cfg.json`,
     `gguf/DuplexCascade-PT-{bf16,q4_k_m}.gguf`. q4_k_m GGUF verified generating
     `<|user finish speaking|>`-led duplex responses at ~170 t/s on the 4080.
- Write `train_cfg.json` in the repo/local model dir exactly as `server.py`
  expects (`{"model": {"name": ..., "trust_remote_code": false}}`), plus
  `model_state.safetensors` (merged state dict, keys compatible with `model.py`
  load path) and the `tokenizer/` snapshot. This keeps `server.py`
  `_load_model_state_dict` path working.
- `model.py`: switch `enable_lora_adapter` default mode to the QLoRA target set
  ("all linear") and make adapter loading tolerant of Unsloth-saved adapter
  (no `modules_to_save` requirement mismatch; adapt if needed).

## 7. Inference stack (open-source backends)

### 7.1 STT — `stt_service.py` (new)
- `faster-whisper` (CTranslate2) model `small` (default) with `language="pt"`,
  `beam_size=5`, VAD (bundled) ON.
- Streaming approach: incremental chunked transcription on a sliding 0.6 s
  window; the transcript is diffed against the previously-emitted words via
  longest-common-prefix and only the new suffix is sent as
  `{"type":"Word",...}` events to the bridge.
- Async websockets server (websockets 17, msgpack) on `:31607`; accepts 24 kHz
  float32 PCM frames, resamples 24k→16k with `scipy.signal.resample_poly`
  before Whisper. Emits `{"type":"Eos"}` on client `Eos`.

### 7.2 TTS — `tts_service.py` (new)
- `pip install pocket-tts`; `TTSModel.load_model(language="portuguese")`
  (the available pt option; `portuguese_24l` is not a selectable language in
  pocket-tts 2.1.0).
- Voice: `get_state_for_audio_prompt(<voice>)` with the language default voice
  (`rafael` for pt) — an embedding from the predefined-voice catalog (no
  voice-cloning model needed). The empty `init_states(...)` state produces
  truncated audio; the voice-conditioned state is required for full-length
  synthesis.
- Streaming: `generate_audio_stream(voice_state, text)` yields 24 kHz chunks;
  split into 1920-sample (80 ms) frames as `{"type":"Audio"}` msgpack messages;
  `{"type":"Eos"}` on client `Eos`. GPU on the 4080, faster than real time.

### 7.3 `server.py` adaptations (bridge)
- **LLM backend is llama.cpp** (`--llm-api-base`, default
  `http://127.0.0.1:8080/v1`) serving the fine-tuned DuplexCascade-PT GGUF. The
  bridge builds the ChatML duplex prompt by hand and calls the raw
  `/v1/completions` endpoint (streaming, with a non-streaming fallback) — the
  chat endpoint's template mangles `<|...|>` special tokens in assistant history.
- **`-sp` flag required on llama-server** so duplex special tokens
  (`<|user is speaking|>`, ...) are emitted in completions output; without it the
  API silently drops them.
- Browser protocol unchanged (24 kHz f32 PCM frames in, binary f32 audio out,
  JSON `user_asr` / `assistant_text` / `assistant_special` / `audio_control`).
- Turn completion: the bridge accumulates ASR words and, after a silence of
  `--overlap-window-s`, sends the full user turn and streams the assistant
  micro-turn. **The assistant turn is primed with `<|user finish speaking|>`**
  (appended to the `assistant\n` header in the prompt) so the fine-tuned model
  continues with the actual answer — without the prime it degenerates into
  `<|user is speaking|>` / `<|user is thinking|>` loops and echoes the user's
  words back instead of answering (fixed 2026-08-14). A system directive
  ("você é um assistente de voz...") also steers output.
- Generic `--stt-ws` / `--tts-ws` args (defaults `:31607` / `:31608`);
  pt-BR-safe `filter_period_for_tts` (strips only sentence-final periods).
- Static demo serving adapted to **websockets 17** (`process_request(conn, req)`
  returning a `Response`; the old tuple return is unsupported).
- Backend process management: `run_services.sh` launching llama-server + stt +
  tts + bridge; `README` updated for the new stack.

## 8. Evaluations (§4 of the new paper)

1. **PT full-duplex turn-taking eval**: build a pt-BR analogue of Full-Duplex-Bench
   by synthesizing the 6 base phenomena scenarios with Qwen3-TTS (§5.3) on pt-BR
   text templates (pause handling, backchannel, smooth turn taking, user
   interruption; synthetic + candor-style); same TOR/JSD/latency metrics +
   Averaged Turn-Taking Accuracy definition from the original.
2. **PT conversational intelligence**: pt-BR text suites for instruction following
   and reasoning (e.g. available pt subsets of MMLU/ARC/IFEval — exact choice per
   availability) + human-likeness of turn-taking via listening test on synthetic
   audio (MOS) — small-scale.
3. **ASR**: Whisper pt WER/CER on Common Voice pt dev/test — report as chain-level
   eval across Whisper sizes (small vs medium).
4. **TTS**: hi-fi/metric (WER of synthesized speech re-recognized by Whisper pt =
   intelligibility), latency.
5. **Comparison table**: DuplexCascade-PT vs (original) DuplexCascade numbers
   copied from the paper where the benchmark is shared (EN benchmarks retained as
   a second table for continuity), vs pipeline baselines.

6. **Training config final decisions** (2026-08-12): base model
   `Qwen/Qwen3-4B-Instruct-2507` (was `Qwen3-8B` until 2026-08-12 — changed for VRAM
   headroom), **2,000 steps** full run (DECIDED 2026-08-12: keep embed fine-tuning,
   ~12 h on the 4080; 5k would be ~29 h), pocket-tts `portuguese_24l`.

## 9. Non-goals (v1)
- No Whisper fine-tuning (unless needed, §5.3c).
- No on-device/browser-side inference (existing web UI kept as-is).
- No distributed training (single 4080).
- No streaming ASR from scratch / no SotA on EN Full-Duplex-Bench: primary target
  is a **reproducible pt-BR full-duplex system** on open-source components + honest
  evaluation on the pt-BR eval set.

## 10. Deliverables & repo layout
```
duplex_cascade/
  SPEC.md
  DuplexCascade/            (vendored inference base; adapted for v2)
  duplex_cascade_pt_article/        → new article sources
    main.tex  figs/  refs.bib       (LaTeX, IEEE-style like original)
  training/
    train_qlora.py          (unsloth QLoRA, custom weighted/masked loss)
    prep_dataset.py         (from §5 data pipeline)
  data/                     (builders + configs; not the raw corpora)
  services/
    stt_service.py          (faster-whisper, :31607)
    tts_service.py          (pocket-tts, :31608)
  eval/                     (pt turn-taking eval, asr/tts evals)
  run_services.sh
  README.md                 (updated instructions for the new stack)
```
Final top-level output: `duplex_cascade_article_v2_pt.pdf` (the new article) +
runnable code + trained adapter/merged model locally (with instructions to publish
to HF if the user opts in).

## 11. Milestones (execution order)
1. **M1 — Data**: dialogue generator (pt-BR, 50k) → duplex micro-turn builder (pt-BR
   backchannel vocab, `<BC/>` markers) → smoke pipeline on 2k dialogues.
2. **M2 — Training**: Unsloth QLoRA smoke run (64–512 samples, 100–500 steps) on
   GPU; verify custom loss masks/weights; full 5k-step run; export merged +
   adapter + `train_cfg.json`.
3. **M3 — Services**: `stt_service.py` (whisper pt + VAD streaming, word events),
   `tts_service.py` (pocket-tts pt streaming); integrate with `server.py`; end-to-end
   browser demo in pt-BR.
4. **M4 — Eval**: pt turn-taking eval set (Qwen3-TTS synthesized), Whisper+Common
   Voice pt ASR eval, TTS eval, text/LLM benchmarks; produce result tables.
5. **M5 — Article**: LaTeX rewrite of the paper (structure of original §1–§7,
   content replaced/updated: QLoRA+Unsloth §3/§4, pt-BR data §3.3, Whisper/pocket-tts
   §4.1/inference, new eval §4.2–§4.5, updated related work for QLoRA/peft-ecosystem
   & open-source TTS/STT), compile to `duplex_cascade_article_v2_pt.pdf`.

## 12. Risks & open questions
- **Training time on 4080**: worst case >24 h for 5k steps → fallback: 2.5–3k steps
  or seq packing with shorter seq (3072); results stay meaningful for the article.
- **Qwen3 thinking mode**: fine-tune format contains no think markers; if the model
  still emits them during inference, prepend a direct-answer system instruction
  (§6.1).
- **Weights for special tokens are rare**: weighted loss + LR is the mitigation;
  monitor per-token F1 of the 6 tokens during eval instead of only CE.
- **pocket-tts pt voice quality**: if built-in pt voice surfaces unavailable at
  runtime, clone from a Common Voice pt reference (verified: `get_state_for_audio_prompt`
  accepts arbitrary wav).
- **Whisper word-streaming latency**: if incremental short-context transcripts are
  unstable in pt-BR (accents, colloquial), fall back to fixed 0.6 s chunk +
  overlap transcription (matches micro-turn cadence) — still "streaming" at turn
  granularity.
- **Data licensing**: synthetic data generated by open models (Qwen3, Apache-2.0) —
  fine; Common Voice pt — CC0; pocket-tts — CC BY 4.0 (attribution in paper/app);
  Whisper — MIT.
- **Decision points**: ~~base model~~ (RESOLVED: `Qwen/Qwen3-4B-Instruct-2507`, changed
  from `Qwen3-8B` on 2026-08-12 for VRAM fit on the 16 GB 4080; dense `Qwen3ForCausalLM`,
  unsloth fast path applies), ~~steps~~ (RESOLVED:
  5k with 2k smoke), ~~pocket-tts variant~~ (RESOLVED: `portuguese_24l`); DIALOGUE
  generator (RESOLVED: llama.cpp instance, OpenAI-compatible API at
  `http://127.0.0.1:8080/v1`, currently `Gemma_4_E4B/gemma-4-E4B-it-UD-Q4_K_XL.gguf`;
  switched from `Qwen3.5-9B-UD` for better pt-BR fluency; thinking disabled via
  `chat_template_kwargs: {"enable_thinking": false}`), whether to also publish on HF.
  Each defaults as recommended above; user may override.

## 13. Immediate next steps (backlog to begin after spec approval)
1. Scaffold repo layout (§10).
2. Implement `data/` dialogue generator + duplex builder; run on 2k dialogues to
   validate shapes/loss masking.
3. Implement `training/train_qlora.py` + custom weighted-loss trainer; 100-step
   smoke run on the 4080.
4. Implement `services/stt_service.py`, `services/tts_service.py`; adapt `server.py`;
   bring up the demo end-to-end with the base (unfine-tuned) LLM first.
5. Kick off full training run; meanwhile build eval sets and the LaTeX article
   skeleton.