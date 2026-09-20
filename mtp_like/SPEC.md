# SPEC — mtp_like: MTP-Style Duplex Protocol Companion for Any LLM

Version: 0.1 (draft)
Date: 2026-09-18

## 1. Objective

Build a **new, model-agnostic way to give any LLM the DuplexCascade turn-taking
behavior** described in `sft/duplex_cascade_article.pdf`, without fine-tuning the
main (big) LLM.

The original approach (`sft/`) fine-tunes the conversational LLM itself
(Qwen3-4B-Instruct-2507) so that it interleaves its answer text with the duplex
control tokens (`<|user is speaking|>`, `<|user finish speaking|>`, ...). This
**couples answer generation with conversation control**: turn-taking skill and
answer quality live in the same weights, so every new backbone needs a full SFT
run, and the model can only be used by systems that understand its special
tokens.

This project replaces that coupling with a **small companion model that plays the
role of a MultiToken-Prediction (MTP) head for the duplex protocol**:

> While the big model writes the answer, the companion (0.8B) predicts, step by
> step, **only the tokens of the duplex cascade protocol** — i.e. which control
> tag (if any) fires at the current point of the conversation. On non-tag
> positions it emits nothing (an empty string that is NOT added to the context).
> Only when a duplex tag fires is it injected into the conversation context and
> acted upon by the orchestrator.

Because the companion's output vocabulary is just `{∅, the 6–7 duplex tags}` and
its input is the same duplex conversation stream, this behavior can be attached
to **any** LLM backend (text-only, no special tokens required on the big model,
no VAD — the companion learns turn-taking from the ASR word stream + generated
text, exactly as the paper's supervised data teaches).

Concrete target:

- **Companion (trained):** `Qwen/Qwen3.5-0.8B` (Qwen3.5 Small family, Apache-2.0,
  text-only loading path).
- **Big model (untouched, off-the-shelf):** `Qwen/Qwen3.5-4B`.
- **Data:** the exact same duplex micro-turn dataset used by `sft/` training
  (`sft/data/duplex_train.jsonl`, produced by `sft/data/build_duplex_dataset.py`
  from the 50k pt-BR dialogues) — same seeds, same phenomena, same supervision
  semantics, but the companion learns the **tag stream**, not the answer text.

Final deliverables: runnable code (data prep, training, runtime orchestrator,
eval) and a trained 0.8B companion that enables a stock Qwen3.5-4B to do
full-duplex pt-BR conversation in the existing `sft/` demo stack (STT/TTS/bridge).

## 2. Background: what `sft/` does today (baseline to build on)

- `sft/data/build_duplex_dataset.py` converts plain pt-BR dialogues into
  interleaved user/assistant **micro-turns** and simulates the 6 interaction
  phenomena (paper §3.3): random micro-turn lengths, natural pauses
  (`<|no voice|>` → `<|user is speaking|>`), user interruptions
  (`<|user interruption|>`), user backchannels (`<|user backchannel|>`),
  system backchannels (`<|system backchannel|>`, beta variant), thinking pauses
  (`<|user is thinking|>`). Output items: `{role, token_ids, text, special,
  weight}` (token_ids are Qwen3-4B tokenizer ids).
- `sft/training/prep_dataset.py` + `train_qlora.py` fine-tune the big model with
  QLoRA, weighted/masked CE: loss only on assistant micro-turns, per-tag weights
  (`<|user finish speaking|>`=10, `<|user interruption|>`=5, ...).
- `sft/DuplexCascade/server.py` + `sft/services/` (faster-whisper STT, pocket-tts
  TTS) run the live full-duplex browser demo with the fine-tuned GGUF served by
  llama.cpp (`-sp` flag to emit special tokens).

**Key insight reused here:** the duplex training data already defines a clean
*control signal* — for every micro-turn, the "next protocol action" is known.
`sft/` trains the answer model to produce that signal inline; `mtp_like/` trains
a tiny separate model to produce exactly that signal as its only output.

## 3. The MTP-like companion (novel contribution)

### 3.1 Analogy with MTP (DeepSeek-style)

MultiToken Prediction appends a small module to the main model that, from the
main model's hidden state and the next token's embedding, predicts **one token
ahead in parallel** with the main model's own next-token prediction — the main
model keeps doing its job, MTP just looks ahead.

The companion does the same *role* split, but at the protocol level:

| | Main model (Qwen3.5-4B) | Companion (Qwen3.5-0.8B, trained here) |
|---|---|---|
| Predicts | the next **answer** token | the next **duplex control token** (or ∅) |
| Vocabulary | full 248k | `{∅, <\|user is speaking\|>, <\|user finish speaking\|>, <\|user interruption\|>, <\|user backchannel\|>, <\|user is thinking\|>, <\|system backchannel\|>}` |
| Input | duplex context (user + assistant turns) | the SAME duplex context |
| Runs | generates speech content | alongside / one step ahead, on every user word event and sentence boundary |
| Trained? | **no** (off-the-shelf) | **yes** (SFT on the sft duplex dataset, tag targets only) |

"One step ahead" is what makes it preemptive: while the big model is writing the
answer, the companion can already predict that an `<|user interruption|>` fires
at the current boundary, so the orchestrator can stop the utterance *before* the
model would have reached that state — no VAD needed, the signal is learned from
the same supervised phenomena of the paper.

### 3.2 Why this is a different (novel) method vs `sft/`

1. **Decoupling**: answer quality (big model) and turn-taking skill (companion)
   are learned independently. No SFT of the big model at all.
2. **Model agnosticism**: the companion speaks a tiny protocol vocabulary and
   reads ordinary text + a few markers. Any LLM (any tokenizer, any template,
   no special tokens) can be dropped in — Qwen3.5-4B, Qwen3-4B, llama.cpp GGUF,
   vLLM, etc. The duplex tags are orchestration state, not big-model vocabulary.
3. **Efficiency**: training a 0.8B on a tag-stream task is minutes-to-hours on a
   single 4080 (vs ~12 h for the sft QLoRA run); inference is a cheap sidecar
   (~ms per query) that can even share the GPU with the serving big model.
4. **Controllability**: thresholds, per-tag confidence gating, and hard rules
   (e.g. never interrupt during a filler) live in the orchestrator, not in
   learned weights of the answer model.

## 4. Architecture (runtime)

```
Browser (24 kHz PCM) ⇄ bridge/orchestrator (adapted from sft/DuplexCascade/server.py)
        │                                  │
        ▼                                  ▼
  STT (faster-whisper pt,          TTS (pocket-tts pt)
  per-word events)                 24 kHz PCM out
        │                                  │
        └──────────► ORCHESTRATOR ◄────────┘
                        │  │
        duplex context (tags + text + <|no voice|> markers)
        │                 │
        ▼                 ▼
  COMPANION (0.8B)   BIG MODEL (Qwen3.5-4B, stock)
  predicts next tag  generates answer text only
  (∅ / 6 tags)
```

### 4.1 Duplex context (shared state)

The orchestrator keeps the conversation as a list of items — the same shape as
the training data:

- `user` items: ASR words accumulated into chunks (`text`), or the marker
  `<|no voice|>` after an ASR-word gap (absence of words, **not VAD** — same
  trick as the paper's VAD-free design).
- `assistant` items: text chunks emitted by the big model, plus the **tags**
  emitted by the companion (they are items in the stream and are rendered into
  the context the companion reads).

### 4.2 Orchestration loop (MTP-like cadence)

The companion is queried on every **event** that can change the protocol state:

1. **User word event** (new ASR words): render context → companion → tag:
   - `<|user is speaking|>` → user mid-utterance: pause big-model generation/TTS
     if the assistant was speaking; keep waiting.
   - `<|user finish speaking|>` → user turn complete: trigger the big model to
     answer (prime = user turn; companion emits this instead of the sft
     `<|user finish speaking|>`-primed assistant header).
   - `<|user interruption|>` → stop generation + TTS immediately, drop pending
     audio, let the user talk.
   - `<|user backchannel|>` → user said a short acknowledgment ("aham", "tá"):
     keep the assistant's current utterance going (no hard stop).
   - `<|system backchannel|>` (beta) → assistant should backchannel: play a
     short canned pt-BR backchannel (or 1–2 word big-model prompt).
2. **Sentence boundary** (big model emitted `.?!…`): render context → companion →
   tag: usually ∅ (keep talking) or a transition tag (e.g. `<|user is
   thinking|>` when the user goes silent after the assistant's turn).
3. **Silence timer** (no ASR words for `--word-gap-s`, default 0.6 s): append a
   `<|no voice|>` user item → companion → usually `<|user is thinking|>`
   (post-answer) or `<|user is speaking|>` (mid-user-turn pause), which is the
   learned, VAD-free silence handling.

Emitted tags are appended to the duplex context (they are real context items for
the companion) and translated by the orchestrator into actions for the big
model + TTS. The big model itself never sees special tokens — at most it sees a
natural-language control line in its prompt (see decision D4).

### 4.3 Prediction format

The companion is a causal LM with a **small linear head** over the last hidden
state: `7 classes` = `{EMPTY, 6 tags}` (7 tags if the beta `<|system
backchannel|>` variant is trained). `EMPTY` means "emit nothing — do not add
anything to the context" (the user's "empty string that would not be considered
in the context"). At inference a tag fires only if `P(tag) ≥ θ_tag`
(per-tag threshold, tuned on eval).

## 5. Training data (same dataset as sft, tag targets only)

### 5.1 Source

Reuse the sft duplex pipeline outputs unchanged (same seeds → same data):

```
../sft/data/duplex_train.jsonl      # produced by ../sft/data/build_duplex_dataset.py
```

Each line: `{id, scenario, items: [{role, token_ids, text, special, weight}], meta}`.
`token_ids` use the Qwen3-4B tokenizer; the companion re-tokenizes from `text`
with the Qwen3.5 tokenizer (both Qwen3.5 Small models share one tokenizer —
verify in M0). The 7 duplex specials are `add_special_tokens`'d into the
companion's tokenizer.

### 5.2 Target stream (the core transformation)

For item `i`, define `target(i)` = the tag the assistant should emit **after
item i has entered the context** — read off directly from the sft supervision:

| item i | target(i) |
|---|---|
| user chunk, non-final chunk of its turn | `<|user is speaking|>` |
| user chunk, final chunk of its turn | `<|user finish speaking|>` |
| user chunk, first chunk of an interrupting turn | `<|user interruption|>` |
| user chunk, backchannel word | `<|user backchannel|>` |
| user chunk with `<BC/>` (beta) | `<|system backchannel|>` |
| `<|no voice|>` during user's turn (pause) | `<|user is speaking|>` |
| `<|no voice|>` after system turn (thinking pairs) | `<|user is thinking|>` |
| any assistant item (plain text or tag-bearing) | `EMPTY` |

Notes:

- The companion NEVER predicts `<|no voice|>` (training artifact, orchestrator
  state only) and never emits answer text.
- The signal that separates `<|user is speaking|>` from `<|user finish
  speaking|>` is exactly the paper's chunk-boundary supervision: non-final
  chunks end mid-sentence, the final chunk ends at a sentence boundary. The
  companion learns this turn-end cue from the data — this is the learned
  replacement for VAD.
- **Streaming-aware supervision (SAS) augmentation**: at inference the user
  words arrive incrementally, so for every user chunk we also add *partial*
  prefixes (first k tokens) to the training stream, all labeled with the tag of
  their containing chunk EXCEPT that partial prefixes of the *final* chunk are
  labeled `<|user is speaking|>` (only the complete final chunk fires
  `<|user finish speaking|>`). This teaches: "words still arriving → keep
  saying speaking; complete sentence unit → finish".
- **Class imbalance**: `EMPTY` dominates (every assistant item). Mitigations:
  (a) weighted CE with `TOKEN_WEIGHT` from `sft/data/common.py` re-applied to
  tags and a small weight (e.g. 0.1–0.3) for `EMPTY`; (b) optionally focal loss;
  (c) per-tag decision thresholds at inference.

### 5.3 Barge-in (controller) augmentation — DECIDED

The companion is the **system controller**: it must learn that the user can
barge in mid-utterance. The sft data already simulates interruptions (0.76% of
boundaries), and the prep adds dedicated controller records:

- for random points inside every system turn, the current assistant item is
  rendered as an **open item** (partial content, no `<|im_end|>` — speech in
  progress) followed by:
  - the user's next real words → `<|user interruption|>` (plus a partial-prefix
    variant: the FIRST words of the barge-in),
  - `<|no voice|>` → `no_tag` (silence during speech = KEEP TALKING),
  - a sampled pt-BR backchannel word → `<|user backchannel|>` (a short
    acknowledgment is NOT a barge-in).
- This is what makes the trained companion "the controller": it distinguishes
  barge-in (hard stop) from backchannel (keep talking) from silence (keep
  talking), without VAD.

### 5.3b Silence semantics (turn-end) — DECIDED (iterated)

The runtime queries the companion on ASR silence by appending `<|no voice|>`.
The data must teach what silence means in each state. For every user content
chunk we add `chunk + <|no voice|>` records:

- non-final chunk + `<|no voice|>` → `<|user is speaking|>` (paused mid-speech),
- FINAL chunk + `<|no voice|>` → `<|user finish speaking|>` (finished + silent
  = START ANSWERING — the sft data never shows this: finish always follows the
  last chunk directly),
- partial final + `<|no voice|>` → `<|user is speaking|>` (paused before
  completing the sentence),
- **minimal records** `[u chunk, u no voice|]` for EVERY turn-final chunk
  (weight 5): the exact runtime query shape for a short single-batch utterance.
  Without them the model never fires finish on a 2-item context (only 16 such
  records existed in the corpus → out-of-distribution at runtime).

Record-level `loss_weight` multipliers (SPEC 5.3c) make the rare but critical
controller patterns dominate the gradient.

### 5.4 Sequence format

Same ChatML conventions as `sft/training/prep_dataset.py`: user items rendered
as `<|im_start|>user\n … <|im_end|>`, assistant items as `<|im_start|>assistant\n
… <|im_end|>`, with the tag tokens and `<|no voice|>` inside the assistant/user
content (they are real tokens in the companion's vocab). The companion's
prediction is supervised at **item boundaries** (one classification per item,
using the last hidden state of the final content token of item i), not
per-token — 1 target per item, drastically fewer labels than a next-token SFT.
(LM-style "generate the next tag token" is the fallback, D3.)

## 6. Training recipe (companion)

| Choice | Value |
|---|---|
| Model | `principled-intelligence/Qwen3.5-0.8B-text-only` (Qwen3_5ForCausalLM, 0.75B — the Qwen3.5-0.8B LM without the vision tower; DECIDED: unsloth's load of the full `Qwen/Qwen3.5-0.8B` misroutes the multimodal forward on text inputs) |
| Stack | **unsloth 2026.9.7** (FastLanguageModel + UnslothTrainer; "Fast Qwen3_5 patching"), LoRA r=32 α=32 on all linears + `modules_to_save=[embed_tokens, lm_head]` (trains the 8 new special-token embeddings, `embedding_learning_rate`) |
| Fast path | `fla` + `causal-conv1d` compiled against torch 2.11/cu130 with the pip-installed CUDA 13.0 nvcc (`nvidia-cuda-nvcc`, see §12) — measured ~3100 tok/s train vs ~20 tok/s torch fallback |
| Precision | bf16, **no gradient checkpointing** (unsloth GC + 4-bit are ~10x slower on the GatedDeltaNet layers: 296 tok/s vs 3155 tok/s) |
| Data | `/mnt/f/duplex_cascade_runs/mtp_like/prepped_policy` (946k records: 10k main + 200k SAS + 155k barge + 553k silence + 28k rule + 458k instruction-mixed) |
| Loss | weighted CE over the full vocab at boundary positions only (custom `PolicyTrainer`): tags per `TOKEN_WEIGHT`, `no_tag` weight 0.3, masked elsewhere, `max_grad_norm=1.0` — plus the **SDFT distill term** (v6): forward-KL at content positions vs the frozen base (teacher = same weights, adapters disabled, top-k 32 support, `--kl-weight 0.3`) to preserve language/instruction-following |
| Eval | per-tag P/R/F1 via restricted argmax over the protocol vocabulary; logits restricted on GPU via `preprocess_logits_for_metrics` (full-vocab gather OOMs system RAM) |
| Seq len | 1024 (tail-truncated — keeps the boundary label, which is the record's final position; see §12 D11) |
| Optimizer | AdamW (unsloth), lr 1e-4, warmup 50, **batch 1 × grad-accum 16 = 16** (batch 2 = 15.8 GB VRAM → WSL2 dxg host-memory OOM, §12 D9), ~5 s/step on the 4080 |
| Steps | smoke: 200 on 2k records (eval_acc 0.79); full (v6): 3000 (~5 h) |
| Export | adapter via `model.save_pretrained`; merge to bf16 with `training/export_policy.py` (sft LEARNINGS #4: merge onto bf16, never 4-bit) — unsloth's `save_pretrained_merged`/`unsloth_save_model` assumes Llama-style `self_attn` layers and fails on Qwen3_5 |

## 6b. Promptable controller (v6)

The companion reads a **system instruction** (the rule) in its context and
fires `<|system take floor|>` / `<|system handover|>` when the rule matches:

- **Data**: `build_rule_records` (per dialogue, seeded) renders the rule as
  the first system item + a trigger chunk labeled with the rule's tag
  (positives weight 5.0, incl. the silence-query variant) and negatives
  (rule + non-trigger chunk → the normal tag, weight 1.0). The CPF rule uses
  the real dictation shape — variable digit count (1–6) and order from a rich
  number set (`zero`…`nove`, `dez`, `onze`, `doze`, `vinte`, `trinta`, `cem`,
  `mil`), comma-form chunks, the full alphabet a–z as letters — so the model
  learns the number-vs-letter *concept*; CPF negatives (digit chunks + a
  non-letter word: `meu, cpf, é, o, número, espera`) never fire.
- **Instruction mix**: ~50% of main/sas/barge/silence records get a generic
  system instruction ("Gerencie a conversa normalmente, com turnos naturais.")
  so the base controller behaviors survive the instruction conditioning.
- **Runtime rule**: `--policy-rule "..."` (default in `run_services.sh`).
  Rendered as the first system item by `Orchestrator._context`. The rule text
  must match the training templates (the demo's letter rule is verbatim).
- **Thresholds**: `system_take_floor` 0.50, `system_handover` 0.30
  (`DEFAULT_THRESHOLDS`) — digit-only streams rate 0.08–0.40, letters in the
  dictation rhythm 0.60–0.86.
- **Instant interrupt**: take_floor responds with `--interrupt-phrase`
  ("Apenas números são aceitos no CPF.") via local TTS, sub-second — the big
  model's 2–4 s generation never cuts the user off. The interrupted
  utterance's remaining deltas are not re-fed (no double fire) and its
  `UtteranceEnd` transcript is dropped (`interrupted_turn` flag in the
  bridge).
- **STT normalization**: `digit_words` in the bridge converts "1, 2, 3, a" →
  "um, dois, três, a" before feeding the companion (the trained shape).
- **Mid-utterance feeding**: words are fed to the companion in BOTH states
  (answering = barge-in; listening = prompt-rule detection), chunked on
  commas so each digit/letter is its own `[u chunk, a tag]` pair;
  `Orchestrator.on_user_utterance` wipes the partial-fed chunks
  (`_turn_start`) before the authoritative final transcript.
- **Big-model history**: the orchestrator derives the dialogue history from
  the items (protocol tags/no-voice filtered, consecutive user chunks merged)
  and passes it to `iter_sentences(user_turn, history)` — a stateless big
  model otherwise repeats itself (the bank assistant re-asked for the CPF).
- **Known gap**: a letter after a SINGLE digit chunk is a weak signal
  (0.17–0.60); after 2+ digit chunks it fires reliably. The planned fix is a
  data variant with more digit-count diversity.

## 7. Inference stack

- **Big model**: `Qwen/Qwen3.5-4B` stock, text-only, thinking mode disabled
  (`chat_template_kwargs: {"enable_thinking": false}` or a direct-answer system
  prompt — it is a multimodal checkpoint; use the text path). Served via
  llama.cpp GGUF (`unsloth/Qwen3.5-4B-GGUF` exists) or vLLM/sglang/HF.
- **Companion**: 0.8B sidecar, HF `AutoModel`/text-only class, flash attention,
  per-query latency target < 20–50 ms on the 4080.
- **Bridge**: adapt `sft/DuplexCascade/server.py` → orchestrator loop (§4.2);
  reuse `sft/services/stt_service.py` (word events) and `tts_service.py`
  (pocket-tts pt) unchanged; browser protocol unchanged.
- **No VAD**: user-speech state comes from ASR word events + word-gap timer +
  the companion's learned tags (§4.2.3).

## 8. Evaluation

1. **Tag-level accuracy (primary)**: on the held-out duplex eval split, render
   the ground-truth context and compare the companion's argmax tag per item to
   `target(i)`: per-tag precision/recall/F1, plus a single "correct decision"
   rate. Report the effect of per-tag thresholds and the SAS augmentation.
2. **Streaming simulation**: replay eval dialogues with user items fed word by
   word and assistant items token by token; measure tag F1, **detection latency**
   (how many words after the true boundary before `<|user finish speaking|>` /
   `<|user interruption|>` fires), and false-positive rates per tag.
3. **End-to-end**: orchestrator + stock Qwen3.5-4B in the sft demo stack on the
   pt-BR Full-Duplex-Bench-style scenarios (from `sft/eval/`): turn-taking
   accuracy, TOR/JSD, latency; side-by-side vs the sft fine-tuned model.
4. **Model-agnostic check**: swap the big model (e.g. Qwen3-4B-Instruct-2507 or
   a GGUF) and re-run (3) with zero companion retraining.

## 9. Repo layout (deliverables)

```
mtp_like/
  SPEC.md
  README.md
  data/
    common.py                  # tags, weights, no_voice marker (mirrors sft/data/common.py)
  training/
    prep_policy_dataset.py     # sft duplex jsonl → item-boundary target stream (+SAS)
    train_policy.py            # 0.8B fine-tune + head
    export_policy.py           # merged/GGUF + tokenizer snapshot
  policy/
    duplex_policy.py           # companion wrapper: context → (tag, confidence)
    orchestrator.py            # duplex loop: events, tag actions, big-model control
    bridge.py                  # adapted sft/DuplexCascade/server.py wiring
  eval/
    eval_policy.py             # tag F1 / thresholds / latency
    eval_streaming.py          # word-by-word replay
    eval_e2e.py                # live stack turn-taking
  run_services.sh              # big model + companion + STT + TTS + bridge
  requirements.txt
```

## 10. Milestones (execution order)

- **M0 — Verify**: Qwen3.5-0.8B/4B availability + shared tokenizer; text-only
  loading (transformers≥5.2.0, `Qwen3_5ForCausalLM`); thinking-mode disable;
  llama.cpp GGUF serving of the 4B.
- **M1 — Data**: `prep_policy_dataset.py` (target stream + SAS); stats on tag
  distribution; confirm train/eval split matches sft's seed.
- **M2 — Train**: `train_policy.py` smoke (small lr, 1k items) → full run;
  tag-level eval (§8.1) and iterate on weighting/thresholds.
- **M3 — Runtime**: `duplex_policy.py` + `orchestrator.py` + `bridge.py`;
  end-to-end with stock Qwen3.5-4B in the browser demo; streaming sim (§8.2).
- **M4 — Eval & writeup**: §8.3–8.4 comparisons vs `sft/`; README + optional
  short article describing the MTP-like companion method.

## 11. Decision points & risks

- **D1 — Companion checkpoint**: DECIDED
  `principled-intelligence/Qwen3.5-0.8B-text-only` (Qwen3.5-0.8B LM, text-only,
  0.75B). The full `Qwen/Qwen3.5-0.8B` checkpoint also loads as
  `Qwen3_5ForCausalLM` with plain transformers but misroutes its multimodal
  forward under unsloth; the trimmed variant is verified end-to-end.
- **D2 — Big model**: `Qwen/Qwen3.5-4B` (user-decided). Risk: multimodal
  checkpoint + hybrid GDN/SSM arch; thinking mode must be disabled; llama.cpp
  serving should be verified early (M0). Any text LLM is swappable by design.
- **D3 — Head vs LM-style**: LM-style boundary supervision (labels = tag token
  id or the label-only `<|no tag|>` token at item boundaries, full-vocab CE,
  restricted-argmax at inference) — DECIDED (no custom head code; standard
  masked-label training; "emit nothing" = `<|no tag|>` never enters the context).
- **D4 — Big-model awareness of tags**: default = orchestrator-only (big model
  sees plain user/assistant text); optional = inject a natural-language
  control line ("o usuário interrompeu — pare de falar") into the big model's
  prompt so it can also react (e.g. restart after interruption).
- **D5 — Toolchain (verified 2026-09-18)**: unsloth was updated 2026.8.19 →
  **2026.9.7** (native "Fast Qwen3_5 patching"; older versions had no qwen3_5
  support at all). The GatedDeltaNet fast path needs `fla` +
  `causal-conv1d`. causal-conv1d has **no wheel** for torch 2.11/cu130 and the
  system nvcc (12.1) mismatches torch's CUDA 13.0 — fixed by installing the
  pip CUDA 13.0 toolkit (`nvidia-cuda-nvcc==13.0.88` → `nvidia/cu13`), copying
  the missing `nv/target` + `cub`/`thrust`/`cuda/std` headers from the system
  CUDA 12.1 install, swapping in a newer `ptxas` from `nvidia-cuda-nvcc==13.4.92`
  (nvcc 13.0.88 emits PTX 9.4 that its own ptxas rejects), adding the
  `libcudart.so` symlink, and building the wheel MANUALLY against the venv's
  torch 2.11 (`setup.py bdist_wheel` — pip's build isolation resolves torch
  2.14 and silently upgrades the venv, breaking torchvision/unsloth's pin).
- **D6 — Prep data bugs found by eval-driven debugging (2026-09-19)**: (1) the
  `DialogueEncoder.content_cache` was keyed by item index and shared across
  dialogues → every dialogue after the first got another dialogue's TEXT as
  input (labels were right, content wrong); eval dropped from 0.98 to 0.70.
  (2) `build_silence_records` used `segments[:i+1]` (including the chunk) and
  re-added the chunk → every silence record DUPLICATED the user chunk; the
  model memorized the duplicate pattern. (3) `Dataset.from_list` OOMs at
  460k+ records → prep now streams rows to parquet. ALWAYS verify prep output
  against the source dialogue (tail items), not just the first record.
- **D7 — Runtime query shapes must match training exactly**: user items are
  ALWAYS rendered closed (`<|im_end|>\n`) even for partial content (SAS shape);
  only an in-progress ASSISTANT item is open (barge-in shape). The orchestrator
  closes user chunks only when NEW words arrive (attaching the previous tag),
  so the silence query sees `[.., u chunk, a tag, .., u final-chunk, u no voice|]`
  — the exact training shape. The learned turn-end signal is the sentence-final
  punctuation of the final chunk + silence (ASR must punctuate; faster-whisper
  does by default).
- **D8 — Big model**: Qwen3.5-4B verbalizes its thinking ("thinking / Thinking
  Process:") regardless of `enable_thinking=False` and system directives (the
  sft SPEC's documented rejection reason). The demo defaults to
  `Qwen3-4B-Instruct-2507` (no thinking, cached); the companion is
  model-agnostic by design and `--big-model Qwen/Qwen3.5-4B` remains selectable
  (e.g. under llama.cpp serving, to be verified).
- **Risks**: class imbalance (weighting + thresholds); turn-end detection
  without VAD (punctuation cues from data + SAS; Whisper punctuates by
  default); companion false interruptions (θ + filler guard from
  `sft/LEARNINGS.md` #10); the rare tags (`backchannel` 0.03%,
  `interruption` 0.76%) need the weighted loss + barge-in augmentation —
  watch `eval_user_interruption_f1` / `eval_user_backchannel_f1` during
  training.
- **D9 — WSL2 OOM (dxg host-memory mirror)**: under WSL2 the NVIDIA dxg
  driver mirrors VRAM into the VM's host RAM and it counts against the
  process's anon-RSS. Training at batch 2 × 1024 tokens peaked at 15.8 GB
  VRAM → ~31 GB anon-RSS on a 31 GB VM → deterministic `oom-kill` at ~step
  57 (three identical kills: pids 29594/30873/31934, same 50.5 GB VM).
  FIX: `--per-device-batch-size 1 --grad-accum 16` halves the forward VRAM
  (~10.8 GB, RSS ~3.1 GB) at the SAME effective batch and per-record speed.
- **D10 — Unsloth collator overwrite**: `UnslothTrainer._prepare_dataset`
  SILENTLY replaces the passed `data_collator` with
  `DataCollatorForSeq2Seq` whenever the dataset has a `labels` column
  (called during `UnslothSFTTrainer.__init__` AND at `train()`). That
  dropped the `loss_weight` column (rule weights ×5) and removed the
  `max_seq_length` cap. FIX: `PolicyTrainer` stashes the real collator in
  `_real_collator` BEFORE `super().__init__()` and re-forces
  `self.data_collator` after every `_prepare_dataset` pass.
- **D11 — truncation keeps the tail**: when a record exceeds
  `max_seq_length`, the collator keeps the LAST tokens (`v[-max_len:]`),
  not the head — the boundary label is the record's final position and
  must survive (the head/instruction is dropped instead).
- **D12 — storage**: datasets and models live on `/mnt/f/duplex_cascade_runs/mtp_like/`
  (`prepped_policy`, `runs/`), mirroring the `distill` project; the repo
  keeps only code. All script defaults point there.

## 12. Immediate next steps

1. M0 verification script (model IDs, tokenizer equality, text-only load, one
   duplex context render with the Qwen3.5 tokenizer).
2. `data/common.py` + `training/prep_policy_dataset.py`; run on
   `../sft/data/duplex_train.jsonl` and print tag distribution + example rows.
3. `training/train_policy.py` smoke run; tag F1 on the eval split.
4. `policy/` runtime; end-to-end demo with stock Qwen3.5-4B.