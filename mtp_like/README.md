# mtp_like — MTP-Style Duplex Protocol Companion for Any LLM

A **novel approach to full-duplex turn-taking** (the DuplexCascade paper,
`sft/duplex_cascade_article.pdf`): instead of fine-tuning the conversational
LLM to both answer AND emit the duplex control tokens, a **small companion
model** (Qwen3.5-0.8B, fine-tuned here) plays the role of an MTP-style head
for the *protocol*:

> While the big model writes the answer, the companion predicts, step by
> step, **only the tokens of the duplex cascade protocol** — which control
> tag (if any) fires at the current point of the conversation. On non-tag
> positions it emits nothing (an empty string that is NOT added to the
> context). Only fired tags enter the conversation and are acted upon by the
> orchestrator.

The companion therefore acts as the **controller** of the system: it decides
when the user finished speaking (`<|user finish speaking|>` → start the
answer), when the user barged in (`<|user interruption|>` → hard stop), when a
backchannel is not an interruption (`<|user backchannel|>` → keep talking),
when silence means "keep talking" (`no tag`), when the user is composing
(`<|user is thinking|>`), and — since v6 — when a **prompt rule** fires
(`<|system take floor|>` → the system interrupts the user NOW,
`<|system handover|>` → hand control to the big model). No VAD is used
anywhere: turn-taking is learned from the ASR word stream + generated text.

Because the companion speaks a tiny protocol vocabulary and reads ordinary
text, this behavior can be attached to **any LLM** — Qwen3.5-4B, Qwen3-4B,
llama.cpp GGUFs, vLLM — without fine-tuning it.

## The promptable controller (v6)

The companion is trained to **read a system instruction** in its context
(`--policy-rule "..."`) and fire two extra tags when the rule matches:

- `<|system take floor|>` — the rule fired and the SYSTEM interrupts the user
  mid-speech. The bridge responds **instantly** (`--interrupt-phrase`, local
  TTS, sub-second) — the big model is too slow to cut the user off.
- `<|system handover|>` — the rule fired; control goes to the big model,
  which decides what to do (respond now, or nothing).

The training data builds synthetic rule scenarios (per-dialogue): the rule
text as a system item, a trigger chunk (with realistic dictation shape —
variable digit count, commas, full alphabet for the letter rule), positives
(weight 5) and negatives (rule + non-trigger chunk → the normal tag), so the
model learns the *concept* (number vs letter) rather than memorized words.
An instruction mix (~50% of regular records) keeps the base controller
behaviors intact, and the **SDFT distill loss** (tag CE + forward-KL to the
frozen base, `--kl-weight 0.3 --top-k 32`) preserves the companion's language
and instruction-following ability.

Runtime details that matter for the letter rule (see SPEC §12 D11/D12):
- STT digits are normalized to words (`digit_words` in `demo/server.py`):
  `"1, 2, 3, a"` → `"um, dois, três, a"` — the trained dictation shape.
- Words are fed to the companion **mid-utterance in both states** (answering =
  barge-in, listening = prompt-rule take_floor), chunked on commas so each
  digit/letter is its own `[u chunk, a tag]` pair.
- `system_take_floor` fires at probability ≥ 0.50 (digit-only streams stay at
  0.08–0.40; letters in the dictation rhythm fire at 0.60–0.86).
- After the interrupt, the rest of the utterance is not re-fed (no double
  fire) and the `UtteranceEnd` transcript is dropped.

## Quick start (browser demo)

```bash
./run_services.sh          # STT :31607 + TTS :31608 + bridge :31606
# open http://localhost:31606, click "Start Microphone", speak in pt-BR
```

Requires: the unsloth venv (`/home/penhfel/unsloth_uv`), faster-whisper,
pocket-tts, the trained companion (`/mnt/f/duplex_cascade_runs/mtp_like/runs/final_v6/merged`) and a
stock big model (default `Qwen/Qwen3-4B-Instruct-2507`, cached).

The bridge (`demo/server.py`) runs the orchestrator in-process: STT words →
companion query → tag → start/stop the stock big model → TTS to the browser.
The browser UI is the same duplex demo as `sft/`, plus a **companion state
panel** (footer) that shows every controller decision live — the fired tag
and its top-3 probabilities — so you can watch the model trigger.

Demo flow (Banco Penha persona, `BIG_SYSTEM` in the orchestrator): say
"quero um cartão de crédito" → the big model asks for the CPF (it has
conversation history now — it never re-asks); dictate "1, 2, 3, x, 5" → the
companion fires `<|system take floor|>` on the letter and the system cuts you
off with "Apenas números são aceitos no CPF.".

## Architecture

```
Browser (24 kHz PCM) ⇄ demo/server.py (bridge) ⇄ services/stt_service.py (Whisper pt)
        │                                          (word events, no VAD — word-gap timer)
        ▼
  Orchestrator (policy/orchestrator.py) ── the companion is the controller
        │  ● user words / silence / sentence boundary events
        │  ● queries the companion (policy/duplex_policy.py, 0.8B, ~ms)
        │  ● tags drive the system: start/pause/stop answer + TTS
        │  ● prompt rules (system item) → take_floor / handover
        ▼
  Big model (stock Qwen3-4B / Qwen3.5-4B, generates answer text only)
        │  ● receives the CONVERSATION HISTORY (protocol tags filtered)
        ▼
  services/tts_service.py (pocket-tts pt) ⇄ browser
```

The duplex context is the same interleaved user/assistant item stream the
companion was trained on; fired tags are real context items.

## Training (reproduce the companion)

1. **Data** — the exact sft duplex dataset (10k pt-BR dialogues from
   `sft/data/build_duplex_dataset.py`):

   ```bash
   python training/prep_policy_dataset.py \
     --duplex ../sft/data/duplex_train_continue.jsonl --out /mnt/f/duplex_cascade_runs/mtp_like/prepped_policy
   ```
   Builds ~946k training records: the main micro-turn stream plus the
   augmentations that teach the controller behaviors:
   - **SAS** (streaming-aware supervision): partial user chunks → speaking,
   - **barge-in**: open assistant item + user words → interruption (+
     backchannel and silence negatives),
   - **silence semantics**: user chunk + `<|no voice|>` → finish/speaking,
     including minimal `[u chunk, u no voice|]` records for single-batch
     utterances,
   - **prompt-rule scenarios** (`--with-rules`): the rule as a system item +
     trigger positives (weight 5) + negatives → `system take floor` /
     `system handover`,
   - **instruction mix**: ~50% of the regular records get a generic system
     instruction so the model keeps following rules.
   Each record carries a `loss_weight` so rare controller patterns dominate.

2. **Train** (unsloth QLoRA on the text-only Qwen3.5-0.8B, with the SDFT
   distill loss):

   ```bash
   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
   python training/train_policy.py --dataset /mnt/f/duplex_cascade_runs/mtp_like/prepped_policy \
     --max-steps 3000 --per-device-batch-size 1 --grad-accum 16 \
     --kl-weight 0.3 --top-k 32 \
     --out-dir /mnt/f/duplex_cascade_runs/mtp_like/runs/final_v6
   ```
   ~5 h on a single RTX 4080. Batch 1 × accum 16 is mandatory on this box:
   batch 2 peaks at 15.8 GB VRAM and the WSL2 dxg driver mirrors VRAM into
   host RAM → deterministic OOM (see SPEC §12 D9).

3. **Export** the merged bf16 companion:

   ```bash
   python training/export_policy.py --adapter /mnt/f/duplex_cascade_runs/mtp_like/runs/final_v6/adapter \
     --out /mnt/f/duplex_cascade_runs/mtp_like/runs/final_v6/merged
   ```

### Environment notes (the toolchain saga — see SPEC §12 D5)

- unsloth **2026.9.7** (has native "Fast Qwen3_5 patching").
- The GatedDeltaNet fast path needs `fla` + `causal-conv1d`. causal-conv1d has
  no wheel for torch 2.11/cu130: install the pip CUDA 13.0 toolkit
  (`nvidia-cuda-nvcc`), copy the missing headers (`nv/target`, `cub`,
  `thrust`, `cuda/std`) from the system CUDA 12.1 install, swap in a newer
  `ptxas` (from `nvidia-cuda-nvcc==13.4.92`), add the `libcudart.so` symlink,
  and build the wheel manually against the venv's torch (`setup.py
  bdist_wheel` — pip build isolation silently upgrades torch and breaks the
  venv).
- Always verify prep output against the source dialogues (see SPEC §12 D6 for
  the two data bugs found by eval-driven debugging).
- `UnslothTrainer._prepare_dataset` silently replaces the passed collator
  with `DataCollatorForSeq2Seq` when the dataset has a `labels` column (it
  drops `loss_weight` and the max-length cap). `PolicyTrainer` re-forces the
  real collator (SPEC §12 D10).
- Datasets and trained models live on `/mnt/f/duplex_cascade_runs/mtp_like/`
  (this repo keeps only code).

## Evaluation

```bash
python eval/eval_policy.py --model /mnt/f/duplex_cascade_runs/mtp_like/runs/final_v6/merged   # tag metrics + controller probes
python eval/eval_e2e.py --companion /mnt/f/duplex_cascade_runs/mtp_like/runs/final_v6/merged  # scripted duplex conversation
```

Final companion (v6) on the held-out duplex split (3,222 boundaries):

| tag | precision | recall | F1 |
|---|---|---|---|
| no tag (emit nothing) | 1.00 | 0.95 | 0.973 |
| `<|user is speaking|>` | 0.99 | 0.92 | 0.953 |
| `<|user finish speaking|>` | 0.79 | 0.98 | 0.877 |
| `<|user interruption|>` | 0.95 | 0.99 | 0.969 |
| `<|user backchannel|>` | 1.00 | 0.98 | 0.99 |
| `<|user is thinking|>` | 0.90 | 1.00 | 0.949 |
| `<|system take floor|>` | 1.00 | 1.00 | 1.00 |
| `<|system handover|>` | 1.00 | 1.00 | 1.00 |

Overall tag accuracy **0.956** (v5: 0.975 — v6 trades a little accuracy for
the instruction-following capability). Controller probes (runtime query
path): barge-in during assistant speech → interruption (0.99); partial
barge-in → interruption (0.99); backchannel "aham" → backchannel (0.83);
silence during assistant speech → keep talking; finished question + silence →
finish (0.99, starts the answer). Promptable-controller probes: CPF rule +
letter (3-digit rhythm) → take_floor (1.00); CPF rule + ONE digit + letter
(live dictation shape) → take_floor (0.60–0.86 in runtime probes); rule +
digits only → no fire; **no rule + letter → no fire** (the model never fires
the system tags without an instruction); handover rule + trigger → handover
(1.00); rule + unrelated words → no fire.

The learned turn-end signal is sentence-final punctuation + silence, so the
ASR must punctuate (faster-whisper does by default).

## Known limitations

- The big model demo defaults to `Qwen3-4B-Instruct-2507`: Qwen3.5-4B
  verbalizes its thinking ("thinking / Thinking Process:") regardless of
  `enable_thinking=False` (the same reason the sft project rejected it).
  The companion itself is model-agnostic.
- `system_backchannel` is not in the sft base dataset (it is a beta variant).
- The letter rule fires reliably once the dictation rhythm is established
  (2+ digit chunks); a letter after a single digit is a weaker signal
  (0.17–0.60) — a training-data variant with more digit-count variety is the
  planned fix.
- The remaining probe miss: after the assistant's answer + silence the model
  emits no tag instead of `<|user is thinking|>` — harmless (both mean wait).

## Repo layout

```
mtp_like/
  SPEC.md                       design + decision log
  data/common.py                protocol tokens, weights, backchannels
  training/
    prep_policy_dataset.py      sft duplex jsonl -> policy records (SAS + barge-in + silence + rules)
    train_policy.py             unsloth QLoRA + weighted loss + SDFT distill + per-tag eval
    export_policy.py            adapter -> merged bf16 + tokenizer
  policy/
    duplex_policy.py            companion sidecar: context render + thresholds + guards
    orchestrator.py             the controller loop (CLI demo too; prompt rules; big-model history)
  demo/
    server.py                   browser bridge (:31606), web/ UI (companion state panel)
  eval/
    eval_policy.py              tag F1 + controller probes (incl. prompt-rule probes)
    eval_e2e.py                 scripted duplex conversation with a stock big model
  run_services.sh               launch STT + TTS + bridge
  /mnt/f/duplex_cascade_runs/mtp_like/runs/final_v6/       adapter + merged model (the trained companion)
```

`SPEC.md` is the working design document and decision log.