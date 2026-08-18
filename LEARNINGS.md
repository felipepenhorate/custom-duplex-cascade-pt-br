# Training Learnings — DuplexCascade-PT

A candid log of the problems hit and solutions found while training and serving
the pt-BR full-duplex model. These are the non-obvious gotchas that cost the most
time. Read this before re-running the [recipe](README.md#recipe-recreate-this-approach-for-a-new-language--dataset).

---

## 1. The "only answers in 1–2 short sentences" problem

**Symptom:** the fine-tuned model always produced very short assistant replies,
even when the question warranted a longer answer.

**Root cause:** the M1 dialogue generator instructed *"respostas curtas do
Assistente. Nada de textos longos"*, and — critically — the duplex chunker
(`build_duplex_dataset.py`) fixed every **assistant micro-turn at exactly 10
tokens**. So the training data only ever contained ~10-token assistant bursts,
and the model faithfully learned to speak only in ~10-token bursts.

**Fix (two parts):**
1. **Longer source dialogues** — a new generator (`build_long_dialogues.py`) that
   mixes *short* (weight 6), *long* (weight 2) and *mixed* (weight 2) styles, so a
   fraction of assistant turns are 2–4 sentence, multi-sentence replies (up to 220
   words, validation-adjusted from 180).
2. **Variable micro-turn chunking** — `build_duplex_dataset.py` now samples system
   chunk sizes in a range (`--system-chunk-min 10 --system-chunk-max 48`, default)
   instead of a fixed 10. ~16% of assistant turns are now >15 tokens.

**Lesson:** if the model's output *length* is wrong, suspect the **training data
length distribution**, not the model. Also: never fix a micro-turn length constant
unless you want that exact length.

## 2. Continue from the merged model, not the base

**Symptom:** re-training the second stage from the base model lost the learned
turn-taking behavior.

**Root cause:** the 7 special-token embeddings and the turn-taking pattern were
already learned in the first stage. Starting from the base re-randomized / lost
them.

**Fix:** `continue_finetune.sh` continues from `$RUNS_ROOT/full/export/merged_bf16`.
In `train_qlora.py`, `lazy_init_special_embeddings` only runs when
`n_added > 0` — when continuing from a merged model the specials already exist and
their embeddings are **preserved** (re-randomizing them would wipe the learned
behavior).

## 3. Special-token vocabulary / vocab mismatch (Qwen3-4B-Instruct-2507)

**Symptom:** adapter incompatible with the base; llama.cpp conversion failed with
"cannot find tokenizer merges in model file".

**Root cause:** `Qwen3-4B-Instruct-2507` ships model rows `vocab_size=151936` but an
HF tokenizer of only **151669** base tokens. Training adds 7 specials → tokenizer
151676, and `resize_token_embeddings(151676)` truncates 260 dead tail rows.

**Fix:** the adapter is only compatible with the **tokenizer saved inside the
adapter dir** (151676) + `resize_token_embeddings` before
`PeftModel.from_pretrained`. Load order that works:
```
tok = AutoTokenizer.from_pretrained(<adapter_dir>)
model.resize_token_embeddings(len(tok))
PeftModel.from_pretrained(model, <adapter_dir>)
```
And **always `tok.save_pretrained(merged_dir)`** during export.

## 4. Merge onto bf16, not 4-bit

**Symptom:** `merge_and_unload()` on a 4-bit-loaded base kept quantized bnb weights
and tripped transformers' tied-weight / weight-conversion checks. Unsloth's
`unsloth_save_model(merged_4bit_forced)` also raised `NotImplementedError`.

**Fix:** load the base in **bf16** (`dtype=torch.bfloat16, load_in_4bit=False`) and
merge there. The LoRA deltas are dtype-independent, so the merge is clean bf16 and
converts to GGUF. (`export_model.py` does this.)

## 5. GGUF export toolchain

**Symptom:** llama.cpp conversion failed; the `llama-quantize` binary was missing
after a trimmed llama.cpp rebuild. Unsloth's `save_to_gguf` failed trying to download
its converter script.

**Fix:** use llama.cpp's own toolchain directly:
`convert_hf_to_gguf.py --outtype bf16` then `build/bin/llama-quantize ... q4_k_m`.
`continue_finetune.sh` auto-builds `llama-quantize` (`cmake --build ... --target
llama-quantize`) if absent. The Qwen3-4B-Instruct-2507 tokenizer hash
(`4f53cda1...`) must be registered as `qwen2` in `conversion/base.py`
`get_vocab_base_pre`.

## 6. `-sp` is REQUIRED on llama-server

**Symptom:** the duplex special tokens (`<|user is speaking|>`, ...) silently
disappeared from completions.

**Fix:** launch llama-server with the **`-sp`** flag so the special tokens are
emitted in the completions output. Without it the API silently drops them.

## 7. The assistant-voice echo / audio bug (inference)

**Symptom:** assistant audio got cut off or never played in the browser.

**Root cause:** the bridge `rx_loop` sent `audio_control "stop"` on **every** TTS
`Eos`, which cleared the browser `playQueue` before playback.

**Fix:** on `Eos`, only extend the echo gate
(`assistant_playing_until = time.time() + 0.6`); `stop` is only sent for real
interruptions (`handle_special`).

## 8. `done=False` was a bridge bug, not a training defect

**Symptom:** single LLM calls returned only the first chunk because `<|im_end|>`
is the EOS token.

**Root cause:** the bridge was re-priming every micro-turn.

**Fix:** run the **duplex micro-turn loop** — up to 10 turns; **prime with
`<|user finish speaking|>` only on turn 0**; subsequently feed `<|no voice|>` as the
user turn and call again; break on `<|user is thinking|>` / `<|user is speaking|>` /
`<|user interruption|>`. Verified `done=True` in 0.44 s E2E; the model naturally ends
with thinking pairs.

## 9. OOM / segfault during training

**Symptom:** QLoRA training segfaulted / OOM'd partway through (around step ~746 in
the first long run).

**Root cause:** a 4B QLoRA run sits right at the 16 GB VRAM ceiling — the failure was
a **fragmented-allocation** failure, not a single oversized example.

**Fix:**
- Free VRAM before training: stop the Gemma generator (`:8081`) and any serving
  llama-server (`:8080`) — `continue_finetune.sh` does this automatically.
- `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce fragmentation.
- Conservative hyperparameters: `--per-device-batch-size 1 --grad-accum 16`.
- Cap prep length with `--max-seq 2048`.

## 10. Filler-word repetition after continuation

**Symptom:** the continued model sometimes repeated fillers (`Opa! Opa!`, `Entendi!`).

**Fix:** filter consecutive repeats in the bridge (`_is_filler()` regex + skip when
the previous sent chunk was also a filler; the filler still goes into
`micro_history`).

## 11. TTS quality trade-offs

- The built-in pt default voice is `rafael`. An empty `init_states(...)` state
  produced **truncated audio** — the voice-conditioned state
  (`get_state_for_audio_prompt`) is required for full-length synthesis.
- Original (default) speed is ~3.8 words/s. A slower, more natural pacing was
  trialed (time-stretch + sentence pauses) and **reverted** because the user
  preferred the original — the speed/pause knobs were removed from the TTS service
  and `run_services.sh`. Keep it simple unless you need the pacing.

## 12. General VRAM / runtime notes

- Embed/lm_head fine-tuning (`modules_to_save`) costs **~21 s/step** on a 16 GB
  4080 → 2k steps ≈ 12 h. LoRA-only is ~1 s/step but leaves embeddings frozen.
- Unsloth must be imported **before** transformers/trl/peft (it patches them).
- `ensure_weight_tying = True` must be set on the peft config (the adapter is
  otherwise not mergeable — see the peft warning about `tie_word_embeddings=True`).
