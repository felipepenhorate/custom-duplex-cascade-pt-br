# DuplexCascade-Distill

A new training methodology for the DuplexCascade micro-turn model that **learns
the duplex turn-taking protocol without regressing general capabilities** (the
current `duplex_cascade/` fine-tune drifts on function calling / tool use / etc.).

The method adapts **Self-Distillation Fine-Tuning (SDFT)** — "Self-Distillation
Enables Continual Learning" (arXiv:2601.19897) — to the duplex setting **without
RL and without on-policy rollouts**:

> The frozen original base model (prompted as a DuplexCascade system) generates
> the assistant content. During fine-tuning we (a) learn the special tags exactly
> as today (weighted CE at tag positions) and (b) keep the **logit distribution of
> everything else identical to the frozen base model** (per-token forward-KL at
> content positions, via a `disable_adapters()` teacher forward — no second model
> copy, no extra VRAM). Optionally a small capability-anchor set (function calling
> etc.) is distilled too.

See `SPEC.md` for the full design, feasibility, and plan.

## What's here (M1 status)

```
duplex_cascade_distill/
├── SPEC.md                       # design / feasibility / plan
├── data/
│   ├── common.py                 # fork: tokens, loss weights, pt-BR helpers
│   ├── teacher_prompts.py        # duplex-trigger system prompt builder
│   ├── build_teacher_duplex.py   # M1: teacher generation + duplex post-processing
│   ├── build_duplex_dataset.py   # fork: deterministic micro-turn/tag builder
│   └── build_anchor_prompts.py   # M1: capability-anchor prompt set (D2 variant)
├── training/                     # M2 (next): train_distill.py, prep_dataset.py fork
├── eval/                         # M3 (next): capability-retention evals
└── logs/                         # launch helpers
```

M1 status: the smoke gate is **passed** — the teacher (frozen `Qwen3-4B-Instruct-2507`)
generates coherent, duplex-consistent pt-BR content (short turns, follow-up
questions, no tags, no thinking tokens), the duplex builder places tags correctly,
and the output is consumed unchanged by the existing `prep_dataset.py`. **M1 target
= 10k dialogues** (not 50k); the run is served via llama.cpp (`:8082`,
`Qwen3-4B-Instruct-2507-Q8_0.gguf`, `--parallel 4`) at ~70 dialogues/min → ~2 h.

M2 status: code written and validated (loss math passes on a CPU synthetic model):
- `training/prep_dataset.py` (fork) — duplex rows + **anchor rows** (D2) + SDFT
  **artifact masking** (`--mask-first-tokens`, drops the first content tokens after
  each `<|user finish speaking|>`).
- `training/train_distill.py` (fork) — `DuplexDistillTrainer` implementing the
  two-term loss: weighted tag CE + content **forward-KL to the frozen base** via a
  `no_grad` `model.disable_adapter()` forward (teacher top-k support, `--top-k`).
  `--kl-weight` (λ), `--teacher-temperature` (τ), `--top-k` (k) are CLI args.
- `logs/run_smoke_train.sh` — smoke-run helper.

**M2 smoke passed** (2026-08-21, after M1 completed): 512 duplex rows + 300 anchors,
100 steps, B=1/seq 2048, lr 1e-5, λ=0.3, τ=1.0, k=32 → **no OOM** with the two-forward
`disable_adapter()` teacher path; loss 4.2 → 0.65; **tag accuracy 0 → 75.8%**; content
KL vs the frozen base bounded (~0.3–1.0 nats during training). ~10 s/step (embed
fine-tune + teacher forward) → a 2k-step full run ≈ 5–6 h on the 4080. Adapter:
`/mnt/f/duplex_cascade_runs/distill/smoke/adapter`. Next: λ/τ/k sweep → full 2k-step run
→ export → M3 capability-retention evals.

M3 status (**evaluation complete** — see `eval/RESULTS.md`):

| Model | Function calling | General | Tag accuracy | KL drift |
|---|---|---|---|---|
| base | 100% | 100% | 0% | 0.51 |
| **distill** (this project) | **100%** | **100%** | **96.4%** | **1.07** |
| sft (DuplexCascade-PT) | 55.6% | 41.7% | 98.0% | 4.76 |

The distill model learns the duplex tags (96.4%) **without** the capability regression:
function calling and general stay at base level, while the plain SFT baseline collapses
(FC 55.6%, general 41.7%). The content forward-KL to the frozen base is the mechanism
that preserves capabilities. Reproduce with `./logs/run_eval.sh`.

Standard benchmarks (lm-eval, subset 2000): **distill stays within 1–2 pp of base**
on HellaSwag (0.566 vs 0.584), ARC-Challenge (0.563 vs 0.584) and MMLU (0.698 vs 0.705);
**sft drops MMLU to 0.660** (−4.5 pp). Full table in `eval/RESULTS.md`; reproduce with
`./logs/run_benchmarks.sh`.

Extra benchmarks (gsm8k ×300, truthfulqa, ifeval ×40): **distill beats base on
TruthfulQA (0.649 vs 0.626) and IFEval (loose 0.762 vs 0.730)**; its one regression is
GSM8K (0.777 vs 0.890, long-form CoT). **sft drops everywhere** (GSM8K 0.827, TruthfulQA
0.601, IFEval 0.603/0.524). Full table in `eval/RESULTS.md`; reproduce with
`./logs/run_benchmarks_extra2.sh`.

## M4 — article

`article/main.tex` — *"Self Distillation using only supervised fine tuning: a duplex
cascade case study"* (IEEE style, 4 pages, includes the method derivation, both result
tables, and a discussion of the GSM8K caveat). Build:

```bash
cd article && pdflatex main.tex && pdflatex main.tex   # -> main.pdf
```

## M4 — real-time GUI + GGUF

The distilled model is exported to `q4_k_m` and served by an adapted
DuplexCascade GUI (same layout as `duplex_cascade/DuplexCascade/`):

- **GGUF**: `/mnt/f/duplex_cascade_runs/distill/full/export/gguf/DuplexCascade-Distill-q4_k_m.gguf`
  (built from `.../export/merged_bf16` via `convert_hf_to_gguf.py` f16 →
  `llama-quantize` q4_k_m, 2.38 GB / 4.95 BPW; f16 intermediate kept alongside).
- **GUI**: `DuplexCascade/` (server.py, model.py, web/, requirements.txt, README.md)
  + `services/` (stt, tts) + `run_services.sh`. The base is the most-updated
  **`custom-duplex-cascade-pt-br`** server line (STT write lock, Eos/LLM timeouts,
  in-place history fix, traceback logging). The distill adaptations: the
  **system prompt** → the DuplexCascade trigger used to generate the teacher
  content (`data/teacher_prompts.py`, content-only) plus an explicit turn-close
  instruction ("emita `<|user is thinking|>` quando terminar"), because the
  distill model needs the prompt on-distribution and, unlike the SFT baseline,
  does not spontaneously emit the turn-close tag. The duplex micro-turn loop and
  web UI are otherwise unchanged.

```bash
./run_services.sh      # llama-server :8080 + STT :31607 + TTS :31608 + bridge :31606
# then open http://localhost:31606
```

See `DuplexCascade/README.md` for details and the GGUF build commands.

### Known issue: degenerate repetition on substantive questions (fixed 2026-08)

The v1-trained distill model, served by the GUI with `--max-new-tokens 96`, could
collapse into a **degenerate token loop** on substantive questions (e.g. *"como
funciona hardware?"* → *"…ver ou sentir. ő ő ő ő ő…"* then filler rambling
*"vou esperar sua próxima palavra…"*). This was NOT a model-distillation problem
per se — it was a combination of three generation-side issues, all now fixed:

1. **Truncation**: 96 max tokens cut real answers mid-sentence; the `<|no voice|>`
   continuation after a truncation is where the loop started. → bridge default
   `--max-new-tokens` is now **256** so answers complete in one micro-turn.
2. **No repetition penalty**: llama.cpp's `repeat-penalty` was at the default 1.0
   (off). → set to **1.15** in `run_services.sh`, `logs/run_llama.sh`, and the
   bridge payload.
3. **Micro-turn loop ignored `finish_reason`**: after ANY non-closing chunk it
   re-prompted with `<|no voice|>`, so a naturally-finished answer (the model
   emitted `<|im_end|>`, `finish_reason="stop"`) was prodded into rambling, which
   polluted the history and made later turns degenerate. → the loop now **ends the
   turn on `finish_reason="stop"`** and only continues on `"length"` (truncation).
   Plus a `_strip_degenerate_tail` guard drops `ő ő ő`-style runs before TTS.

The content itself was also improved: the inference system prompt was bumped to the
**v2** teacher trigger (`data/teacher_prompts.py`, `TRIGGER_VERSION="v2"`) — *"Responda
de forma natural e completa … não se repita"* — which the trained model follows
(it retained instruction-following), giving complete, non-repeating answers without
retraining. **If you want the complete-answer behavior native (not prompt-following),
regenerate everything with the v2 trigger and retrain** — see `SPEC.md` §6 "R0".

## M1 usage

### 1. Teacher content + aligned duplex sequences (`build_teacher_duplex.py`)

Two backends:

```bash
# (a) in-process, exact bf16 base model (GPU, serial) — used for the smoke run
python data/build_teacher_duplex.py --in-process \
  --dialogues /home/penhfel/github/duplex_cascade/data/dialogues_pt.jsonl \
  --out data/teacher_duplex_train.jsonl --limit 2000

# (b) llama.cpp OpenAI-compatible API (parallel, cheap) — full run
#     serve the BASE Qwen3-4B-Instruct GGUF first (see logs/run_teacher_api.sh),
#     then:
python data/build_teacher_duplex.py \
  --api-base http://127.0.0.1:8082/v1 --workers 4 \
  --dialogues /home/penhfel/github/duplex_cascade/data/dialogues_pt.jsonl \
  --out data/teacher_duplex_train.jsonl
```

Output record: `{id, src_id, scenario, messages (teacher-rewritten), teacher{...},
items (duplex micro-turns, same schema as build_duplex_dataset.py), meta}`. User
turns are kept from the validated source corpus; only assistant turns are
regenerated by the teacher. If a generation keeps failing, it falls back to the
source assistant text (`--no-fallback-source` to disable). `--allow-tags` lets the
teacher attempt the full tag protocol (for the trigger-prompt ablation, SPEC §8).

### 2. Anchor prompts (optional, D2 variant)

```bash
python data/build_anchor_prompts.py --n-per-category 150 --out data/anchor_prompts.jsonl
# optionally merge an external function-calling JSONL:
python data/build_anchor_prompts.py --n-per-category 150 --input ../path/fc.jsonl --out data/anchor_prompts.jsonl
```

## Next (M2)

Fork `training/prep_dataset.py` (attach teacher top-k targets when cached) and
`training/train_qlora.py` → `training/train_distill.py` with the two-term loss
(tags weighted CE + content forward-KL vs `disable_adapters()` teacher forward +
optional anchor KL), then the smoke run, λ/τ/k sweep, full run, and export.
See `SPEC.md` §6 M2.

## Notes

- The HF base-model cache lives at `/mnt/f/huggingface/` (`HF_HOME` is already set).
- The 7 duplex special tokens are only in the *data* (via the builder's tokenizer);
  the teacher itself is never prompted to output them in content-only mode.
- In-process generation is serial (~6 dialogues/min) — fine for a quick smoke, too
  slow for the M1 run. The 10k run uses the base GGUF served via llama.cpp
  (`logs/run_teacher_api.sh`, port 8082, `--parallel 4`).