# SPEC — DuplexCascade-Distill: Self-Distillation Fine-Tuning for the Duplex Micro-Turn Protocol (No RL)

Version: 1.0 (draft)
Date: 2026-08-21
Status: proposal — to be approved before implementation

## 1. Objective

Produce a **new training methodology** for the DuplexCascade micro-turn model
(fork of `duplex_cascade/`, currently `DuplexCascade-PT`, Qwen3-4B-Instruct-2507,
QLoRA/Unsloth) that **learns the duplex turn-taking protocol without regressing the
model's general capabilities** (function calling, tool use, reasoning, instruction
following).

The method borrows the **Self-Distillation Fine-Tuning (SDFT)** idea from
*"Self-Distillation Enables Continual Learning"* (Shenfeld, Damani, Hübötter &
Agrawal, MIT / Improbable AI Lab / ETH Zurich, arXiv:2601.19897) and adapts it to
the duplex setting in a way that **does not require reinforcement learning** and
**does not require on-policy rollouts**.

The core idea, in one sentence:

> Generate the assistant content from the **frozen original base model** prompted
> with a "duplex cascade" system trigger; record its per-token logits. Then
> fine-tune the student so that it (a) emits the duplex special tags where the
> deterministic micro-turn builder says they belong (supervised CE, as today) and
> (b) keeps the **logit distribution over everything else identical to the frozen
> base model** (KL-divergence to the base's per-token distribution).

Because the "everything else" term is a **dense, token-level KL to the base model's
own distribution**, the student is anchored to the pretrained policy everywhere the
tags are not explicitly supervised. This is the mechanism that should stop the
reported regression (e.g., function calling degrading after tag learning), without
any reward function, PPO/GRPO, or importance sampling.

## 2. Problem statement (what is broken today)

In `duplex_cascade/`, the fine-tune (`training/train_qlora.py`) trains the model to
predict the 7 conversational special tokens (`<|user is speaking|>`,
`<|user finish speaking|>`, `<|user interruption|>`, `<|user backchannel|>`,
`<|user is thinking|>`, `<|system backchannel|>`, plus `<|no voice|>`) interleaved
with the dialogue text, with loss applied only on system micro-turn positions.

The reported symptom:

- After training, the model does turn-taking well but **regresses on other tasks**
  (function calling is the concrete example): general instruction-following and tool
  behavior degrade.

This is the textbook **catastrophic forgetting of prior capabilities under off-policy
SFT**: plain SFT on the micro-turn protocol pushes the whole next-token distribution
toward "duplex-ish text + tags" everywhere, including on prompts that have nothing to
do with duplex dialogue.

## 3. The borrowed idea: Self-Distillation Fine-Tuning (SDFT)

SDFT (arXiv:2601.19897) frames continual learning without RL:

- **Teacher** = the *same* model conditioned on an expert demonstration: `π(·|x, c)`.
- **Student** = the base policy conditioned only on the task: `π_θ(·|x)`.
- **Loss** = `D_KL(π_θ(·|x) ∥ π(·|x, c))` (reverse KL in theory; **forward KL works
  best in practice**, §3). Token-level decomposition; the **analytic per-token
  estimator** (full marginalization over the vocabulary at each step) is the most
  stable and best-performing estimator (Appendix A.1).
- **On-policy**: the student samples its own trajectories `y ~ π_θ(·|x)` and the
  teacher scores them. This is what gives the anti-forgetting benefit.
- **EMA teacher** (`α ∈ {0.01, 0.02, 0.05}`): frozen-base teacher underperforms
  (doesn't track progress); current-student-as-teacher diverges; EMA is the stable
  compromise (Appendix A.3).
- **Mask the first few tokens** of each generated response in the loss, to suppress
  teacher "learned artifacts" like *"Based on the example..."* (Discussion).

Key empirical result (Table 5): on Tool Use / Science Q&A / Medical, plain SFT
crashes the *previous-capabilities* average from ~65.5 → ~53–56, while SDFT holds it
at ~64.5 **and** improves new-task accuracy above SFT (70 vs 63–66).

Cost: ~2.5× FLOPs and ~4× wall-clock vs SFT (single on-policy rollout per prompt).

### What we adopt vs. what we change

| SDFT ingredient | Adopt? | DuplexCascade-Distill adaptation |
|---|---|---|
| Teacher = model conditioned on demonstration | **Yes** | Teacher = frozen **base** model (Qwen3-4B-Instruct-2507) reading the full duplex-annotated sequence (tags + dialogue) — i.e., conditioned on the "demonstration" that includes the micro-turn protocol. |
| Student = base policy | **Yes** | Student = base + QLoRA adapters (same as today). |
| On-policy rollouts from the student | **No** (user requirement: no RL, minimal complexity) | Replaced by **dense per-token logit matching on a fixed, aligned training set**. This is offline, but the teacher is the *base model itself*, so the KL term directly implements "make everything else the same as the original model". |
| Reverse KL | **No** | **Forward KL** (CE to the teacher's soft distribution) — the paper's recommended practical choice. |
| Analytic per-token KL | **Yes (approximated)** | Full-vocab KL is computed on the fly via a **second forward pass** (see §5.2), or cached as **top-k logits** (see §5.3). |
| EMA teacher | Optional | Default teacher = **frozen base** (it is precisely the "keep everything else the same" reference the user wants). EMA of the student is an optional Phase-2 variant for the *tag/content* signal. |
| Mask first tokens | **Yes** | Mask the first few content tokens of each assistant micro-turn to avoid copying teacher surface artifacts. |
| Dual use of teacher for tags | No | Tags are **never** taken from the teacher's (imperfect) generation — they are placed deterministically by the existing duplex builder. The teacher only supplies the *content* distribution. This removes SDFT's main "weak-ICL teacher" risk (see §6). |

## 4. Proposed method

### 4.1 Roles

- **Teacher (frozen)**: the *original* `Qwen/Qwen3-4B-Instruct-2507` weights, **no
  adapters**, never updated. When fed the duplex-annotated sequence it produces, at
  every position, the base model's next-token distribution — the distribution that
  "knows" function calling, reasoning, etc.
- **Trigger prompt (teacher context)**: a system message that frames the model as a
  duplex cascade assistant and describes the micro-turn/tag protocol. Used **when
  generating the teacher content** (§4.2) and optionally **prepended to the input**
  when computing teacher logits, so the teacher's content distribution is consistent
  with being *inside* a duplex dialogue (a raw base model would otherwise find the
  tag-conditioned prefixes "surprising").
- **Student**: base + QLoRA (all linear, r=16 α=16, dropout 0, `modules_to_save`
  embed/lm_head) — identical config to `duplex_cascade/training/train_qlora.py`.

### 4.2 Teacher content generation + logit recording

For each dialogue in the corpus (reuse `data/build_dialogues.py` output, pt-BR):

1. Build the **teacher prompt** = duplex-trigger system message + the user turns
   (ChatML, exactly like the inference server builds duplex prompts).
2. **Generate** the assistant turn with the **frozen base model** (temperature ~0.7,
   top-p 0.95; the existing llama.cpp server or an in-process forward both work).
   The output will **not** follow the protocol reliably — that is expected and
   explicitly accounted for (tags are not taken from this generation).
3. Run the assistant content through the **existing duplex micro-turn builder**
   (`data/build_duplex_dataset.py` semantics: split into system micro-turns,
   insert `<|user finish speaking|>` / `<|no voice|>` / `<|user is speaking|>` /
   probabilistic interruptions & backchannels). This yields the **demonstration
   sequence** — the exact ChatML sequence with tags at supervised positions.
4. **Record teacher logits**: run a forward pass of the frozen base model over the
   demonstration sequence and store the per-position next-token distribution:
   - *Primary*: computed **on the fly during training** via a
     `model.disable_adapters()` forward (zero extra VRAM, no storage), using only
     the teacher's **top-k** logits (`k ≈ 32`) as the KL support (see §5.2).
   - *Alternative*: **offline cache** of teacher top-k logits per position
     (`training/cache_teacher_logits.py`), so training itself is a normal SFT step
     plus a cheap KL term (see §5.3).

### 4.3 Training loss

Let `T` be the set of positions whose supervised label is a duplex special token and
`C` the set of positions whose label is regular content (both on the system side;
user-side positions stay masked as today). `w_t` = existing per-special-token weight
(`TOKEN_WEIGHT`), `λ` = KL weight, `τ` = teacher temperature.

```
L = Σ_{t∈T} w_t · CE(softmax(student_logits_t), y_t)          (tags: as today)
  + λ · Σ_{t∈C} CE(softmax(student_logits_t), softmax(teacher_logits_t / τ))   (content: match base)
  + λ_a · Σ_{anchors} CE(softmax(student_logits_t), softmax(base_logits_t / τ)) (optional anchor set)
```

- **Tags** are learned exactly as today (weighted CE; `<|user finish speaking|>=10`,
  `<|user interruption|>=5`, ...). No KL on tag positions.
- **Content** is distilled from the frozen base model instead of being hard-fit to
  the exact dialogue string. Because the teacher is conditioned on the same tags and
  prefix, its argmax ≈ the target text anyway; the KL just forbids the sharpening /
  distributional drift that causes regression.
- **Anchor set (optional but recommended)** — a small curated batch of
  **non-duplex capability prompts** (function calling / tool use, general QA,
  reasoning, safe-chat), scored by the frozen base model. This term directly pins the
  LoRA delta to ≈ 0 on general prompts, i.e. it is a single-stage, concurrent version
  of the paper's "Re-invoke". This is the cheapest, most direct guard against the
  reported function-calling regression.

### 4.4 Variants (decision points, see §8)

| Variant | Teacher | Rollouts | Cost | Notes |
|---|---|---|---|---|
| **D1 (primary, proposed)** | Frozen base, logits on aligned duplex sequences | None (offline) | ~1.5–2× SFT | Matches user's "keep everything else the same" |
| **D2** | Frozen base, logits on aligned duplex sequences **+ anchor set** | None | ~1.6–2.2× SFT | Recommended: strongest anti-forgetting |
| **D3** | **EMA of student** (α≈0.02) | None | ~1.6–2.2× SFT | Tracks student progress; SDFT §A.3 |
| **D4 (stretch)** | EMA/frozen teacher | Student on-policy rollouts | ~4× SFT | Full SDFT; only if D1–D3 leave a measurable forgetting gap |

## 5. Feasibility

### 5.1 Hardware

Same single **RTX 4080 16 GB**, CUDA 13.1, torch 2.10.0+cu130, unsloth 2026.4.4
(`/home/penhfel/unsloth_uv`), 1 TB disk, 31 GB RAM — identical to the current project.
Base: `Qwen/Qwen3-4B-Instruct-2507` (4-bit QLoRA ≈ 5–7 GB, as today).

### 5.2 The critical trick: teacher forward without a second copy

Because the student is *base + LoRA*, the teacher is *the same tensors without the
adapters*. PEFT provides a `disable_adapters()` context manager, so the teacher
forward costs **no extra VRAM**:

```python
with torch.no_grad(), model.disable_adapters():
    teacher_logits = model(input_ids, attention_mask=attention_mask).logits
```

KL support = the **teacher's top-k** next tokens (`k = 32`, ids from the frozen pass),
and the student's logits are gathered at those ids via `selective_log_softmax`
(already used in `train_qlora.py`). Memory peak for the extra forward is bounded by
one `[B, T, vocab]` logit tensor (~2.4 GB fp16 at B=2, T=4096) — fits comfortably next
to the ~7 GB QLoRA footprint.

### 5.3 Cost estimates

| Scheme | Per-step cost (4080) | 2k-step run | vs today (~12 h SFT) |
|---|---|---|---|
| SFT (today) | ~21 s (embed fine-tune) | ~12 h | 1× |
| D1/D2 (frozen teacher forward, no_grad) | ~27–32 s | ~16–19 h | ~1.4–1.6× |
| D2 with offline top-k logit cache | ~22–24 s | ~13–14 h | ~1.1× |
| D4 (on-policy, paper-style) | ~50–80 s | ~28–45 h | ~2.5–4× |

D2 with the offline cache is the cheapest path to the full benefit: one-time teacher
pass over the dataset writes compact `topk_ids` + `topk_logits` per position
(~6.5 GB for 10k × 2k positions × k=32, fp16), then training is a normal SFT step
plus a small gather+KL.

### 5.4 Alignment (why the two sequences line up)

The supervised tags come from the **deterministic builder** on the **teacher's
generated content**. The teacher logits are then recorded on the **final aligned
sequence** (tags included), so every position has a teacher distribution by
construction. No re-alignment of separately-sampled streams is needed — this is the
SDFT "demonstration-conditioned teacher" applied at the token level.

### 5.5 The "small model ICL" risk is mostly avoided

SDFT's failure mode at 3B scale (weak in-context learning → teacher guidance worse
than SFT, Fig. 5) comes from trusting the teacher to *infer the correct behavior*
from a demonstration. **We do not**: tag placement is deterministic, and the teacher
is only asked to supply the base model's own content distribution, which a 4B
instruct model does natively well. The residual risk is whether the *trigger prompt*
produces coherent, duplex-consistent content — testable immediately (M1 gate, §9).

### 5.6 Verdict

**Feasible.** The design reuses the entire existing data/training/export pipeline of
`duplex_cascade/`, adds one forward pass (or a one-time cache), and changes only the
loss on content positions. Everything runs on the existing single 4080. The main
unknowns are hyper-parameters (λ, τ, k) and trigger-prompt quality, both cheap to
resolve with the smoke run.

## 6. Steps to execute

### M1 — Data: teacher content + aligned duplex sequences

1. `data/teacher_prompts.py` — duplex-trigger system message + prompt builder
   (ChatML, mirrors `DuplexCascade/server.py` prompt construction).
2. `data/build_teacher_duplex.py` — for each dialogue: teacher generation (base
   model, temperature 0.7 / top-p 0.95, via the existing llama.cpp instance or
   in-process forward) → duplex micro-turn builder (reuse
   `duplex_cascade/data/build_duplex_dataset.py` logic) → emit JSONL items
   (role / text / special-tag markers), identical schema to today.
3. Smoke gate (200–2k dialogues): manual review that teacher content is coherent
   pt-BR and that the builder's tags land correctly.
4. Optional (for D2 cache path): `data/build_anchor_prompts.py` — a few hundred
   function-calling + general-capability prompts (can seed from
   `../function_calling_is_all_you_need/`).
5. Optional offline cache: `training/cache_teacher_logits.py` — frozen base forward
   over the aligned sequences, write `topk_ids`/`topk_logits` (k=32) per position to
   the prepped dataset.

### M2 — Training

6. `training/prep_dataset.py` (fork) — tokenize aligned sequences as today; add a
   flag to attach cached teacher top-k targets when present.
7. `training/train_distill.py` (fork of `train_qlora.py`) — `DuplexDistillTrainer`
   subclassing `UnslothTrainer`:
   - same masked/weighted tag loss as today (§4.3, term 1),
   - content-position forward-KL vs `disable_adapters()` teacher forward (term 2)
     and, if enabled, the anchor KL (term 3),
   - mask the first 2–4 content tokens of each assistant micro-turn (SDFT
     artifact-masking),
   - teacher temperature τ and KL weight λ as CLI args (sweep later).
8. **Smoke run**: 64–512 sequences, 100–300 steps, lr 1e-5–2e-4. Verify: tag F1 not
   worse than the SFT baseline, base-model distribution drift (per-position KL vs
   base) lower than the SFT baseline, no OOM.
9. **Hyper-parameter mini-sweep**: λ ∈ {0.1, 0.3, 1.0}, τ ∈ {1.0, 2.0}, k ∈ {16, 32};
   pick by (tag F1, capability retention) Pareto.
10. **Full run**: 2k steps (≈16–19 h on 4080 for D2 on-the-fly, ≈13–14 h with the
    cache). Export merged bf16 + GGUF via existing `export_model.py` /
    `export_gguf.py`.

### M3 — Evaluation (the point of the project)

11. `eval/eval_turntaking.py` — reuse the duplex tag/turn-taking eval from
    `duplex_cascade/` (tag F1 per special token, micro-turn latency/shape stats).
12. `eval/eval_capabilities.py` — **the regression probe**:
    - **Function calling / tool use** (the reported failure): a fixed function-calling
      eval set, exact-match on the tool call.
    - General: IFEval (or a subset), a reasoning/QA slice (pt-BR MMLU subset or
      ARC), HumanEval-style if available.
    - **Distribution drift**: average per-token KL (student ∥ base) on a held-out
      mix of duplex + general prompts — the direct measure of "everything else kept
      the same".
13. **Comparison table**: Base / SFT (`DuplexCascade-PT` current) / **D2 (this
    project)**; columns = tag F1, function-calling accuracy, general-bench average,
    KL drift, step time. This is the paper-quality evidence that the method works.

### M4 — Write-up

14. `README.md` with the recipe (mirroring `duplex_cascade/README.md` style) and,
    if desired, a short article/LaTeX doc reusing the `duplex_cascade_pt_article`
    scaffolding.

### R0 — Regenerate everything with the v2 trigger (standalone recipe)

**When to use**: the shipped distill model was trained on v1 teacher content
(*"Responda de forma curta"*). The v2 trigger (`data/teacher_prompts.py`,
`TRIGGER_VERSION="v2"` — *"Responda de forma natural e completa … não se repita"*)
fixes the substantive-question repetition at **inference** via prompt-following
(the trained model retained instruction-following; verified). Regenerating the
teacher data with v2 + retraining makes the complete-answer behavior **native**
and re-anchors the content distribution to the v2 prompt — recommended if
prompt-following ever proves fragile, or as the "final" model.

Expected wall-clock on the 4080: teacher gen ≈ 2 h, prep ≈ 20 min, top-k cache
≈ 1–2 h, full train ≈ 13–14 h, GGUF export ≈ 30 min. **Stop all llama.cpp/GPU
servers before training** (16 GB ceiling).

1. **Serve the frozen base (teacher)** — `logs/run_teacher_api.sh`:
   llama.cpp on `:8082` serving `/home/penhfel/Models/Qwen3-4B-Instruct-2507-Q8_0.gguf`
   with `--parallel 4 -sp` (the teacher must be the ORIGINAL base, not the distill GGUF).

2. **Regenerate teacher content** (v2 prompt is the default now):
   ```bash
   cd /home/penhfel/github/duplex_cascade_distill
   /home/penhfel/unsloth_uv/bin/python data/build_teacher_duplex.py \
     --dialogues /home/penhfel/github/duplex_cascade/data/dialogues_combined.jsonl \
     --out data/teacher_duplex_train_v2.jsonl \
     --api-base http://127.0.0.1:8082/v1 \
     --workers 4 --max-new-tokens 256 \
     --temperature 0.7 --top-p 0.95
   ```
   **IMPORTANT**: use a NEW `--out` filename — the script resumes by skipping
   src_ids already in the output file, so writing to `teacher_duplex_train.jsonl`
   would skip all 10k dialogues. Throughput ≈ 70 dlg/min → ≈ 2 h for the full 10k.
   *Gate*: check response-length stats (median should be higher than the v1 ≈ 106
   chars, more complete answers) + eyeball ~20 samples for coherent, complete,
   non-repeating pt-BR with no tags.

3. **Prep** (mirrors the v1 full-run config, `max-seq 1024`):
   ```bash
   /home/penhfel/unsloth_uv/bin/python training/prep_dataset.py \
     --duplex data/teacher_duplex_train_v2.jsonl \
     --anchors data/anchor_prompts.jsonl \
     --out /mnt/f/duplex_cascade_runs/distill/v2_prepped \
     --max-seq 1024 --mask-first-tokens 2 \
     --tokenizer Qwen/Qwen3-4B-Instruct-2507
   ```

4. **Cache teacher logits** (offline top-k; the v1 full run used this to halve
   peak VRAM and reach ~26 s/step):
   ```bash
   /home/penhfel/unsloth_uv/bin/python training/cache_teacher_logits.py \
     --dataset /mnt/f/duplex_cascade_runs/distill/v2_prepped \
     --out /mnt/f/duplex_cascade_runs/distill/v2_prepped_cached --top-k 32
   ```

5. **Full train** (mirrors the v1 full run: 10,249 train / 16 eval, 4 epochs,
   2,000 steps, B=1, grad-accum 16, lr 1e-4 / embed 2e-4, λ=1.0, τ=1.0, k=32):
   ```bash
   PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
   /home/penhfel/unsloth_uv/bin/python training/train_distill.py \
     --dataset /mnt/f/duplex_cascade_runs/distill/v2_prepped_cached \
     --out-dir /mnt/f/duplex_cascade_runs/distill/v2 \
     --max-seq-length 1024 --per-device-batch-size 1 --grad-accum 16 \
     --max-steps 2000 --lr 1e-4 --embedding-lr 2e-4 --warmup-steps 10 \
     --kl-weight 1.0 --teacher-temperature 1.0 --top-k 32 \
     --teacher-cache /mnt/f/duplex_cascade_runs/distill/v2_prepped_cached \
     --export-merged --no-eval
   ```
   Consider a 100–300-step smoke first (tag F1 ≈ SFT baseline, KL drift < SFT, no OOM).

6. **Export GGUF** (f16 → q4_k_m):
   ```bash
   cd ~/llama.cpp
   python convert_hf_to_gguf.py /mnt/f/duplex_cascade_runs/distill/v2/export/merged_bf16 \
     --outfile /mnt/f/duplex_cascade_runs/distill/v2/export/gguf/DuplexCascade-Distill-v2-f16.gguf \
     --outtype f16
   build/bin/llama-quantize \
     /mnt/f/duplex_cascade_runs/distill/v2/export/gguf/DuplexCascade-Distill-v2-f16.gguf \
     /mnt/f/duplex_cascade_runs/distill/v2/export/gguf/DuplexCascade-Distill-v2-q4_k_m.gguf \
     q4_k_m
   ```

7. **Serve + verify**: `GGUF=/mnt/f/duplex_cascade_runs/distill/v2/export/gguf/DuplexCascade-Distill-v2-q4_k_m.gguf ./run_services.sh`
   (the launch adds `--repeat-penalty 1.15`; the bridge defaults to
   `--max-new-tokens 256`). In the GUI ask *"como funciona hardware?"* and a
   small-talk turn — expect complete, non-repeating answers that close cleanly.
   Re-run M3 evals (`./logs/run_eval.sh`, `./logs/run_benchmarks.sh`,
   `./logs/run_benchmarks_extra2.sh`) and confirm no capability regression
   (function calling / general must stay ≈ base).

## 7. Deliverables & repo layout

```
duplex_cascade_distill/
  SPEC.md
  README.md
  data/
    common.py                 (fork; tokens/weights/pt-BR helpers)
    teacher_prompts.py        (M1: duplex-trigger system prompt builder)
    build_teacher_duplex.py   (M1: teacher generation + duplex post-processing)
    build_anchor_prompts.py   (M1: capability-anchor prompts)
    build_duplex_dataset.py   (reuse from duplex_cascade/)
  training/
    prep_dataset.py           (M2: optional teacher top-k targets)
    train_distill.py          (M2: DuplexDistillTrainer)
    cache_teacher_logits.py   (M2 optional: offline top-k cache)
    export_model.py           (reuse)
    export_gguf.py            (reuse)
  eval/
    eval_turntaking.py        (M3)
    eval_capabilities.py      (M3: function calling + general + KL drift)
  logs/                       (launch helpers)
```

## 8. Risks & open questions

- **λ / τ tuning**: too little KL → regression returns; too much → tags degrade or
  content collapses to base-argmax (short, generic answers). Mitigate with the M2
  Pareto sweep; start λ=0.3, τ=1.0.
- **Trigger-prompt quality**: the teacher's content must be *duplex-plausible*
  (micro-turn-sized, natural, pt-BR). If the frozen base ignores the trigger, fall
  back to D1 using the *existing* dialogue content (no new generation) — the KL term
  still works because the teacher is conditioned on the aligned sequence.
- **Masking artifacts**: apply SDFT's first-tokens mask to assistant micro-turns;
  monitor for "Como um sistema duplex..."-style prefixes in eval.
- **Memory spikes**: the teacher `no_grad` forward still materializes full logits;
  bound with bf16 + B≤2, or chunk over the sequence. Fallback: top-k cache.
- **User-side positions**: remain masked (labels = -100) exactly as today; no KL
  there either (they are fixed input).
- **Does content-KL fight tag learning?**: tags have no KL term and keep their high
  weights (10/5/...), so they should win; the smoke run's tag-F1 gate checks this.
- **Single-seed statistics**: replicate the paper's 3-seed convention for the final
  tables.
- **Decision points**: D1 vs D2 (anchor set) vs D3 (EMA teacher) vs D4 (on-policy);
  on-the-fly teacher vs offline top-k cache; k and τ values. Defaults recommended
  above; user may override.

## 9. Plan (execution order / gates)

| # | Milestone | Gate to proceed |
|---|---|---|
| 1 | Scaffold repo layout (§7) | files in place |
| 2 | M1: trigger prompt + teacher generation smoke (2k dialogues) | teacher content coherent, tags land correctly |
| 3 | M1: full teacher content pass (**10k dialogues**, the M1 target) + optional anchor set + optional top-k cache | aligned dataset + cache ready |
| 4 | M2: `train_distill.py` + smoke run (100–300 steps) | tag F1 ≈ SFT baseline, KL drift < SFT, no OOM |
| 5 | M2: λ/τ/k sweep + full 2k-step run + export | chosen config; merged bf16 + GGUF |
| 6 | M3: turntaking + capability evals, comparison table | evidence of no regression (esp. function calling) |
| 7 | M4: README/article | docs |
| 8 | **R0 (optional)**: regenerate everything with the v2 trigger + full retrain (§6 R0) | v2 GGUF answers "como funciona hardware?" complete & non-repeating natively; M3 evals show no regression |

## 10. References

- Shenfeld, Damani, Hübötter & Agrawal. *Self-Distillation Enables Continual
  Learning.* arXiv:2601.19897v2. (SDFT: teacher = demonstration-conditioned model,
  on-policy forward-KL, EMA teacher, analytic per-token estimator, artifact masking,
  ~2.5× FLOPs.)
- Yang, Fujita, Sudo (sbintuitions). *DuplexCascade: Full-Duplex Speech-to-Speech
  Dialogue...* (the protocol / tags / micro-turn builder being preserved).
- `duplex_cascade/` repo: existing QLoRA/Unsloth pipeline this project forks.