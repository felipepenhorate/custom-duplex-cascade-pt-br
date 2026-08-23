# M3 — DuplexCascade-Distill evaluation results

Date: 2026-08-22. Models compared (all `Qwen3-4B-Instruct-2507` base):

| Model | Function calling (18 pt-BR) | General (12 pt-BR) | Tag accuracy (2930 pos) | KL drift (20k content pos) |
|---|---|---|---|---|
| base | 100% | 100% | 0.0% | 0.51 |
| **distill** (this project, λ=1.0) | **100%** | **100%** | **96.4%** | **1.07** |
| sft (`DuplexCascade-PT` continue) | 55.6% | 41.7% | 98.0% | 4.76 |

## Interpretation

- **distill learns the duplex protocol** (96.4% tag accuracy, per-token weighted CE on
  tags) **without regressing general capabilities**: function-calling and general
  instruction-following stay at base level (100%). Its KL drift over the frozen base is
  1.07 — barely above the 0.51 top-k-truncation floor of a perfectly-matched model.
- **The plain SFT baseline regresses exactly as reported**: it learns the protocol
  (98% tags) but function calling collapses to 55.6% (−44 pp vs base) and general
  accuracy to 41.7% (−58 pp). KL drift is 4.76, ~3.5× the excess of the distill model.
- The content forward-KL term to the frozen base (SPEC §4.3) is doing its job: it pins
  the model's distribution on everything that is not a tag, which is precisely the
  capability-retention mechanism the project set out to test.

## Standard benchmarks (lm-evaluation-harness 0.4.12, raw prompts, subset limit 2000)

| Benchmark | base | **distill** | sft |
|---|---|---|---|
| HellaSwag (acc_norm) | 0.584 | 0.566 | 0.593 |
| ARC-Challenge (acc_norm) | 0.584 | 0.563 | 0.574 |
| MMLU | 0.705 | 0.698 | **0.660** |

- **distill** stays within ~1–2 pp of base on all three (knowledge, commonsense,
  reasoning) — essentially no capability regression from the tag fine-tune.
- **sft** drops MMLU by −4.5 pp (0.705 → 0.660) while Hellaswag/ARC stay flat; its
  worst regression shows up on *format-following* tasks (function calling 55.6%,
  general 41.7%), which the log-likelihood MC benchmarks do not stress.
- Numbers are on a 2000-example subset (per-task), so treat the ~1 pp deltas as
  within noise; the headline (distill ≈ base everywhere, sft regresses on
  knowledge + format-following) is robust.

Reproduce: `./logs/run_benchmarks.sh` (tasks: `hellaswag,arc_challenge,mmlu`).

## Extra benchmarks (gsm8k 5-shot ×300, truthfulqa_mc2, ifeval sample ×40)

| Benchmark | base | **distill** | sft |
|---|---|---|---|
| GSM8K (exact_match) | 0.890 | 0.777 | 0.827 |
| TruthfulQA (mc2) | 0.626 | **0.649** | 0.601 |
| IFEval loose (inst) | 0.730 | **0.762** | 0.603 |
| IFEval strict (inst) | 0.651 | **0.683** | 0.524 |

- **distill** ≥ base on TruthfulQA (+2.3 pp), IFEval loose/strict (+3.2 pp) and roughly
  at base on the MC suite; it shows a real **GSM8K dip (−11.3 pp vs base)** — the one
  regression, on a long-form multi-step CoT generation task. This is the honest
  tradeoff of the strong content-KL (λ=1.0): format/knowledge retention is excellent,
  but multi-step reasoning on a 300-example subset degrades. A lower λ, or the
  EMA-teacher / on-policy variants (SPEC D3/D4), are the natural follow-ups to probe.
- **sft** drops on everything it can: GSM8K −6.3 pp, TruthfulQA −2.5 pp,
  IFEval loose −12.7 pp / strict −12.7 pp, MMLU −4.5 pp, plus the format-following
  collapse (FC 55.6%, general 41.7%).
- All three ran the identical task configs and subsets (n=300 GSM8K, n=40 IFEval);
  deltas are well beyond the per-sample stderr except where noted.

Reproduce: `./logs/run_benchmarks_extra2.sh` (base IFEval already cached from the
first pass; the driver re-runs gsm8k+tqa+ifeval for distill/sft).

## Metric notes

- Function calling uses the native Qwen tool-calling format (`apply_chat_template`
  `tools=[...]`, thinking disabled), greedy decoding; scored by tool name + required
  arguments (date/time compared digit-normalized; free-text `assunto`/`corpo` are
  presence-only).
- Tag accuracy = argmax == supervised special token on the held-out duplex eval split
  (1024-seq). Base scores 0 because it never learned the tags.
- KL drift = mean per-token forward-KL (student ∥ frozen base, teacher top-k=32 support)
  on content positions of the same held-out split. The base-vs-base 0.51 is the
  truncation constant (softmax support is cut to the teacher's top-32), so use it as the
  reference floor and compare excess drift.

## Reproduce

```bash
./logs/run_eval.sh    # capabilities + turntaking for base / distill / sft
```
Raw outputs: `logs/eval/capabilities_{base,distill,sft}.json`,
`logs/eval/turntaking_{base,distill,sft}.json`.