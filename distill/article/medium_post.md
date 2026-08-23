# Teaching an LLM to Take Turns Without Forgetting Everything Else

### Self-distillation with plain supervised fine-tuning: how we trained a full-duplex voice model that keeps its function-calling skills intact

*Based on our paper "Self Distillation using only supervised fine tuning: a duplex cascade case study." Code, data and models are open: [duplex_cascade_distill](https://github.com/lumierenoir/duplex_cascade_distill) · [models & data on Hugging Face](https://huggingface.co/lumierenoir/DuplexCascade-PT-BR-V0)*

---

## The problem: a great turn-taker that became a bad assistant

We have been building a **full-duplex voice assistant** in Brazilian Portuguese based on the *duplex cascade* paradigm (Yang, Fujita & Sudo, sbintuitions). The idea is elegant: instead of relying on voice-activity detection to decide who speaks when, the LLM itself drives the conversation by emitting special control tokens mid-stream:

- `<|user is speaking|>` — the user is talking
- `<|user finish speaking|>` — user finished, start replying
- `<|user interruption|>` — the user cut you off
- `<|user backchannel|>` / `<|system backchannel|>` — those little "uhum", "entendi" moments
- `<|no voice|>` — silence

Learning this protocol is easy. The supervision is cheap, synthetic and deterministic. We fine-tuned Qwen3-4B-Instruct-2507 with QLoRA on ~10k synthetic pt-BR micro-turn dialogues and got a model with **98% tag accuracy**.

There was just one problem. The same model that now negotiates turn-taking like a pro had lost most of its ability to be an LLM:

| Model | Function calling | General instructions | Tag accuracy |
|---|---|---|---|
| base | **100%** | **100%** | 0% |
| plain SFT | 55.6% | 41.7% | **98%** |

That is catastrophic forgetting in its most practical form: **a good turn-taker and a bad assistant.**

## Why does plain SFT destroy capabilities?

Supervised fine-tuning is *off-policy*: the model imitates expert text on a fixed offline distribution. When every training sequence is saturated with special tokens and micro-turn-shaped dialogue, the optimizer does not learn "emit tags at the right places" — it drags the **entire next-token distribution** toward tag-heavy dialogue, including on prompts that have nothing to do with spoken conversation. Ask the fine-tuned model to call a function and it answers like it is half of a phone call.

The standard remedy is on-policy RL (PPO/GRPO) with a reward saying "keep being a good assistant while learning tags." But rewards are hard to write for "don't get worse," and RL is expensive and brittle — not what you want for a project on a single RTX 4080.

A recent paper — *Self-Distillation Enables Continual Learning* (SDFT, arXiv:2601.19897) — showed something better: use the model in two roles, a **teacher** conditioned on an expert demonstration and a **student** conditioned only on the query, and minimize a KL divergence between them on the student's own trajectories. Great results… but it still requires on-policy rollouts, roughly 4× the wall-clock of SFT.

## Our key observation: tags and content are separable by token position

Here is the trick that let us skip both RL and rollouts. In the duplex setting, the "new skill" and "everything else" live at **different token positions**:

- **Tags** sit at supervised positions placed by a deterministic builder.
- **Content** is everything else.

So we split the objective in two:

```
L(θ) = Σ_{t ∈ tags}   w_t · CE(π_θ(·|x<t), y_t)          ← learn the protocol
     + λ · Σ_{t ∈ content} KL( π_0(·|x<t) ‖ π_θ(·|x<t) )  ← stay anchored to the base
```

Term 1 is exactly the original recipe: weighted cross-entropy on the special tokens (`<|user finish speaking|>` gets weight 10, interruptions 5, and so on).

Term 2 is the new part: at every *content* position, pull the student's distribution toward the **frozen base model's own distribution** with a forward KL. The teacher is literally "what would the original model say here?" — so everywhere there is no explicit tag supervision, the LoRA delta is pinned to approximately zero. No reward, no sampling from the student, no second model copy.

## Making it practical (the part that made it cheap)

Three implementation details took this from "nice idea" to "runs at SFT cost on a 16 GB GPU":

**1. The teacher forward is free.** Our student is *base + LoRA*. So the teacher is the same tensors with the adapters disabled — one extra `no_grad` forward pass under PEFT's `disable_adapter()`. Zero additional VRAM, no second model in memory.

**2. Precompute the teacher once.** Even better: we ran the frozen base over the whole training corpus *once* and cached its top-k (k=32) next-token ids/logits for every position (~36 minutes). Training then reads cached targets — the KL term becomes a cheap gather + cross-entropy. The full 2,000-step run costs about the same as plain SFT.

**3. Where does the content come from?** From the teacher itself. We prompted the *frozen* base with a "you are a duplex voice assistant" trigger and regenerated every assistant turn of our 10k pt-BR dialogues (temperature 0.7, top-p 0.95). At first it does not follow the protocol faithfully — that is expected and irrelevant. We keep only its *content*; a deterministic builder slices it into micro-turns and places all the tags. Two SDFT details carried over too: we mask the first couple of content tokens after each turn boundary (so the student doesn't copy teacher surface artifacts), and we add ~300 "anchor" prompts (function calling, general QA) distilled the same way, explicitly pinning behavior outside the duplex protocol.

## Results

We trained a QLoRA adapter (r=16, trainable embeddings/LM head so the new special-token rows can be learned) for 2,000 steps — about six hours on one RTX 4080 — and compared three models: the base, our **distill** model, and the original plain-SFT baseline.

| | base | **distill** | sft |
|---|---|---|---|
| Tag accuracy | 0% | **96.4%** | 98.0% |
| Function calling (pt-BR) | 100% | **100%** | 55.6% |
| General instruction/QA | 100% | **100%** | 41.7% |
| KL drift vs base (nats) | 0.51* | **1.07** | 4.76 |

\* the measurement floor of a perfectly-matched model (we score over the teacher's top-32 support only).

The distill model learns essentially the same protocol as SFT (96.4% vs 98%) while staying at **base level** on everything else — and its distributional drift is barely above the floor, versus ~3.5× worse for SFT.

Standard benchmarks tell the same story (lm-evaluation-harness, identical configs for all models):

| Benchmark | base | **distill** | sft |
|---|---|---|---|
| HellaSwag (acc-norm) | 0.584 | 0.566 | 0.593 |
| ARC-Challenge (acc-norm) | 0.584 | 0.563 | 0.574 |
| MMLU | 0.705 | 0.698 | 0.660 |
| GSM8K (5-shot) | 0.890 | 0.777 | 0.827 |
| TruthfulQA (mc2) | 0.626 | **0.649** | 0.601 |
| IFEval loose / strict | 0.730 / 0.651 | **0.762 / 0.683** | 0.603 / 0.524 |

Within noise of base on knowledge and reasoning, actually *above* base on TruthfulQA and IFEval — while the SFT baseline drops broadly.

### The honest caveat

GSM8K: −11 pp vs base. Long-form multi-step chain-of-thought is exactly where a strong anchoring term (λ = 1.0) bites: the content distribution is heavily pinned to a 4B model's fragile CoT, and any shift hurts. Lower λ, or an EMA-teacher variant that tracks the student's progress, are the obvious next experiments.

## This is bigger than duplex dialogue

The crux of the method is **positional separation**: a small, well-localized set of supervised positions carries the new behavior; everything else is anchored to the frozen base. That describes a very common fine-tuning pattern we call *partial-answer supervision* — the target output is already largely correct, and only a small identifiable part must be edited.

The most immediate target: **turning an LLM into a guardrail classifier**. Fine-tune on content-moderation data where the verdict (`ok` / `flagged`, or a category label) is the "tag" and the one-line justification plus everything else is the "content." Plain SFT collapses the model onto label tokens and erodes general ability; our recipe learns the decision boundary while keeping language and knowledge intact.

Other settings where the edit lands at predictable positions:

- schema/format fix-ups (correct semantics, wrong JSON keys or tool-call syntax)
- claim correction in RAG / medical / legal pipelines
- code repair where only the buggy lines change
- citation insertion

When the edit location is not known a priori, you can often align outputs to a canonical template first (exactly what our micro-turn builder does for dialogue) so the edits collapse onto predictable positions; otherwise full SDFT with rollouts remains the fallback. There is also a nice theoretical echo here: *RL's Razor* (arXiv:2509.04259) argues forgetting is governed by the KL to the base on the new task — partial-answer supervision is precisely where an explicit per-token KL constraint is cheapest, because the supervised support is small.

## Takeaways

1. Off-policy SFT on a narrow behavioral protocol drags the whole output distribution with it. Expect regressions.
2. If the new behavior lives at predictable token positions, split the loss: supervised CE there, forward-KL to the frozen base everywhere else.
3. With base + LoRA, the teacher costs nothing (`disable_adapter()`), and an offline top-k cache brings the run back to SFT cost.
4. It works: protocol learned, capabilities kept, no RL, single 16 GB GPU.

Everything is open — training code, evaluation harness, the teacher-generated dataset, adapters and GGUFs:

- Code & paper: [github.com/lumierenoir/duplex_cascade_distill](https://github.com/felipepenhorate/custom-duplex-cascade-pt-br)
- Models & data: [huggingface.co/lumierenoir/DuplexCascade-PT-BR-V0](https://huggingface.co/lumierenoir/DuplexCascade-PT-BR-V0)
- Voice demo stack: [github.com/lumierenoir/duplex_cascade](https://github.com/felipepenhorate/custom-duplex-cascade-pt-br)

**References**

- Shenfeld et al., *Self-Distillation Enables Continual Learning* (SDFT), arXiv:2601.19897
- Yang, Fujita & Sudo, *DuplexCascade*, arXiv:2603.09180
- Chu et al., *SFT Memorizes, RL Generalizes*, ICML 2025
- Agarwal et al., *On-Policy Distillation of Language Models*, ICLR 2024
- Lu & Thinking Machines Lab, *On-Policy Distillation*, 2025
- Shenfeld, Pari & Agrawal, *RL's Razor*, ICLR 2026

---

*Suggested Medium tags: `machine-learning`, `llm`, `fine-tuning`, `deep-learning`, `tutorial`.*
