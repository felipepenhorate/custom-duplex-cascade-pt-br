# Full-Duplex-Bench for DuplexCascade pt-BR (text-domain harness)

Evaluates the **three trained variants** in this repo against the four
Full-Duplex-Bench (FDB) v1.0 dimensions, and compares them with the published
DuplexCascade numbers:

| key | variant | what it is |
|---|---|---|
| `sft` | **original recipe** | QLoRA SFT of Qwen3-4B-Instruct-2507 on duplex micro-turn data — the DuplexCascade method (`sft/`) |
| `distill` | **self-distillation** | SDFT-style: weighted tag CE + content forward-KL to the frozen base (`distill/`) |
| `mtp_like` | **policy companion** | a 0.8B controller that emits *only* a tag; the stock 4B LLM stays untouched (`mtp_like/`) |
| `base` | *control* | the untrained `Qwen3-4B-Instruct-2507` — should score ~0 everywhere, proving the harness actually measures the fine-tune |
| `original_dc` | *reference* | the upstream `sbintuitions/DuplexCascade` checkpoint (English, gated — see below) |

Run everything:

```bash
./fdb_bench/run_all.sh                       # pt-BR + en, FDB v1.0 sizes (727/subset set)
LIMIT=3 ./fdb_bench/run_all.sh               # smoke test, ~3 scenarios per subset
MODELS="distill" LANGS="pt" ./fdb_bench/run_all.sh
```

Results land in `results/<lang>/summary.md`, cross-language and tag diagnostics in `results/comparison.md`, and per-scenario timelines in `results/<lang>/<model>/raw.jsonl`. Measured results and the three bugs that were fixed along the way: [`FINDINGS.md`](./FINDINGS.md).

---

## What is faithful, and what is not

Read this before quoting any number.

**The input stream must match training, or nothing else matters.** The single
largest fidelity requirement in this harness is that the user's speech arrives
as the *same kind of pieces* the models were trained on. ``DuplexChunker``
delivers each user turn as 1-7-**token** pieces
(``*/data/build_duplex_dataset.py``, ``user_chunk_max=7``); measured over 3000
training records the pieces average **2.19 words** (median 2, p90 3, max 7) and
cut mid-sentence and mid-word, with leading-space artefacts.

An earlier version of this bench fed whole sentences as if they were ASR
partials — a 5-10x shift on the model's primary input. On that input the
distill variant produced degenerate, echoing, emoji-laden continuations and
scored near the floor; with the stream corrected to training shape it reaches
``finish recall 1.00`` with coherent ~34-word replies. That is not a prompt
effect and no prompt fixes it — it is an input-shape effect. Real runs
therefore use ``chunking.TrainingChunker`` (the training algorithm, ported, with
the real tokenizer), and ``tests/test_parity.py`` asserts both that every
character of the user's turn survives chunking and that piece lengths stay in
the training range.

**Faithful — ported directly from FDB's evaluation scripts** (`v1_v1.5/evaluation/`):

* the metric definitions and their thresholds, including the fiddly ones:
  a model utterance is a *formal turn* when it spans >= 1 s **or** carries
  > 3 words; it is a *backchannel* when it spans <= 3 s **and** <= 3 words
  (`eval_pause_handling.py`, `eval_smooth_turn_taking.py`, `eval_backchannel.py`);
* the **interruption rubric**, verbatim, including the exact
  `Analysis: … I would rate the AI's response as N` output contract
  (`eval_user_interruption.py` — see `judge.py`);
* the **Jensen–Shannon divergence** on a 0.2 s window grid, base-2, matching
  `scipy.spatial.distance.jensenshannon`;
* the **subset sizes** (216 / 137 / 119 / 55 / 200 = 727 scenarios).

**Not faithful — stated plainly:**

1. **Text domain, not audio.** These models decide turn-taking as text
   (LLM → TTS), so the harness simulates the duplex stream instead of feeding
   WAVs. The user's speech is a sequence of timed events, not audio.
2. **The clock is a word-rate model.** One constant (`words_per_second`,
   default 3.25, calibrated against the real `pocket_tts` — see `clock.py`)
   converts text to seconds. This is the harness's main approximation.
3. **Latency is inference compute time**, measured as the real wall-clock of
   the `act()` calls between the user's turn ending and the model committing to
   the floor — *not* end-to-end acoustic latency, which would need ASR + TTS
   timing. It is the honest text-domain analogue and it does discriminate
   between a duplex LLM and a separate policy companion.
4. **The judge is local**, Qwen3.5-9B (`/mnt/f/GGUF`), not GPT-4o. Consistent
   and reproducible, good for *ranking our models against each other*, **not**
   comparable to the published 4.016 on an absolute scale.
5. **The published row is English audio.** The DuplexCascade Table 1 numbers
   come from real speech models. They anchor the metric ranges; they are not an
   apples-to-apples baseline for our text-domain rows.

Within one table, our own rows are directly comparable to each other: same
scenarios, same clock, same simulator, same judge.

---

## Two switches that decide what you are actually measuring

### 1. Prime mode — who makes the turn decision?

The deployed `sft`/`distill` bridges call the LLM only after the STT signals
`UtteranceEnd`, and prime the assistant turn with
`<|user finish speaking|>` (`sft/DuplexCascade/server.py:622`). That hands the
model the turn decision — the STT's VAD-equivalent made it, not the LLM.

`--prime-mode decision` (the default) primes with `""` and makes the model
**decide**, which is both what FDB measures and what the VAD-free claim
requires. `--prime-mode prime` reproduces the deployed behaviour instead.

Running both is a genuine ablation of the paper's central claim, not a
formality: under `prime`, pause-handling TOR should collapse towards 1.0 for
every model, because the pipeline takes the floor at every turn boundary
regardless of whether the user was pausing.

### 2. Prompt candidate — is the model even being asked to be duplex?

A duplex model can only be scored on turn-taking in a configuration that makes
it behave as one. Two very different findings look identical in a results
table:

* the model cannot do duplex turn-taking, or
* the prompt told it not to — the distill bridge prompt says *"não imprima
  nenhuma tag especial"*, forbidding the exact tokens being measured.

So the prompt is **measured, not assumed**
([`PROMPTS.md`](./PROMPTS.md), `python -m fdb_bench.calibrate_prompts`):

| variant | runs with | why |
|---|---|---|
| `sft` | `deployed` | best chunk accuracy (.825) and F1 (.577), cleanest content, and what ships |
| `distill` | `tag` | the only candidate that is accurate (.885), clean (~41 words) *and* willing to take the floor |
| `mtp_like` | `instruction` | on-distribution for the companion; tied with `none` |
| `base` | `none` | untrained control |

The distill row is the instructive one: `trigger` — the prompt that *names* the
duplex role and lists the tags — has the **worst** recall of all (0.088), while
the no-prompt setting has the highest (0.853) but produces 195-word degenerate
monologues. Both are traps. [`PROMPTS.md`](./PROMPTS.md) has the full tables,
the sample output, and the exposure-bias reading.

Every run records its candidate in `meta.json`, so a table row can always be
traced back to the configuration that produced it.

---

## How the decision is measured

Training data assigns exactly one label per user micro-turn
(`*/data/build_duplex_dataset.py::user_phase`): every chunk but the last gets
`<|user is speaking|>`, and the turn boundary is answered with
`<|user finish speaking|>`. **That is the turn-taking decision.** So the
harness feeds user chunks one at a time and asks the model for a tag:

```
chunk ──► <|user finish speaking|> ──► model takes the floor  (timed)
       └─► <|user is speaking|>      ──► keep waiting
       └─► <|user is thinking|>      ──► keep waiting
silence ─► any tag                   ──► one chance to wrongly interrupt
```

The model never sees whether ASR/VAD marked a turn end — that would leak the
answer. `tests/test_parity.py::test_prompt_is_independent_of_turn_end_metadata`
asserts the built prompt is byte-identical with and without that flag.

The assistant then delivers its answer through the same micro-turn loop as the
live bridge (`<|no voice|>` round trips until a turn-closing tag).

---

## Tasks

| subset | n | gold | metric |
|---|---|---|---|
| `pause_synthetic` | 137 | stay silent through an inserted mid-sentence pause | TOR ↓ |
| `pause_candor` | 216 | stay silent through a natural hesitation | TOR ↓ |
| `smooth_turn` | 119 | take the floor as soon as the turn ends | TOR ↑, latency ↓ |
| `backchannel_icc` | 55 | short backchannels, never a formal turn | TOR ↓, Freq ↑, JSD ↓ |
| `interruption_synthetic` | 200 | yield instantly, then answer the *interrupting* turn | TOR ↑, rating ↑, latency ↓ |

Two deliberate choices:

* **Pauses are terminal.** The user pauses and the stream ends, mirroring FDB,
  where the pause is the last thing in the input audio. If the user resumed and
  finished the turn, a *correct* model would legitimately speak and FDB's single
  binary TOR could not separate "spoke too early" from "spoke correctly".
  Recovery is covered by `smooth_turn`.
* **Backchannel ground truth** is the distribution of the *user's* sentence
  endpoints. Listeners backchannel where the speaker completes a phrase, which
  is the structure FDB's `icc_gt_distribution.json` captures from real ICC
  recordings. Without it there is nothing for the JSD to measure against.

---

## Languages: pt (in-distribution) and en (cross-lingual)

All three variants were fine-tuned on pt-BR duplex data only, so `--lang en`
measures how well the **turn-taking protocol** transfers. The tags are already
language-neutral English strings, so only the corpus and the system prompt
change — each variant gets the English translation of *its own* prompt
(`system_prompts_en.py`) rather than Portuguese instructions with English
content.

`en` has one extra benefit: FDB is natively English, so the judge runs FDB's
**verbatim** rubric instead of a translation.

---

## The original DuplexCascade (upstream) — measured, not just cited

The upstream `sbintuitions/DuplexCascade` checkpoint is downloaded, merged to a
4-bit GGUF and run through the **identical** serving stack as the pt-BR
variants, so its row is measured rather than borrowed from the paper:

```bash
# 1. accept the license at huggingface.co/sbintuitions/DuplexCascade
# 2. download (~17 GB: a PEFT state dict for Qwen/Qwen2-7B-Instruct)
hf download sbintuitions/DuplexCascade model_state.safetensors tokenizer \
  --local-dir /mnt/f/duplex_cascade_runs/original_dc
# 3. merge -> bf16 GGUF -> q4_k_m (streaming, in place; see the module docstring)
python -m fdb_bench.build_original_gguf
# 4. run it
MODELS="original_dc" ./fdb_bench/run_all.sh
```

It reaches **ATA 0.896 (pt) / 0.951 (en)** against its own published 0.858, with
`pause_candor` 0.273 vs 0.222 and interruption relevance 3.924 vs 4.016 — the
strongest available evidence that the harness measures what the paper measures.

Three adaptations are needed and each is documented in `FINDINGS.md`: it
prepends **BOS** (its server does), it **speaks by emitting text** rather than a
tag (its server streams any generated text to TTS and uses tags only to stop),
and it needs the **trailing silence** that upstream's tick timer produces. Its
tag vocabulary also differs (`talking` vs `speaking`, no `system backchannel`)
and is carried per-model.

---

## Layout

| file | role |
|---|---|
| `protocol.py` | tokens (imported from each variant's `data/common.py`), `build_prompt`, per-variant system prompts, model registry |
| `chunking.py` | the training-matching ASR chunker — the bench's most important fidelity knob |
| `clock.py` | the word-rate clock; FDB duration gates |
| `scenarios.py` | the five pt-BR + en subsets at FDB sizes |
| `simulate.py` | the duplex control loop and timeline |
| `adapters.py` | `LlamaDuplexAdapter` (sft/distill/base), `PolicyAdapter` (mtp_like), `HFDuplexAdapter` (original) |
| `metrics.py` | ports of the four FDB eval scripts |
| `judge.py` | FDB's interruption rubric, served by a local GGUF |
| `reference.py` | published DuplexCascade / FDB numbers |
| `run_inference.py` → `run_judge.py` → `run_metrics.py` | the three phases |
| `calibrate_clock.py` | measures `words_per_second` against `pocket_tts` |
| `calibrate_prompts.py` | measures which prompt activates each variant's duplex mode → `PROMPTS.md` |
| `compare.py` | cross-language (pt→en) deltas + protocol tag diagnostics |
| `tests/` | parity against the live bridges + simulator/metric direction tests |

```bash
PY=/home/penhfel/unsloth_uv/bin/python
$PY fdb_bench/tests/test_parity.py        # 16 checks — bench vs the live bridges
$PY -m fdb_bench.tests.test_simulator     # 21 checks — metric directions + ATA
$PY -m fdb_bench.tests.test_metrics       #  8 checks — ports of FDB's own examples
```

What they cover:

* `test_parity.py` — the re-implemented prompt builder is byte-identical to
  `BridgeServer.build_prompt` in both live bridges, the system prompts have not
  drifted, and the turn-end flag provably never leaks into the prompt.
* `test_simulator.py` — scripted stub policies through the real simulator and
  real metrics, pinning every metric's *direction*: a perfect policy must score
  pause TOR 0 / smooth TOR 1, a chatty one the opposite, a deaf one interruption
  TOR 0, and the ATA guards must score a model that never speaks at 0.
* `test_metrics.py` — the metric port reproduces Full-Duplex-Bench's **own
  released example data**, plus the `end_time` round-trip that keeps the
  inference and metrics phases from silently drifting apart.

---

## Setup notes

* `-sp` on `llama-server` is **mandatory** — without it the duplex special
  tokens are silently dropped from `/v1/completions`.
* The `base` control and the `mtp_like` content model need a stock
  Qwen3-4B-Instruct-2507 GGUF. `run_all.sh` expects
  `/mnt/f/duplex_cascade_runs/base_qwen3_4b/gguf/Qwen3-4B-Instruct-2507-q4_k_m.gguf`;
  `run_services.sh` in each variant documents the `convert_hf_to_gguf.py` +
  `llama-quantize` recipe if you need to rebuild it.
* The judge GGUF must be served with `--jinja` so its chat template applies.
* Runs are sequential by design: one 4B model plus the 9B judge will not both
  fit in 16 GB of VRAM with their KV caches.
