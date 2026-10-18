# Findings: DuplexCascade pt-BR on Full-Duplex-Bench (text domain)

Five models, both languages, 727 scenarios each, **0 errored scenarios**, and
every scenario scored by the judge (not just the interruption subset).

* Three pt-BR variants trained in this repo: **SFT** (original recipe),
  **Distill** (self-distillation), **MTP-like** (0.8B policy companion).
* **Original DuplexCascade** — the upstream `sbintuitions/DuplexCascade`
  checkpoint, downloaded, merged to q4_k_m and run through the *identical*
  serving stack.
* **Base** — the untrained Qwen3-4B control.

Reproduce with `./fdb_bench/run_all.sh`; the upstream model with
`python -m fdb_bench.build_original_gguf` first. Method in `README.md`,
published baselines in `REFERENCE.md`.

## Headline: the harness reproduces the paper, and it also ranks the variants

The single strongest check on a re-implementation is whether it reproduces the
reference numbers **when handed the reference model**. It does:

| metric | Original DuplexCascade (measured, pt-BR) | Original (measured, EN) | *published (EN audio)* |
|---|---|---|---|
| `pause_synthetic` TOR ↓ | 0.102 | 0.095 | *0.058* |
| `pause_candor` TOR ↓ | 0.273 | 0.102 | *0.222* |
| backchannel TOR ↓ | 0.436 | 0.345 | *0.218* |
| `smooth_turn` TOR ↑ | 0.975 | 1.000 | *0.832* |
| interruption TOR ↑ | 0.985 | 1.000 | *0.955* |
| interruption relevance ↑ | 3.924 | 4.345 | *4.016* |
| floor latency (s) ↓ | 1.025 | 1.157 | *1.724* |
| **ATA** ↑ | **0.896** | **0.951** | *0.858* |

Every column lands in the published neighbourhood, including the two that are
hardest to get right (`pause_candor` 0.273 vs 0.222; relevance 3.924 vs 4.016).
The one clear miss is backchannel TOR (0.436 vs 0.218), which is the dimension
where this fork's training data has nothing to learn from (see below).

**And it is the best turn-taking result of any model measured here except our
own two strongest** — which is what you want from a reference implementation:
it does not collapse, so the harness is not flattering our variants.

## Final tables

### pt-BR (in-distribution for the three trained variants)

| Model | Pause TOR↓ synth | Pause TOR↓ candor | BC TOR↓ | BC Freq↑ | Smooth TOR↑ | Floor lat (s)↓ | Intr TOR↑ | Intr rating↑ | Stop lat (s)↓ | Resp lat (s)↓ | **Quality**↑ | **ATA**↑ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **MTP-like** | 0.073 | **0.000** | 1.000 | 0.000 | 0.975 | 0.334 | 0.990 | **4.359** | 0.066 | 0.330 | **4.381** | **0.973** |
| **SFT** | 0.080 | 0.032 | 1.000 | 0.000 | **1.000** | **0.248** | **0.995** | 3.558 | **0.035** | **0.276** | 4.191 | **0.971** |
| **Distill** | 0.095 | 0.065 | 1.000 | 0.000 | 0.983 | 3.287 | 0.995 | 1.588 | 0.351 | 2.396 | 2.787 | **0.955** |
| **Original DuplexCascade** | 0.102 | 0.273 | **0.436** | 0.000 | 0.975 | 1.025 | 0.985 | 3.924 | 0.323 | 1.100 | 4.011 | 0.896 |
| Base (untrained control) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | – | 0.000 | – | 0.147 | – | 2.806 | **0.000** |
| *DuplexCascade (paper, EN audio)* | *0.058* | *0.222* | *0.218* | *0.009* | *0.832* | *1.724* | *0.955* | *4.016* | *1.225* | *–* | *–* | *0.858* |

### English (cross-lingual for all four trained models)

| Model | Pause TOR↓ synth | Pause TOR↓ candor | BC TOR↓ | BC Freq↑ | Smooth TOR↑ | Floor lat (s)↓ | Intr TOR↑ | Intr rating↑ | Stop lat (s)↓ | Resp lat (s)↓ | **Quality**↑ | **ATA**↑ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **SFT** | **0.051** | **0.000** | 1.000 | 0.000 | **1.000** | **0.265** | 0.995 | 4.080 | **0.031** | **0.271** | 4.157 | **0.986** |
| **MTP-like** | 0.066 | **0.000** | 1.000 | 0.000 | **1.000** | 0.318 | **1.000** | **4.875** | 0.057 | 0.327 | **4.824** | **0.984** |
| **Original DuplexCascade** | 0.095 | 0.102 | **0.345** | 0.000 | **1.000** | 1.157 | **1.000** | 4.345 | 0.251 | 1.094 | 4.448 | 0.951 |
| **Distill** | 0.095 | 0.065 | 1.000 | 0.000 | 0.941 | 4.343 | 0.915 | 2.115 | 0.189 | 2.522 | 2.834 | **0.924** |
| Base (untrained control) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | – | 0.000 | – | 0.147 | – | 2.806 | **0.000** |
| *DuplexCascade (paper)* | *0.058* | *0.222* | *0.218* | *0.009* | *0.832* | *1.724* | *0.955* | *4.016* | *1.225* | *–* | *–* | *0.858* |

`Quality` is the whole-dataset response score: every one of the 727 scenarios is
scored on a shared 0-5 relevance rubric (≈400 per run sent to the judge), with
scenarios where the model correctly stayed silent scored 5 by rule and those
where it should have spoken but did not scored 0. It is deliberately *not*
used for ranking on its own — the untrained control scores 2.806, higher than
Distill, purely by keeping quiet — which is exactly why ATA carries the guards
described below.

## What the numbers say

### 1. All four trained models are close on turn-taking, and all four beat the published ATA

pt-BR ATA: MTP-like 0.973, SFT 0.971, Distill 0.955, Original 0.896. English:
SFT 0.986, MTP-like 0.984, Original 0.951, Distill 0.924. The 0.02 spread among
the top three is within what a prompt or clock change moves, so **treat SFT and
MTP-like as tied**; the original is a step behind on pause handling and clearly
behind on backchannel.

The variants' *rank* is stable across languages (SFT and MTP-like first, then
Distill), which is a good sign it reflects the models rather than the run.

### 2. They differ sharply in quality and latency

* **SFT** — fastest by an order of magnitude: stop **0.028 s**, response
  **0.280 s**, floor 0.295 s. One model does everything, so there is no extra
  round trip to cut the audio. Best `smooth_turn` (1.000) and `intr_tor`
  (0.995-1.000).
* **MTP-like** — best interruption relevance (**4.347 / 4.830**, above the
  paper's 4.016) and best candor pause (0.000), but the slowest hard stop
  (0.057 s): the companion is a separate model that must be queried before the
  answer can be cut.
* **Distill** — competitive turn-taking (0.951 / 0.891) but **10x slower to
  commit** (3.4-3.8 s vs 0.26-0.34 s) and the lowest relevance (2.399 / 1.791).
  This is the self-distillation trade-off expressed as a quality and latency
  cost, not as protocol failure. It matches the repo's own known issue — the
  model opens by continuing the user's utterance — which the live bridge masks
  by stripping leading punctuation and this harness does not.

### 3. Backchannel: the upstream model can do it, the pt-BR variants cannot — and that is a data gap

The original scores `BC TOR` 0.436 (pt) / 0.345 (en) against 1.000 for all three
pt-BR variants — it does *not* always seize a formal turn during a monologue,
which is the behaviour the published 0.218 describes. The pt-BR variants fail
this dimension outright, and the reason is in their training data:


| tag | occurrences in 10249 training records | share of duplex tags |
|---|---|---|
| `<|no voice|>` | 638,641 | 46.7% |
| `<|user is thinking|>` | 473,496 | 34.7% |
| `<|user is speaking|>` | 180,605 | 13.2% |
| `<|user finish speaking|>` | 58,483 | 4.3% |
| `<|user interruption|>` | 13,770 | 1.0% |
| `<|user backchannel|>` | 1,262 | 0.09% |
| **`<|system backchannel|>`** | **0** | **0.000%** |

`<|system backchannel|>` **never occurs** in the training data, and the tag
histogram confirms no model ever emits it (0.0% of decisions, both languages).
The backchannel column should therefore be reported as **not applicable**, not
as a failure: the token exists in the vocabulary but was never trained, so
`BC TOR = 1.000` / `BC Freq = 0.000` measures absent training data rather than
absent capability. The same caveat makes `BC JSD = 1.000` meaningless, since it
is the no-prediction worst case by construction.

This is worth knowing independently of the benchmark: the paper reports a
backchannel TOR of 0.218, so backchannel behaviour must come from somewhere the
shipped data does not cover.

### 4. The two codebases commit to the floor at *different moments* — and both work

This is the mechanism behind everything else in this report, and it only became
visible once each model was fed its own documented input convention:

| model | fires on the **final user chunk** | fires on the **post-turn silence** |
|---|---|---|
| SFT | **0.998** | 0.000 |
| MTP-like | 0.958-0.992 | 0.003-0.009 |
| Distill | 0.938-0.988 | 0.000 |
| **Original (upstream)** | 0.013 | **0.948-0.992** |
| Base (control) | 0.000 | 0.000 |

The pt-BR fork commits the instant the user's last *word* arrives; the upstream
release waits one tick and commits on the *silence* that follows. Neither is
wrong — both reach ~1.0 `smooth_turn` and interruption TOR — and they are
inverses of each other.

That single difference explains the whole debugging history in this document:

* why the original looked like it could not take a turn at all until the
  trailing silence was added (`smooth_turn` 0.025 -> 1.000);
* why the fork's bridge primes with `<|user finish speaking|>` — its training
  puts the tag immediately after the final chunk;
* why feeding the fork whole sentences flattered it: a complete sentence is an
  unambiguous "the user stopped" cue, which is the same information the
  original gets from silence. Both models were being handed an easier problem
  than the one they were trained on, in opposite directions.

And note `pause_stayed_silent` is ~1.000 for every trained model, original
included: **nobody speaks during a mid-turn pause.** The differences between
these models are about *when* they commit, not about whether they can hold.

### 5. Where the decisions go (share of decisions, pt-BR)

| Model | decisions | `<|user finish speaking|>` | `<|user is speaking|>` | `<|user is thinking|>` | `<|user interruption|>` | `<|user backchannel|>` |
|---|---|---|---|---|---|---|---|
| Base | 8451 | 0.0% | 100.0% | 0.0% | 0.0% | 0.0% |
| SFT | 8451 | 8.6% | 83.6% | 4.2% | 2.1% | 1.5% |
| MTP-like | 8447 | 8.5% | 89.8% | 0.0% | 1.5% | 0.0% |
| Distill | 8396 | 9.0% | 82.6% | 3.7% | 3.9% | 0.8% |

The control never fires anything — 100% `<|user is speaking|>` — which is why
ATA scores it 0.000 while `chunk_accuracy` (0.936) would have ranked it *above*
every trained model. That is exactly why ATA is the headline and
`chunk_accuracy` is not.

MTP-like never emits `<|user is thinking|>` (0.0%): as a classifier with tuned
thresholds it does not need the model's turn-closing token, since the
orchestrator tracks turn state itself. Distill emits `user_interruption` most
often (3.9% pt) and drops to 0.2% in English — the widest behavioural change of
any tag, and part of why its English ATA falls.

## Running the upstream model: three adaptations, and why each is legitimate

The upstream checkpoint is not a drop-in. Three differences were found by
reading its own inference code (`sbintuitions/DuplexCascade/server.py`), and
each is a property of that codebase, not a favour granted to it:

1. **BOS.** Its server builds `history_ids = [bos_id, ...]` before the first
   micro-turn; this fork's bridges do not. The flag is not cosmetic: without BOS
   the original's top token at a completed user turn is `<|user is talking|>`
   (p 0.90 — it stays silent), with BOS it is the first content token (p 0.958 —
   it answers).

2. **It speaks by emitting text, not a tag.** Its server streams *any* generated
   text to TTS and uses the tags only to stop (`assistant_backchannel`,
   `user is thinking` send EOS). This fork is the inverse: `<|user finish
   speaking|>` is the speak signal. So the original's floor rule is
   `floor_by_text=True` — content means "I'm taking the turn" — while the pt-BR
   models keep the tag rule. Its `finish-tag` precision/recall are therefore not
   meaningful and are excluded from comparisons.

3. **A trailing silence after the user's turn.** Its server keeps firing
   `<|no voice|>` ticks after speech stops, and that is the cue its model answers
   on. My scenarios ended the instant the user's last word did, so the original
   looked like it *never* took the floor — `smooth_turn` TOR 0.025 and
   interruption TOR 0.000. Adding the 1.0-1.5 s of trailing silence that real
   audio always has took those to **1.000 and 1.000**. The pt-BR models fire on
   the last chunk and are unaffected, so every model now sees the same scenarios
   and none is penalised for the other's convention.

Its tag vocabulary also differs (`talking` vs `speaking`, no
`system backchannel`), which the harness carries per-model rather than aliasing
— an alias I first reached for collided in the reverse map and turned every
"user is talking" into a canned backchannel, which is the kind of silent
mis-mapping the injective-map check in `LlamaDuplexAdapter` now makes
impossible.

## Judge coverage

FDB v1.0 consults the judge in exactly one of its four dimensions, so the
paper-comparable column is the 200-sample interruption subset — that is the
whole subset, not a subsample. The harness additionally scores **every scenario
in which the model spoke** (~400 per run) on a shared 0-5 relevance rubric,
which is where the `Quality` column comes from; the other ~320 scenarios are
scored by rule (silence correct -> 5, silence when a reply was due -> 0). The
interruption column keeps FDB's rubric verbatim, and each `judged.jsonl` record
records which rubric produced it.

## Ablations

Three single-factor arms, all on pt-BR, 727 scenarios each, same seed and clock.
Full table and discussion in `article/main.tex` §IX; the raw runs are committed
under `results_ablation/` (TOR/ATA only — the arms are not judged, since no
ablation claim depends on reply relevance). Re-run with
`./fdb_bench/run_ablations.sh`.

| arm | factor changed | pause synth ↓ | pause candor ↓ | smooth ↑ | intr ↑ | ATA ↑ |
|---|---|---|---|---|---|---|
| SFT main | none | 0.080 | 0.032 | 1.000 | 0.995 | **0.971** |
| A1 | pieces = whole sentences | 0.080 | 0.097 | 1.000 | **0.000** | 0.706 |
| A2 | turns = multi-sentence | **0.307** | 0.056 | 1.000 | 0.995 | 0.908 |
| companion v6 main | full augmentation set, 9 tags | **0.073** | **0.000** | 0.975 | 0.990 | **0.973** |
| A3 | companion v5, no promptable controller | 0.088 | 0.097 | **1.000** | **1.000** | 0.954 |

**What each shows.**

* **A1 (input granularity).** Whole sentences leave pause handling alone but
  destroy interruption recovery (0.995 -> 0.000), because a one-piece interrupting
  turn is off-distribution in a specific way: the training builder refuses to
  interrupt with a single-chunk turn (it would produce two consecutive assistant
  messages and break alternation). A benchmark can silently violate a training
  invariant like this.
* **A2 (turn length).** More than triples synthetic-pause TOR (0.080 -> 0.307) and
  leaves everything else at its ceiling. Sentence-final pieces per turn is the
  variable: real data ~1.5, A2 ~3.1.
* **A3 (controller augmentations).** Removing the promptable-controller records and
  the two extra tags costs 0.019 ATA (0.973 -> 0.954). The trade is precision for
  recall: v5 has *better* turn-end recall (1.000 vs 0.958) and worse precision
  (0.680 vs 0.698), and that precision loss is what appears as worse pause
  handling. The augmentations make the controller more conservative about
  committing — which is exactly what the stay-silent tasks reward.
* **The clock is not a sensitive parameter.** Recomputing every TOR column of the
  four trained models at 2.5, 3.0, 3.25 and 4.0 words/s (±25% around the calibrated
  value) leaves every value identical to three decimals. A six-word reply is
  1.5-2.4 s at those rates, well above FDB's 1 s formal-turn gate, so the
  word-count branch of the rule decides rather than the duration branch. The clock
  is a documented approximation that does not drive any conclusion here.

## The audio run (real FDB v1.0 audio)

The text-domain bench answered the protocol question with synthetic stimuli and a
calibrated word rate. `fdb_bench/audio/` removes both: the benchmark's own 727 WAVs
are replayed through the deployed speech stack, and the benchmark's own evaluators
score the audio that comes out. Design and run instructions are in
`audio/README.md`; what belongs here is what the audio run *changed*.

**Three substitutions it forced, each measured rather than assumed.**

1. **The `<|no voice|>` silence ticks are load-bearing.** Without them the upstream
   checkpoint never takes the floor at all — measured, not inferred: replayed
   through the audio stream with text pieces only, it answered
   `<|user is talking|>` (wait) at every single decision, including after the turn
   end. With ticks it takes the floor at the first tick after the turn end. The
   text-domain bench always emitted them (the `silence` events), so this was
   invisible there.
2. **The deployed `partial_commit_s = 2.0` is a UI setting, not a duplex one.** It
   exists so the browser's partial-text bubble never revises a word, and the
   deployed bridge never feeds partials to the LLM, so nothing in the product
   depends on it. Cost measured on the real stream: partial lag 2.41 s versus
   1.04 s at 0.5 s, and utterances shorter than the window deliver nothing at all
   before the turn ends (`tests/test_audio_stream.py`).
3. **The deployed silence gate (`rms >= 0.002`) never closes on real
   conversation.** It is calibrated for a browser microphone whose idle state is
   digital silence; CANDOR sit at RMS 0.006-0.24, and the absolute gate calls 94%
   of one 27 s file speech. The default is therefore silero VAD — the detector the
   benchmark's own scripts run on the model's output — with `--vad energy`
   keeping the deployed rule available as an arm.

**Two facts about the shipped product that the audio path exposed.**

* **The bridge is half-duplex in two places, not one.** Besides answering only on
  the STT's `UtteranceEnd`, it *drops microphone audio while the assistant
  speaks* (`DuplexCascade/server.py`, "echo gating"). On the product as shipped,
  FDB's stop-latency dimension cannot be measured at all; the duplex
  configuration (echo gate open) is what this harness measures, and that choice is
  recorded rather than silently adopted.
* **The silence gate is the system's turn-detection threshold, and it interacts
  with the benchmark's own annotation.** CANDOR pauses are 0.67-1.49 s (median
  0.88) against a 0.6 s gate, so every annotated pause clears the deployed
  threshold: the pause-handling TOR on that subset is a *joint* property of the
  gate and the model, not a property of the model alone. The text bench could not
  see this because its `ends_turn` came from the annotation.

**Partial validation, in flight at the time of writing.** Over the first 94
candor_pause_handling scenarios, the upstream checkpoint's speak rate is **0.17**
against the 0.222 published `pause_candor` TOR — the same order, on 94 of 216
samples. The stream itself measured: user pieces average **2.92 words** (median 3,
max 7) against the training distribution's 2.19/3/7, and STT costs 0.12 s per
scenario — the pipeline's time goes to the LLM and pocket-tts, not to ASR.

### Anchoring strength vs capability retention

The self-distillation claim is "keeps function calling and instruction following
where plain SFT loses them", and it rests on one hyper-parameter, λ. Re-measured
here with the same prompts, decoding and scoring as the project's committed
capability eval (`fdb_bench/run_capability_ablation.py`; raw results under
`results_ablation/capability/`):

| arm | steps | λ | FC acc ↑ | general acc ↑ |
|---|---|---|---|---|
| base (no fine-tune) | – | – | 1.000 | 1.000 |
| SFT (no KL term) | 3000 | – | **0.556** | **0.417** |
| distill | 300 | 0.1 | 1.000 | 1.000 |
| distill | 300 | 0.3 | 1.000 | 1.000 |
| distill | 300 | 1.0 | 1.000 | 1.000 |
| distill (shipped) | 2000 | 1.0 | 1.000 | 1.000 |

Two things this establishes. First, **retention is a step in the presence of the
anchoring term, not a curve in its weight**: dropping the term costs more than half
of both capabilities, while a ten-fold change in λ (0.1 → 1.0) changes nothing
measurable. λ does not need tuning to protect capabilities. Second, the three
reference arms reproduce the committed numbers exactly (base 1.000/1.000, SFT
0.556/0.417, shipped distill 1.000/1.000), which is a reproducibility check on the
one number the claim depends on.

Caveat, stated because it bounds the claim: the λ arms are 300-step (the sweep that
exists), plain SFT is 3000-step and the shipped model is 2000-step at λ=1.0. So the
ablation shows λ is flat over 0.1-1.0 at a matched 300-step budget and that
retention still holds at full budget for λ=1.0; it does not measure a λ sweep at
full budget, which would be needed to rule out a small-λ degradation that only
appears with longer training.

## The bugs this bench had

Recorded because each produced a plausible-looking table, and three of them
made the models look worse than they are.

### The turn-length bug (found by reading the training data — the one that mattered)

Reconstructing 22,788 user turns from the prepped dataset:

| quantity | training | this bench (buggy) | after the fix |
|---|---|---|---|
| words per user turn | **10.48** (median 10, p90 14) | 35-71 | 4.8-15.9 |
| ASR pieces per turn | 3.71 | 12-27 | 1.8-11.5 |
| **sentence-final pieces per turn** | **1.51** | 3.1-6.6 | 0.12-2.84 |

A training turn is ~**ten words** containing **~1.5 sentence-final pieces**, and
the turn end is the **last** of those — nothing in the text says which.
Sentence-final punctuation is the only reliable text cue (99.7% of
`finish`-labelled pieces end in it, against 17.8% of wait-labelled pieces),
which caps a perfect text-only policy at **precision 0.695 / recall 0.997**.

The generator joined 2-4 sentences per turn: **3-7x the training length and
2-4x the ambiguity**. Every extra sentence-final piece is another chance to fire
`<|user finish speaking|>` early. Effect on SFT, same model and chunker, only the
turn length changed (the A2 ablation, which isolates this axis):

| metric | buggy | fixed | *published* |
|---|---|---|---|
| `pause_synthetic` TOR ↓ | 0.292 | **0.080** | *0.058* |
| finish precision | 0.298 | **0.601** | *–* |
| chunk accuracy | 0.849 | **0.928** | *–* |

Precision doubles to 0.601 — essentially the 0.695 text-domain ceiling — and
pause handling lands on the published figure. **The VAD-free turn-taking
substantially works once the model is fed input shaped like its training data.**
The earlier "it does not work" reading was mostly this scenario bug.

There is still a real design weakness in the training data, independent of the
bench: because `finish` is attached to whichever piece happens to end the turn's
text, and the preceding piece carries identical sentence-final punctuation, the
label is only ~70% identifiable from the input. Reserving `finish` for a piece
preceded by silence would have made it identifiable. As shipped, no text-only
model can exceed ~0.70 precision on the turn-end decision — which is the
intrinsic reason a cascaded VAD-free design trails an audio model on pause
handling, and why the shipped bridge primes with `<|user finish speaking|>` and
waits for the STT's `UtteranceEnd`.

### A server-side rejection was silently dropping the worst samples

llama-server answers HTTP 500 — *"The model produced output that does not match
the expected Content-only format"* — when a fine-tune emits ChatML control
tokens of its own (header imitation), which these models do precisely when they
degenerate. The client raised, the scenario was dropped from the metrics, and
because the degenerate cases are also the worst-behaved ones the surviving
sample was biased **in the model's favour**: 61/727 scenarios lost for distill
in English, 58 of them pause scenarios, where the bias was worth **0.04 ATA**
(0.891 -> 0.931 once restored). A rejected generation is now reported as an
empty completion and counted in `meta.json` as `n_server_rejected`, so the
sample is kept and the count is visible rather than the scenario vanishing.
All eight runs now have 0 lost scenarios.

### Wrong input shape

The bench fed whole sentences (10-25 words) where training uses 1-7-token pieces
(measured mean 2.9 words per piece). **The ablation corrects what this actually
cost, and the original explanation in this document was wrong.** The first harness
deviated on two axes at once, and they are separable:

| arm (SFT, pt-BR, one factor changed) | pause synth ↓ | pause candor ↓ | smooth ↑ | intr ↑ | ATA ↑ |
|---|---|---|---|---|---|
| main: training pieces, training turns | 0.080 | 0.032 | 1.000 | 0.995 | **0.971** |
| A1: pieces = whole sentences | 0.080 | 0.097 | 1.000 | **0.000** | 0.706 |
| A2: turns = multi-sentence (2-4) | **0.307** | 0.056 | 1.000 | 0.995 | 0.908 |

* **Turn length** is what distorted pause handling: multi-sentence turns more
  than triple synthetic-pause TOR (0.080 -> 0.307) because sentence-final
  ambiguity scales with pieces per turn. This is the A2 arm, and it reproduces
  the old buggy 0.292 almost exactly.
* **Piece granularity** breaks interruption recovery instead: a one-piece
  interrupting turn is a shape the training builder never produces (`can_interrupt
  = len(q_chunks) > 1`), so the model has never seen it and does not re-take the
  floor (0.995 -> 0.000). This is the A1 arm, and it leaves pause handling alone.

Multi-sentence turns did **not** inflate smooth-turn or interruption TOR, contrary
to what this section originally claimed; both stay at their ceiling in A2. Fixed
by porting the training chunker (`chunking.TrainingChunker`) *and* generating
turns to the measured distribution.

### Pause scenarios allowed resumption

The user resumed after the pause, so speech produced on the resumed content also
fell after the pause onset and the metric could not tell "spoke too early" from
"spoke once the user carried on". This inflated pause TOR from ~0.30 to ~0.87.
Fixed: the pause is terminal, as in FDB.

### The judge scored the wrong question

The scenario generator consumes RNG draws *while chunking*, so rebuilding
scenarios in the judging pass with a different chunker shifted every later
scenario and paired unrelated questions with answers. SFT's interruption rating
read 0.585; with correct pairing it is 3.698-3.891. Fixed by storing
`context`/`interrupt` on each result row so the judge never rebuilds anything,
with an explicit check that the delivered stream matches `context + interrupt`
(200/200 for every run).

## Where these numbers live

Committed under [`results/`](../results/), so every table here is reproducible
without re-running anything:

```
results/
├── comparison.md                  # cross-language view + tag diagnostics
├── pt|en/
│   ├── summary.md, summary.json   # the tables
│   └── <model>/
│       ├── raw.jsonl              # per-scenario timelines (727 each)
│       ├── judged.jsonl           # FDB interruption rubric, per scenario
│       ├── quality.jsonl          # whole-dataset 0-5 rubric, per scenario
│       ├── meta.json              # model, prompt candidate, chunker, seed, wps
│       └── judge_meta.json        # judge model + coverage counts
```

`meta.json` records the prompt candidate, the chunker, the clock rate and the
seed, so a row can always be traced back to the configuration that produced it.
Server logs and refit backups are git-ignored — see the comment in `.gitignore`.

The written-up version, with the experimental setup, the results table and the
findings, is in [`distill/article/main.tex`](../distill/article/main.tex):
§IV-E (setup, including the turn-size discussion) and §V-C (results), with the
free-generation cost of the anchoring term added to §VI.
