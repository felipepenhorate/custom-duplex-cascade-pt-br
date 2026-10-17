# Which system prompt each variant runs with, and why

> **Superseded by the ASR-chunking fix — read this box first.** The tables
> further down were measured while the bench was feeding whole sentences to the
> models. The training stream is 1-7-token pieces (mean 2.19 words,
> `chunking.py`), and that input-shape error dominated every number in them —
> including the prompt ranking it claims to establish. With the stream
> corrected, the no-prompt setting that looked "degenerate" (195-word echoing
> rambles) reaches `finish recall 1.00` with coherent ~34-word replies, so the
> ranking below no longer holds and the tables are **not** evidence.
>
> What survives: the *method* (measure, do not assume) and the defaults it
> landed on, which are also the principled ones — the **training distribution**
> for both duplex LLMs:

| variant | runs with | why |
|---|---|---|
| `sft` | `none` | `data/duplex_train*.jsonl` carries no system item, so this is the distribution the protocol was learned in |
| `distill` | `none` | as above |
| `mtp_like` | `instruction` | the companion's training data *does* carry a generic instruction on ~50% of records (`policy/prep_policy_dataset.py:120`) |
| `base` | `none` | untrained control |
| `original_dc` | `deployed` | the upstream release is English and ships with a prompt |

No candidate is invented by the bench, and none was selected on the evaluation
metrics. Re-running `calibrate_prompts.py` on the corrected chunker would settle
the ranking properly; until then, prefer the training-distribution setting and
treat prompt sensitivity as an open question.
## How to read the tables

| column | meaning |
|---|---|
| `finish P` | precision of `user_finish speaking` — of the times the model took the floor, how often the turn had actually ended |
| `finish R` | recall — of the turns that genuinely ended, how often it took the floor |
| `chunk acc` | all four outcomes / all chunk decisions |
| `pause silent` | fraction of silence ticks where it correctly did **not** take the floor |
| `non-Latin` | fraction of letters outside the Latin script (catches cross-language degeneration) |
| `words` | mean length of a spoken utterance — *this is the trap column* |

**`words` is why recall alone cannot be trusted.** A model that rambles for 200
words always "takes the floor", so recall looks great while the behaviour is
degenerate. A 200-word utterance is also ~60 s of speech, i.e. not duplex
behaviour at all, and it inflates every duration-based FDB metric to a formal
turn.

## sft — 58 scenarios, 252 decisions (pt-BR)

| candidate | prompt len | finish P | finish R | finish F1 | chunk acc | pause silent | words | verdict |
|---|---|---|---|---|---|---|---|---|
| `none` | 0 | 0.363 | 0.971 | 0.528 | 0.766 | 1.000 | 26.7 | over-eager |
| **`deployed`** | 76 | **0.429** | 0.882 | **0.577** | **0.825** | 1.000 | 25.2 | over-eager |
| `tag` | 261 | 0.267 | 0.912 | 0.413 | 0.651 | 1.000 | — | over-eager |

Content was clean for all three (non-Latin 0.000, repeats 0.000, empty 0.000).
**Chosen: `deployed`** — best accuracy and F1, cleanest, and it is what the
shipped bridge uses. The untrained `base` control runs with `none`.

## distill — 58 scenarios, 252 decisions (pt-BR)

| candidate | prompt len | finish P | finish R | finish F1 | chunk acc | pause silent | words | verdict |
|---|---|---|---|---|---|---|---|---|
| `none` | 0 | 0.397 | **0.853** | 0.542 | 0.806 | 0.997 | **195.5** | degenerate |
| `deployed` | 879 | 0.500 | 0.353 | 0.414 | 0.865 | 1.000 | 15.4 | suppresses the turn decision |
| **`tag`** | 261 | **0.593** | 0.471 | **0.525** | **0.885** | 1.000 | 40.8 | — |
| `trigger` (teacher prompt) | 564 | 0.750 | **0.088** | 0.158 | 0.873 | 1.000 | 165.3 | never takes the floor |
| `trigger_tags` | 903 | 0.342 | 0.382 | 0.361 | 0.817 | 0.995 | 70.4 | — |

`deployed` is `distill/DuplexCascade/server.py`'s own prompt — the one the live
bridge ships. `trigger` / `trigger_tags` are `TRIGGER_SYSTEM` and
`TRIGGER_SYSTEM_ALLOW_TAGS` imported verbatim from
`distill/data/teacher_prompts.py`.

This table is the substantive result, and it is not the one intuition predicts:

* **`trigger` — the "duplex role" prompt — has the worst recall of all
  (0.088).** Naming the duplex role and listing the tags does *not* make this
  model commit to the floor; it rambles for 165 words and stays quiet. Adding
  the explicit "use the special tags" instruction (`trigger_tags`) only gets
  recall to 0.382.
* **`none` has the highest recall (0.853) but is degenerate.** Sample output
  under `none`:

  > `! 🌿  \nPlantei alface, tomate e manjericão — e agora **h Ah, entendi! Você quer dizer: **"Plantei…"** 👉 Então sim — **é uma ótima combinação!**`

  A leading stray `!`, then an echo of the user's own words, then emoji and
  markdown — 195 words on average. Its recall is an artefact of never stopping.
* **`deployed` (the shipped prompt) caps recall at 0.353** because it instructs
  the model *not* to print tags. That is the honest explanation for why the
  distill variant needs the bridge to hand it the turn decision.
* **`tag`** — a plain tag-permitting prompt — is the best compromise and is
  what the bench uses.

So the honest conclusion the bench should support is **not** "distill cannot do
duplex" but "distill's autonomous turn-taking is fragile under every prompt,
and much weaker than SFT's" — which lines up with the repo's own teacher-forced
tag accuracy (0.9638 vs 0.9795) and its documented note that the distill model
"does NOT spontaneously emit `<|user is thinking|>` to end a turn in free
generation (the SFT baseline does)".

Note the gap between teacher-forced accuracy (~96%) and free-generation recall
(0.35–0.47): the tag *is* in the model's distribution — `prep_dataset.py` masks
only the first content tokens, and the tag is predicted by the position before
it, so it was supervised — but errors compound once the model is generating on
its own. That is exposure bias, not a missing capability.

## mtp_like — 15 scenarios (pt-BR)

| candidate | finish P | finish R | chunk acc | pause silent | words |
|---|---|---|---|---|---|
| `none` | 0.231 | 1.000 | 0.765 | 0.956 | 23.7 |
| **`instruction`** | 0.222 | 1.000 | 0.753 | 0.972 | 25.1 |

A statistical tie. `instruction` is the companion's on-distribution setting —
its training data carries a system instruction on ~50% of records
(`policy/prep_policy_dataset.py:120`) — so it is chosen for distribution
fidelity. The companion is queried as a *classifier*, so content quality is
governed by the stock content model, not by this prompt.

An earlier run of this comparison made `instruction` appear pathologically slow.
That was a bug in the adapter, not a property of the prompt: the companion's
system item was being added twice (once by the simulator, once by the adapter),
and the duplicate also leaked into the content model's messages. Fixed; the two
candidates now run in the same time.

## Why our mid-turn precision is low across the board

Every model posts precision well below 1.0, and the reason is a property of the
**text domain**, not of the models: the scenario chunker breaks the user's turn
at clause boundaries, so a mid-turn chunk and a turn-ending chunk are frequently
identical as text — both end in `.`. In real audio they are distinguished by
prosody (final lengthening, pitch fall, pause length), and that information does
not survive transcription. DuplexCascade's own pause-handling numbers rest on
exactly those cues.

What this means for reading the results:

* `pause silent` (≈0.95–1.0 everywhere) is the clean measure on the pause task —
  it is about holding the floor during silence, where no text changes at all.
* `finish recall` is the clean measure of commitment — taking the floor when the
  user really did finish.
* `chunk acc` and precision are **lower bounds**: some of the "false positives"
  are decisions that no text-only system could get right.
