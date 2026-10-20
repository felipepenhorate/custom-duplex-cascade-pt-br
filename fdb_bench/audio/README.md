# Audio Full-Duplex-Bench harness

The text-domain bench (`../README.md`) replays **synthetic** timed text and turns
the model's words into seconds with a calibrated word rate. This harness removes
both approximations: it runs the **real FDB v1.0 audio** through the deployed
speech stack and lets the benchmark's own evaluators score the audio that comes
out.

```
real FDB input.wav
   └─ silero VAD (the benchmark's own detector) ────────────────► turn end / barge-in
   └─ faster-whisper `medium`, deployed cadence ────────────────► user micro-turn pieces
          └─ the model decides (same protocol as the text bench)
                 └─ pocket-tts (the deployed voice) ─────────────► assistant audio
                        └─ rendered output.wav, timed on the input's clock
                               └─ peak-normalised ───────────────► parakeet (EN) / whisper large-v3 (pt)
                                      └─ the benchmark's eval_*.py ─► metrics
```

## What is now real, and what is still substituted

| component | text bench | audio bench |
|---|---|---|
| user speech | synthetic timed text | **real FDB WAVs** (727 scenarios, 3.6 h of audio) |
| user transcript | written to match the training distribution | **faster-whisper `medium`** on the real audio (errors, casing, punctuation included) |
| turn-end cue | annotation (`ends_turn` from the GT transcript) | **silero VAD**, 0.6 s silence, the benchmark's own detector |
| assistant duration | `words / 3.25` (calibrated clock) | **measured** from the synthesised audio |
| assistant speech | not produced | **pocket-tts** (the deployed voice) |
| barge-in | truncate the text timeline | **truncate real playback** |
| scoring | FDB's rules ported to text | **upstream `eval_*.py`**, unmodified |
| `output.json` | n/a | **parakeet-tdt-0.6b-v2** (upstream's own recipe); whisper large-v3 on the pt-BR set, which parakeet cannot read |
| interruption judge | local Qwen3.5-9B | local Qwen3.5-9B (upstream uses `gpt-4-turbo`; no API key here) |

## How the user stream is built

`sft/services/stt_service.py` is the deployed streaming recogniser, and its
semantics are ported verbatim in `stt.py`: 24 kHz frames resampled to 16 kHz, a
beam-1 pass every **0.6 s** emitting words whose audio is older than the commit
window, and one beam-5 pass over the whole utterance when a **0.6 s** silence gap
ends it, which then emits every remaining word plus the turn-end signal. All
timestamps are in audio seconds from the start of the WAV, so the stream shares
the origin the benchmark's annotations and metric scripts use.

Three deliberate departures, each measurable:

1. **Settled words are kept.** The deployed bridge discards them and hands the LLM
   only the final bulk transcript at the end of the utterance — a VAD-gated
   half-duplex front end. Feeding settled words as they settle is the micro-turn
   stream the protocol was trained on, and the only substitution this harness
   makes.
2. **The commit window is 0.5 s, not the deployed 2.0 s.** The 2.0 s window exists
   so the browser's partial-text bubble never revises a word; the bridge never
   feeds partials to the LLM, so nothing in the product depends on it. Measured
   cost of the deployed value: partials lag the speech by 2.41 s instead of
   1.04 s, and utterances shorter than the window produce no partials at all
   (`tests/test_audio_stream.py::test_commit_window_delays_the_words`).
3. **The silence gate is silero, not frame energy.** The deployed service calls a
   frame "speech" at `rms >= 0.002`, which is calibrated for a browser microphone
   whose idle state is digital silence. CANDOR sit at RMS 0.006–0.24, so that gate
   calls 94% of real conversation speech and never fires a turn end. silero is what
   the benchmark's own scripts run on the model's output, so the system's notion of
   "the user stopped talking" matches the benchmark's notion of silence;
   `--vad energy` keeps the deployed rule available as an arm.

Pieces are cut to the training maximum (`user_chunk_max=7`); the emitted stream is
asserted to be lossless and to stay within that bound.

## The pt-BR set: a re-voicing, and what makes it faithful

The pt-BR set is the same 727 scenarios with the user side in Portuguese: the
benchmark's audio is segmented with silero, each span is translated, spoken by
pocket-tts, and put back on the timeline. `build_pt_audio.py` does it; the
result lives in `/mnt/f/fdb_data/pt_v1.0`.

**The build policy is: nothing stretched, nothing padded.** The Portuguese
speaker says the same thing at their own speed; the gap between two consecutive
spans -- the real pause the benchmark annotates -- keeps its length exactly; and
the scene shifts by the accumulated difference between the Portuguese speech and
the English span it replaces. The annotations (`pause.json`,
`turn_taking.json`, `interrupt.json`) are re-timed with the same map, so the set
is internally consistent: the same events, at the same place in the scene.

That policy replaced the obvious one, and the reason is worth recording because
the obvious one looked fine. The first build time-fitted every translated span
into the English span it replaced and filled the remainder with digital silence.
The Portuguese TTS is shorter than the English span in 61% of the segments
(median 0.91 of the slot; 30.4% needed more than 1.33x of expansion to fill),
so 30.4% of the segments carried a pad. **That pad was silence inside the
user's turn**, and the harness's turn-end signal is a 0.6 s silence gate: the
padding produced turn-end invitations in the middle of a sentence. Measured on
the traces, invitations landing in silence the benchmark does *not* annotate as
a pause went from 58 (English) to 303 (Portuguese) over the same 353 scenarios,
and at every class of invitation every model took the floor 1.2-2.0x more often
on the re-voiced audio -- including the English-trained upstream model, which
cannot be a language-skill effect. The full diagnosis is in
`OOD_PARADOX.md`.

Stretching to fill the slot instead was rejected on measurement: 30.4% of the
segments would have needed more than 1.33x of expansion and 10.8% more than 2x,
which is a larger distortion than a speaker's uniform rate difference.

**`check_pt_structure.py` is the gate**, and it is what the first build lacked.
It compares the pt audio with the recording it was made from -- speech fraction,
spans, gaps of 0.6 s or more, and silence outside the benchmark's own pause
windows -- and fails if the two shapes differ. A set that fails it is not a
Portuguese benchmark; it is a different test.

**Recognition is level-normalised, in both sets.** The two pocket-tts voices
differ by 12.6 dB (english -18.5 dB RMS, portuguese -31.1 dB), and whisper's VAD
-- which runs before the decoder -- rejects a reply it cannot find speech in: on
the pt set 23 of 119 smooth scenarios had audible audio and an empty
transcript, so the model was scored as mute. Every rendered file is therefore
peak-normalised before it is recognised (`score.normalise_for_recognition`), so
the recogniser reads what was said rather than how loud the voice was. The
English renders were already at a normal level (median peak 0.58), and the same
rule is applied to them.

**The scorer is language-matched.** `parakeet-tdt-0.6b-v2` is the benchmark's own
recipe and it is English-only, so it reads a Portuguese reply as silence: it
inflates the pause columns (an unread reply looks like good restraint) and
deflates smooth-turn and interruption. The pt set is therefore scored with
`faster-whisper large-v3 --language pt` (`rescore_pt.py`), and the English set
keeps parakeet. Both scorers run on normalised audio.

## A known accounting gap (found on the pt-BR set)

`Adapter.last_call_s` -- the number the session bills into latency -- times only
the *first* generation of a decision. When the model takes the floor, the adapter
then runs a micro-turn loop (up to `MAX_MICRO_TURNS = 10` further generations,
copied from the deployed bridge), and that loop's wall time is **not** included.

It went unnoticed on the English set because the upstream model closes its turns
quickly there (1.1 assistant turns per scenario, ~38 words). On the pt-BR set the
same model rambles through Portuguese: 1.6 turns per scenario and the full loop
every time, which is why the pt sweep runs at 19.1 s/scenario against 7.0 while
the *recorded* per-scenario compute stays at ~8 s. The missing ~11 s is that
loop.

Consequences, stated so the two sets are read correctly:

* **turn-taking numbers are unaffected** -- invitations and the decision to take
  the floor do not depend on the clock;
* **latency columns are under-reported for rambling models**, so the pt-BR
  floor-latency figures are optimistic by roughly the loop time (up to a few
  seconds per turn). The English figures carry the same, smaller, bias;
* fixing it changes the meaning of the latency column, so it should be applied to
  both sets and both re-run rather than patched mid-sweep. The sweep in flight
  therefore keeps the old accounting throughout.

## Latency, precisely

An assistant turn is placed at the moment its triggering piece became available to
the model **plus** the model's measured inference time; a barge-in stop lands at
the speech onset **plus** the model's reaction time. That is the pipeline's honest
latency floor — it excludes network jitter and audio-device buffering — and it is
why these latencies are larger than the text bench's compute-only numbers. The
deployed ASR cadence (0.6 s) and commit window bound how fast any of it can
happen, which is a result rather than a defect: the text bench could not see it.

## Barge-in policy

The primary configuration is the duplex one (echo gate open), matching the text
bench: the user's voice cuts whatever is playing and the stop is timed from the
reaction. The shipped bridge instead **drops microphone audio while the assistant
speaks** (`DuplexCascade/server.py`, "echo gating"), so on the product as shipped
FDB's stop-latency dimension cannot be measured at all. That choice is recorded
here rather than silently adopted.

## Running it

```bash
# one model, all 727 scenarios, inference then scoring
./fdb_bench/audio/run_audio_all.sh

# quick look
MODELS=original_dc LIMIT=5 ./fdb_bench/audio/run_audio_all.sh

# a single model by hand (LLM served separately, judge on :8099)
python -m fdb_bench.audio.run_audio --model sft --out results_audio/decision/sft
python -m fdb_bench.audio.run_audio --model sft --out results_audio/decision/sft --score-only
```

Requirements: FDB v1.0 unpacked under `/mnt/f/fdb_data/v1.0` (`dataset.py` says
how), `faster-whisper`, `pocket-tts`, `silero-vad`; scoring additionally needs
NeMo + `nvidia/parakeet-tdt-0.6b-v2`, which live in `/home/penhfel/nemo_uv`
because NeMo pins `transformers<5`. A llama.cpp server with the judge GGUF must be
up on `:8099` for the relevance column.

## Where the files go

The rendered audio is the bulk of a run (hundreds of MB) and the repo keeps only
what a reader needs to check a number, so the two are split:

```
/mnt/f/fdb_audio/<mode>/<model>/<subset>/<id>/output.wav    # the audio
results_audio/<mode>/<model>/
    <subset>/<id>/output.json   # parakeet transcript in the benchmark's schema
    <subset>/<id>/trace.json    # the session: pieces, decisions, tags, turns, barge-ins
    metrics.json                # official task metrics + audio roll-up
    meta.json                   # completion marker (resume-safe)
```

Override the audio location with ``--audio-root`` (or ``AUDIO_ROOT=`` for
``run_audio_all.sh``). Scoring needs `output.wav`, `output.json` and the task
annotation in one folder, so it builds that folder under ``/tmp/opencode/audio_score``
with the audio **symlinked** in and copies nothing but the annotations; the
parakeet transcript is written back into the repo as evidence.
