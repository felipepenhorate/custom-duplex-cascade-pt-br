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
                               └─ parakeet-tdt-0.6b-v2 ──────────► output.json
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
| `output.json` | n/a | **parakeet-tdt-0.6b-v2** (upstream's own recipe) |
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

Outputs per model (`results_audio/<mode>/<model>/`):

```
<subset>/<id>/output.wav    # what the benchmark's evaluators read
<subset>/<id>/output.json   # parakeet transcript in the benchmark's schema
<subset>/<id>/trace.json    # the session: pieces, decisions, tags, turns, barge-ins
<subset>/<id>/<annotation>.json   # copied in: the evaluators read it beside the audio
metrics.json                # official task metrics + audio roll-up
meta.json                   # completion marker (resume-safe)
```
