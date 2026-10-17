# Reference numbers: what we compare against

Two different baselines are used, and they mean different things. Keep them
apart.

## 1. The published DuplexCascade model (the "original")

Table 1 of the DuplexCascade paper (arXiv:2603.09180), reproduced verbatim in
the vendored `sft/DuplexCascade/assets/full-duplex-bench.png`:

| Model | Pause TOR↓ synth | Pause TOR↓ candor | BC TOR↓ | BC Freq↑ | BC JSD↓ | Smooth TOR↑ | Smooth lat↓ | Intr TOR↑ | Intr GPT-4o↑ | Intr lat↓ | ATA↑ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **DuplexCascade** | **0.058** | **0.222** | **0.218** | 0.009 | 0.949 | 0.832 | 1.724 | 0.955 | **4.016** | 1.225 | **0.858** |
| DuplexCascade-β | 0.343 | 0.458 | 0.309 | 0.034 | 0.811 | 0.899 | 0.567 | 0.950 | 4.011 | 0.850 | 0.748 |

For context, other models from the same table: dGSLM (0.466 ATA), Moshi
(0.395), Freeze-Omni (0.489), PersonaPlex (0.759), MiniCPM-Duplex (0.598),
Gemini Live (0.778).

### These are not an apples-to-apples baseline

The published numbers are **English, audio-domain**, produced by a real speech
model driven through the full cascaded ASR–LLM–TTS stack with real audio
timestamps. Our rows are **text-domain** numbers from a simulated clock. The
gaps that matter:

| | published | ours |
|---|---|---|
| language | English | pt-BR (in-distribution) + en (cross-lingual) |
| backbone | Qwen2-7B-Instruct | Qwen3-4B-Instruct-2507 |
| input | real audio through streaming ASR | timed text events |
| durations | measured from audio | word-rate model (3.25 words/s, calibrated vs `pocket_tts`) |
| latency | acoustic end-to-end | inference compute time |
| judge | GPT-4o | Qwen3.5-9B (local) |
| ATA | paper's definition | ours (see below) |

So: use the published row to see **where the metric ranges sit** and whether our
harness produces numbers in a plausible regime. Use **our own rows against each
other** for any actual conclusion — same scenarios, same clock, same simulator,
same judge, same decoding.

The one genuinely comparable number is the **direction**: the published
DuplexCascade is excellent on pause handling (0.058 / 0.222) and interruption
(0.955) but only 0.832 on smooth turn-taking and weak on backchannel
(0.218 TOR, 0.009 Freq) — backchannel is a known hard spot for this
architecture, and it is the dimension where we expect the variants to differ
most.

### A note on `ATA`

FDB's released code does not implement "Averaged Turn-Taking Accuracy" — the
paper defines no such function. Both the published ATA and ours are therefore
conventions, not a shared estimator. Ours averages the four headline TORs
oriented so higher is better, with two guards (see
`metrics.averaged_turn_taking_accuracy`): an unmeasured interruption scores 0
rather than being dropped, and pause credit is only granted when the model
demonstrably takes the floor elsewhere — otherwise a model that never speaks
would score 0.667 while being useless.

## 2. The upstream checkpoint (downloadable, gated)

`https://huggingface.co/sbintuitions/DuplexCascade` — English, base
`Qwen/Qwen2-7B-Instruct`, shipped as a raw `model_state.safetensors`
(17.4 GB). Gated, so the weights 403 until the license is accepted:

```bash
# accept at https://huggingface.co/sbintuitions/DuplexCascade first
hf download sbintuitions/DuplexCascade model_state.safetensors tokenizer \
  --local-dir /mnt/f/duplex_cascade_runs/original_dc
MODELS="original_dc" ./fdb_bench/run_all.sh
```

`fdb_bench/original_dc.py` then reuses the vendored loader
(`sft/DuplexCascade/model.py`) and wraps it in `HFDuplexAdapter`, which is the
in-process twin of the llama.cpp adapter: identical prompt construction,
micro-turn loop and **greedy** decoding. That keeps the comparison from being
confounded by the serving backend or by sampling.

Running it gives a fourth, *directly measured* row — the only way to get a
true in-harness number for the original architecture. Expect it to be a hard
test for our protocol harness too: the upstream model is English, our adapter
drives it with the English system prompt, and its turn-taking mechanics should
match the paper's row closely if the harness is faithful.

## 3. FDB v1.0 baselines

From the official FDB README, for reading the metric columns:

| Model | Pause TOR↓ synth | Pause TOR↓ candor | BC TOR↓ | BC Freq↑ | BC JSD↓ | Smooth TOR↑ | Smooth lat↓ | Intr TOR↑ | Intr GPT-4o↑ | Intr lat↓ |
|---|---|---|---|---|---|---|---|---|---|---|
| Moshi | 0.985 | 0.980 | 1.000 | 0.001 | 0.957 | 0.941 | 0.265 | 1.000 | 0.765 | 0.257 |
| dGSLM | 0.934 | 0.935 | 0.691 | 0.015 | 0.934 | 0.975 | 0.352 | 0.917 | 0.201 | 2.531 |
| GPT-Realtime | 0.010 | 0.120 | 0.000 | 0.007 | 0.980 | 1.000 | 1.470 | 0.970 | 3.850 | 1.500 |

These show the trade-off the benchmark is built around: a model that is very
good at staying quiet (GPT-Realtime, pause TOR 0.010) also backchannels almost
never (Freq 0.007), while a model that never takes a turn (Moshi on
backchannel, TOR 1.000) fails the opposite way.
