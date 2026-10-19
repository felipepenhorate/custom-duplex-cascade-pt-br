# Audio Full-Duplex-Bench: real FDB v1.0 audio

Produced by `fdb_bench/audio/` (design and caveats in `fdb_bench/audio/README.md`). Every number comes from the benchmark's own `eval_*.py` reading `output.wav`/`output.json`, where `output.json` is produced by `nvidia/parakeet-tdt-0.6b-v2` exactly as upstream's `get_transcript/asr.py` does. 727 scenarios per model.

## Official task metrics

## Take-over rates, everything tested

The same five columns the benchmark reports TOR for. Arrow = the good direction; pause and backchannel rate a model that speaks too eagerly, smooth turn-taking and interruption rate a model that answers.

| run | model | pause_synth ↓ | pause_candor ↓ | bc_tor ↓ | smooth_tor ↑ | intr_tor ↑ | ATA ↑ (ours) | ATA (paper conv.) |
|---|---|---|---|---|---|---|---|---|
| *published* | *DuplexCascade (EN audio, GPT-4o)* | *0.058* | *0.222* | *0.218* | *0.832* | *0.955* | *0.858* | *0.858* (0.877 ours)* |
| audio (real WAVs, official evaluators) | Distill (pt-BR) | 0.088 | 0.060 | 0.109 | 0.160 | 0.360 | 0.160 | 0.653 |
| audio (real WAVs, official evaluators) | Original (upstream, EN) | 0.241 | 0.310 | 0.491 | 0.647 | 0.410 | 0.626 | 0.603 |
| audio (real WAVs, official evaluators) | SFT (pt-BR) | 0.613 | 0.486 | 0.836 | 0.555 | 0.605 | 0.515 | 0.445 |
| audio (in progress, audio-truth reading) | MTP companion (pt-BR) (353/727 scenarios) | 0.942 | 0.912 | — | — | — | — | — |
| text (synthetic stimuli, en) | Base (control) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | — |
| text (synthetic stimuli, en) | Distill (pt-BR) | 0.095 | 0.065 | 1.000 | 0.941 | 0.915 | 0.924 | — |
| text (synthetic stimuli, en) | MTP companion (pt-BR) | 0.066 | 0.000 | 1.000 | 1.000 | 1.000 | 0.984 | — |
| text (synthetic stimuli, en) | Original (upstream, EN) | 0.095 | 0.102 | 0.345 | 1.000 | 1.000 | 0.951 | — |
| text (synthetic stimuli, en) | SFT (pt-BR) | 0.051 | 0.000 | 1.000 | 1.000 | 0.995 | 0.986 | — |

Reading the two runs side by side: the text bench feeds clean transcripts and knows where turns end, so it flatters every model; the audio bench feeds ASR output and a real silence gate, which is where the over-eager pt-BR models separate from the upstream one.

## Why the pause columns move between the two runs

Per-scenario TOR is a compound of the model's willingness to speak and the number of invitations the harness creates. The invitation-level rate separates them: in the text bench an invitation is the annotated turn end of a clean transcript, in the audio run it is silero firing on a real conversation gap.

| model | text: P(floor \| invitation) | audio: P(floor \| turn end) | audio: P(floor \| silence tick) | invitations/scenario |
|---|---|---|---|---|
| Distill (pt-BR) | 0.081 | 0.393 | 0.000 | 3.9 |
| Original (upstream, EN) | 0.119 | 0.125 | 0.234 | 3.9 |
| SFT (pt-BR) | 0.086 | 0.507 | 0.000 | 3.9 |

The upstream English model barely moves across the two harnesses; the pt-BR models move four- to five-fold. That is the cross-lingual effect the audio run exists to expose: on fragmentary English audio the pt-BR models cannot tell an unfinished turn from a finished one, so they take the floor at ~40% of every invitation where the English model takes 7.9%.

| model | pause_synth ↓ | pause_candor ↓ | bc_tor ↓ | bc_freq ↑ | bc_jsd ↓ | smooth_tor ↑ | smooth_lat ↓ | intr_tor ↑ | intr_rel ↑ | intr_lat ↓ |
|---|---|---|---|---|---|---|---|---|---|---|
| *published DuplexCascade (EN audio, GPT-4o)* | *0.058* | *0.222* | *0.218* | *0.009* | *0.949* | *0.832* | *1.724* | *0.955* | *4.016* | *1.225* |
| Distill (pt-BR) | 0.088 | 0.060 | 0.109 | 0.031 | 0.906 | 0.160 | 2.754 | 0.360 | 0.236 | 4.958 |
| Original (upstream, EN) | 0.241 | 0.310 | 0.491 | 0.018 | 0.905 | 0.647 | 1.379 | 0.410 | 0.427 | 2.335 |
| SFT (pt-BR) | 0.613 | 0.486 | 0.836 | 0.117 | 0.705 | 0.555 | 0.264 | 0.605 | 0.455 | 1.418 |

## Where the pipeline's time goes, and when a reply arrives too late

| model | mean LLM s | mean pocket-tts s | replies audible only past the end of the input |
|---|---|---|---|
| Distill (pt-BR) | 8.47 | 5.66 | 0.148 |
| Original (upstream, EN) | 3.73 | 5.35 | 0.029 |
| SFT (pt-BR) | 1.59 | 4.62 | 0.028 |

The last column is the pipeline's own latency made visible: a reply that begins after the input ends is audible only because the render carries a 4 s tail. FDB's own outputs are exactly input-length, so those replies would be scored as silence there — the trace keeps both readings (`spoke` vs `audible_within_input`).
