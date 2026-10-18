# Audio Full-Duplex-Bench: real FDB v1.0 audio

Produced by `fdb_bench/audio/` (design and caveats in `fdb_bench/audio/README.md`). Every number comes from the benchmark's own `eval_*.py` reading `output.wav`/`output.json`, where `output.json` is produced by `nvidia/parakeet-tdt-0.6b-v2` exactly as upstream's `get_transcript/asr.py` does. 727 scenarios per model.

## Official task metrics

## Take-over rates, everything tested

The same five columns the benchmark reports TOR for. Arrow = the good direction; pause and backchannel rate a model that speaks too eagerly, smooth turn-taking and interruption rate a model that answers.

| run | model | pause_synth ↓ | pause_candor ↓ | bc_tor ↓ | smooth_tor ↑ | intr_tor ↑ | ATA ↑ (ours) | ATA (paper conv.) |
|---|---|---|---|---|---|---|---|---|
| *published* | *DuplexCascade (EN audio, GPT-4o)* | *0.058* | *0.222* | *0.218* | *0.832* | *0.955* | *0.858* | *0.858* (0.877 ours)* |
| audio (real WAVs, official evaluators) | Distill (pt-BR) | 0.226 | 0.269 | 0.345 | 0.706 | 0.605 | 0.704 | 0.694 |
| audio (real WAVs, official evaluators) | MTP companion (pt-BR) | 0.920 | 0.829 | 0.782 | 0.891 | 0.935 | 0.519 | 0.459 |
| audio (real WAVs, official evaluators) | Original (upstream, EN) | 0.088 | 0.190 | 0.182 | 0.630 | 0.945 | 0.824 | 0.823 |
| audio (real WAVs, official evaluators) | SFT (pt-BR) | 0.825 | 0.713 | 0.673 | 0.866 | 0.930 | 0.564 | 0.517 |
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
| Distill (pt-BR) | 0.081 | 0.294 | 0.000 | 3.7 |
| MTP companion (pt-BR) | 0.086 | 0.441 | 0.001 | 3.7 |
| Original (upstream, EN) | 0.119 | 0.079 | 0.278 | 3.7 |
| SFT (pt-BR) | 0.086 | 0.397 | 0.000 | 3.7 |

The upstream English model barely moves across the two harnesses; the pt-BR models move four- to five-fold. That is the cross-lingual effect the audio run exists to expose: on fragmentary English audio the pt-BR models cannot tell an unfinished turn from a finished one, so they take the floor at ~40% of every invitation where the English model takes 7.9%.

| model | pause_synth ↓ | pause_candor ↓ | bc_tor ↓ | bc_freq ↑ | bc_jsd ↓ | smooth_tor ↑ | smooth_lat ↓ | intr_tor ↑ | intr_rel ↑ | intr_lat ↓ |
|---|---|---|---|---|---|---|---|---|---|---|
| *published DuplexCascade (EN audio, GPT-4o)* | *0.058* | *0.222* | *0.218* | *0.009* | *0.949* | *0.832* | *1.724* | *0.955* | *4.016* | *1.225* |
| Distill (pt-BR) | 0.226 | 0.269 | 0.345 | 0.041 | 0.874 | 0.706 | 3.817 | 0.605 | 1.488 | 2.700 |
| MTP companion (pt-BR) | 0.920 | 0.829 | 0.782 | 0.111 | 0.722 | 0.891 | 0.612 | 0.935 | 4.396 | 1.048 |
| Original (upstream, EN) | 0.088 | 0.190 | 0.182 | 0.007 | 0.964 | 0.630 | 1.483 | 0.945 | 3.667 | 1.457 |
| SFT (pt-BR) | 0.825 | 0.713 | 0.673 | 0.116 | 0.713 | 0.866 | 0.329 | 0.930 | 3.108 | 0.654 |

## Where the pipeline's time goes, and when a reply arrives too late

| model | mean LLM s | mean pocket-tts s | replies audible only past the end of the input |
|---|---|---|---|
| Distill (pt-BR) | 8.89 | 5.84 | 0.168 |
| MTP companion (pt-BR) | 2.68 | 6.75 | 0.062 |
| Original (upstream, EN) | 3.42 | 4.42 | 0.028 |
| SFT (pt-BR) | 0.98 | 3.39 | 0.060 |

The last column is the pipeline's own latency made visible: a reply that begins after the input ends is audible only because the render carries a 4 s tail. FDB's own outputs are exactly input-length, so those replies would be scored as silence there — the trace keeps both readings (`spoke` vs `audible_within_input`).
