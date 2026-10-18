# Audio Full-Duplex-Bench: real FDB v1.0 audio

Produced by `fdb_bench/audio/` (design and caveats in `fdb_bench/audio/README.md`). Every number comes from the benchmark's own `eval_*.py` reading `output.wav`/`output.json`, where `output.json` is produced by `nvidia/parakeet-tdt-0.6b-v2` exactly as upstream's `get_transcript/asr.py` does. 727 scenarios per model.

## Official task metrics

| model | pause_synth ↓ | pause_candor ↓ | bc_tor ↓ | bc_freq ↑ | bc_jsd ↓ | smooth_tor ↑ | smooth_lat ↓ | intr_tor ↑ | intr_rel ↑ | intr_lat ↓ |
|---|---|---|---|---|---|---|---|---|---|---|
| *published DuplexCascade (EN audio, GPT-4o)* | *0.058* | *0.222* | *0.218* | *0.009* | *0.949* | *0.832* | *1.724* | *0.955* | *4.016* | *1.225* |
| Distill (pt-BR) | 0.226 | 0.269 | 0.345 | 0.041 | 0.874 | 0.706 | 3.817 | 0.605 | 1.488 | 2.700 |
| Original (upstream, EN) | 0.088 | 0.190 | 0.182 | 0.007 | 0.964 | 0.630 | 1.483 | 0.945 | 3.667 | 1.457 |
| SFT (pt-BR) | 0.825 | 0.713 | 0.673 | 0.116 | 0.713 | 0.866 | 0.329 | 0.930 | 3.108 | 0.654 |

## Where the pipeline's time goes, and when a reply arrives too late

| model | mean LLM s | mean pocket-tts s | replies audible only past the end of the input |
|---|---|---|---|
| Distill (pt-BR) | 8.89 | 5.84 | 0.168 |
| Original (upstream, EN) | 3.42 | 4.42 | 0.028 |
| SFT (pt-BR) | 0.98 | 3.39 | 0.060 |

The last column is the pipeline's own latency made visible: a reply that begins after the input ends is audible only because the render carries a 4 s tail. FDB's own outputs are exactly input-length, so those replies would be scored as silence there — the trace keeps both readings (`spoke` vs `audible_within_input`).
