# pt-BR Full-Duplex-Bench: real audio, two scorers

`parakeet` is the benchmark's own recipe (`get_transcript/asr.py`) and is **English-only**; `whisper-pt` re-scores the identical audio and annotations with `faster-whisper large-v3 --language pt`. The models answer in Portuguese for 84-98% of turns on this set, so only the whisper rows are interpretable -- the parakeet rows are kept to show what the upstream recipe reports when it cannot read the replies.

| model | scorer | pause_syn ↓ | pause_cand ↓ | BC TOR ↓ | smooth ↑ | smooth lat ↓ | intr ↑ | relevance ↑ | intr lat ↓ | **ATA** ↑ |
|---|---|---|---|---|---|---|---|---|---|---|
| Original DuplexCascade (upstream, EN) | parakeet | 0.241 | 0.310 | 0.491 | 0.647 | 1.38 | 0.410 | 0.43 | 2.33 | **0.626** |
| Original DuplexCascade (upstream, EN) | whisper-pt | 0.343 | 0.481 | 0.509 | 0.773 | 0.86 | 0.860 | 1.96 | 0.69 | **0.702** |
| SFT (pt-BR) | parakeet | 0.613 | 0.486 | 0.836 | 0.555 | 0.26 | 0.605 | 0.45 | 1.42 | **0.515** |
| SFT (pt-BR) | whisper-pt | 0.883 | 0.880 | 0.855 | 0.874 | 0.14 | 0.880 | 2.93 | 0.52 | **0.498** |
| Self-distillation (pt-BR) | parakeet | 0.088 | 0.060 | 0.109 | 0.160 | 2.75 | 0.360 | 0.24 | 4.96 | **0.160** |
| Self-distillation (pt-BR) | whisper-pt | 0.146 | 0.102 | 0.200 | 0.345 | 2.26 | 0.555 | 1.62 | 3.55 | **0.345** |
| Companion (MTP-style) | parakeet | 0.445 | 0.440 | 0.782 | 0.546 | 0.91 | 0.610 | 0.48 | 2.38 | **0.568** |
| Companion (MTP-style) | whisper-pt | 0.869 | 0.838 | 0.873 | 0.798 | 0.44 | 0.840 | 3.17 | 1.15 | **0.483** |
| Self-distillation + deployed prompt (prime) | whisper-pt | 0.628 | 0.463 | 0.436 | 0.445 | 0.32 | 0.780 | 2.01 | 0.50 | **0.445** |
| Self-distillation + training prompt | whisper-pt | 0.102 | 0.056 | 0.036 | 0.017 | 0.25 | 0.100 | 2.70 | 0.99 | **0.017** |

## English reference (parakeet is the right scorer there)

| model | pause_syn ↓ | pause_cand ↓ | BC TOR ↓ | smooth ↑ | smooth lat ↓ | intr ↑ | relevance ↑ | intr lat ↓ | **ATA** ↑ |
|---|---|---|---|---|---|---|---|---|---|
| Original DuplexCascade (upstream, EN) | 0.088 | 0.190 | 0.182 | 0.630 | 1.48 | 0.945 | 3.67 | 1.46 | **0.824** |
| SFT (pt-BR) | 0.825 | 0.713 | 0.673 | 0.866 | 0.33 | 0.930 | 3.11 | 0.65 | **0.564** |
| Self-distillation (pt-BR) | 0.226 | 0.269 | 0.345 | 0.706 | 3.82 | 0.605 | 1.49 | 2.70 | **0.704** |
| Companion (MTP-style) | 0.920 | 0.829 | 0.782 | 0.891 | 0.61 | 0.935 | 4.40 | 1.05 | **0.519** |
