# Full-Duplex-Bench on real audio: pt-BR (in-distribution) vs English (out-of-distribution)

The three pt-BR variants are fine-tuned on pt-BR duplex data, so the Portuguese set is their **in-distribution** set and the English recording is **out-of-distribution**. Both rows are measured by the same harness: the same silence gate, the same rendering, the same level normalisation before recognition, and the benchmark's own `eval_*.py`. They differ in the language of the set and in the recogniser that reads the assistant's replies (whisper large-v3 `--language pt` vs the benchmark's English-only parakeet).

The Portuguese set is a re-voicing: the benchmark's own audio is segmented, translated, spoken by pocket-tts and put back on the timeline with the gaps preserved and the annotations re-timed (`build_pt_audio.py`; `check_pt_structure.py` is the gate that compares the rebuilt audio with the recording it was made from).

## Portuguese (in-distribution for the three trained variants)

| model | pause_syn ↓ | pause_cand ↓ | BC TOR ↓ | smooth ↑ | smooth lat ↓ | intr ↑ | relevance ↑ | intr lat ↓ | **ATA** ↑ |
|---|---|---|---|---|---|---|---|---|---|

## English (out-of-distribution for the three trained variants)

| model | pause_syn ↓ | pause_cand ↓ | BC TOR ↓ | smooth ↑ | smooth lat ↓ | intr ↑ | relevance ↑ | intr lat ↓ | **ATA** ↑ |
|---|---|---|---|---|---|---|---|---|---|
| Original DuplexCascade (upstream) | 0.088 | 0.190 | 0.182 | 0.630 | 1.48 | 0.945 | 3.67 | 1.46 | **0.824** |
| SFT (pt-BR) | 0.825 | 0.713 | 0.673 | 0.866 | 0.33 | 0.930 | 3.11 | 0.65 | **0.564** |
| Self-distillation (pt-BR) | 0.226 | 0.269 | 0.345 | 0.706 | 3.82 | 0.605 | 1.49 | 2.70 | **0.704** |
| Companion (MTP-style) | 0.920 | 0.829 | 0.782 | 0.891 | 0.61 | 0.935 | 4.40 | 1.05 | **0.519** |
