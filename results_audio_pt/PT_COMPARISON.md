# Full-Duplex-Bench on real audio: pt-BR (in-distribution) vs English (out-of-distribution)

The three pt-BR variants are fine-tuned on pt-BR duplex data, so the Portuguese set is their **in-distribution** set and the English recording is **out-of-distribution**. Both rows are measured by the same harness: the same silence gate, the same rendering, the same level normalisation before recognition, and the benchmark's own `eval_*.py`. They differ in the language of the set and in the recogniser that reads the assistant's replies (whisper large-v3 `--language pt` vs the benchmark's English-only parakeet).

Two ATA columns: the project's convention with its two guards, and the plain four-term mean (`raw`), which is what the published row is. The reply-language columns are the local Qwen3.5-9B judging each assistant turn `pt`, `en`, or another language: the share of turns in Portuguese and the share of scenarios whose answers are, then the languages that showed up in the rest.
The Portuguese set is a re-voicing: the benchmark's own audio is segmented, translated, spoken by pocket-tts and put back on the timeline with the gaps preserved and the annotations re-timed (`build_pt_audio.py`; `check_pt_structure.py` is the gate that compares the rebuilt audio with the recording it was made from).

## Portuguese (in-distribution for the three trained variants)

| model | pause_syn ↓ | pause_cand ↓ | BC TOR ↓ | smooth ↑ | smooth lat ↓ | intr ↑ | relevance ↑ | intr lat ↓ | **ATA** ↑ | ATA raw | reply in pt | other langs | non-latin drift |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Original DuplexCascade (upstream) | 0.234 | 0.315 | 0.182 | 0.639 | 1.31 | 0.870 | 1.94 | 0.97 | **0.740** | 0.740 | | 83.5% / 83.5% | xx:5, es:3, zh:1 | 0.7% / 0.1%|
| SFT (pt-BR) | 0.883 | 0.843 | 0.909 | 0.849 | 0.26 | 0.855 | 3.10 | 0.51 | **0.494** | 0.494 | | 98.6% / 98.6% | - | 0.0% / 0.0%|
| Self-distillation (pt-BR) | 0.212 | 0.157 | 0.364 | 0.378 | 2.36 | 0.590 | 1.56 | 3.85 | **0.378** | 0.650 | | 79.0% / 77.3% | ko:149 | 19.0% / 18.8%|
| Self-distillation, production configuration (deployed prompt, decision mode) | 0.380 | 0.148 | 0.145 | 0.143 | 0.22 | 0.585 | 3.36 | 0.48 | **0.143** | 0.550 | | 81.4% / 73.1% | ko:13, vi:11, pl:3 | 7.6% / 4.8%|

## English (out-of-distribution for the three trained variants)

| model | pause_syn ↓ | pause_cand ↓ | BC TOR ↓ | smooth ↑ | smooth lat ↓ | intr ↑ | relevance ↑ | intr lat ↓ | **ATA** ↑ | ATA raw | reply in pt | other langs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Original DuplexCascade (upstream) | 0.088 | 0.190 | 0.182 | 0.630 | 1.48 | 0.945 | 3.67 | 1.46 | **0.824** | 0.824 | | 0.0% / 0.0% | - | 1.0% / 0.0%|
| SFT (pt-BR) | 0.825 | 0.713 | 0.673 | 0.866 | 0.33 | 0.930 | 3.11 | 0.65 | **0.564** | 0.564 | | 25.6% / 22.3% | - | 0.1% / 0.0%|
| Self-distillation (pt-BR) | 0.226 | 0.269 | 0.345 | 0.706 | 3.82 | 0.605 | 1.49 | 2.70 | **0.704** | 0.704 | | 0.0% / 0.0% | ko:296 | 61.1% / 60.6%|
| Companion (MTP-style) | 0.920 | 0.829 | 0.782 | 0.891 | 0.61 | 0.935 | 4.40 | 1.05 | **0.519** | 0.519 | | 0.0% / 0.0% | - | 0.0% / 0.0%|

## The distilled model's degeneration, and the prompt that fixes it

The language judge labels a reply by the language it judges the turn to be in, and it reads a turn from its first word. The distilled model opens many replies with a stray non-Latin token ("아! You're saying you know *a lot* of people ..."), so the judge calls those turns Korean even though the body of the reply is in the model's own language. The `main language` column is the same judgement on the same reply with that leading token removed: it is what the model was actually saying.

| run | reply: pt / en / other | main language: pt / en / other | replies opening with a stray token | words per turn | replies | relevance | floor latency | ATA |
|---|---|---|---|---|---|---|---|---|---|
| PT, no prompt | 79.0% / 9.7% / 11.3% | 88.9% / 8.3% / 2.7% | 18.8% | 34.4 | 1.56 | 2.36 s | 0.378 (raw 0.650) |
| PT, production prompt | 81.4% / 12.6% / 5.9% | 82.9% / 15.2% / 1.9% | 4.8% | 14.7 | 3.36 | 0.22 s | 0.143 (raw 0.550) |
| EN, no prompt | 0.0% / 72.0% / 28.0% | 0.2% / 97.1% / 2.7% | 60.6% | 38.7 | 1.49 | 3.82 s | 0.704 (raw 0.704) |

## The difference the rebuild was about

PT is the in-distribution set for the trained variants, so a negative number means the model did better on its own language. The upstream model is English-trained, so for it the sign is expected to point the other way.

| model | ATA EN (OOD) | ATA PT (ID) | Δ (PT−EN) |
|---|---|---|---|
| Original DuplexCascade (upstream) | 0.824 | 0.740 | -0.084 |
| SFT (pt-BR) | 0.564 | 0.494 | -0.070 |
| Self-distillation (pt-BR) | 0.704 | 0.378 | -0.326 |
