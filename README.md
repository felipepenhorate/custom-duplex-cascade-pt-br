# DuplexCascade pt-BR — SFT and Self-Distill versions

This repository contains **two versions** of the DuplexCascade pt-BR full-duplex
speech model (cascaded ASR–LLM–TTS, micro-turn turn-taking with special tags,
based on [DuplexCascade](https://github.com/sbintuitions/DuplexCascade),
arXiv:2603.09180):

| Folder | Version | Training | General-capability retention |
|---|---|---|---|
| [`sft/`](./sft/) | **DuplexCascade-PT** (original) | pure SFT — QLoRA fine-tune of Qwen3-4B-Instruct-2507 on duplex dialogue | regresses (function calling 55.6%, general 41.7% in eval) |
| [`distill/`](./distill/) | **DuplexCascade-Distill** (new) | **self-distillation** — SDFT-inspired (arXiv:2601.19897): frozen base generates content, QLoRA learns tags (weighted CE) + forward-KL to the base at content positions | **kept** (function calling 100%, general 100%) |

The whole point of the `distill/` version: learn the duplex micro-turn protocol
without the capability regression that the plain SFT fine-tune shows.

## Laya typed-decision companion (v7)

[`laya_approach/`](./laya_approach/) adds a 322M-parameter **Laya multilingual**
typed decision companion to the cascade. It chooses one of seven protocol tags
from the stream event and hands the floor to the same stock Qwen3-4B content
model; it never writes content itself. v7 introduces an explicit
`stream: text|audio` contract and a deferred-silence rule for audio, which fixes
the over-eager `user_finish_speaking` behavior.

The full training code, data pipeline, evaluation scripts and model card are in
[`laya_approach/`](./laya_approach/README.md). The released checkpoint is
[`lumierenoir/laya-duplex-cascade-pt-br`](https://huggingface.co/lumierenoir/laya-duplex-cascade-pt-br).

Averaged turn-taking accuracy (ATA) against the existing arms:

| | pt text | en text | pt audio | en audio |
|---|---:|---:|---:|---:|
| **Laya companion v7** | **1.000** | **1.000** | **0.849** | **0.862** |
| MTP-like (0.8B companion) | 0.973 | 0.984 | 0.520 | 0.515 |
| SFT | 0.971 | 0.986 | - | - |
| Distill | 0.955 | 0.924 | - | - |
| Original DuplexCascade | 0.896 | 0.951 | - | - |

Text results use the word chunker; the MTP-like text baseline used the training
chunker, so that comparison is cross-arm. Audio ATA is computed from the task
TORs with the same orientation and smooth-turn gate. Full tables:
[`laya_approach/ATA.md`](./laya_approach/ATA.md).

Commercial/permissive training recipe (MIT-only data, no CANDOR/ICC):
[`laya_approach/COMMERCIAL.md`](./laya_approach/COMMERCIAL.md).

## Onda Travel voice demo

[`laya_approach/demo/`](./laya_approach/demo/) is a runnable product example for
the Laya companion: a bilingual voice-first travel storefront. The browser
streams 24 kHz audio into faster-whisper; Laya decides when the turn ends or a
barge-in starts; Qwen3-4B answers and Phonon 40m multilingual speaks. The demo
handles travel filtering, small talk and general questions, switches the page
and TTS between English and pt-BR mid-conversation, and keeps the visible cards
in sync with the spoken answer.

```bash
# from the repository root, with the GPU environment active
./laya_approach/demo/run_demo.sh
# open http://localhost:7860

# no models, text-only UI
./laya_approach/demo/run_demo.sh --mock

# explicit environment
PYTHON=/path/to/venv/bin/python ./laya_approach/demo/run_demo.sh
```

The demo defaults to `models/laya-companion`, `models/qwen3-4b-instruct-2507`,
`models/phonon` and `models/phonon-voices`; download commands, model licenses,
flags, the demo script and troubleshooting are in
[`laya_approach/demo/README.md`](./laya_approach/demo/README.md).

## Which one do I want?

- **Distill (recommended)** — `distill/`: same duplex behavior, but keeps
  general QA / reasoning / function-calling at base-model level. Served by the
  real-time GUI and shipped as `DuplexCascade-Distill-q4_k_m.gguf`.
- **SFT** — `sft/`: the original fine-tune. Better turn-taking tag accuracy
  (98% vs 96.4%) but drops general capabilities. Useful as the SFT baseline.

Both share the same project layout and GUI (mic in / streaming ASR / LLM /
streaming TTS / audio out). Full details, method, eval tables and the data
pipelines live in each subfolder's `README.md` + `SPEC.md`:

- [`sft/`](./sft/) → `sft/README.md`, `sft/SPEC.md`
- [`distill/`](./distill/) → `distill/README.md`, `distill/SPEC.md`
  (incl. §6 R0 — how to regenerate everything with the v2 trigger)

## Article

The three approaches are written up together in **[`article/`](./article/)** —
one paper with a section per approach, a shared evaluation, the free-generation
results and the ablations:

```bash
cd article && pdflatex main.tex && pdflatex main.tex   # -> main.pdf
```

The per-approach write-ups ([`distill/article/main.tex`](./distill/article/main.tex),
[`mtp_like/article/mtp_duplex.tex`](./mtp_like/article/mtp_duplex.tex)) are kept as
records of each project; the combined article supersedes them as the single
citable document.

## Full-Duplex-Bench results

The three variants are evaluated against a pt-BR + English Full-Duplex-Bench
(FDB v1.0) harness in [`fdb_bench/`](./fdb_bench/) — same four dimensions as the
paper's Table 1 (pause handling, backchannel, smooth turn-taking, user
interruption), 727 scenarios per subset set, with the interruption ratings
judged by a local Qwen3.5-9B GGUF:

```bash
./fdb_bench/run_all.sh                 # all variants, pt-BR + en
LIMIT=3 ./fdb_bench/run_all.sh         # quick smoke
```

**Results are committed** under [`results/`](./results/) — per-model
`raw.jsonl` / `judged.jsonl` / `quality.jsonl` plus `summary.md` and the
cross-language [`comparison.md`](./results/comparison.md) — so every number below
is reproducible from the repository. See:

- [`fdb_bench/FINDINGS.md`](./fdb_bench/FINDINGS.md) — results, the mechanism
  behind them, and the six bugs found and fixed along the way
- [`fdb_bench/README.md`](./fdb_bench/README.md) — what the harness does and does
  not reproduce
- [`fdb_bench/REFERENCE.md`](./fdb_bench/REFERENCE.md) — the published numbers it
  compares against
- [`distill/article/main.tex`](./distill/article/main.tex) §V-E / §VI-C — the
  experiments, results and findings written up in the article

Headline (pt-BR, 727 scenarios per model, ATA ↑):

| Model | ATA | Pause↓ syn | Pause↓ cand | Smooth↑ | Intr↑ | Relevance↑ | Quality↑ |
|---|---|---|---|---|---|---|---|
| MTP-like companion | **0.973** | 0.073 | **0.000** | 0.975 | 0.990 | 4.36 | **4.38** |
| SFT (original recipe) | **0.971** | 0.080 | 0.032 | **1.000** | **0.995** | 3.56 | 4.19 |
| Distill | 0.955 | 0.095 | 0.065 | 0.983 | 0.995 | 1.59 | 2.79 |
| *Original DuplexCascade (upstream, measured here)* | *0.896* | *0.102* | *0.273* | *0.975* | *0.985* | *3.92* | *4.01* |
| Base (control) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | – | 2.81 |
| *Published DuplexCascade (EN audio)* | *0.858* | *0.058* | *0.222* | *0.832* | *0.955* | *4.02* | *–* |

All three trained variants beat the published ATA; the upstream release,
re-measured with the same harness, lands on its own published row (pause
0.102/0.273 vs 0.058/0.222, ATA 0.896 vs 0.858). The one dimension where the
upstream model wins is backchannel (TOR 0.436 vs 1.000) — a **data gap**, not a
capability gap: `<|system backchannel|>` occurs **zero** times in this project's
training data. See [`fdb_bench/FINDINGS.md`](./fdb_bench/FINDINGS.md).

## Running the GUI

Both versions use the same four-component stack (llama.cpp + faster-whisper +
pocket-tts + bridge) via each folder's `run_services.sh`:

```bash
cd distill && ./run_services.sh   # or: cd sft && ./run_services.sh
# open http://localhost:31606
```

Model runs (adapters / `merged_bf16` / GGUF) are large and live on the HDD at
`/mnt/f/duplex_cascade_runs/` (see each README); they are **not** committed.
Point `GGUF=/path/to/model.gguf` to switch models.

## Notes

- Trained models, `*.gguf`, `*.safetensors`, generated datasets (`**/data/*.jsonl`,
  `**/data/prepped_*/`) and logs are git-ignored — they are rebuilt by each
  folder's data/training pipeline (see the SPECs).
- `sft/` = the original project moved verbatim; `distill/` = the new
  self-distillation project.

## Known issues

### Distill: reply opens by continuing the user's utterance (mitigated 2026-08)

The distill model occasionally opens its reply with the **continuation of the
user's own sentence** instead of a fresh answer — e.g. user: *"Qual é a capital
do Brasil?"* → assistant: *", a capital do Brasil é Brasília."* (leading comma
+ echo of the user's words).

**Why:** the teacher (frozen base) sometimes autocompleted the user turn when
generating the assistant content under the v1 *"Responda de forma curta"*
trigger, and the content forward-KL anchors that mode into the trained model.
The v2 trigger at inference reduces it but does not remove it.

**Current mitigation:** the bridge strips leading whitespace + continuation
punctuation from every micro-turn before TTS/transcript/history
(`_strip_leading_continuation` in `distill/DuplexCascade/server.py`).

**Planned fix (future):** regenerate the teacher training data with the **v2
trigger** ("responda de forma natural e completa … que responda de fato ao que o
usuário perguntou … não se repita") and retrain — see `distill/SPEC.md` §6 R0 —
so the continuation artifact is gone from the model natively instead of being
stripped at inference.

## Citation

If you use this work, please cite the accompanying article
([`distill/article/main.tex`](./distill/article/main.pdf)):

```bibtex
@article{fonseca2026selfdistillation,
  title   = {Self Distillation using only supervised fine tuning: a duplex cascade case study},
  author  = {Fonseca, Felipe Penhorate Carvalho da},
  year    = {2026}
}
```