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