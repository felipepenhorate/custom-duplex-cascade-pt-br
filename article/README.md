# Combined article

**One write-up for the three approaches** in this repository:

| section | covers | source project |
|---|---|---|
| IV | **Approach A** — supervised fine-tuning (the original DuplexCascade recipe) | [`sft/`](../sft) |
| V | **Approach B** — self-distillation by token position (SDFT-style) | [`distill/`](../distill) |
| VI | **Approach C** — an MTP-style protocol companion | [`mtp_like/`](../mtp_like) |
| VII–VIII | shared evaluation and results, including the free-generation Full-Duplex-Bench run | [`fdb_bench/`](../fdb_bench) |
| IX | ablations | [`fdb_bench/run_ablations.sh`](../fdb_bench/run_ablations.sh) |

## Build

```bash
cd article
pdflatex main.tex && pdflatex main.tex     # twice, for the cross-references
```

Produces `main.pdf` (10 pages, no overfull boxes, 18 references).

## Relationship to the per-approach articles

`distill/article/main.tex` and `mtp_like/article/mtp_duplex.tex` are the original
per-approach write-ups and are kept as records of each project. **This directory
supersedes them as the single citable document**: it contains their methods and
results, restructured into one narrative, plus the shared evaluation, the
cross-approach comparison and the ablations that neither had.

Numbers here are the committed ones under [`../results/`](../results) and
[`../results_ablation/`](../results_ablation); the harness and every caveat are
documented in [`../fdb_bench/README.md`](../fdb_bench/README.md) and
[`../fdb_bench/FINDINGS.md`](../fdb_bench/FINDINGS.md).

## Caveat on the published comparison

The published DuplexCascade row is English audio from the original paper. Our rows
are pt-BR (and English text) evaluated with a text-domain harness: durations come
from a calibrated speaking rate, latency is inference compute, and the interruption
judge is a local 9B model rather than GPT-4o. The upstream model was re-measured
with the same harness and reproduces its own published numbers (ATA 0.896 vs.
0.858), which is what makes the comparison meaningful; the published row itself is
a range check, not a baseline.
