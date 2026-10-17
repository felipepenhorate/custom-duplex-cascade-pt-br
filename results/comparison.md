# DuplexCascade pt-BR — Full-Duplex-Bench comparison

Text-domain FDB v1.0 harness. See `fdb_bench/README.md` for what is
and is not faithful, and `fdb_bench/REFERENCE.md` for the published
numbers. Within a table, our rows are directly comparable to each
other; the paper row is a range check, not a baseline.


## Headline

**en** — best ATA: SFT (original recipe, continue) [none] (0.986); Base (untrained Qwen3-4B control) [none]: ATA 0.000, pause 0.000/0.000, smooth 0.000, intr 0.000, rating -; Distill (self-distillation) [none]: ATA 0.924, pause 0.095/0.065, smooth 0.941, intr 0.915, rating 2.115; MTP-like (0.8B policy companion + stock Qwen3-4B) [instruction]: ATA 0.984, pause 0.066/0.000, smooth 1.000, intr 1.000, rating 4.875; Original DuplexCascade (sbintuitions, q4_k_m) [none]: ATA 0.951, pause 0.095/0.102, smooth 1.000, intr 1.000, rating 4.345; SFT (original recipe, continue) [none]: ATA 0.986, pause 0.051/0.000, smooth 1.000, intr 0.995, rating 4.080

**pt** — best ATA: MTP-like (0.8B policy companion + stock Qwen3-4B) [instruction] (0.973); Base (untrained Qwen3-4B control) [none]: ATA 0.000, pause 0.000/0.000, smooth 0.000, intr 0.000, rating -; Distill (self-distillation) [none]: ATA 0.955, pause 0.095/0.065, smooth 0.983, intr 0.995, rating 1.588; MTP-like (0.8B policy companion + stock Qwen3-4B) [instruction]: ATA 0.973, pause 0.073/0.000, smooth 0.975, intr 0.990, rating 4.359; Original DuplexCascade (sbintuitions, q4_k_m) [none]: ATA 0.896, pause 0.102/0.273, smooth 0.975, intr 0.985, rating 3.924; SFT (original recipe, continue) [none]: ATA 0.971, pause 0.080/0.032, smooth 1.000, intr 0.995, rating 3.558

## All runs

| Model | Lang | Pause TOR↓ synth | Pause TOR↓ candor | BC TOR↓ | BC Freq↑ | BC JSD↓ | Smooth TOR↑ | Floor lat↓ | Intr TOR↑ | Intr rating↑ | Stop lat↓ | Resp lat↓ | ATA↑ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Base (untrained Qwen3-4B control) [none] | en | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | - | 0.000 | - | 0.147 | - | 0.000 |
| Base (untrained Qwen3-4B control) [none] | pt | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | - | 0.000 | - | 0.147 | - | 0.000 |
| Distill (self-distillation) [none] | en | 0.095 | 0.065 | 1.000 | 0.000 | 1.000 | 0.941 | 4.343 | 0.915 | 2.115 | 0.189 | 2.522 | 0.924 |
| Distill (self-distillation) [none] | pt | 0.095 | 0.065 | 1.000 | 0.000 | 1.000 | 0.983 | 3.287 | 0.995 | 1.588 | 0.351 | 2.396 | 0.955 |
| MTP-like (0.8B policy companion + stock Qwen3-4B) [instruction] | en | 0.066 | 0.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.318 | 1.000 | 4.875 | 0.057 | 0.327 | 0.984 |
| MTP-like (0.8B policy companion + stock Qwen3-4B) [instruction] | pt | 0.073 | 0.000 | 1.000 | 0.000 | 1.000 | 0.975 | 0.334 | 0.990 | 4.359 | 0.066 | 0.330 | 0.973 |
| Original DuplexCascade (sbintuitions, q4_k_m) [none] | en | 0.095 | 0.102 | 0.345 | 0.000 | 1.000 | 1.000 | 1.157 | 1.000 | 4.345 | 0.251 | 1.094 | 0.951 |
| Original DuplexCascade (sbintuitions, q4_k_m) [none] | pt | 0.102 | 0.273 | 0.436 | 0.000 | 1.000 | 0.975 | 1.025 | 0.985 | 3.924 | 0.323 | 1.100 | 0.896 |
| SFT (original recipe, continue) [none] | en | 0.051 | 0.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.265 | 0.995 | 4.080 | 0.031 | 0.271 | 0.986 |
| SFT (original recipe, continue) [none] | pt | 0.080 | 0.032 | 1.000 | 0.000 | 1.000 | 1.000 | 0.248 | 0.995 | 3.558 | 0.035 | 0.276 | 0.971 |

## Cross-lingual transfer (pt → en)

Deltas are oriented so **+ is always better**. English is
out-of-distribution for all three variants (they were fine-tuned
on pt-BR duplex data only), so this is protocol transfer, not
language skill.

| Model | Pause TOR↓ synth Δ(en−pt) | Pause TOR↓ candor Δ(en−pt) | BC TOR↓ Δ(en−pt) | BC Freq↑ Δ(en−pt) | BC JSD↓ Δ(en−pt) | Smooth TOR↑ Δ(en−pt) | Floor lat↓ Δ(en−pt) | Intr TOR↑ Δ(en−pt) | Intr rating↑ Δ(en−pt) | Stop lat↓ Δ(en−pt) | Resp lat↓ Δ(en−pt) | ATA↑ Δ(en−pt) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Base (untrained Qwen3-4B control) [none] | -0.000 | -0.000 | -0.000 | +0.000 | -0.000 | +0.000 | - | +0.000 | - | -0.000 | - | +0.000 |
| Distill (self-distillation) [none] | -0.000 | -0.000 | -0.000 | +0.000 | -0.000 | -0.042 | -1.055 | -0.080 | +0.527 | +0.162 | -0.125 | -0.031 |
| MTP-like (0.8B policy companion + stock Qwen3-4B) [instruction] | +0.007 | -0.000 | -0.000 | +0.000 | -0.000 | +0.025 | +0.016 | +0.010 | +0.516 | +0.009 | +0.003 | +0.011 |
| Original DuplexCascade (sbintuitions, q4_k_m) [none] | +0.007 | +0.171 | +0.091 | +0.000 | -0.000 | +0.025 | -0.133 | +0.015 | +0.421 | +0.072 | +0.006 | +0.055 |
| SFT (original recipe, continue) [none] | +0.029 | +0.032 | -0.000 | +0.000 | -0.000 | +0.000 | -0.017 | +0.000 | +0.523 | +0.004 | +0.005 | +0.015 |

## Protocol diagnostics — where the decisions go

Share of decisions spent on each tag. A model with a low pause TOR
but a high `user_is_speaking` share is genuinely waiting; one with
many `no_tag` or `user_interruption` emissions is misfiring.

| Model | Lang | decisions | user_finish_speaking % | user_is_speaking % | user_is_thinking % | user_interruption % | user_backchannel % | system_backchannel % | no_tag % |
|---|---|---|---|---|---|---|---|---|---|
| Base (untrained Qwen3-4B control) [none] | en | 8858 | 0.0 | 99.9 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Distill (self-distillation) [none] | en | 9158 | 7.9 | 85.7 | 4.2 | 1.5 | 0.8 | 0.0 | 0.0 |
| MTP-like (0.8B policy companion + stock Qwen3-4B) [instruction] | en | 8842 | 8.2 | 91.8 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Original DuplexCascade (sbintuitions, q4_k_m) [none] | en | 7640 | 8.8 | 89.6 | 0.0 | 1.5 | 0.0 | 0.0 | 0.0 |
| SFT (original recipe, continue) [none] | en | 8858 | 7.7 | 73.4 | 17.0 | 0.4 | 1.5 | 0.0 | 0.0 |
| Base (untrained Qwen3-4B control) [none] | pt | 9747 | 0.0 | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Distill (self-distillation) [none] | pt | 9740 | 7.8 | 83.8 | 3.8 | 3.9 | 0.7 | 0.0 | 0.0 |
| MTP-like (0.8B policy companion + stock Qwen3-4B) [instruction] | pt | 9725 | 7.5 | 92.0 | 0.0 | 0.4 | 0.0 | 0.0 | 0.1 |
| Original DuplexCascade (sbintuitions, q4_k_m) [none] | pt | 8121 | 9.5 | 87.8 | 1.3 | 1.0 | 0.4 | 0.0 | 0.0 |
| SFT (original recipe, continue) [none] | pt | 9747 | 7.4 | 73.7 | 17.1 | 0.5 | 1.3 | 0.0 | 0.0 |
