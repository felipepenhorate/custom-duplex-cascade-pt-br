# Duplex prompt calibration (pt)

Measured by `python -m fdb_bench.calibrate_prompts`. Each row is one
system-prompt candidate run over the same scenario sample; the verdict
column says whether it activates duplex mode (see `calibrate_prompts.verdict`).

### pt — prompt candidates

| Model | Candidate | prompt len | finish P | finish R | finish F1 | chunk acc | pause silent | non-Latin | repeats | empty | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| distill | `none` | 0 | 0.397 | 0.853 | 0.542 | 0.806 | 0.997 | 0.003 | 0.000 | 0.000 | **OVER-EAGER** |
| distill | `deployed` | 879 | 0.500 | 0.353 | 0.414 | 0.865 | 1.000 | 0.001 | 0.000 | 0.000 | **OK** |
| distill | `tag` | 261 | 0.593 | 0.471 | 0.525 | 0.885 | 1.000 | 0.000 | 0.000 | 0.000 | **OK** |
| distill | `trigger` | 564 | 0.750 | 0.088 | 0.158 | 0.873 | 1.000 | 0.002 | 0.000 | 0.000 | **OK** |
| distill | `trigger_tags` | 903 | 0.342 | 0.382 | 0.361 | 0.817 | 0.995 | 0.007 | 0.000 | 0.000 | **OVER-EAGER** |
