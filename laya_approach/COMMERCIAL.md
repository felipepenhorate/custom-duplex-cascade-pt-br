# Training a commercial / permissive Laya companion

The published v7 checkpoint is **CC BY-NC 4.0** because it uses CANDOR and ICC
annotations. This recipe produces a **permissive** model (Apache-2.0 or MIT) by
keeping only MIT components:

| component | status |
|---|---|
| Laya multilingual base | Apache-2.0 |
| `DuplexCascade-PT-BR-V0` dialogues | MIT |
| translated pilot of those dialogues | MIT |
| FDB **synthetic** pause / interruption sets | MIT |
| FDB CANDOR and ICC | **excluded** (CC BY-NC 4.0) |
| this training/eval code | MIT/Apache-2.0-compatible |

The trade-off is real: the commercial model no longer sees CANDOR smooth-turn,
CANDOR pause or ICC backchannel data. Text turn-taking is still trainable from
the MIT dialogues; audio is limited to the two synthetic FDB subsets. Do not
claim the v7 audio/corpus numbers for this narrower model.

## 1. Get the v7 decision data

From this GitHub checkout, or from the public HF release:

```bash
# Option A: local research mix
SRC=laya_approach/data/generated_v7_mix

# Option B: download the published decision rows (1.0 GB repo; this pulls only
# the relevant files)
huggingface-cli download lumierenoir/laya-duplex-cascade-pt-br \
    --repo-type model \
    --include "training_data/decisions/*" \
    --local-dir data/laya_release
SRC=data/laya_release/training_data/decisions
```

The mix contains `state.stream` (`text` / `audio`) and
`_meta.subset`. The filter uses those fields.

## 2. Drop every CANDOR/ICC-derived row

```bash
python -m laya_approach.data.filter_commercial \
    --src "$SRC" \
    --out data/generated_v7_commercial
```

Expected output from the v7 mix:

```text
train: 40,871 rows  (30,000 text + 10,871 synthetic audio)
val:   23,870 rows
test:  23,719 rows
dropped train: candor_pause_handling 4,469
               candor_turn_taking    2,876
               icc_backchannel       2,942
```

The filter keeps every `stream=text` row (MIT source dialogues + MIT translated
pilot) and only these audio subsets:

- `synthetic_pause_handling`
- `synthetic_user_interruption`

It writes a per-split audit with row counts, labels, streams and dropped
subsets. Verify the audit contains **no** `candor_*` or `icc_*` entries before
training.

## 3. Train from the Apache-2.0 Laya base

Same recipe as v7, new output directory:

```bash
python -m laya_approach.training.train_laya_companion \
    --data-dir data/generated_v7_commercial \
    --out-dir laya_approach/runs/laya_companion_commercial \
    --max-train-rows 35000 --epochs 1 \
    --batch-size 16 --grad-accum 1 \
    --lr 2e-4 --head-lr 5e-4 \
    --lora-r 32 --lora-alpha 64 \
    --log-every 25 --calibrate --save-merged
```

The trainer re-samples 35k stratified rows, calibrates on the filtered val set,
and writes `merged/` with `policy_cfg.json`. Keep `--max-train-rows` at 35k or
lower; the filtered train set is 40,871 rows, so one epoch is still 2,188
steps at batch 16.

## 4. Evaluate only the compatible arms

```bash
# decision quality
python -m laya_approach.eval.tune_thresholds \
    --model laya_approach/runs/laya_companion_commercial/merged \
    --data-dir data/generated_v7_commercial --split val --write-cfg

python -m laya_approach.eval.evaluate_laya \
    --model laya_approach/runs/laya_companion_commercial/merged \
    --data-dir data/generated_v7_commercial --split test

# audio: only the two synthetic subsets are in-distribution
python -m fdb_bench.audio.run_audio \
    --model laya_companion_commercial --lang pt \
    --data-root laya_approach/data/fdb_pt_audio/v1.0 \
    --stt-trace-root custom-duplex-cascade-pt-br/results_audio_pt/decision/mtp_like \
    --audio-root laya_approach/data/fdb_audio_commercial_pt \
    --out laya_approach/data/fdb_audio_results/laya_commercial_pt \
    --tasks synthetic_pause_handling,synthetic_user_interruption \
    --tts-language portuguese --skip-score
```

Do **not** report the CANDOR/ICC columns as held-out results for this model —
those distributions were deliberately removed. If you need smooth-turn or
backchannel coverage under a permissive license, add MIT-licensed dialogue /
audio data for those behaviors and regenerate the same `stream=text` and
`stream=audio` decision rows.

## 5. Licensing the result

- **Model weights:** Apache-2.0 (derivative of the Apache-2.0 Laya base) or MIT
  if you prefer; keep the Laya attribution and Apache notice.
- **Training data:** the filtered decisions contain only MIT source dialogues
  and MIT synthetic FDB subsets. The FDB synthetic subsets are MIT per the
  Full-Duplex-Bench dataset README; CANDOR/ICC rows are gone.
- **Code:** MIT or Apache-2.0; add the repository LICENSE of your choice.
- **Attribution:** cite Laya (`convaiinnovations/laya`), DuplexCascade
  (`lumierenoir/DuplexCascade-PT-BR-V0`) and Full-Duplex-Bench.

This is engineering guidance, not legal advice. If you redistribute a model
trained with this recipe, run the audit and keep the filter script with the
release so the component provenance is reproducible.

## One-command version

```bash
python -m laya_approach.data.filter_commercial \
    --src laya_approach/data/generated_v7_mix \
    --out data/generated_v7_commercial \
&& python -m laya_approach.training.train_laya_companion \
    --data-dir data/generated_v7_commercial \
    --out-dir laya_approach/runs/laya_companion_commercial \
    --max-train-rows 35000 --epochs 1 \
    --batch-size 16 --lr 2e-4 --head-lr 5e-4 \
    --lora-r 32 --lora-alpha 64 --calibrate --save-merged
```
