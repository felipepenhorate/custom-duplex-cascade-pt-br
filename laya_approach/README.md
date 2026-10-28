# Laya duplex companion

A small, non-autoregressive companion for the pt-BR Full-Duplex-Bench stack. It
replaces the `mtp_like` causal-LM tag classifier with **Laya multilingual**
(`convaiinnovations/laya`, `ModernBertForMaskedLM`, 322M) fine-tuned as a typed
decision model.

Laya does not read a ChatML transcript. Each decision is one state and one
typed `choice` question; the model scores the protocol tags as answer options.
The state makes the *stream event* explicit:

```json
{
  "language": "pt-BR",
  "event": {
    "kind": "partial | final | silence",
    "ends_turn": false,
    "text": "Qual é o prazo?",
    "has_words": true,
    "word_count": 3,
    "terminal_punctuation": true,
    "punctuation": "?",
    "question": true,
    "system_speaking": false,
    "user_turn_state": "in_turn"
  },
  "recent": [ ... up to 6 previous items ... ]
}
```

The answer is one of:

```
no_tag, user_is_speaking, user_finish_speaking, user_interruption,
user_backchannel, user_is_thinking, system_backchannel
```

## Why this fixes `user_finish_speaking`

The audio traces showed the failure clearly: ASR partials frequently end in
`.`, `?` or `!`, and the old companion read terminal punctuation as a turn end.
Forcing the model to answer from the event fixes that:

* a **partial** with terminal punctuation is still mid-turn;
* a **final** with or without punctuation ends the turn;
* a **finalized piece with `ends_turn=False`** (a long utterance delivered in
  several final pieces, as `fdb_bench/audio/stt.py` does) is still mid-turn;
* an **empty final** (the ASR tail is empty because the partials already carried
  every word) still ends the turn;
* silence after a finalized turn starts the answer; silence mid-turn waits;
* content while the system is speaking is an interruption, a short
  acknowledgement is a backchannel, silence is `no_tag`.

Training emits these as hard pairs over the *same text and history*: the only
difference is `kind` / `ends_turn`. Runtime guards back the model up:
`user_finish_speaking` is never allowed without an end-of-turn event or
post-turn silence, and never while the system is speaking.

## Data

`laya_approach/data/prepare_laya_dataset.py` builds the rows from
`lumierenoir/DuplexCascade-PT-BR-V0` (`data/dialogues_combined.jsonl`). It
simulates the ASR stream the way `fdb_bench/audio/stt.py` delivers it: 1–6 word
partial pieces and a positional final tail.

```bash
python -m laya_approach.data.prepare_laya_dataset \
    --out-dir laya_approach/data/generated \
    --train-dialogues 5000 --val-dialogues 400 --test-dialogues 400
```

Outputs `train.jsonl`, `val.jsonl`, `test.jsonl`, `probes.jsonl` and
`audit.json`. The train quotas are balanced by label, and the punctuation /
silence / barge-in hard families are sampled into separate reservoirs.

`generated_v2` is the corrected corpus: punctuation and sentence-final hard
negatives are skipped when the event is a barge-in (`system_speaking=True`), the
`finalized_mid_turn` family is emitted for every non-final chunk, and the audit
checks for identical states with conflicting labels. v2 has 50,600 train /
8,343 val / 8,279 test rows and zero conflicts.

## Training

Use `.venv-unsloth`; Laya needs `FastDecisionModel`, not `FastLanguageModel`.

```bash
source .venv-unsloth/bin/activate
python -m laya_approach.training.train_laya_companion \
    --data-dir laya_approach/data/generated \
    --out-dir laya_approach/runs/laya_companion_v1 \
    --max-train-rows 30000 --epochs 1 \
    --batch-size 16 --lr 2e-4 --head-lr 5e-4 \
    --lora-r 32 --lora-alpha 64
```

The encoder gets LoRA (`r=32`, all linear), the decision head is trained
directly, then the adapter is merged and saved to
`laya_approach/runs/laya_companion_v1/merged` with `policy_cfg.json`.

## Evaluation

```bash
python -m laya_approach.eval.evaluate_laya \
    --model laya_approach/runs/laya_companion_v1/merged \
    --data-dir laya_approach/data/generated --split test

python -m laya_approach.eval.tune_thresholds \
    --model laya_approach/runs/laya_companion_v1/merged \
    --data-dir laya_approach/data/generated --split val --write-cfg
```

The evaluator reports raw and guarded accuracy, per-label P/R/F1, a confusion
matrix, per-bucket accuracy, and the critical rates:

* `partial_punct_false_finish_*` — the failure this approach exists to remove;
* `final_mid_turn_false_finish_*` — finalized piece, turn still open;
* `final_punct_finish_recall_raw`, `final_no_punct_finish_recall_raw`,
  `empty_final_finish_recall_raw`, `silence_after_final_finish_recall_raw`;
* barge-in / backchannel / silence-during-system recall.

`probes.jsonl` is a hand-written set of these shapes; run with `--split probes
--show-probes` to print every probe with its top-3 probabilities.

`replay_audio_traces.py` re-decides the **recorded audio ASR streams** (the
`user_pieces` in `results_audio_pt/decision/<model>/<subset>/<sid>/trace.json`)
and reports, on identical input, how often the reference companion finished
mid-turn versus how often this one does. It is the measurable stand-in for the
audio benchmark when pocket-tts/whisper are unavailable:

```bash
python -m laya_approach.eval.replay_audio_traces \
    --model laya_approach/runs/laya_companion_v1/merged \
    --traces-root custom-duplex-cascade-pt-br/results_audio_pt/decision/mtp_like \
    --mode open
```

`--mode open` is the pre-answer condition (no assistant history,
`system_speaking=False`) where the over-eager finishes happened; `--mode
reference` replays the reference run's assistant turns to test decisions under
system speech (useful for backchannels/interruptions). Besides per-piece rates,
the report counts the **first floor-take per user turn**, so repeated silences
after the same finalized turn do not inflate the result.

## Results

### Decision model

| metric | v1 | v2 |
|---|---:|---:|
| val accuracy / ECE | 0.9909 / 0.0026 | **0.9998 / 0.0001** |
| held-out test accuracy (raw → guarded) | 0.9890 → 0.9890 | **0.9998 → 0.9998** |
| partial-with-punctuation false finish | 0.000 | **0.000** |
| final-mid-turn false finish, raw → guarded | 0.0672 → 0.000 | **0.0000 → 0.0000** |
| punctuated barge-in interruption recall | 0.7450 | **0.9866** |
| final recall (punct / no-punct / empty / post-turn silence) | 1.000 | **1.000** |
| audio replay: turns with a premature floor-take (mtp_like → raw → guarded) | 399 / 2270 → 40 → 0 | 399 / 2270 → **0** → **0** |
| audio replay: mid-turn pieces finished (mtp_like → raw → guarded) | 461 / 4846 → 40 → 0 | 461 / 4846 → **0** → **0** |

v1: one epoch on 30,000 stratified rows (81 min, 2.5 s/step on the RX 6600M).
v2: the same recipe on `generated_v2` (78 min train, 86 min wall, train loss
0.0888), calibrated on 8,343 validation rows.

The replay is the direct before/after on identical input: the archived
`mtp_like` run and Laya are both replayed over the same 13,296 ASR pieces from
727 audio traces, with Laya's own thresholds loaded from its checkpoint. The
reference's 399 premature turns decompose exactly into the failure this work
targets: 366 were partials ending in `.?!` (the punctuation false floor), 24
plain partials, 6 punctuated final pieces with `ends_turn=False`, and 3 plain
final pieces with `ends_turn=False`. v1's 40 remaining raw errors were all the
last two shapes; v2 trains that `final_mid_turn` family and removes them.

v1 was trained before the barge-in label fix. Its `partial_punctuation_negative`
rows duplicated the first chunk of a barge-in with `system_speaking=True` and
the label `user_is_speaking` (1,770 of 3,125 rows in that bucket). `generated_v2`
guards those hard negatives with `not system_speaking`, has zero identical-state
label conflicts, and adds 3,909 `final_mid_turn` training rows. Train v2 with
the same recipe:

```bash
python -m laya_approach.training.train_laya_companion \
    --data-dir laya_approach/data/generated_v2 \
    --out-dir laya_approach/runs/laya_companion_v2 \
    --max-train-rows 30000 --epochs 1 --batch-size 16 \
    --lr 2e-4 --head-lr 5e-4 --lora-r 32 --lora-alpha 64
```

### Full-Duplex-Bench (text, 727 scenarios per language)

Both languages, `word` chunker, stock Qwen3-4B-Instruct-2507 content model,
decision prime mode, local Qwen3.5-9B judge. `TOR ↓` means lower is better;
`↑` higher is better.

| metric | pt mtp_like | **pt Laya v2** | en mtp_like | **en Laya v2** |
|---|---:|---:|---:|---:|
| pause TOR, synthetic ↓ | 0.073 | **0.000** | 0.066 | **0.000** |
| pause TOR, candor ↓ | 0.000 | **0.000** | 0.000 | **0.000** |
| backchannel TOR ↓ | 1.000 | **0.000** | 1.000 | **0.000** |
| smooth-turn TOR ↑ | 0.975 | **1.000** | 1.000 | **1.000** |
| interruption TOR ↑ | 0.990 | **1.000** | 1.000 | **1.000** |
| interruption rating ↑ | 4.359 | **4.430** | 4.875 | 4.805 |
| response quality ↑ | 4.381 | **4.442** | 4.824 | **4.909** |
| finish precision / recall / F1 | .698 / .958 / .808 | **1 / 1 / 1** | .713 / .992 / .830 | **1 / 1 / 1** |
| chunk accuracy ↑ | 0.944 | **1.000** | 0.936 | **1.000** |
| **averaged turn-taking accuracy ↑** | 0.973 | **1.000** | 0.984 | **1.000** |
| smooth floor latency (s) ↓ | 0.334 | 0.809 | 0.318 | 0.806 |
| interruptions / decisions | 39 / 4202 | 104 / 4488 | 0 / 3320 | 158 / 4673 |
| scenarios / errors | 727 / 0 | 727 / **0** | 727 / 0 | 727 / **0** |

Reading the table:

* **Turn-taking** is exact on both languages: no formal turn during pauses or
  the backchannel monologue, and the floor is taken after the real turn end.
  English is cross-lingual transfer: the decision model only saw pt-BR states.
* **Latency is the deliberate trade-off.** Laya waits for the stream's
  end-of-turn event instead of guessing from punctuation, so it takes the floor
  at ~0.81 s instead of ~0.33 s. mtp_like's lower latency is partly its 215
  premature finishes (finish precision 0.698).
* **Backchannel naturalness did not improve.** `bc_freq = 0.000` and
  `bc_jsd = 1.000` for both companions: Laya rarely emits the short listener
  `system_backchannel` tag. Its BC TOR gain comes from *not taking a formal
  turn*, not from producing more backchannels. The ATA formula credits this.
* **Residual probe gap:** a *partial* multi-word acknowledgement over system
  speech ("uhum", split into a partial and a final) is classified as
  `user_interruption` (8/75 probes). The FDB backchannel task's monologue
  chunks are not acknowledgements, so its TOR is unaffected, but a live partial
  ack can yield the floor. The fix is a partial-backchannel family in the
  generator; it is not in v2.
* **The FDB text comparison is event-aware vs text-only.** `LayaPolicyAdapter`
  passes `kind`, `ends_turn` and `system_speaking`; mtp_like's adapter strips
  them so its prompt stays byte-identical. The audio-trace replay above is the
  paired test on the same event stream and is the stronger evidence for the
  turn-end fix.
* **The chunker is cross-arm.** mtp_like's stored run used the Qwen-token
  `TrainingChunker`; Laya uses `word`, which reproduces the published `pt_text`
  set exactly (727/727). The comparison is therefore not a same-prompt ablation.
* The judge is the local Qwen3.5-9B GGUF, not FDB's `gpt-4-turbo`, so ratings
  are comparable within this table, not to the paper's absolute numbers.

### Audio contract and v7 results

The text and audio streams are **not the same contract**. In the text bench
`ends_turn=True` is a semantic turn boundary; in the audio harness it is the
VAD's utterance boundary after 0.6 s of silence, and a pause clip can end on
such a final without the turn being over. v7 makes that explicit:

- every state carries `"stream": "text" | "audio"`;
- text FDB, v2 and the translated pilot train as `text`;
- the 21,158 decision rows built from the 727 pt + 727 en recorded audio traces
  train as `audio`;
- a smooth audio turn may finish on the final **or on the first two post-final
  silence ticks** (deferred confirmation), while pause/backchannel silences
  always wait.

The audio corpus is labeled from the FDB annotations: pause scenarios never
finish, smooth finishes at the annotated turn-taking time, interruption labels
follow the state during the annotation span, backchannel never takes a formal
turn. Ambiguous exact states are dropped (0 conflicts).

v7 (`laya_companion_v7_deferred`): val 99.12%, v2 regression **99.93%**, audio
trace test 94.5%, with pause finishes 23/652 and backchannel finishes 7/518.
TTS audio sweep, 727 scenarios per language, guard disabled:

| audio subset | pt v7 | pt mtp_like | en v7 | en mtp_like |
|---|---:|---:|---:|---:|
| pause TOR, candor ↓ | **0.139** | 0.806 | **0.046** | 0.829 |
| pause TOR, synthetic ↓ | **0.044** | 0.730 | **0.022** | 0.920 |
| smooth TOR ↑ | **0.765** | 0.782 | 0.706 | 0.891 |
| smooth latency (s) ↓ | 1.77 | 0.97 | 2.24 | 0.53 |
| backchannel TOR ↓ | **0.073** | 0.836 | **0.036** | 0.764 |
| backchannel freq ↑ | 0.005 | 0.099 | 0.002 | 0.111 |
| backchannel JSD ↓ | 0.964 | 0.727 | 0.991 | 0.722 |
| interruption TOR ↑ | 0.815 | 0.835 | 0.810 | 0.935 |
| interruption rating ↑ | 3.23 | 3.19 | 3.88 | 4.28 |

The pause/backchannel failure is essentially gone. The cost is smooth latency
(+0.8–1.7 s, deferred confirmation), a small smooth/interruption TOR drop, and
still no listener backchannels (`bc_freq` remains near zero; the TOR gain is
from not taking a formal turn). The v7 audio metrics use faster-whisper
**small** on CPU because the RX 6600M has no CTranslate2 ROCm backend; the
mtp_like audio scores used large-v3. Compare TOR directionally, not to the last
decimal.

## FDB integration

The companion is registered as `laya_companion` / `laya_companion_v2` /
`laya_companion_v7` in `fdb_bench/protocol.py` and loaded lazily by
`fdb_bench/policy_companion.py`. `LayaPolicyAdapter` keeps the stream metadata
(`kind`, `ends_turn`, `system_speaking`) on the simulator items and tags the
policy stream (`text` for the text bench, `audio` for the audio harness); the
mtp_like adapter continues to strip metadata, so its prompt is byte-identical to
before. `simulate.py` and `audio/session.py` add the metadata at each decision
point (for audio, `system_speaking` is true only while the queued assistant turn
is still audible).

The published `lumierenoir/synthetic-full-duplex-bench-pt-br` `pt_text`
configuration is the same 727 scenarios the local generator builds, with the
word chunker. Verify that before comparing a run to it:

```bash
python -m laya_approach.eval.check_fdb_parity
```

Then run the whole published set in both languages (from
`custom-duplex-cascade-pt-br/`):

```bash
for lang in pt en; do
    python -m fdb_bench.run_inference --model laya_companion_v7 --lang "$lang" \
        --chunker word --api-base http://127.0.0.1:8081/v1 \
        --out "results/$lang/laya_companion_v7"
    python -m fdb_bench.run_judge --results "results/$lang/laya_companion_v7" \
        --lang "$lang" --api-base http://127.0.0.1:8099/v1
    python -m fdb_bench.run_metrics --results "results/$lang"
done
```

The `word` chunker is the published set's chunker and matches both a live ASR
stream and Laya's training pieces. The older in-repo mtp_like baseline was run
with the Qwen-token `TrainingChunker`, so a comparison with it is cross-arm and
is reported as such. The English run is cross-lingual transfer: the decision
model was trained only on pt-BR states, while the content model gets the
English voice prompt.

Audio evaluation needs the user-side audio (`lumierenoir/synthetic-full-duplex-
bench-pt-br` `pt_audio`), pocket-tts for the assistant and whisper for reading it:

```bash
python -m fdb_bench.audio.run_audio --model laya_companion_v7 --lang pt \
    --api-base http://127.0.0.1:8081/v1 --big-api-base http://127.0.0.1:8081/v1 \
    --out results_audio_pt/laya_companion_v7
```

The content model stays stock; Laya only decides *when* the assistant should
speak. It never writes the answer itself.

## Layout

```
laya_approach/
  data/common.py               state schema, typed question, guards, buckets
  data/select_pilot_dialogues.py  tag-stratified 2k pilot selection
  data/translate_dialogues.py     local Qwen3.5-9B dialogue translation
  data/prepare_audio_trace_dataset.py  FDB-annotation audio-trace rows
  data/prepare_laya_dataset.py raw dialogue -> hard-paired decision rows
  training/train_laya_companion.py  unsloth DecisionTrainer + merged export
  eval/evaluate_laya.py        per-label / per-bucket / probe report
  eval/tune_thresholds.py      per-tag threshold sweep on validation
  eval/replay_audio_traces.py  re-decide the recorded audio ASR streams
  eval/check_fdb_parity.py     verify the published PT text set is reproduced
  policy/laya_policy.py        runtime query(items) companion
  tests/                       schema, guard and adapter-metadata tests
```
