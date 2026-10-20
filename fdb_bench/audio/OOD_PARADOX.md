# Why the out-of-distribution set scored better than the in-distribution one

**Status:** diagnosed, not yet fixed. Every number below is reproducible from
`results_audio*/` and `/mnt/f/fdb_audio*/`; the code paths are named.

## The paradox

The pt-BR variants are fine-tuned on pt-BR duplex data, so Portuguese audio is
their *in-distribution* set and the English FDB audio is *out-of-distribution*.
The evaluation says the opposite:

| model | EN (parakeet) | PT (whisper-pt) | Δ (PT−EN) |
|---|---|---|---|
| Original DuplexCascade (upstream, English-trained) | 0.824 | 0.702 | −0.122 |
| SFT | 0.564 | 0.498 | −0.066 |
| Self-distillation | 0.704 | 0.345 | **−0.359** |
| Companion (MTP-style) | 0.519 | 0.483 | −0.036 |

The out-of-distribution set wins for all three trained models. It should not.

**It is not a model property. It is the measurement.**

## The decisive test: take the recogniser out

Upstream's take-over rule (`fdb_bench/audio/fdb_eval/eval_pause_handling.py`,
`eval_smooth_turn_taking.py`) is *session-level*: the model "took the floor" if
the transcript of the whole `output.wav` has ≥1 s of speech or >3 words. So it
can be computed from the TTS ground truth (the trace) instead of from the ASR.
Re-running the same rule on the traces, with no recogniser in the loop:

| model | Δ (PT−EN), ASR-based | Δ (PT−EN), TTS-truth | share of the gap that is the recogniser |
|---|---|---|---|
| Original | −0.122 | −0.127 | ~0% (real behaviour) |
| SFT | −0.066 | −0.029 | 56% |
| Self-distillation | −0.359 | **−0.044** | **88%** |
| Companion | −0.036 | −0.015 | 58% |

For self-distillation, 88% of the "Portuguese is worse" gap is the recogniser
not hearing the reply. The models do not behave differently enough to explain it.

## Defect 1 — the pt audio is not the benchmark (30% of it is synthetic silence)

`fdb_bench/audio/build_pt_audio.py` translates the English *ASR transcript*,
synthesises each silero span with pocket-tts, and **pads the remainder of the
span with digital silence** so the English timeline is preserved
(`stretch.py::fit_to_slot` pads whenever the rate would fall below 0.75).

Measured over the whole set (`translation.json` metadata, 2738 segments):

* **832 segments (30.4%) are padded** — Portuguese TTS is shorter than the
  English span, so the gap is filled with silence *inside the user's turn*;
* in `candor_pause_handling`, **12.2% of the user's speech timeline is synthetic
  silence**; in `candor_turn_taking` 8.3%;
* 896 segments were clamped (TTS could not fit the slot) and 61 truncated
  mid-word;
* 94 segments (3.4%) are effectively untranslated — the Portuguese voice reads
  English, e.g. `"Magoo uh enemy"` → `"Magoo, inimigo"`;
* per-segment gain runs 1.01 → **92.78** (median 2.96), so the level varies
  wildly between segments.

Consequence, measured on the traces (353 pause scenarios per set):

| | EN | PT |
|---|---|---|
| speech as a fraction of the timeline | 0.80 | 0.63 |
| silence per scenario (candor_pause) | 2.19 s | 4.23 s |
| silence outside the benchmark's own pause windows ("phantom" silence) | 1.07 s | **3.08 s** |
| gaps ≥0.6 s per scenario (candor_pause) | 1.26 | 1.73 |
| gaps ≥0.6 s per scenario (icc_backchannel) | 1.60 | **4.67** |

The harness's turn-end signal is a **0.6 s silence gate** on the VAD
(`fdb_bench/audio/stt.py`), so every artificial gap is a *new invitation* to
take the floor. Invitations that land in silence the benchmark does **not**
annotate as a pause (i.e. mid-utterance, created by the padding): **58 in EN,
303 in PT** over the same 353 scenarios. At every class of invitation, every
model answers more often on the pt audio:

| model | annotated pause (EN→PT) | user speaking | end of input |
|---|---|---|---|
| Original | 8.6% → 17.1% | 5.3% → 11.0% | 7.7% → 13.3% |
| SFT | 36.6% → 56.0% | 27.2% → 48.8% | 44.5% → 53.2% |
| Self-distillation | 28.8% → 38.9% | 24.9% → 44.1% | 31.2% → 43.2% |
| Companion | 36.0% → 55.6% | 34.8% → 42.4% | 56.4% → 65.6% |

The upstream model is **English-trained and never saw Portuguese**, so its
doubling on the pt audio cannot be a language-skill effect: the pt *input* makes
every model chattier.

## Defect 2 — the take-over rule is session-level, so every extra invitation counts

`eval_pause_handling.py` never opens `pause.json`. TOR = "did the model say
≥4 words or ≥1 s *anywhere* in the session". One answer during a pause zeroes
the whole scenario, and Defect 1 gives the model ~5× more chances to answer.
The pause columns are therefore a measure of *chattiness* under a silence gate,
not of pause sensitivity.

## Defect 3 — the rendered reply is never level-normalised, and the pt voice is 12.6 dB quieter

The *input* audio is peak-normalised per segment (`build_pt_audio.py`, target
0.7). The *assistant's* rendered audio is not normalised anywhere
(`render.py`). The two voices:

| voice | RMS |
|---|---|
| english | −18.5 dB |
| portuguese | **−31.1 dB** |

and the model-specific tails are worse: the pt render of self-distillation's
replies peaks at 0.008 (−42 dBFS), frame RMS −59.9 dB, against −11.8 dB for the
same scenario in EN. `faster-whisper` with `vad_filter=True` then rejects it and
returns an empty transcript — or hallucinates (`"Obrigado por assistir"`,
`"Legenda por Sônia Ruberti"`). On `candor_turn_taking`, **23 of 119 pt
scenarios have audible audio and an empty transcript**; the model is scored as
mute. That is the 88% of distill's gap.

## Defect 4 — a barge-in can zero the reply

Every single zero-duration turn in the corpus was stopped by a barge-in
(140/140, 92/92, 385/385, 185/185 in EN; 327/328, 148/148, 631/631, 332/332 in
PT). The stop lands at `max(turn.start, min(barge_in, natural_end))`, so a reply
that begins at the same instant the user resumes renders as nothing at all.
The pt set makes this worse: the models start earlier relative to the
annotation (self-distillation's first speech lands **before** the benchmark's
turn end in 50% of the pt smooth scenarios vs 38% in EN; SFT 77% vs 55%), so
more replies are born inside a barge-in.

## Defect 5 — the ATA guard is a cliff

`fdb_bench/metrics.py::averaged_turn_taking_accuracy`:

```python
if math.isnan(smooth) or smooth < DUPLEX_CAPABLE_SMOOTH_TOR:   # 0.5
    return smooth
```

Self-distillation's pt smooth TOR is 0.345 (below the guard, because Defects 3
and 4 made the renderer and the recogniser miss its replies), so ATA is *pinned*
to 0.345 instead of the 0.413 the four-term mean gives. The guard exists to
stop silence from scoring well; here it converts a measurement failure into a
catastrophic ATA.

## Defect 6 — the benchmark's own scorer is English-only

`nvidia/parakeet-tdt-0.6b-v2` cannot read Portuguese replies. On the pt set it
inflates the pause columns (unreadable replies look like silence, which the
pause task rewards) and deflates smooth/interruption. The distortion is
per-model and bidirectional:

| model | PT parakeet | PT whisper-pt | Δ |
|---|---|---|---|
| Original | 0.626 | 0.702 | +0.076 |
| SFT | 0.515 | 0.498 | −0.017 |
| Self-distillation | 0.160 | 0.345 | +0.185 |
| Companion | 0.568 | 0.483 | −0.085 |

So the published comparison mixes two different scorers across the two rows.

## The control that says the residual is small

The text-domain harness holds the timing structure fixed and swaps only the
*content* language. There, out-of-distribution ≈ in-distribution, with mixed
signs: SFT 0.986 (en) / 0.971 (pt), Companion 0.984 / 0.973, Self-distillation
0.924 / 0.955. If the pt content itself were the problem, the audio run would
not be 5-10× worse than the text run.

## Fixes, in order of impact

1. **Rebuild the pt audio without the padding.** Compress the Portuguese TTS
   into the English slot (or re-time the whole utterance) instead of padding
   with silence (`stretch.py::fit_to_slot`: raise `min_rate`, or re-time
   downstream segments when a span is short). Then the pt audio's silence
   structure matches the original recording, and the 0.6 s gate stops inventing
   invitations. **This is the fix for Defect 1 and it is a data rebuild.**
2. **Peak-normalise the rendered assistant audio** before it is written
   (`render.py`), exactly as the input side is normalised. Removes Defect 3.
3. **Score every set with a language-matched recogniser**, and print the
   scorer next to every number. Removes Defect 6.
4. **Report ATA with and without the 0.5 guard**, or apply the guard to the
   *rendered* behaviour rather than the recognised one. Removes Defect 5.
5. Optionally: report a windowed pause metric computed from `pause.json` (the
   benchmark's own annotation) next to upstream's session-level one, since the
   latter is what Defect 1 feeds on.

## What is still open

* How much of the residual (TTS-truth Δ of −0.015 to −0.127) is the artificial
  silence versus a genuine cross-lingual effect? The text-domain control says
  the cross-lingual part is small (≈ −0.055 for the upstream model), but it
  uses a different corpus, so it bounds rather than settles the question.
* The causal test (rebuild → re-run → re-score) has not been run yet.

## Remediation (what was changed, and what has to be re-measured)

The pt-BR audio results were deleted (`results_audio_pt*`, the rendered audio,
the set itself) and the set was rebuilt. Four changes, in order of impact:

1. **The set is rebuilt without padding.** `build_pt_audio.py` no longer
   time-fits a span into its English slot; the Portuguese speech keeps its own
   rate, the gaps keep their length, and the annotations are re-timed with the
   same map (`plan_timeline`, `make_remap`, `remap_annotation`). The old
   policy is kept in `time_fit` with a comment recording why it was dropped.
2. **A synthesised span must occupy its slot the way the original span did**
   (`fit_span_to_slot`). The recording's span is one utterance -- the detector
   merges speech separated by less than 100 ms, so a span holds at most a
   100 ms hitch and begins and ends on speech. The synthesiser does neither: it
   opens and closes with silence and pauses for longer than 100 ms at
   punctuation, and every such pause **split the Portuguese span in two**. The
   structure gate caught this on the first rebuild: on the backchannel subset
   the rebuilt audio had 12.9 silero spans per scenario against the
   recording's 9.4, with 1.35 extra gaps of 0.6 s or more *inside* the user's
   turns -- the same defect as the padding, one level down. The edges are now
   trimmed to the speech and the internal hitches are shortened below the
   detector's own merge threshold.
3. **`check_pt_structure.py` is a gate.** It compares the rebuilt audio with
   the recording it was made from -- speech fraction, spans, gaps >= 0.6 s,
   silence outside the benchmark's own pause windows -- and fails the build if
   the two shapes differ. This is the check the first build lacked, and it is
   what caught (2).
4. **Recognition is level-normalised, in both sets**
   (`score.normalise_for_recognition`), and the pt set is scored with a
   Portuguese recogniser while the English set keeps parakeet. The English set
   is re-scored from its existing renders, so the two rows are measured
   identically.

Everything downstream of the audio (the renders, the traces, the metrics, the
tables, and the article's audio section) is regenerated by the same chain, so
no number in the repository still depends on the defective set.
