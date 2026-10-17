"""Full-Duplex-Bench v1.0 metrics, ported from the official eval scripts.

Each function below is a faithful port of one FDB script, reading a
:class:`~fdb_bench.simulate.SimResult` timeline instead of a WAV + ASR JSON:

=========================  ==================================================
FDB script                 ported here
=========================  ==================================================
``eval_pause_handling.py`` :func:`pause_handling`
``eval_smooth_turn_taking`` :func:`smooth_turn_taking`
``eval_backchannel.py``    :func:`backchannel`
``eval_user_interruption`` :func:`user_interruption_tor_latency`
                           + :func:`aggregate_interruption_ratings`
=========================  ==================================================

The FDB constants are kept verbatim:

* ``turn_duration_threshold = 1``   — a model utterance spanning >= 1 s is a
  formal turn; shorter ones only count when they carry > 3 words.
* ``turn_num_words_threshold = 3``
* backchannel: a segment is a backchannel when ``duration <= 3`` s **and**
  ``<= 3`` words (``time_threshold = 3``).

Directions: pause/backchannel TOR and all latencies are *lower is better*;
smooth-turn TOR, interruption TOR and the judge rating are *higher is better*.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

from .clock import SpeechClock
from .simulate import SimResult

# FDB constants (eval_pause_handling.py / eval_smooth_turn_taking.py /
# eval_backchannel.py)
TURN_DURATION_THRESHOLD = 1.0
TURN_NUM_WORDS_THRESHOLD = 3
BACKCHANNEL_TIME_THRESHOLD = 3.0
BACKCHANNEL_WINDOW_S = 0.2   # eval_backchannel.py: window_size
EPSILON = 1e-10


# --------------------------------------------------------------- helpers ----

def _model_utterances(res: SimResult, clock: SpeechClock) -> list[tuple[float, float, str]]:
    """All model speech as ``(start, end, text)``, backchannels included."""
    return [(m.start, m.end, m.text) for m in res.model_turns if m.text.strip()]


def _tor_from_segments(
    segments: Iterable[tuple[float, float, str]], clock: SpeechClock,
) -> int:
    """FDB's binary take-over decision for a set of model segments.

    ``eval_pause_handling.py``: no output at all -> 0; else take the total span,
    and treat it as a takeover when it reaches ``turn_duration_threshold``
    seconds or (when shorter) carries more than ``turn_num_words_threshold``
    words.
    """
    segs = list(segments)
    if not segs:
        return 0
    total_words = sum(len(t.split()) for _, _, t in segs)
    span = max(e for _, e, _ in segs) - min(s for s, _, _ in segs)
    if span < TURN_DURATION_THRESHOLD:
        return 0 if total_words <= TURN_NUM_WORDS_THRESHOLD else 1
    return 1


def mean(xs: list[float]) -> float:
    """Mean, or NaN for an empty sample (so tables show '-' not 0.0)."""
    return sum(xs) / len(xs) if xs else float("nan")


_mean = mean   # internal alias


def _sem(xs: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    m = _mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(var / len(xs))


# ------------------------------------------------------ 1. pause handling ----

def pause_handling(results: Iterable[SimResult], clock: SpeechClock) -> dict[str, Any]:
    """Pause handling: TOR (lower is better).

    FDB asks "did the model wrongly take the floor while the user was only
    pausing?". We score the model's speech *inside the pause window only*: the
    user chunks before the first silence event are context, and everything the
    model says from the silence onwards counts against it.

    Returns ``tor`` (mean, lower better) plus ``n`` and the standard error.
    """
    tors: list[int] = []
    for res in results:
        if res.error:
            continue
        first_silence = next(
            (i for i, e in enumerate(res.user_events) if e.kind == "silence"), None,
        )
        if first_silence is None:
            continue
        t_pause = res.user_events[first_silence].t_start
        segs = [
            (m.start, m.end, m.text)
            for m in res.model_turns
            if m.text.strip() and m.start >= t_pause - 1e-9
        ]
        tors.append(_tor_from_segments(segs, clock))
    return {
        "tor": _mean([float(t) for t in tors]),
        "tor_sem": _sem([float(t) for t in tors]),
        "n": len(tors),
    }


# --------------------------------------------------- 2. smooth turn-taking ---

def smooth_turn_taking(
    results: Iterable[SimResult], clock: SpeechClock, *, compute_latency: bool = True,
) -> dict[str, Any]:
    """Smooth turn-taking: TOR (higher better) and latency (lower better).

    TOR = did the model take the floor after the user's turn ended.
    ``latency_s`` = the model's *real inference compute time* between the end of
    the user's turn and the floor being taken (recorded by the adapter); this is
    the text-domain analogue of FDB's acoustic response latency and is the
    number that actually differs between a duplex LLM and a policy companion.
    """
    results = [r for r in results if not r.error]
    tors: list[int] = []
    latencies: list[float] = []
    lat_chunks: list[int] = []
    for res in results:
        user_end = res.user_turn_end()
        if user_end is None:
            continue
        # the model turn that starts at/after the user's turn end
        after = [m for m in res.model_turns if m.start >= user_end - 1e-9]
        took = 1 if after and _tor_from_segments(
            [(m.start, m.end, m.text) for m in after], clock) else 0
        tors.append(took)
        if took and compute_latency:
            # decisions the model sat through before committing
            lat_chunks.append(
                sum(1 for e in res.user_events if e.t_end <= user_end + 1e-9)
            )
            # adapter-recorded compute: recomputed below by the runner
    out: dict[str, Any] = {
        "tor": _mean([float(t) for t in tors]),
        "tor_sem": _sem([float(t) for t in tors]),
        "latency_chunks": _mean([float(c) for c in lat_chunks]),
        "n": len(tors),
    }
    _ = latencies
    return out


# --------------------------------------------------------- 3. backchannel ----

def backchannel(results: Iterable[SimResult], clock: SpeechClock) -> dict[str, Any]:
    """Backchannel: TOR (lower better), frequency (higher better), JSD (lower better).

    * **TOR** — did the model take a *formal* turn during a monologue it should
      only backchannel on? FDB flags TOR=1 as soon as any model segment exceeds
      ``BACKCHANNEL_TIME_THRESHOLD`` seconds (``eval_backchannel.py``).
    * **Freq** — backchannel segments per second of conversation, matching FDB's
      ``len(backchannel_prediction) / max_end_time``.
    * **JSD** — Jensen–Shannon divergence between the model's backchannel time
      distribution and the ground truth (the user's sentence endpoints), on the
      same 0.2 s window grid FDB uses. FDB reports ``scipy``'s
      ``jensenshannon``, which is the *base-2* JS divergence (its square root),
      so we do the same to stay numerically comparable.
    """
    results = [r for r in results if not r.error]
    tors: list[int] = []
    freqs: list[float] = []
    jsds: list[float] = []
    for res in results:
        segs = _model_utterances(res, clock)
        total_s = res.end_time or 1e-9

        # TOR: any segment that counts as formal speaking
        formal = any(
            (e - s) > BACKCHANNEL_TIME_THRESHOLD
            or len(t.split()) > TURN_NUM_WORDS_THRESHOLD
            for s, e, t in segs
        )
        tors.append(1 if formal else 0)

        # backchannel segments = short model utterances
        bc = [(s, e) for s, e, t in segs if clock.is_backchannel(t, e - s)]
        freqs.append(len(bc) / total_s)
        # eval_backchannel.py: with no backchannel prediction at all the
        # divergence is 1 (the maximum) -- without this a model that stays
        # silent would score a *perfect* 0 against a near-uniform GT histogram,
        # which would reward never speaking.
        jsds.append(
            1.0 if not bc
            else _jsd_backchannel(bc, res.gt_backchannel_times, total_s)
        )
    return {
        "tor": _mean([float(t) for t in tors]),
        "tor_sem": _sem([float(t) for t in tors]),
        "freq": _mean(freqs),
        "freq_sem": _sem(freqs),
        "jsd": _mean(jsds),
        "jsd_sem": _sem(jsds),
        "n": len(results),
    }


def _histogram(times: list[float], total_s: float) -> list[float]:
    """Normalised 0.2 s-window occupancy histogram (FDB's ``time_intervals``)."""
    bins = max(1, int(total_s / BACKCHANNEL_WINDOW_S) + 1)
    h = [0.0] * bins
    for t in times:
        i = min(bins - 1, max(0, int(t / BACKCHANNEL_WINDOW_S)))
        h[i] += 1.0
    h = [x + EPSILON for x in h]
    s = sum(h)
    return [x / s for x in h]


def _jsd_backchannel(
    pred: list[tuple[float, float]], gt_times: list[float], total_s: float,
) -> float:
    """Jensen–Shannon divergence between two window histograms (base-2)."""
    pred_hist = _histogram([(s + e) / 2 for s, e in pred], total_s)
    gt_hist = _histogram(gt_times, total_s)
    # resize to a common length, as FDB does with its interpolation
    n = max(len(pred_hist), len(gt_hist))
    p = _resample(pred_hist, n)
    q = _resample(gt_hist, n)
    m = [(a + b) / 2 for a, b in zip(p, q)]
    kl_pm = _kl(p, m)
    kl_qm = _kl(q, m)
    div = 0.5 * kl_pm + 0.5 * kl_qm
    return math.sqrt(max(0.0, div))   # scipy.spatial.distance.jensenshannon


def _kl(p: list[float], q: list[float]) -> float:
    return sum(a * math.log2(a / b) for a, b in zip(p, q) if a > 0 and b > 0)


def _resample(h: list[float], n: int) -> list[float]:
    if len(h) == n:
        return h
    if len(h) == 1:
        return h * n
    out = []
    for i in range(n):
        pos = i * (len(h) - 1) / max(1, n - 1)
        lo = int(math.floor(pos))
        hi = min(len(h) - 1, lo + 1)
        frac = pos - lo
        out.append(h[lo] * (1 - frac) + h[hi] * frac)
    s = sum(out) or 1.0
    return [x / s for x in out]


# -------------------------------------------------- 4. user interruption -----

def user_interruption_tor_latency(results: Iterable[SimResult]) -> dict[str, Any]:
    """User interruption: TOR (higher better), yield latency (lower better).

    TOR = after the user barges in, does the model take the floor and answer?
    ``yield_s`` = inference compute time from the barge-in to the model
    stopping (hard-stop latency), ``response_s`` = compute from the barge-in to
    the model re-taking the floor.
    """
    results = [r for r in results if not r.error]
    tors: list[int] = []
    yields: list[float] = []
    responses: list[float] = []
    n_unscored = 0
    for res in results:
        # FDB computes TOR over EVERY sample in the subset (a model that never
        # takes the turn scores 0 and stays in the denominator). An interruption
        # scenario with no `interrupt_end` means the harness failed to complete
        # the barge-in, so it is counted as "did not take the floor" rather than
        # silently dropped — dropping it would flatter a model whose behaviour
        # (emitting <|user interruption|> itself) is what tripped the harness.
        if res.interrupt_end is None:
            n_unscored += 1
            tors.append(0)
            continue
        took = 1 if res.took_floor_after_interrupt and res.post_interrupt_text.strip() else 0
        tors.append(took)
        if res.response_compute_s:
            responses.append(res.response_compute_s)
        if res.yield_compute_s:
            yields.append(res.yield_compute_s)
    return {
        "tor": _mean([float(t) for t in tors]),
        "tor_sem": _sem([float(t) for t in tors]),
        "response_compute_s": _mean(responses),
        # stop latency, the interruption analogue of FDB's stop-latency column
        "yield_compute_s": _mean(yields),
        "n": len(tors),
        # should be 0: any non-zero value is a harness defect, not a model result
        "n_harness_unscored": n_unscored,
    }


def tag_decision_accuracy(results: Iterable[SimResult]) -> dict[str, Any]:
    """Turn-taking protocol accuracy at every decision point.

    At each user chunk the scenario knows whether that chunk *ended* the user's
    turn (``gold_ends_turn``); the model either emitted
    ``<|user finish speaking|>`` (take the floor) or did not. Counting those
    gives the protocol's precision/recall directly, which the FDB TORs only
    reflect indirectly — a model can post a good pause TOR simply by never
    speaking, and that shows up here as abysmal recall.

    ``silence`` decisions (the pause ticks) have no gold label: the correct
    answer there is always "do not take the floor", so they are scored as
    ``gold_ends_turn=False`` in the ``pause_*`` figures rather than mixed into
    the chunk-level accuracy.
    """
    tp = fp = fn = tn = 0
    pause_correct = pause_total = 0
    on_silence = silence_after_end = 0
    for res in results:
        if res.error:
            continue
        for d in res.decisions:
            if d["kind"] == "silence":
                if d["gold_ends_turn"]:
                    # a silence right after the user's turn ended. Taking the
                    # floor here is correct, and the FDB TOR columns already
                    # measure whether the model did; counting it in the
                    # chunk-level precision/recall as well would count two gold
                    # "opportunities" per turn and double-penalise a model that
                    # simply answers one tick later instead of on the final word.
                    silence_after_end += 1
                    on_silence += 1 if d["pred_finish"] else 0
                else:
                    # a genuine mid-turn pause: firing here is wrong
                    pause_total += 1
                    pause_correct += 0 if d["pred_finish"] else 1
                continue
            gold, pred = d["gold_ends_turn"], d["pred_finish"]
            if gold and pred:
                tp += 1
            elif gold and not pred:
                fn += 1
            elif pred and not gold:
                fp += 1
            else:
                tn += 1
    n = tp + fp + fn + tn
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    return {
        "n_decisions": n,
        "finish_precision": prec,
        "finish_recall": rec,
        "finish_f1": (2 * prec * rec / (prec + rec)
                      if (tp + fp) and (tp + fn) and (prec + rec) > 0 else float("nan")),
        "chunk_accuracy": (tp + tn) / n if n else float("nan"),
        "pause_stayed_silent": (pause_correct / pause_total) if pause_total else float("nan"),
        # of the turns that ended, how many got an immediate tag on the FINAL
        # chunk rather than one tick later on the trailing silence. A promptness
        # measure, not a correctness one -- correctness is smooth/interruption TOR.
        "floor_on_post_turn_silence": (on_silence / silence_after_end
                                       if silence_after_end else float("nan")),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def aggregate_quality_scores(per_sid: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Whole-dataset response quality (the shared 0-5 relevance rubric).

    Covers every scenario, not just the interruption subset. Scenarios where the
    model correctly stayed silent score 5 by rule and those where it should have
    spoken but did not score 0, so the mean reflects behaviour rather than only
    the subset the judge happened to see.
    """
    rated = [v["rating"] for v in per_sid.values()
             if v.get("rated") and isinstance(v.get("rating"), (int, float))]
    by_task: dict[str, list[float]] = {}
    for v in per_sid.values():
        if v.get("rated") and isinstance(v.get("rating"), (int, float)):
            by_task.setdefault(v.get("task", "?"), []).append(float(v["rating"]))
    return {
        "quality": _mean([float(r) for r in rated]),
        "quality_n": len(rated),
        "quality_n_all": len(per_sid),
        "quality_by_task": {k: _mean(v) for k, v in sorted(by_task.items())},
        "quality_from_judge": sum(
            1 for v in per_sid.values() if v.get("source") == "judge"),
    }


def aggregate_interruption_ratings(
    per_sid: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate the judge's 0-5 relevance scores (FDB's "GPT-4o" column).

    FDB reports the mean over samples where TOR == 1 (it only asks the judge
    about models that actually took the turn). We follow that, and also report
    the mean over *all* scenarios with 0 substituted for "never took the floor",
    so a model cannot score well by refusing to answer.
    """
    rated = [
        v["rating"] for v in per_sid.values()
        if v.get("rated") and isinstance(v.get("rating"), (int, float))
    ]
    all_sids = list(per_sid)
    with_zero = [
        (per_sid[s].get("rating", 0) if per_sid[s].get("rated") else 0)
        for s in all_sids
    ]
    return {
        "judge_rating": _mean([float(r) for r in rated]),
        "judge_rating_all": _mean([float(r) for r in with_zero]),
        "judge_n": len(rated),
        "judge_n_all": len(all_sids),
    }


# ------------------------------------------------------------ aggregated ----

@dataclass
class FDBRow:
    """One model's row of the FDB table."""

    model: str
    lang: str
    pause_synthetic_tor: float = float("nan")
    pause_candor_tor: float = float("nan")
    bc_tor: float = float("nan")
    bc_freq: float = float("nan")
    bc_jsd: float = float("nan")
    smooth_tor: float = float("nan")
    smooth_latency_s: float = float("nan")
    intr_tor: float = float("nan")
    intr_rating: float = float("nan")
    intr_latency_s: float = float("nan")
    ata: float = float("nan")

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


# A model must take the floor at least this often when the user's turn ends for
# its silence on the pause tasks to count as *restraint* rather than as an
# inability to speak at all.
DUPLEX_CAPABLE_SMOOTH_TOR = 0.5


def averaged_turn_taking_accuracy(row: dict[str, Any]) -> float:
    """FDB's "Averaged Turn-Taking Accuracy" column.

    FDB does not define this column in the released code. Ours is the mean of
    the four headline TOR numbers, each oriented so higher is better (pause TOR
    is inverted — staying silent through a pause is correct, so it contributes
    ``1 - tor``).

    Two guards keep the number honest, because otherwise *silence scores well*:
    a model that never speaks gets TOR 0 on both pause tasks and on the
    backchannel task for free, and would average 0.667 while being useless.

    1. an unmeasured interruption (the model never took the floor at all) is
       scored **0**, not dropped;
    2. the pause credit is only granted when the model demonstrably takes the
       floor elsewhere (``smooth_tor >= DUPLEX_CAPABLE_SMOOTH_TOR``); below that
       the score collapses to ``smooth_tor``.
    """
    smooth = row.get("smooth_tor", float("nan"))
    if math.isnan(smooth) or smooth < DUPLEX_CAPABLE_SMOOTH_TOR:
        return smooth
    intr = row.get("intr_tor", float("nan"))
    vals = [
        1.0 - row.get("pause_synthetic_tor", float("nan")),
        1.0 - row.get("pause_candor_tor", float("nan")),
        smooth,
        0.0 if math.isnan(intr) else intr,
    ]
    vals = [v for v in vals if not math.isnan(v)]
    return _mean(vals) if vals else float("nan")
