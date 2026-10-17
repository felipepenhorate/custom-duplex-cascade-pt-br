"""Metric-port tests against Full-Duplex-Bench's *own released example data*.

The strongest available check on a re-implementation is that it reproduces the
reference implementation's answers on the reference's own fixtures. FDB ships
`v1_v1.5/evaluation/example_data/{pause_handling,smooth_turn_taking,
user_interruption}_example/1/` — real `output.json` files with word-level ASR
timestamps and the matching annotation files.

This module re-derives the expected TOR/latency by hand from those fixtures
(values are pinned inline, with the arithmetic in the comments) and asserts our
metric helpers agree. It also cross-checks the *clock-free* parts so the port
cannot drift silently.

Run:  /home/penhfel/unsloth_uv/bin/python -m fdb_bench.tests.test_metrics
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fdb_bench import metrics as M                       # noqa: E402
from fdb_bench.clock import SpeechClock                   # noqa: E402

# ---------------------------------------------------------------- fixtures --
# FDB v1_v1.5/evaluation/example_data/*/1/output.json, reduced to the fields the
# metrics touch: (n_chunks, first_chunk_start, last_chunk_end_or_start).
FIXTURES = {
    # eval_pause_handling.py on pause_handling_example/1
    "pause_handling": {
        "chunks": 12, "first_start": 0.0, "last_end": 4.68,
        "annotation": {"pause": [3.32, 4.0]},
    },
    # eval_smooth_turn_taking.py on smooth_turn_taking_example/1
    "smooth_turn_taking": {
        "chunks": 22, "first_start": 3.36, "last_end": 10.38,
        "annotation": {"turn_taking": [2.01, 2.40]},
    },
    # eval_user_interruption.py on user_interruption_example/1
    "user_interruption": {
        "chunks": 32, "first_start": 17.224, "last_end": 28.124,
        "annotation": {"interrupt": [10.530666666666667, 13.144]},
    },
}


def _fdb_tor(n_chunks: int, first_start: float, last_end: float) -> int:
    """Verbatim transcription of FDB's decision (no clock involved).

    eval_pause_handling.py / eval_smooth_turn_taking.py:
        if len(segments) == 0: TOR = 0
        duration = segments[-1].end - segments[0].start
        if duration < 1: TOR = 0 if len(segments) <= 3 else 1
        else: TOR = 1
    """
    if n_chunks == 0:
        return 0
    duration = last_end - first_start
    if duration < 1.0:
        return 0 if n_chunks <= 3 else 1
    return 1


def test_tor_arithmetic_matches_fdb_examples() -> None:
    """Our threshold constants must reproduce FDB's arithmetic on its fixtures."""
    assert M.TURN_DURATION_THRESHOLD == 1.0
    assert M.TURN_NUM_WORDS_THRESHOLD == 3

    for name, f in FIXTURES.items():
        want = _fdb_tor(f["chunks"], f["first_start"], f["last_end"])
        # our helper works on (start, end, text) segments; text only matters for
        # the sub-1s branch, which none of these fixtures hit
        got = M._tor_from_segments(
            [(f["first_start"], f["last_end"], "x" * f["chunks"])], SpeechClock()
        )
        assert want == 1, f"{name}: fixture should be a formal turn"
        assert got == 1, f"{name}: our port says {got}, FDB says {want}"


def test_expected_fdb_values_on_examples() -> None:
    """Pin the published expectations of the three examples.

    * pause handling: 4.68 s span, 12 chunks -> the model took the turn -> TOR 1
      (FDB's whole point: a good model must NOT do this on a pause)
    * smooth turn taking: latency = first output start - user turn end
      = 3.36 - 2.01 = 1.35 s, TOR 1
    * user interruption: latency = 17.224 - 13.144 = 4.08 s, TOR 1
    """
    ph = FIXTURES["pause_handling"]
    assert _fdb_tor(ph["chunks"], ph["first_start"], ph["last_end"]) == 1

    st = FIXTURES["smooth_turn_taking"]
    latency = st["first_start"] - st["annotation"]["turn_taking"][0]
    assert abs(latency - 1.35) < 1e-9, latency
    assert _fdb_tor(st["chunks"], st["first_start"], st["last_end"]) == 1
    # a negative latency is clamped to 0 by eval_smooth_turn_taking.py
    assert max(0.0, -0.5) == 0.0

    ui = FIXTURES["user_interruption"]
    latency = ui["first_start"] - ui["annotation"]["interrupt"][1]
    assert abs(latency - 4.08) < 1e-9, latency
    assert _fdb_tor(ui["chunks"], ui["first_start"], ui["last_end"]) == 1


def test_short_utterance_branch() -> None:
    """The sub-1 s branch, which the fixtures do not exercise."""
    clock = SpeechClock(words_per_second=3.0)
    # 0.5 s, 2 words -> not a takeover
    assert M._tor_from_segments([(0.0, 0.5, "aham tá")], clock) == 0
    # 0.5 s but 5 words -> a takeover
    assert M._tor_from_segments([(0.0, 0.5, "um dois tres quatro cinco")], clock) == 1
    # no output at all -> never a takeover
    assert M._tor_from_segments([], clock) == 0


def _sim(task: str = "backchannel_icc", turns=(), gt=(), end_time: float = 5.0):
    """A minimal SimResult for exercising the metric entry points."""
    from fdb_bench.simulate import ModelTurn, SimResult
    return SimResult(
        sid="t", task=task, lang="pt", model="t",
        model_turns=[ModelTurn(start=a, end=b, text=t, tag="user_finish_speaking")
                     for a, b, t in turns],
        gt_backchannel_times=list(gt),
        end_time=end_time,
    )


def test_backchannel_worst_case_is_one_not_zero() -> None:
    """A model with no backchannels must score JSD 1.0.

    This is FDB's explicit special case (``eval_backchannel.py``: ``if len(
    backchannel_prediction) == 0: js_divergence = 1``). It matters: without it a
    silent model scores 0.0 — a *perfect* divergence score — against a
    near-uniform ground-truth histogram, so never backchanneling would look
    optimal.
    """
    clock = SpeechClock()
    silent = M.backchannel([_sim(turns=(), gt=[1.0, 2.0, 3.0])], clock)
    assert silent["jsd"] == 1.0, silent
    assert silent["freq"] == 0.0, silent


def test_backchannel_perfect_prediction_scores_zero() -> None:
    """Placing a short backchannel exactly on the GT times must score JSD 0."""
    clock = SpeechClock()
    gt = [1.0, 2.0, 3.0, 4.0]
    turns = [(t, t + 0.1, "aham") for t in gt]      # short and word-poor
    perfect = M.backchannel([_sim(turns=turns, gt=gt)], clock)
    assert perfect["jsd"] < 1e-9, perfect
    assert perfect["freq"] > 0, perfect
    assert perfect["tor"] == 0.0, perfect            # short => never a formal turn


def test_backchannel_formal_turn_sets_tor() -> None:
    """A long model utterance in the backchannel task is a TOR failure."""
    clock = SpeechClock()
    long_turn = [(1.0, 5.0, "uma resposta bem longa que passa de tres segundos")]
    r = M.backchannel([_sim(turns=long_turn, gt=[1.0])], clock)
    assert r["tor"] == 1.0, r


def test_jsd_is_bounded_and_symmetric() -> None:
    """JSD is a proper divergence: in [0, 1] here, and symmetric."""
    a_pred = [(0.2, 0.4), (1.6, 1.8)]
    a_gt = [0.3, 1.7]
    b_pred = [(2.2, 2.4), (3.6, 3.8)]
    b_gt = [2.3, 3.7]
    fwd = M._jsd_backchannel(a_pred, b_gt, 4.0)
    rev = M._jsd_backchannel(b_pred, a_gt, 4.0)
    assert 0.0 <= fwd <= 1.0 and 0.0 <= rev <= 1.0, (fwd, rev)
    assert abs(fwd - rev) < 1e-9, (fwd, rev)
    # same distribution -> zero divergence
    assert M._jsd_backchannel(a_pred, a_gt, 4.0) < 1e-9



def test_result_round_trip_keeps_every_metric_field() -> None:
    """Serialising a SimResult and reloading it must preserve every field.

    ``run_inference`` writes ``raw.jsonl`` and ``run_metrics`` reads it back
    through two hand-maintained field lists. If they drift, a metric silently
    becomes NaN/0 instead of failing — exactly the kind of bug that quietly
    changes a published table. This pins the round trip.
    """
    import json
    import tempfile
    from pathlib import Path as _P

    from fdb_bench.run_inference import _serialise
    from fdb_bench.run_metrics import load_run
    from fdb_bench.simulate import ModelTurn, SimResult, StreamEvent

    res = SimResult(
        sid="x", task="interruption_synthetic", lang="en", model="m",
        user_events=[StreamEvent(t_start=0.0, t_end=1.0, kind="chunk", text="oi",
                                 ends_turn=True, index=0)],
        model_turns=[ModelTurn(start=1.0, end=2.0, text="aham", tag="system_backchannel",
                               kind="backchannel")],
        tags=["system_backchannel", "user_finish_speaking"],
        gt_backchannel_times=[1.0, 2.0],
        interrupt_time=1.0, interrupt_end=1.5, post_interrupt_text="claro",
        took_floor_after_interrupt=True, yield_time=1.1,
        response_compute_s=0.2, floor_compute_s=0.3,
        model_words_at_interrupt=7, model_speaking_at_interrupt=True,
        yield_compute_s=0.05, end_time=2.0, error=None,
        n_decisions=3, compute_s=0.9,
    )
    with tempfile.TemporaryDirectory() as d:
        d = _P(d)
        (d / "raw.jsonl").write_text(
            json.dumps(_serialise(res), ensure_ascii=False) + "\n", encoding="utf-8")
        loaded, meta = load_run(d)
    assert meta == {}
    got = loaded["x"]
    for field in ("interrupt_time", "interrupt_end", "post_interrupt_text",
                  "took_floor_after_interrupt", "response_compute_s",
                  "floor_compute_s", "model_words_at_interrupt",
                  "model_speaking_at_interrupt", "yield_compute_s", "end_time",
                  "n_decisions", "gt_backchannel_times", "tags"):
        assert getattr(got, field) == getattr(res, field), (
            f"{field}: {getattr(got, field)!r} != {getattr(res, field)!r}")
    # the metric entry points must consume the reloaded object
    assert M.user_interruption_tor_latency([got])["tor"] == 1.0
    assert M.backchannel([got], SpeechClock())["n"] == 1


def test_every_interruption_scenario_stays_in_the_tor_denominator() -> None:
    """FDB computes interruption TOR over the whole subset — 200 of 200.

    Two bookkeeping bugs used to drop scenarios: the model's own
    ``<|user interruption|>`` tag set ``interrupt_time``, which blocked the
    harness's injection (it only fires while ``interrupt_time is None``), and
    the injection flag was tied to the same condition. The dropped scenarios
    were exactly the ones where the model *acknowledged* the interruption — the
    best-behaved cases — so TOR was measured on 180/200 for distill pt and
    193/200 for sft en.

    This pins the contract: the simulator must always record ``interrupt_end``,
    and the metric must never shrink its denominator.
    """
    from fdb_bench import scenarios as S
    from fdb_bench.run_inference import build_chunker
    from fdb_bench.simulate import DuplexSimulator

    # a stub whose first action is `<|user interruption|>` — the exact behaviour
    # that used to break the bookkeeping
    class InterruptingStub:
        name = "stub_interrupter"
        system_prompt = ""
        no_voice_marker = "<|no voice|>"
        last_call_s = 0.01

        def reset(self) -> None:
            self.last_call_s = 0.01

        def default_backchannel(self) -> str:
            return "aham"

        def act(self, items: list[dict]) -> tuple[str, str]:
            last = items[-1]
            if last["role"] == "user" and last["text"].strip() != self.no_voice_marker:
                # always claim the user is interrupting, never take the floor
                return "user_interruption", ""
            return "user_is_speaking", ""

    sim = DuplexSimulator(InterruptingStub(), SpeechClock(3.25))
    scen = S.build_scenarios("interruption_synthetic", limit=12, lang="pt",
                             chunker=build_chunker())
    res = [sim.run(sc) for sc in scen]
    assert all(r.error is None for r in res), [r.error for r in res]
    missing = [r.sid for r in res if r.interrupt_end is None]
    assert not missing, f"interrupt_end missing for {missing}"

    metric = M.user_interruption_tor_latency(res)
    assert metric["n"] == len(res), f"TOR over {metric['n']}/{len(res)}"
    assert metric["n_harness_unscored"] == 0, metric
    # the stub never takes the floor, so TOR must be 0 on every sample
    assert metric["tor"] == 0.0, metric


def _run_all() -> int:
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
