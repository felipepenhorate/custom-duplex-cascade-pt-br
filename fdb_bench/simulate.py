"""Text-domain duplex simulator: replays a scenario against a model adapter.

This is the heart of the harness. It reproduces the runtime control loop of
``*/DuplexCascade/server.py`` (for the tag-emitting models) or
``mtp_like/policy/orchestrator.py`` (for the companion) over a virtual clock
and records a timeline the FDB metrics can be computed from.

Control loop
------------
For every user-side event on the stream (an ASR chunk, or a tick of silence):

1. append the event to the conversation as a ``user`` item,
2. ask the adapter what to do → ``(tag, text)``,
3. apply the tag's effect to the timeline,
4. advance the clock.

Tag semantics, matching ``server.py`` (``FINISH_TOKENS`` / ``THINK_TOKENS`` /
``INTERRUPT_TOKENS`` / ``SPEAK_BACKCHANNEL``) and
``orchestrator.py``'s dispatch table:

============================  ==========================================
tag                           effect
============================  ==========================================
``user_finish_speaking``      take the floor; ``text`` is spoken (TTS)
``user_interruption``         the user barged in: hard-stop immediately
``user_is_speaking``          user mid-utterance: keep waiting
``user_is_thinking``          user composing: keep waiting
``user_backchannel``          user acknowledged: keep talking
``system_backchannel``        we should emit a short backchannel
``no_tag``                    emit nothing, keep waiting
============================  ==========================================

The adapter owns its own micro-turn loop (feeding ``<|no voice|>`` until a
turn-closing tag), exactly like ``llm_tick_task`` does in the live bridge, and
appends its own assistant output to ``items``.

Two clocks
----------
``t`` is a single non-decreasing timeline. The user's events advance it by the
speech duration of their text; the model's speech advances it too. So when the
user barges in mid-answer, the interrupting chunk is timestamped *while the
model is still speaking* — the overlap that FDB's interruption task is built to
measure.

Latency
-------
``latency_s`` is the model's **real inference compute time** between the user's
turn ending and the model committing to the floor: the wall-clock of the
``act()`` calls in between (recorded by the adapter, not simulated). That is
what differs between a tag-emitting LLM and a separate policy companion, and it
is the honest text-domain analogue of FDB's acoustic response latency. It is
*not* end-to-end acoustic latency — that would additionally need ASR + TTS
timing. ``latency_chunks`` (how many user micro-turns the model sat through)
is recorded alongside as a clock-free cross-check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .clock import SpeechClock
from .scenarios import Scenario

# Silence tick granularity: the ASR word-gap timer in the live stack. Each tick
# is one opportunity for the model to (wrongly) take the floor during a pause.
SILENCE_TICK_S = 0.5
MAX_SILENCE_TICKS = 12

WAIT_TAGS = ("user_is_speaking", "user_is_thinking", "user_backchannel")
BACKCHANNEL_TAGS = ("system_backchannel",)
NO_TAG = "no_tag"


@dataclass
class StreamEvent:
    """One timed event on the simulated timeline (user side)."""

    t_start: float
    t_end: float
    kind: str                      # "chunk" | "silence"
    text: str = ""
    ends_turn: bool = False        # ASR/VAD finalized the user turn here
    is_backchannel: bool = False   # the user's own short backchannel
    index: int = -1


@dataclass
class ModelTurn:
    """One contiguous stretch of assistant speech."""

    start: float
    end: float
    text: str
    tag: str
    kind: str = "speech"           # "speech" | "backchannel"


@dataclass
class SimResult:
    """Everything the metrics need from one scenario run."""

    sid: str
    task: str
    lang: str
    model: str
    user_events: list[StreamEvent] = field(default_factory=list)
    model_turns: list[ModelTurn] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    # backchannel ground truth, resolved to seconds
    gt_backchannel_times: list[float] = field(default_factory=list)
    # The judge's inputs, copied straight from the scenario so the judging pass
    # never has to rebuild it. Rebuilding is what broke this once: the scenario
    # generator consumes RNG draws *while chunking*, so a rebuild with a
    # different chunker produces different text for every later scenario, and the
    # judge ended up scoring a cooking question against a programming answer.
    context: str = ""
    interrupt: str = ""
    # interruption bookkeeping
    # ``interrupt_time`` is the barge-in ONSET (the user's first interrupting
    # chunk) — that is what stop latency and the overlap diagnostics use.
    # ``interrupt_end`` is the END of the interrupting turn, which is FDB's
    # ``input_end_time``: the reference point for the interruption TOR and the
    # response latency. FDB measures against the latter because a model that
    # starts answering halfway through the interruption (before the user has
    # finished) has not actually handled the interruption.
    interrupt_time: float | None = None
    interrupt_end: float | None = None
    post_interrupt_text: str = ""
    took_floor_after_interrupt: bool = False
    yield_time: float | None = None          # when the model stopped, if it did
    response_compute_s: float = 0.0          # compute to answer the interruption
    floor_compute_s: float = 0.0             # compute from turn-end to taking the floor
    model_words_at_interrupt: int = 0        # diagnostics: was the answer long?
    model_speaking_at_interrupt: bool = False
    yield_compute_s: float = 0.0             # stop latency: barge-in -> hard-stop
    # diagnostics
    error: str | None = None
    n_decisions: int = 0
    n_model_calls: int = 0
    compute_s: float = 0.0
    trace: list[dict[str, Any]] = field(default_factory=list)

    end_time: float = 0.0          # max end over both sides; set at the end of a run
    # Per-decision turn-taking records: the scenario's own gold label for each
    # decision point (did this user chunk end the user's turn?) against what the
    # model chose. This is the harness's direct measure of the protocol, and it
    # is what makes the prompt-candidate comparison in calibrate_prompts.py a
    # measurement rather than a judgement call.
    decisions: list[dict[str, Any]] = field(default_factory=list)
    # the model emitted <|user finish speaking|> but produced no speech at all.
    # These turns have zero duration and are excluded from the text metrics, so
    # without a counter they vanish rather than showing up as the failure they
    # are (the model takes the floor and says nothing).
    empty_floor_takings: int = 0

    def user_turn_end(self) -> float | None:
        """End of the user's first completed turn (the FDB ``input_end_time``)."""
        for e in self.user_events:
            if e.ends_turn:
                return e.t_end
        return None


class Adapter(Protocol):
    """What the simulator needs from a model under test."""

    name: str
    system_prompt: str        # system item prepended to every scenario; "" = none
    no_voice_marker: str      # what a silence tick appends as the user item
    last_call_s: float        # wall-clock of the most recent act() call

    def reset(self) -> None: ...

    def default_backchannel(self) -> str:
        """The short listener backchannel spoken for ``system_backchannel``."""
        ...

    def act(self, items: list[dict]) -> tuple[str, str]:
        """Decide what to do at this point in the stream.

        ``items`` is the duplex context in the exact shape the live bridges
        pass to their models (``{"role", "text"}`` entries). Implementations
        must **append their own assistant output to ``items``** (they may append
        several items — that is the micro-turn loop) and return
        ``(tag_key, spoken_text)``.
        """
        ...


class DuplexSimulator:
    """Drives an :class:`Adapter` over a :class:`Scenario`'s user events."""

    def __init__(self, adapter: Adapter, clock: SpeechClock | None = None) -> None:
        self.adapter = adapter
        self.clock = clock or SpeechClock()

    # -- public API --------------------------------------------------------
    def run(self, scenario: Scenario) -> SimResult:
        self.adapter.reset()
        res = SimResult(
            sid=scenario.sid, task=scenario.task, lang=scenario.lang,
            model=getattr(self.adapter, "name", "?"),
            context=scenario.context, interrupt=scenario.interrupt,
        )
        try:
            self._run_inner(scenario, res)
        except Exception as e:  # one bad scenario must not kill a sweep
            res.error = f"{type(e).__name__}: {e}"
        return res

    # -- internals ---------------------------------------------------------
    def _run_inner(self, scenario: Scenario, res: SimResult) -> None:
        gt = set(scenario.gt_backchannel_times)
        # an empty system_prompt means the training distribution: no system item
        items: list[dict] = (
            [{"role": "system", "text": self.adapter.system_prompt}]
            if self.adapter.system_prompt else []
        )
        t = 0.0
        speaking = False          # the assistant currently has the floor
        cur: ModelTurn | None = None
        model_words = 0           # drives the interruption injection point
        context_done = False      # the user's first turn finished
        injected = False
        answered_context = False     # the model took the floor on the context turn
        # interruption accounting
        interrupt_compute_start: float | None = None
        compute_at_turn_end: float = 0.0
        compute_all_at_interrupt: float = 0.0
        compute_at_interrupt_end: float = 0.0

        def close_speech(at: float) -> None:
            """Stop the assistant at ``at``.

            ``min`` matters: when the user barges in mid-utterance the model's
            already-scheduled audio end is *truncated* to the barge-in point
            (that truncation is what FDB's stop-latency measures), whereas if the
            model had already finished, the recorded end is left alone.
            """
            nonlocal speaking, cur, model_words
            if cur is not None:
                cur.end = min(cur.end, at)
                res.model_turns.append(cur)
                cur = None
            speaking = False

        def open_speech(at: float, text: str, tag: str, kind: str = "speech") -> None:
            nonlocal speaking, cur
            close_speech(at)
            cur = ModelTurn(start=at, end=at + self.clock.duration(text),
                            text=text, tag=tag, kind=kind)
            speaking = True

        def do(tag: str, text: str, at: float) -> None:
            """Apply one tag's effect at time ``at``."""
            nonlocal model_words, interrupt_compute_start, injected, answered_context
            nonlocal compute_all_at_interrupt
            res.tags.append(tag)

            if tag in WAIT_TAGS or tag == NO_TAG:
                return

            if tag in BACKCHANNEL_TAGS:
                # the model should emit a SHORT listener backchannel and stay
                # quiet; it is a real model turn, but a short one
                bc = text or self.adapter.default_backchannel()
                open_speech(at, bc, tag, kind="backchannel")
                close_speech(at + self.clock.duration(bc))
                return

            if tag == "user_interruption":
                if scenario.task == "interruption_synthetic":
                    # NOTE: do NOT set ``interrupt_time`` here. That field is the
                    # *harness's* barge-in anchor, written when the interrupting
                    # turn actually starts. Setting it from the model's own tag
                    # made ``interrupt_end`` unreachable — the later injection
                    # block only fires while ``interrupt_time is None`` — so
                    # exactly the scenarios where the model acknowledged the
                    # interruption were dropped from TOR (20/200 for distill pt).
                    # A model that never yields and one that yields both need to
                    # stay in the denominator.
                    if res.yield_compute_s == 0.0:
                        res.yield_compute_s = res.compute_s - compute_all_at_interrupt
                    res.yield_time = at
                    interrupt_compute_start = res.compute_s
                close_speech(at)
                return

            if tag == "user_finish_speaking":
                if not text.strip():
                    res.empty_floor_takings += 1
                if context_done:
                    answered_context = True
                if res.floor_compute_s == 0.0 and res.compute_s > compute_at_turn_end:
                    res.floor_compute_s = res.compute_s - compute_at_turn_end
                if (scenario.task == "interruption_synthetic"
                        and res.interrupt_end is not None):
                    res.took_floor_after_interrupt = True
                    if interrupt_compute_start is not None:
                        res.response_compute_s = res.compute_s - interrupt_compute_start
                        interrupt_compute_start = None
                open_speech(at, text, tag)
                model_words += len(text.split())
                return

            # unknown tag: treat as "keep waiting" but keep it visible
            res.tags[-1] = f"<unknown:{tag}>"

        for idx, ev in enumerate(scenario.events):
            if ev.kind == "chunk":
                dur = self.clock.duration(ev.text)
                ev_start = t
                # NOTE: ``ends_turn`` rides along for test/stub adapters only.
                # The real adapters never see it: ``_to_history`` builds the
                # prompt from role+text alone, which is the whole point — the
                # model has to *infer* the turn end from the text, exactly as
                # it does live, where the ASR/VAD boundary is never shown to it.
                # tests/test_parity.py asserts the prompt is byte-identical
                # with and without this field.
                items.append({"role": "user", "text": ev.text, "ends_turn": ev.ends_turn})
                res.user_events.append(StreamEvent(
                    t_start=ev_start, t_end=t + dur, kind="chunk", text=ev.text,
                    ends_turn=ev.ends_turn, is_backchannel=ev.is_backchannel, index=idx,
                ))
                if idx in gt:
                    res.gt_backchannel_times.append(t + dur)
                t = max(t, t + dur)

                # Does this chunk barge in? FDB times the interrupting audio to
                # land inside the model's answer, so the trigger is "the model
                # has answered the context turn and the interrupting turn has
                # now started" — *not* "the model happens to still be speaking".
                # Requiring a live speech span meant short answers never got
                # interrupted at all (the model's audio had already ended), and
                # the task silently degraded to a plain smooth-turn scenario.
                # Gated on the *scenario* reaching the interrupting turn, not on
                # the model having answered the context: a model that never takes
                # the floor must still be scored on the interruption (TOR 0), not
                # skipped. Skipping it excused exactly the worst behaviour.
                inject_here = (
                    scenario.task == "interruption_synthetic"
                    and not injected and context_done
                )
                if ev.ends_turn:
                    compute_at_turn_end = res.compute_s
                    if not context_done:
                        context_done = True
                    elif (scenario.task == "interruption_synthetic" and injected
                          and res.interrupt_end is None):
                        # the interrupting turn just finished: this is FDB's
                        # input_end_time, the reference for TOR and latency
                        res.interrupt_end = t
                        compute_at_interrupt_end = res.compute_s
                        interrupt_compute_start = res.compute_s

                # Snapshot the barge-in state BEFORE anything mutates it: this
                # is the moment the user's audio lands on top of the model's,
                # and ``close_speech`` below is the cut that follows from it.
                if inject_here:
                    injected = True
                    if res.interrupt_time is None:
                        res.interrupt_time = t
                    res.model_words_at_interrupt = model_words
                    res.model_speaking_at_interrupt = speaking
                    compute_all_at_interrupt = res.compute_s

                # the user's voice always cuts whatever is playing (this is the
                # truncation FDB's stop-latency measures)
                close_speech(t)

                tag, text = self._decide(
                    items, res, gold_ends_turn=ev.ends_turn, kind="chunk")
                do(tag, text, t)

                if inject_here and tag != "user_interruption":
                    # the model did not acknowledge the barge-in with the
                    # dedicated tag; close_speech above still cut its audio, so
                    # it did yield — record when, and the latency to it
                    res.yield_time = res.yield_time or t
                    if res.yield_compute_s == 0.0:
                        res.yield_compute_s = max(0.0, res.compute_s - compute_all_at_interrupt)

                # Place the interrupting turn INSIDE the model's utterance, the
                # way FDB times its interruption audio. Without this the user's
                # next chunk is timestamped right after the previous user chunk,
                # which is usually *before* the model has even produced its
                # answer — so the barge-in never actually overlapped anything and
                # the interruption task silently became "answer a second
                # question", losing the yield behaviour entirely.
                if (scenario.task == "interruption_synthetic" and not injected
                        and answered_context and cur is not None):
                    offset_s = (scenario.interrupt_after_model_words
                                / max(self.clock.words_per_second, 1e-6))
                    t = max(t, min(cur.start + offset_s, cur.end))

            else:  # silence
                dur = float(ev.duration or SILENCE_TICK_S)
                ticks = min(max(1, int(round(dur / SILENCE_TICK_S))), MAX_SILENCE_TICKS)
                for _ in range(ticks):
                    items.append({"role": "user", "text": self.adapter.no_voice_marker,
                                  "ends_turn": False})
                    t += SILENCE_TICK_S
                    res.user_events.append(StreamEvent(
                        t_start=t - SILENCE_TICK_S, t_end=t, kind="silence", index=idx,
                    ))
                    tag, text = self._decide(
                        items, res, kind="silence",
                        gold_ends_turn=ev.after_turn_end)
                    do(tag, text, t)
                    if tag == "user_finish_speaking":
                        break  # it took the floor during a pause: that is the TOR

        close_speech(t)

        # post-interruption response text = everything said after the
        # interruption ENDED (FDB's out_after_interrupt_text)
        ref = res.interrupt_end if res.interrupt_end is not None else res.interrupt_time
        if ref is not None:
            res.post_interrupt_text = " ".join(
                m.text for m in res.model_turns if m.start >= ref
            ).strip()

        res.end_time = max(
            [e.t_end for e in res.user_events] + [m.end for m in res.model_turns] or [0.0]
        )

    def _decide(
        self, items: list[dict], res: SimResult, *,
        gold_ends_turn: bool = False, kind: str = "chunk",
    ) -> tuple[str, str]:
        tag, text = self.adapter.act(items)
        res.n_decisions += 1
        # adapters report the wall-clock of the whole decision (including their
        # internal micro-turn loop / companion+big-model round trip) in
        # ``last_call_s``; that real compute is the latency signal.
        res.compute_s += float(getattr(self.adapter, "last_call_s", 0.0) or 0.0)
        res.trace.append({"tag": tag, "text": text[:90]})
        res.decisions.append({
            "gold_ends_turn": bool(gold_ends_turn),
            "kind": kind,
            "tag": tag,
            "pred_finish": tag == "user_finish_speaking",
        })
        return tag, text
