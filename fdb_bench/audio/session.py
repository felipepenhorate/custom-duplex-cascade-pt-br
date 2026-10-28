"""The duplex session: real user audio in, real assistant audio out.

The decision semantics are the text-domain benchmark's, unchanged -- the model is
called with the growing item history, returns a tag, and the tag's effect is the
same (wait / take the floor / backchannel / yield). What changes is that both
sides are now audio:

* user events come from :class:`~fdb_bench.audio.stt.AudioUserStream`, so the
  model sees ASR output (punctuation, casing and recognition errors included)
  instead of the benchmark's clean transcript, and the turn-end invitation comes
  from a real silence gate instead of an annotation;
* assistant turns are pocket-tts audio, so their durations -- and every
  behaviour gate that depends on duration -- are measured rather than estimated;
* barge-in truncates real playback, so FDB's stop-latency measure (computed by
  its own scripts from the rendered WAV) reflects an actual stop.

Latency, stated once: an assistant turn is placed at the moment its triggering
piece became available to the model *plus* the model's measured inference time,
and a barge-in stop lands at the speech onset *plus* the model's reaction time.
That is the pipeline's honest latency floor -- it excludes only network jitter
and audio-device buffering -- and it is why these numbers are larger than the
text-domain bench's compute-only latencies.

Barge-in policy is the duplex one (echo gate open), matching the text-domain
bench: the user's voice cuts whatever is playing, and the stop is timed from the
reaction. The shipped bridge instead drops microphone audio while the assistant
speaks (``DuplexCascade/server.py``, "echo gating"), which makes FDB's
stop-latency dimension unmeasurable on the product as shipped. That choice is
recorded here rather than silently adopted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .dataset import AudioScenario
from .stt import AudioUserStream

WAIT_TAGS = ("user_is_speaking", "user_is_thinking", "user_backchannel")
BACKCHANNEL_TAGS = ("system_backchannel",)
FLOOR_TAG = "user_finish_speaking"
BARGE_TAG = "user_interruption"

TTS_SR = 24000


@dataclass
class AudioTurn:
    """One audible assistant turn, timed on the session clock."""

    start: float
    text: str
    tag: str
    audio: np.ndarray            # float32 mono @ 24 kHz
    kind: str = "speech"         # speech | backchannel
    stop_at: float | None = None  # truncation point (barge-in / turn-close)
    compute_s: float = 0.0
    tts_compute_s: float = 0.0

    @property
    def natural_end(self) -> float:
        return self.start + self.audio.size / TTS_SR

    @property
    def end(self) -> float:
        return self.natural_end if self.stop_at is None else min(self.stop_at, self.natural_end)

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def n_words(self) -> int:
        return len(self.text.split())

    def audio_slice(self) -> np.ndarray:
        """The audio that actually played (truncated at a barge-in)."""
        n = int(round(self.duration_s * TTS_SR))
        return self.audio[: max(0, min(n, self.audio.size))]


@dataclass
class AudioResult:
    model: str
    subset: str
    task: str
    sid: str
    mode: str = "decision"
    duration_s: float = 0.0
    user_pieces: list[dict] = field(default_factory=list)
    user_speech_spans: list[list] = field(default_factory=list)
    gt_user_speech_s: float = 0.0
    turns: list[AudioTurn] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    decisions: list[dict] = field(default_factory=list)
    n_decisions: int = 0
    compute_s: float = 0.0
    max_decision_s: float = 0.0
    tts_compute_s: float = 0.0
    stt_s: float = 0.0
    stt_partial_s: float = 0.0
    stt_final_s: float = 0.0
    stt_calls: int = 0
    # metrics-side bookkeeping
    barge_ins: list[dict] = field(default_factory=list)
    yield_compute_s: float = 0.0
    asr_settle_latency_s: float | None = None
    floor_compute_s: float = 0.0
    utt_finalized: int = 0
    utt_skipped: dict = field(default_factory=dict)
    audio_beyond_input_s: float = 0.0
    tts_capped: int = 0          # turns whose tail could never be heard
    tts_capped_words: int = 0
    end_time: float = 0.0
    # interruption-task view
    paused_floor_take: float | None = None  # spoke during a mid-turn pause
    interrupt_span: tuple | None = None
    post_interrupt_text: str = ""
    took_floor_after_interrupt: bool = False
    error: str | None = None

    # --- audio-derived readings of the behaviour gates ------------------
    @property
    def speech_intervals(self) -> list[list[float]]:
        out: list[list[float]] = []
        for t in self.turns:
            if t.duration_s <= 0:
                continue
            if out and t.start - out[-1][1] <= 0.05:
                out[-1][1] = t.end
            else:
                out.append([t.start, t.end])
        return out

    def formal_turn_by_audio(self) -> bool:
        """FDB's rule applied to the TTS ground truth: >= 1 s or > 3 words."""
        return any(
            t.duration_s >= 1.0 or t.n_words > 3
            for t in self.turns if t.duration_s > 0
        )

    def spoke(self) -> bool:
        """Any audible sample at all -- the same test the renderer applies."""
        return any(t.duration_s > 0.0 for t in self.turns)


class AudioDuplexSession:
    """Runs one scenario against one adapter, producing real audio."""

    def __init__(
        self,
        adapter,
        stream: AudioUserStream,
        voice=None,
        *,
        mode: str = "decision",
        max_speech_s: float = 4.0,
    ) -> None:
        if mode not in ("decision", "prime"):
            raise ValueError(mode)
        self.adapter = adapter
        self.stream = stream
        self.voice = voice
        self.mode = mode
        self.max_speech_s = float(max_speech_s)
        self.tts = voice is not None
        self.max_speech_s = float(max_speech_s)

    # ------------------------------------------------------------------ run
    def run(self, scenario: AudioScenario, pcm24: np.ndarray) -> AudioResult:
        res = AudioResult(
            model=getattr(self.adapter, "name", "model"),
            subset=scenario.subset, task=scenario.task, sid=scenario.sid,
            mode=self.mode, duration_s=scenario.duration_s,
            interrupt_span=scenario.interrupt_span,
        )
        if scenario.gt_words:
            res.gt_user_speech_s = float(scenario.gt_words[-1]["timestamp"][1])

        items: list[dict] = (
            [{"role": "system", "text": self.adapter.system_prompt}]
            if getattr(self.adapter, "system_prompt", "") else []
        )

        speaking: AudioTurn | None = None
        pending_barge: float | None = None
        open_span: int | None = None

        def speak(text: str, tag: str, at: float, compute: float,
                  kind: str = "speech") -> None:
            """Queue a turn: playback starts at ``at``, or after whatever is
            still playing (the bridge serialises TTS, it does not mix)."""
            nonlocal speaking
            if not text.strip():
                res.tags.append(f"{tag}:empty")
                return
            full_words = len(text.split())
            if self.tts and self.max_speech_s:
                # Speech that cannot fit before the end of the input is clipped by
                # the render anyway (FDB's own outputs are padded to the input
                # length), so synthesising it is pure waste -- pocket-tts is the
                # slowest component in the loop. The budget is measured from the
                # end of whatever is *already queued*, not from this turn's start:
                # a turn that begins after the queue has run past the window can
                # never be heard at all, and synthesising it was the single
                # biggest cost in the first full sweep.
                base = speaking.natural_end if speaking is not None else max(at, 0.0)
                room = (scenario.duration_s - base) + self.max_speech_s
                budget = int(max(0.0, room) * 3.3)
                if full_words > budget:
                    words = text.split()
                    text = " ".join(words[:budget])
                    res.tts_capped += 1
                    res.tts_capped_words += full_words - budget
                    if budget == 0:
                        res.tags.append(f"{tag}:inaudible")
                        return
            chunk = self.voice.speak(text) if self.tts else None
            audio = chunk.audio if chunk is not None else np.zeros(0, dtype=np.float32)
            tts_c = chunk.compute_s if chunk is not None else 0.0
            res.tts_compute_s += tts_c
            base = speaking.natural_end if speaking is not None else 0.0
            speaking = AudioTurn(
                start=max(at, base), text=text, tag=tag, audio=audio,
                kind=kind, compute_s=compute, tts_compute_s=tts_c,
            )
            res.turns.append(speaking)

        def stop_turn(at: float) -> None:
            """Flush playback: everything queued stops at ``at``."""
            nonlocal speaking
            if speaking is not None:
                speaking.stop_at = max(speaking.start,
                                       min(at, speaking.natural_end))
            speaking = None

        # ---- the stream ---------------------------------------------------
        for ev in self.stream.events(pcm24):
            if ev.kind == "speech_start":
                open_span = len(res.user_speech_spans)
                res.user_speech_spans.append([ev.t_start, None])
                if speaking is not None and speaking.end > ev.t_start:
                    pending_barge = ev.t_start
                continue

            if ev.kind == "speech_end":
                if open_span is not None:
                    res.user_speech_spans[open_span][1] = ev.t_end
                    open_span = None
                continue

            # ---- a text piece: the model decides --------------------------
            if ev.kind == "silence":
                # the protocol's `<|no voice|>` user turn: the model is asked what
                # to do while the user stays quiet
                marker = getattr(self.adapter, "no_voice_marker", "<|no voice|>")
                res.user_pieces.append({
                    "kind": ev.kind, "text": marker,
                    "t_start": ev.t_start, "t_end": ev.t_end, "t_avail": ev.t_avail,
                    "ends_turn": False, "utt": ev.utt_index,
                })
                items.append({
                    "role": "user",
                    "text": marker,
                    "kind": "silence",
                    "ends_turn": False,
                    "system_speaking": bool(
                        speaking is not None and speaking.end > ev.t_avail
                    ),
                })
                t_dec = ev.t_avail
                tag, text = self.adapter.act(items)
                res.n_decisions += 1
                compute = float(getattr(self.adapter, "last_call_s", 0.0) or 0.0)
                res.compute_s += compute
                res.max_decision_s = max(res.max_decision_s, compute)
                res.tags.append(tag if text else f"{tag}:empty")
                res.decisions.append({
                    "kind": "silence", "t": round(t_dec, 3), "piece": marker,
                    "ends_turn": False, "tag": tag, "text": text[:160],
                    "compute_s": round(compute, 3),
                })
                if tag in WAIT_TAGS or tag == "no_tag":
                    continue
                if tag in BACKCHANNEL_TAGS:
                    speak(text, tag, t_dec + compute, compute, kind="backchannel")
                    continue
                if tag == BARGE_TAG:
                    stop_turn(t_dec + compute)
                    continue
                if tag == FLOOR_TAG:
                    if res.floor_compute_s == 0.0:
                        res.floor_compute_s = compute
                    if res.paused_floor_take is None:
                        res.paused_floor_take = t_dec
                    speak(text, tag, t_dec + compute, compute)
                    continue
                res.tags[-1] = f"<unknown:{tag}>"
                continue

            res.user_pieces.append({
                "kind": ev.kind, "text": ev.text,
                "t_start": ev.t_start, "t_end": ev.t_end, "t_avail": ev.t_avail,
                "ends_turn": bool(ev.ends_turn), "utt": ev.utt_index,
            })
            if self.mode == "prime" and ev.kind == "partial":
                continue  # the deployed bridge answers only at the turn end
            if ev.ends_turn:
                res.utt_finalized += 1

            items.append({
                "role": "user",
                "text": ev.text,
                "kind": ev.kind,
                "ends_turn": bool(ev.ends_turn),
                "system_speaking": bool(
                    speaking is not None and speaking.end > ev.t_avail
                ),
            })
            t_dec = ev.t_avail

            tag, text = self.adapter.act(items)
            res.n_decisions += 1
            compute = float(getattr(self.adapter, "last_call_s", 0.0) or 0.0)
            res.compute_s += compute
            res.max_decision_s = max(res.max_decision_s, compute)
            res.tags.append(tag if text else f"{tag}:empty")
            res.decisions.append({
                "kind": ev.kind, "t": round(t_dec, 3), "piece": ev.text[:120],
                "ends_turn": bool(ev.ends_turn), "tag": tag, "text": text[:160],
                "compute_s": round(compute, 3),
            })

            # ---- barge-in ------------------------------------------------
            # The user's voice reached the model while it was speaking, so
            # playback stops after the reaction time. This is the stop whose
            # latency FDB's scripts measure from the rendered audio.
            if pending_barge is not None:
                stop_at = pending_barge + compute
                turn = speaking
                if turn is not None and turn.end > pending_barge:
                    # was it actually audible when the user's voice landed?
                    # (a queued turn that had not started yet is flushed, not
                    # counted as an interrupted turn)
                    playing = turn.start <= pending_barge
                    played = max(0.0, pending_barge - turn.start) if playing else 0.0
                    natural = turn.natural_end - turn.start
                    settle = next(
                        (p["t_avail"] - pending_barge for p in res.user_pieces
                         if p["t_avail"] >= pending_barge), None)
                    stop_turn(stop_at)
                    if playing:
                        res.yield_compute_s = res.yield_compute_s or compute
                        res.asr_settle_latency_s = settle
                        res.barge_ins.append({
                            "speech_onset": round(pending_barge, 3),
                            "stop_at": round(max(stop_at, turn.start), 3),
                            "reaction_s": round(compute, 3),
                            "played_before_stop_s": round(played, 3),
                            "would_have_played_s": round(natural, 3),
                            "asr_settle_s": round(settle, 3) if settle is not None else None,
                        })
                    else:
                        res.barge_ins.append({
                            "speech_onset": round(pending_barge, 3),
                            "stop_at": round(max(stop_at, turn.start), 3),
                            "reaction_s": round(compute, 3),
                            "flush_before_playback": True,
                        })
                pending_barge = None

            # ---- the tag's effect ----------------------------------------
            if tag in WAIT_TAGS or tag == "no_tag":
                continue
            if tag in BACKCHANNEL_TAGS:
                speak(text, tag, t_dec + compute, compute, kind="backchannel")
                continue
            if tag == BARGE_TAG:
                stop_turn(t_dec + compute)
                continue
            if tag == FLOOR_TAG:
                if res.floor_compute_s == 0.0:
                    res.floor_compute_s = compute
                speak(text, tag, t_dec + compute, compute)
                if (scenario.task == "user_interruption"
                        and scenario.interrupt_span is not None
                        and t_dec >= scenario.interrupt_span[1]):
                    res.took_floor_after_interrupt = True
                continue
            res.tags[-1] = f"<unknown:{tag}>"

        if open_span is not None:
            res.user_speech_spans[open_span][1] = scenario.duration_s

        # what was still playing when the input ran out is clipped by the render
        for t in res.turns:
            if t.end > scenario.duration_s:
                res.audio_beyond_input_s += t.end - max(t.start, scenario.duration_s)

        if scenario.interrupt_span is not None:
            ref = scenario.interrupt_span[1]
            res.post_interrupt_text = " ".join(
                t.text for t in res.turns if t.start >= ref
            ).strip()

        res.stt_s = getattr(self.stream, "stt_s", 0.0)
        res.stt_partial_s = getattr(self.stream, "stt_partial_s", 0.0)
        res.stt_final_s = getattr(self.stream, "stt_final_s", 0.0)
        res.stt_calls = getattr(self.stream, "stt_calls", 0)

        skipped: dict = {}
        for u in self.stream.utterances:
            if u.skipped:
                skipped[u.skipped] = skipped.get(u.skipped, 0) + 1
        res.utt_skipped = skipped

        res.end_time = max(
            [scenario.duration_s]
            + [t.end for t in res.turns]
            + [s[1] or 0.0 for s in res.user_speech_spans]
        )
        return res
