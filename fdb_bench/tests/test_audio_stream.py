"""Tests for the audio user stream: piece shape, turn ends, silence ticks.

No whisper: a stub ASR emits one word per 0.7 s of audio, which is what the
stream's bookkeeping has to handle. The stream's *piece-size* behaviour is the
property that has to hold for the audio bench to mean anything -- the protocol
was trained on 1-7-token user pieces -- so it is asserted here rather than
eyeballed from a run.

Run:  python -m fdb_bench.tests.test_audio_stream
"""

from __future__ import annotations

import numpy as np

from fdb_bench.audio.stt import (
    KIND_FINAL, KIND_PARTIAL, KIND_SILENCE, AudioUserStream, energy_spans,
)

SR = 24000
WORD_S = 0.7


class _Word:
    def __init__(self, word, start, end):
        self.word, self.start, self.end = word, start, end


class _Seg:
    def __init__(self, words):
        self.words = words
        self.text = " ".join(w.word for w in words)
        self.start = words[0].start if words else 0.0
        self.end = words[-1].end if words else 0.0


class StubWhisper:
    """One word per 0.7 s of audio, up to ``n_words``; timing is exact."""

    def __init__(self, n_words: int = 40, revision: float = 0.0):
        self.n_words = n_words
        self.revision = revision
        self.calls = 0

    def transcribe(self, pcm, **kw):  # noqa: ARG002
        self.calls += 1
        dur = pcm.size / 16000.0
        words = []
        n = min(self.n_words, int(dur / WORD_S))
        for i in range(n):
            # beam-5 style revision: beam_size is passed through kwargs
            shift = self.revision if kw.get("beam_size", 1) == 1 else 0.0
            words.append(_Word(f"w{i}", i * WORD_S + shift, (i + 1) * WORD_S + shift))
        return [_Seg(words)], None


def _pcm(spans: list[tuple[float, float]], total_s: float, amp: float = 0.2) -> np.ndarray:
    """Speech spans at ``amp``, silence at a low noise floor elsewhere."""
    rng = np.random.default_rng(0)
    out = rng.normal(0.0, 0.0002, int(total_s * SR)).astype(np.float32)
    for s, e in spans:
        a, b = int(s * SR), int(e * SR)
        t = np.arange(b - a) / SR
        out[a:b] = (amp * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
    return out


def _stream(**kw) -> AudioUserStream:
    kw.setdefault("vad", "energy")
    kw.setdefault("rms_threshold", 0.002)
    return AudioUserStream(StubWhisper(), language="en", **kw)


def test_pieces_are_training_sized_and_lossless() -> None:
    pcm = _pcm([(1.0, 20.0)], 22.0)
    st = _stream()
    pieces = [p for p in st.events(pcm) if p.text]
    assert pieces, "no pieces emitted"
    sizes = [len(p.text.split()) for p in pieces]
    assert max(sizes) <= 7, f"piece larger than the training maximum: {sizes}"
    assert all(s >= 1 for s in sizes), sizes
    # every stub word appears exactly once, in order
    words = [w for p in pieces for w in p.text.split()]
    assert words == [f"w{i}" for i in range(len(words))], words[:12]
    print(f"  piece sizes: {sizes[:12]} ... max={max(sizes)} n={len(sizes)}")


def test_turn_end_after_the_silence_gap() -> None:
    pcm = _pcm([(1.0, 6.0)], 8.0)
    st = _stream(silence_s=0.6)
    pieces = [p for p in st.events(pcm) if p.text]
    finals = [p for p in pieces if p.ends_turn]
    assert finals, "no turn end emitted after the user stopped"
    assert finals[-1].kind == KIND_FINAL
    assert abs(finals[-1].t_avail - 6.6) < 0.25, finals[-1].t_avail
    # turn ends are not repeated while the user stays silent
    assert len(finals) == 1, [(p.t_avail, p.text) for p in finals]
    print(f"  turn end at {finals[-1].t_avail:.2f}s (speech ended 6.0s)")


def test_silence_ticks_start_after_the_turn_end() -> None:
    pcm = _pcm([(1.0, 6.0)], 12.0)
    st = _stream(silence_s=0.6, silence_tick_s=0.5)
    events = list(st.events(pcm))
    ticks = [e for e in events if e.kind == KIND_SILENCE]
    first_final = next(e for e in events if e.ends_turn)
    assert ticks, "no silence ticks"
    assert ticks[0].t_avail > first_final.t_avail, (
        f"tick {ticks[0].t_avail} came before the turn end {first_final.t_avail}")
    gaps = [b.t_avail - a.t_avail for a, b in zip(ticks, ticks[1:])]
    assert all(g >= 0.49 for g in gaps), gaps
    # no ticks before the first turn end
    assert not [e for e in events if e.kind == KIND_SILENCE
                and e.t_avail < first_final.t_avail]
    print(f"  ticks at {[round(t.t_avail, 2) for t in ticks[:5]]} (turn end {first_final.t_avail:.2f})")


def test_stream_end_is_an_eos() -> None:
    # audio stops mid-speech: the deployed service finalizes on an explicit Eos,
    # so the stream end must still produce a turn end
    pcm = _pcm([(1.0, 9.0)], 9.0)
    st = _stream()
    pieces = [p for p in st.events(pcm) if p.text]
    finals = [p for p in pieces if p.ends_turn]
    assert finals, "stream end produced no turn end"
    assert finals[-1].kind == KIND_FINAL
    words = sum(len(p.text.split()) for p in pieces)
    assert words >= 8, words
    print(f"  stream-end final: {finals[-1].text[:60]!r}")


def test_commit_window_delays_the_words() -> None:
    """The deployed 2.0 s commit window is a UI setting, not a duplex one.

    It exists so the browser's partial-text bubble never shows a word twice. The
    deployed bridge never feeds partials to the LLM, so nothing in the product
    depends on it; a duplex LLM does. The measurable cost is *lag*: with the
    deployed window the model is told what the user said about two seconds after
    they said it, and on short utterances it is told nothing at all until the
    turn ends.
    """
    pcm = _pcm([(1.0, 6.0), (6.4, 12.0)], 14.0)

    def lag(commit: float) -> tuple[float, int]:
        st = _stream(partial_commit_s=commit)
        pieces = [p for p in st.events(pcm) if p.kind == KIND_PARTIAL]
        lags = [p.t_avail - p.t_end for p in pieces]
        return (sum(lags) / len(lags) if lags else 0.0), len(pieces)

    lag_fast, n_fast = lag(0.5)
    lag_deployed, n_deployed = lag(2.0)
    assert lag_deployed > lag_fast + 1.0, (lag_deployed, lag_fast)
    print(f"  partial lag: commit=0.5s -> {lag_fast:.2f}s over {n_fast} pieces | "
          f"deployed 2.0s -> {lag_deployed:.2f}s over {n_deployed}")


def test_energy_gate_never_closes_on_real_conversation_levels() -> None:
    """The deployed absolute gate is why the VAD default is silero."""
    pcm = _pcm([(0.0, 5.0)], 5.0, amp=0.05)  # quiet, but clearly speech
    pcm += np.random.default_rng(1).normal(0, 0.004, pcm.size).astype(np.float32)
    spans = energy_spans(pcm, rms_threshold=0.002)
    assert len(spans) == 1 and spans[0][1] - spans[0][0] > 4.9, spans
    loud = energy_spans(_pcm([(0.0, 5.0)], 5.0), rms_threshold=0.6)
    assert loud == [], loud
    print(f"  energy gate spans on noisy speech: {[(round(a,2), round(b,2)) for a, b in spans]}")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        print(f"-- {t.__name__}")
        t()
    print(f"\n{len(tests)} audio-stream checks passed")
