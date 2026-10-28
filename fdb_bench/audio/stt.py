"""Real user audio -> the micro-turn piece stream the protocol was trained on.

Replays a benchmark WAV through the deployed STT pipeline's *semantics*,
ported from ``sft/services/stt_service.py``:

* audio is fed as 24 kHz frames and resampled to 16 kHz (``resample_24k_to_16k``);
* while the user talks, a beam-1 pass every 0.6 s emits the words that are
  *settled* -- their audio ended before the end of the buffer minus a commit
  window;
* a 0.6 s silence gap finalizes the utterance: one beam-5 pass over the whole
  utterance emits every remaining word, then the turn-end signal;
* the utterance buffer is cleared at that point, so a fresh utterance starts
  accumulating from the silence (matching ``audio16.clear()``);
* too-short and near-silent utterances are dropped without a turn-end signal,
  exactly as the service does.

Two deliberate departures, both forced by the data and both measurable:

1. **The settled partials are kept.** The deployed bridge throws them away and
   hands the LLM only the final bulk transcript at the end of the utterance -- a
   VAD-gated half-duplex front end. Delivering settled words as they settle is
   the micro-turn stream the protocol was trained on, and it is the only
   substitution this harness makes.
2. **The commit window is 0.3 s, not the deployed 2.0 s.** ``partial_commit_s``
   exists in the shipped service to stop already-displayed words being revised in
   the browser's partial-text bubble. It is a UI concern, and the deployed bridge
   never feeds partials to the LLM, so the value was never tuned for that. Feeding
   settled words to a duplex LLM with a 2.0 s window starves it: measured on
   CANDOR, 27 s of conversation produced 5 pieces where the training distribution
   averages one piece per ~0.7 s. At 0.3 s the emitted pieces match the training
   micro-turn sizes (see ``tests/test_audio_stream.py``), which is the property
   that has to hold for the numbers to mean anything. The deployed 2.0 s remains
   available as an arm.
3. **Which silence gate.** The deployed service marks "the user is still talking"
   with an absolute frame energy (``rms >= 0.002``), which is calibrated for a
   browser microphone whose idle state is digital silence. Real conversation
   audio never goes that quiet -- CANDOR sit at RMS 0.006-0.24, and the absolute
   gate calls 94% of it speech, so it never fires a turn end. The default gate is
   therefore **silero VAD**, the same detector the benchmark's own metric scripts
   run on the model's output, so the system's notion of "the user stopped
   talking" matches the benchmark's notion of silence. ``vad="energy"`` keeps the
   deployed rule available as an arm, and its failure on this data is a result
   rather than a bug.

Everything is expressed in *audio time* -- seconds from the start of the input
WAV -- so the stream is deterministic and shares the origin the benchmark's
annotations and metric scripts use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

WHISPER_SR = 16000
MIC_SR = 24000  # what the browser sends and what the deployed service expects

KIND_PARTIAL = "partial"
KIND_FINAL = "final"
KIND_SPEECH_START = "speech_start"
KIND_SPEECH_END = "speech_end"
KIND_SILENCE = "silence"  # a `<|no voice|>` tick while the user is quiet

_SILERO = None


def silero_model():
    global _SILERO
    if _SILERO is None:
        from silero_vad import load_silero_vad

        _SILERO = load_silero_vad()
    return _SILERO


def resample_24k_to_16k(pcm: np.ndarray) -> np.ndarray:
    """Same 2/3 polyphase decimation the deployed service uses."""
    from scipy.signal import resample_poly

    if pcm.size == 0:
        return pcm
    return resample_poly(pcm, 2, 3)


class TraceUserStream:
    """Replay a recorded ASR stream instead of running Whisper.

    ``results_audio*/decision/<model>/<subset>/<sid>/trace.json`` stores the
    exact ``user_pieces`` the live session fed the model, including availability
    times, ``kind``, ``ends_turn`` and the utterance index.  Re-deciding those
    pieces is the audio analogue of the text bench and the paired way to compare
    two companions on identical ASR input.  No microphone audio is decoded here;
    the WAV is still used for the scenario clock and for rendering.
    """

    def __init__(self, trace: dict) -> None:
        self.trace = trace
        self.spans = [tuple(map(float, span))
                      for span in (trace.get("speech_intervals") or [])]
        self.utterances: list[Utterance] = []
        self.stt_s = 0.0
        self.stt_partial_s = 0.0
        self.stt_final_s = 0.0
        self.stt_calls = 0

    def events(self, pcm24: np.ndarray) -> Iterator[UserPiece]:
        scheduled: list[tuple[float, int, int, UserPiece]] = []
        order = 0
        for start, end in self.spans:
            scheduled.append((start, 0, order, UserPiece(
                t_start=start, t_end=start, kind=KIND_SPEECH_START,
                t_avail=start)))
            order += 1
            scheduled.append((end, 2, order, UserPiece(
                t_start=end, t_end=end, kind=KIND_SPEECH_END, t_avail=end)))
            order += 1
        for piece in self.trace.get("user_pieces") or []:
            kind = piece.get("kind")
            if kind not in (KIND_PARTIAL, KIND_FINAL, KIND_SILENCE):
                continue
            t_avail = float(piece.get("t_avail", piece.get("t_start", 0.0)))
            t_start = float(piece.get("t_start", t_avail))
            t_end = float(piece.get("t_end", t_avail))
            scheduled.append((t_avail, 1, order, UserPiece(
                t_start=t_start, t_end=t_end, kind=kind,
                text=piece.get("text", ""),
                ends_turn=bool(piece.get("ends_turn", False)),
                utt_index=int(piece.get("utt", 0) or 0),
                t_avail=t_avail)))
            order += 1
        scheduled.sort(key=lambda item: item[:3])
        for _, _, _, event in scheduled:
            yield event


def load_wav_24k(path: str | Path) -> np.ndarray:
    """Load a benchmark WAV as mono float32 at the 24 kHz the stack expects."""
    import soundfile as sf

    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != MIC_SR:
        from scipy.signal import resample_poly

        g = np.gcd(int(sr), MIC_SR)
        wav = resample_poly(wav, MIC_SR // g, int(sr) // g).astype(np.float32)
    return np.ascontiguousarray(wav, dtype=np.float32)


@dataclass
class UserPiece:
    """One event on the simulated microphone stream, timed in audio seconds."""

    t_start: float
    t_end: float
    kind: str
    text: str = ""
    ends_turn: bool = False
    utt_index: int = 0
    # When the piece became available to the model. A settled word's audio may
    # have ended seconds earlier (the settle window), and it is the availability
    # time, not the word's own span, that a response is timed against.
    t_avail: float = 0.0

    @property
    def is_text(self) -> bool:
        return self.kind in (KIND_PARTIAL, KIND_FINAL)


@dataclass
class Utterance:
    """Bookkeeping for one STT utterance (buffer clear to buffer clear)."""

    index: int
    t_start: float
    t_end: float
    words: list[tuple[str, float, float]] = field(default_factory=list)
    finalized: bool = False
    skipped: str | None = None  # why the deployed service emitted nothing


def silero_spans(pcm24: np.ndarray, *, min_silence_ms: int = 100,
                 speech_pad_ms: int = 30) -> list[tuple[float, float]]:
    """Speech spans in audio seconds, from the benchmark's own VAD."""
    import torch
    from silero_vad import get_speech_timestamps

    pcm16 = resample_24k_to_16k(np.asarray(pcm24, dtype=np.float32))
    ts = get_speech_timestamps(
        torch.from_numpy(np.ascontiguousarray(pcm16)), silero_model(),
        sampling_rate=WHISPER_SR, return_seconds=True,
        min_silence_duration_ms=min_silence_ms, speech_pad_ms=speech_pad_ms,
    )
    return [(float(t["start"]), float(t["end"])) for t in ts]


def energy_spans(pcm24: np.ndarray, *, step_s: float = 0.02,
                 rms_threshold: float = 0.002,
                 min_silence_ms: int = 100) -> list[tuple[float, float]]:
    """The deployed service's absolute energy gate, expressed as spans."""
    step = max(1, int(round(step_s * MIC_SR)))
    n = int(np.ceil(pcm24.size / step))
    flags = []
    for i in range(n):
        f = pcm24[i * step:(i + 1) * step]
        if f.size == 0:
            break
        flags.append(float(np.sqrt((f.astype(np.float64) ** 2).mean())) >= rms_threshold)
    spans: list[tuple[float, float]] = []
    start: int | None = None
    for i, on in enumerate(flags + [False]):
        if on and start is None:
            start = i
        elif not on and start is not None:
            s, e = start * step_s, i * step_s
            if spans and s - spans[-1][1] < min_silence_ms / 1000.0:
                spans[-1] = (spans[-1][0], e)
            else:
                spans.append((s, e))
            start = None
    return spans


def _norm_word(w: str) -> str:
    """Comparison key for a word: case- and punctuation-insensitive."""
    import re

    return re.sub(r"[^\w']", "", str(w).lower())


def _common_prefix_len(a: list[str], b: list[str]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _split_pieces(words: list[tuple[str, float, float]], *, max_words: int) -> list[list]:
    """Cut a delivered word list into training-sized micro-turn pieces.

    The protocol was trained on pieces of at most 7 tokens
    (``build_duplex_dataset.py``, ``user_chunk_max=7``); a long batch of words
    delivered at once (a lagging partial, or an utterance tail) is split so the
    model still sees the input shape it was trained on.
    """
    out: list[list] = []
    for i in range(0, len(words), max_words):
        out.append(words[i:i + max_words])
    return out


class AudioUserStream:
    """Replays a WAV as the user side of a duplex session.

    ``events()`` yields speech-state changes and text pieces in audio-time
    order. Speech state comes from the configured VAD, which is also the signal
    that decides when the user has stopped talking.
    """

    def __init__(
        self,
        model,
        *,
        language: str = "pt",
        vad: str = "silero",
        silence_s: float = 0.6,
        partial_interval_s: float = 0.6,
        partial_commit_s: float = 0.5,
        min_silence_s: float = 0.3,
        step_s: float = 0.2,
        rms_threshold: float = 0.002,
        max_piece_words: int = 7,
        silence_ticks: bool = True,
        silence_tick_s: float = 0.5,
        max_silence_ticks: int = 12,
    ) -> None:
        if vad not in ("silero", "energy"):
            raise ValueError(vad)
        self.model = model
        self.language = language
        self.vad = vad
        self.silence_s = float(silence_s)
        self.partial_interval_s = float(partial_interval_s)
        self.partial_commit_s = float(partial_commit_s)
        self.min_silence_s = float(min_silence_s)
        self.step_s = float(step_s)
        self.rms_threshold = float(rms_threshold)
        self.max_piece_words = int(max_piece_words)
        self.silence_ticks = bool(silence_ticks)
        self.silence_tick_s = float(silence_tick_s)
        self.max_silence_ticks = int(max_silence_ticks)
        # filled during the run, for the trace
        self.utterances: list[Utterance] = []
        self.spans: list[tuple[float, float]] = []
        self.n_partials = 0
        self.n_finalizations = 0
        self.piece_word_counts: list[int] = []
        self.stt_s = 0.0            # total whisper time (partial + finalize)
        self.stt_partial_s = 0.0
        self.stt_final_s = 0.0
        self.stt_calls = 0

    # ---------------------------------------------------------------- whisper
    def _transcribe(self, pcm16: np.ndarray, beam_size: int):
        import time as _t

        t0 = _t.monotonic()
        try:
            return self.model.transcribe(
            pcm16,
            language=self.language,
            beam_size=beam_size,
            temperature=0.0,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": int(self.min_silence_s * 1000)},
                word_timestamps=True,
            )
        finally:
            dt = _t.monotonic() - t0
            self.stt_s += dt
            self.stt_calls += 1
            if beam_size == 1:
                self.stt_partial_s += dt
            else:
                self.stt_final_s += dt

    @staticmethod
    def _words(segments) -> list[tuple[str, float, float]]:
        out: list[tuple[str, float, float]] = []
        for seg in segments:
            ws = getattr(seg, "words", None)
            if ws:
                for w in ws:
                    t = str(w.word).strip()
                    if t:
                        out.append((t, float(w.start or 0.0), float(w.end or 0.0)))
            else:
                t = str(seg.text).strip()
                if t:
                    start = float(getattr(seg, "start", 0.0) or 0.0)
                    end = float(getattr(seg, "end", start) or start)
                    for piece in t.split():
                        out.append((piece, start, end))
        return out

    def _spans_for(self, pcm24: np.ndarray) -> list[tuple[float, float]]:
        if self.vad == "energy":
            return energy_spans(pcm24, rms_threshold=self.rms_threshold)
        return silero_spans(pcm24)

    # ------------------------------------------------------------------ replay
    def events(self, pcm24: np.ndarray) -> Iterator[UserPiece]:
        """Yield the stream for one input WAV, in audio time."""
        self.spans = self._spans_for(pcm24)
        step = int(round(self.step_s * MIC_SR))
        buf: list[np.ndarray] = []          # 16 kHz frames of the current utterance
        utt_start = 0.0                     # audio time of the utterance buffer start
        emitted_words: list[str] = []       # normalised words already delivered
        prev_norm: list[str] = []           # previous decode, for stability
        utt_index = 0
        last_speech_at = 0.0
        last_partial_at = 0.0
        finalized_at = -1.0                 # a finalization needs speech after this
        speaking_reported = False
        utt = Utterance(index=0, t_start=0.0, t_end=0.0)
        span_i = 0
        span_end_pending: float | None = None
        tick_run = 0
        last_tick_at = 0.0

        n_steps = int(np.ceil(pcm24.size / max(step, 1)))
        for i in range(n_steps):
            frame24 = pcm24[i * step:(i + 1) * step]
            t_end = min((i + 1) * self.step_s, pcm24.size / MIC_SR)
            if frame24.size:
                buf.append(resample_24k_to_16k(frame24))

            # speech-state transitions, at the VAD's own times
            while span_i < len(self.spans) and self.spans[span_i][0] <= t_end:
                if not speaking_reported:
                    speaking_reported = True
                    yield UserPiece(t_start=self.spans[span_i][0],
                                    t_end=self.spans[span_i][0],
                                    kind=KIND_SPEECH_START,
                                    t_avail=self.spans[span_i][0])
                span_end_pending = self.spans[span_i][1]
                span_i += 1
            if (span_end_pending is not None and speaking_reported
                    and t_end >= span_end_pending):
                speaking_reported = False
                yield UserPiece(t_start=span_end_pending, t_end=span_end_pending,
                                kind=KIND_SPEECH_END, t_avail=span_end_pending)
                span_end_pending = None

            ended = [e for _, e in self.spans[:span_i] if e <= t_end]
            last_speech_at = max(ended) if ended else 0.0
            silent_for = t_end - last_speech_at if ended else 0.0

            # `<|no voice|>` ticks: the protocol's silence marker. The live bridge
            # sends it as the user turn between assistant micro-turns and the
            # text-domain bench sends it after a turn end; without it a model that
            # waits to be told the user has stopped never gets the cue -- measured
            # on the upstream checkpoint, which never took the floor at all. Ticks
            # start only after the first turn end, as the bridge does (it waits for
            # the first transcript before calling the LLM), and are capped per run.
            if (self.silence_ticks and not speaking_reported and finalized_at >= 0
                    and tick_run < self.max_silence_ticks
                    and (t_end - last_tick_at) >= self.silence_tick_s):
                last_tick_at = t_end
                tick_run += 1
                yield UserPiece(t_start=t_end, t_end=t_end, kind=KIND_SILENCE,
                                t_avail=t_end)
            if speaking_reported:
                tick_run = 0
                last_tick_at = t_end

            if not buf:
                continue

            pcm16 = np.concatenate(buf) if len(buf) > 1 else buf[0]
            utt.t_end = t_end

            if not ended or silent_for < self.silence_s or last_speech_at <= finalized_at:
                # user is talking (or has not spoken since the last turn end):
                # live partials, settled words only
                if (t_end - last_partial_at) >= self.partial_interval_s:
                    last_partial_at = t_end
                    if pcm16.size >= WHISPER_SR * 0.5:
                        segments, _ = self._transcribe(pcm16, beam_size=1)
                        words = self._words(segments)
                        buf_end = t_end - utt_start
                        cur_norm = [_norm_word(w) for w, _, _ in words]
                        # Only a prefix that two consecutive decodes agree on is
                        # stable: whisper re-decodes the growing buffer and can
                        # revise (or repeat) words near the end. The deployed
                        # service relies on the browser *replacing* the bubble for
                        # this; a duplex LLM sees appended text, so stability has
                        # to be enforced here instead.
                        stable_n = _common_prefix_len(prev_norm, cur_norm)
                        prev_norm = cur_norm
                        cand = [
                            (w, s, e) for (w, s, e) in words[:stable_n]
                            if e <= buf_end - self.partial_commit_s
                            and _norm_word(w)
                        ]
                        if len(cand) > len(emitted_words):
                            new = cand[len(emitted_words):]
                            emitted_words.extend(_norm_word(w) for w, _, _ in new)
                            for chunk in _split_pieces(new, max_words=self.max_piece_words):
                                self.n_partials += 1
                                self.piece_word_counts.append(len(chunk))
                                yield UserPiece(
                                    t_start=utt_start + chunk[0][1],
                                    t_end=utt_start + chunk[-1][2],
                                    kind=KIND_PARTIAL,
                                    text=" ".join(w for w, _, _ in chunk),
                                    utt_index=utt_index,
                                    t_avail=t_end,
                                )
            else:
                # user paused: finalize the utterance (deployed: beam 5, all words)
                if pcm16.size < WHISPER_SR * 0.25:
                    utt.skipped = "too_short"
                    utt.finalized = True
                    utt.words = []
                    self.utterances.append(utt)
                    finalized_at = last_speech_at
                    last_tick_at = t_end
                    buf.clear()
                else:
                    rms_all = float(np.sqrt((pcm16.astype(np.float64) ** 2).mean()))
                    if rms_all < 0.0005:
                        utt.skipped = "digital_silence"
                        utt.finalized = True
                        utt.words = []
                        self.utterances.append(utt)
                        finalized_at = last_speech_at
                        last_tick_at = t_end
                        buf.clear()
                    else:
                        segments, _ = self._transcribe(pcm16, beam_size=5)
                        words = self._words(segments)
                        utt.words = words
                        utt.finalized = True
                        self.utterances.append(utt)
                        self.n_finalizations += 1
                        finalized_at = last_speech_at
                        last_tick_at = t_end
                        # The beam-5 pass is authoritative, and delivery is
                        # positional: whatever the partials already handed over is
                        # not handed over again. A mid-utterance revision is
                        # therefore left alone rather than re-delivered -- a
                        # duplicated word corrupts the prompt, a superseded one
                        # only means the model saw what the ASR actually said at
                        # the time, which is what a live pipeline does.
                        k = min(len(emitted_words), len(words))
                        tail = words[k:]
                        if tail:
                            emitted_words.extend(_norm_word(w) for w, _, _ in tail)
                            chunks = _split_pieces(tail, max_words=self.max_piece_words)
                            for ci, chunk in enumerate(chunks):
                                self.piece_word_counts.append(len(chunk))
                                yield UserPiece(
                                    t_start=utt_start + chunk[0][1],
                                    t_end=utt_start + chunk[-1][2],
                                    kind=KIND_FINAL,
                                    text=" ".join(w for w, _, _ in chunk),
                                    ends_turn=(ci == len(chunks) - 1),
                                    utt_index=utt_index,
                                    t_avail=t_end,
                                )
                        else:
                            # nothing new to say, but the turn still ends
                            yield UserPiece(
                                t_start=t_end, t_end=t_end, kind=KIND_FINAL,
                                text="", ends_turn=True, utt_index=utt_index,
                                t_avail=t_end,
                            )
                        buf.clear()

                if not buf:
                    utt_index += 1
                    utt = Utterance(index=utt_index, t_start=t_end, t_end=t_end)
                    utt_start = t_end
                    emitted_words = []
                    prev_norm = []
                    last_partial_at = t_end

        # The stream ended with an open buffer: the deployed service finalizes on an
        # explicit ``Eos`` from the client (the browser sends it when the user stops
        # the mic), so the end of the input is treated the same way rather than
        # dropped. Without this a real benchmark input whose audio simply stops
        # mid-speech would never produce a turn end, and the model would be scored
        # as if it had chosen to stay silent.
        if buf and not utt.finalized:
            pcm16 = np.concatenate(buf) if len(buf) > 1 else buf[0]
            if pcm16.size >= WHISPER_SR * 0.25:
                rms_all = float(np.sqrt((pcm16.astype(np.float64) ** 2).mean()))
                if rms_all >= 0.0005:
                    segments, _ = self._transcribe(pcm16, beam_size=5)
                    words = self._words(segments)
                    utt.words = words
                    utt.finalized = True
                    utt.t_end = pcm24.size / MIC_SR
                    self.utterances.append(utt)
                    self.n_finalizations += 1
                    k = min(len(emitted_words), len(words))
                    tail = words[k:]
                    if tail:
                        emitted_words.extend(_norm_word(w) for w, _, _ in tail)
                    chunks = _split_pieces(tail, max_words=self.max_piece_words) or [[]]
                    for ci, chunk in enumerate(chunks):
                        if chunk:
                            self.piece_word_counts.append(len(chunk))
                            yield UserPiece(
                                t_start=utt_start + chunk[0][1],
                                t_end=utt_start + chunk[-1][2],
                                kind=KIND_FINAL,
                                text=" ".join(w for w, _, _ in chunk),
                                ends_turn=(ci == len(chunks) - 1),
                                utt_index=utt_index,
                                t_avail=utt.t_end,
                            )
                        elif ci == len(chunks) - 1:
                            yield UserPiece(
                                t_start=utt.t_end, t_end=utt.t_end, kind=KIND_FINAL,
                                text="", ends_turn=True, utt_index=utt_index,
                                t_avail=utt.t_end,
                            )
                else:
                    utt.skipped = "digital_silence"
                    self.utterances.append(utt)
            else:
                utt.skipped = "too_short"
                self.utterances.append(utt)
