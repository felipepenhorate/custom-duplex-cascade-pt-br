"""Voice I/O for the Onda Travel demo.

STT is a small per-session endpointing loop on top of faster-whisper.  It emits
*incremental* ASR pieces, the exact shape Laya was trained on, and a final
event when the speaker has been quiet long enough to call the turn over.

TTS is Gradium Phonon 40m multilingual, loaded from the local checkpoint.  It
streams 24 kHz float32 chunks, which the websocket bridge forwards to the
browser unchanged.
"""

from __future__ import annotations

import asyncio
import re
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import resample_poly

SAMPLE_RATE = 24_000
FRAME_SAMPLES = 480  # 20 ms


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _tail_words(text: str, count: int = 3) -> str:
    words = _clean_text(text).split()
    return " ".join(words[-count:]) if words else ""


def _common_prefix(a: str, b: str) -> str:
    limit = min(len(a), len(b))
    index = 0
    while index < limit and a[index] == b[index]:
        index += 1
    return a[:index]


def resample_to_16k(pcm: np.ndarray) -> np.ndarray:
    if pcm.size == 0:
        return pcm
    return resample_poly(pcm, 2, 3).astype(np.float32, copy=False)  # 24k -> 16k


class STTEngine:
    """One faster-whisper model shared by all sessions."""

    def __init__(
        self,
        model_name: str = "small",
        *,
        device: str = "cuda",
        compute_type: str = "float16",
    ) -> None:
        from faster_whisper import WhisperModel

        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)
        self.lock = threading.Lock()

    def transcribe(
        self,
        pcm24: np.ndarray,
        *,
        language: str | None,
        beam_size: int,
        word_timestamps: bool = False,
    ) -> tuple[str, str | None, float]:
        pcm16 = resample_to_16k(pcm24)
        if pcm16.size < 1_600:  # < 0.1 s
            return "", language, 0.0
        with self.lock:
            segments, info = self.model.transcribe(
                pcm16,
                language=language,
                beam_size=beam_size,
                temperature=0.0,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 250},
                condition_on_previous_text=False,
                word_timestamps=word_timestamps,
            )
            words: list[str] = []
            for segment in segments:
                if getattr(segment, "words", None) and word_timestamps:
                    words.extend(word.word for word in segment.words)
                else:
                    words.append(segment.text)
            text = _clean_text(" ".join(words))
        detected = getattr(info, "language", None)
        prob = float(getattr(info, "language_probability", 0.0) or 0.0)
        return text, detected, prob


class StreamingASR:
    """Endpointing + incremental partials for one browser session.

    Call :meth:`feed` with 24 kHz float32 frames and :meth:`start` once.  The
    callbacks run on the event loop:

    ``on_partial(piece, full_text)``
    ``on_final(full_text, final_piece, language)``
    """

    def __init__(
        self,
        engine: STTEngine,
        *,
        on_partial: Callable[[str, str], Any],
        on_final: Callable[[str, str, str | None], Any],
        language: str | None = None,
        endpoint_silence: float = 0.85,
        partial_interval: float = 0.75,
        max_utterance: float = 25.0,
    ) -> None:
        self.engine = engine
        self.on_partial = on_partial
        self.on_final = on_final
        self.language = language
        self.endpoint_silence = endpoint_silence
        self.partial_interval = partial_interval
        self.max_utterance = max_utterance

        self._audio: list[np.ndarray] = []
        self._audio_samples = 0
        self._last_speech_at = time.monotonic()
        self._last_partial_at = 0.0
        self._noise_floor = 0.0015
        self._speech = False
        self._speech_started_at = 0.0
        self._last_hypothesis = ""
        self._partial_busy = False
        self._closed = False
        self._task: asyncio.Task | None = None
        self._level_cb: Callable[[float, bool], Any] | None = None

    # ---------------------------------------------------------------- control
    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def close(self) -> None:
        self._closed = True
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None

    def reset(self) -> None:
        self._audio.clear()
        self._audio_samples = 0
        self._last_hypothesis = ""
        self._last_partial_at = 0.0
        self._speech = False
        self._speech_started_at = 0.0

    def set_level_callback(self, callback: Callable[[float, bool], Any]) -> None:
        self._level_cb = callback

    # ------------------------------------------------------------------ audio
    def feed(self, pcm24: np.ndarray) -> None:
        if self._closed or pcm24.size == 0:
            return
        pcm = np.asarray(pcm24, dtype=np.float32).reshape(-1)
        self._audio.append(pcm)
        self._audio_samples += int(pcm.size)

        rms = float(np.sqrt(np.mean(np.square(pcm)))) if pcm.size else 0.0
        threshold = max(0.0045, self._noise_floor * 2.6)
        now = time.monotonic()
        if rms >= threshold:
            if not self._speech:
                self._speech_started_at = now
            self._speech = True
            self._last_speech_at = now
        else:
            self._noise_floor = 0.98 * self._noise_floor + 0.02 * rms
        if self._level_cb is not None:
            self._level_cb(min(1.0, rms * 18.0), self._speech)

        # Keep a hard cap on the utterance so a forgotten mic cannot exhaust
        # memory or make whisper decode forever.
        if self._audio_samples > int(SAMPLE_RATE * self.max_utterance):
            self._trim_to(max_seconds=self.max_utterance)

    def _trim_to(self, max_seconds: float) -> None:
        max_samples = int(SAMPLE_RATE * max_seconds)
        while self._audio_samples > max_samples and self._audio:
            drop = self._audio.pop(0)
            self._audio_samples -= int(drop.size)

    def _pcm(self) -> np.ndarray:
        if not self._audio:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self._audio)

    # ------------------------------------------------------------------- loop
    async def _run(self) -> None:
        while not self._closed:
            await asyncio.sleep(0.12)
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(f"[asr] tick error: {type(exc).__name__}: {exc}", flush=True)

    async def _tick(self) -> None:
        if not self._audio:
            return
        now = time.monotonic()
        duration = self._audio_samples / SAMPLE_RATE
        silence_for = now - self._last_speech_at

        # No speech yet: throw away ambient noise after two seconds.
        if not self._speech:
            if duration > 2.0 and silence_for > 1.0:
                self.reset()
                self._noise_floor = max(0.001, self._noise_floor * 0.99)
            return

        # Silence long enough: endpoint the utterance.
        if silence_for >= self.endpoint_silence and duration >= 0.25:
            await self._finalize()
            return

        # The user has been talking for a while without a pause.  Keep a
        # rolling partial transcript, but let the final decoder see the whole
        # buffer once the endpoint arrives.
        if (
            not self._partial_busy
            and duration >= 0.9
            and (now - self._last_partial_at) >= self.partial_interval
        ):
            await self._partial()

    async def _partial(self) -> None:
        self._partial_busy = True
        self._last_partial_at = time.monotonic()
        pcm = self._pcm()
        try:
            text, _detected, _prob = await asyncio.to_thread(
                self.engine.transcribe,
                pcm,
                language=self.language,
                beam_size=1,
                word_timestamps=False,
            )
        finally:
            self._partial_busy = False
        text = _clean_text(text)
        if not text:
            return
        previous = self._last_hypothesis
        piece = ""
        if not previous:
            piece = text
            self._last_hypothesis = text
        elif text.startswith(previous):
            piece = text[len(previous) :].strip()
            self._last_hypothesis = text
        elif previous.startswith(text):
            # A shorter revision: keep the words already sent and wait.
            return
        else:
            # Whisper revised an earlier word.  Laya cannot retract a piece, so
            # keep the settled history and wait for the final decoder to emit
            # the corrected tail.
            prefix = _common_prefix(previous, text)
            if len(prefix) < len(previous):
                return
            piece = text[len(previous) :].strip()
            self._last_hypothesis = text
        if piece:
            await self.on_partial(piece, text)

    async def _finalize(self) -> None:
        pcm = self._pcm()
        had_speech = self._speech
        previous = self._last_hypothesis
        self.reset()
        if not had_speech or pcm.size == 0:
            return
        text, detected, prob = await asyncio.to_thread(
            self.engine.transcribe,
            pcm,
            # Finals auto-detect so a user can switch language mid-conversation.
            language=None,
            beam_size=5,
            word_timestamps=False,
        )
        text = _clean_text(text)
        if not text:
            return
        if detected in ("pt", "en") and prob >= 0.55:
            self.language = detected
        if not previous:
            piece = text
        elif text.startswith(previous):
            piece = text[len(previous) :].strip()
        else:
            piece = _tail_words(text, 3)
        if not piece:
            # The partials already carried every word.  A short duplicate tail
            # is the least ambiguous way to mark the turn end for Laya.
            piece = _tail_words(text, 3)
        # Keep the next utterance's partial diff independent.  reset() already
        # cleared the hypothesis; setting it back to the final text here would
        # make every partial of the *next* turn compare against this one and
        # silently emit nothing.
        await self.on_final(text, piece, detected if prob >= 0.55 else None)


class PhononTTS:
    """Two Phonon instances (one normalizer per language), same voice."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        voices_dir: str | Path | None = None,
        voice: str = "Manuela-Lu",
    ) -> None:
        import ptts

        self.ptts = ptts
        self.voice = voice
        self.sample_rate = SAMPLE_RATE
        model = str(model_path)
        catalog = str(voices_dir) if voices_dir else None
        kwargs: dict[str, Any] = {"quant": "q8"}
        if catalog:
            kwargs["voice_catalog"] = catalog
        self.engines = {
            "en": ptts.TTS(model, lang="en", **kwargs),
            "pt": ptts.TTS(model, lang="pt", **kwargs),
        }
        # Touch the voice once so the first real answer does not pay the file
        # lookup on the audio path.
        for engine in self.engines.values():
            try:
                engine.voices  # noqa: B018 - property lookup by design
            except Exception:
                pass

    def _engine(self, language: str):
        return self.engines["pt" if language.startswith("pt") else "en"]

    async def stream(
        self,
        text: str,
        *,
        language: str,
        cancel: threading.Event,
    ):
        """Yield 24 kHz float32 chunks as they are generated."""
        engine = self._engine(language)
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue()

        def produce() -> None:
            try:
                with engine.stream(text, voice=self.voice) as audio:
                    for chunk in audio:
                        if cancel.is_set():
                            break
                        pcm = np.asarray(chunk, dtype=np.float32).reshape(-1)
                        if pcm.size:
                            loop.call_soon_threadsafe(_safe_put, queue, pcm)
            except Exception as exc:
                print(f"[tts] generation error: {type(exc).__name__}: {exc}", flush=True)
            finally:
                loop.call_soon_threadsafe(_safe_put, queue, None)

        thread = threading.Thread(target=produce, name="phonon-tts", daemon=True)
        thread.start()
        try:
            while True:
                chunk = await queue.get()
                if chunk is None:
                    break
                if cancel.is_set():
                    break
                yield np.clip(chunk, -1.0, 1.0)
        finally:
            cancel.set()
            # Do not join the thread from the event loop: it may be waiting on
            # the model's next 80 ms frame.  It exits on its own when the
            # generator notices the cancel flag.
            await asyncio.sleep(0)


def _safe_put(queue: asyncio.Queue, value: Any) -> None:
    try:
        queue.put_nowait(value)
    except asyncio.QueueFull:
        pass


def strip_for_tts(text: str) -> str:
    """Remove visual markdown that should not be spoken."""
    text = re.sub(r"[*_`#>]+", "", text)
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
