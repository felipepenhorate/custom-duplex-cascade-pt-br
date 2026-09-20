"""M3 - STT service: streaming pt-BR ASR with faster-whisper.

Receives 24 kHz float32 PCM as msgpack `{"type":"Audio","pcm":[...]}` frames
over a websocket and streams back `{"type":"Word","text":...}` events as words
become available, plus `{"type":"Eos"}` on client `Eos`.

Implementation (SPEC 7.1): faster-whisper (CTranslate2) with `language="pt"`,
VAD on, word timestamps. Design: accumulate audio into the current utterance;
when the user pauses (a silence gap of `--silence-s`, checked periodically) or
sends `Eos`, transcribe the whole utterance ONCE and emit its words, then reset
for the next utterance. A background flusher checks for the pause. This avoids
the whisper hallucinations and word-diff bugs that plague incremental
re-transcription of short fragments.

Usage:
  python services/stt_service.py --port 31607 --model medium
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

import msgpack
import numpy as np
import websockets
from scipy.signal import resample_poly

try:
    from faster_whisper import WhisperModel
except ImportError:  # pragma: no cover
    print("[stt] run from the venv with faster-whisper installed")
    sys.exit(1)

INPUT_SR = 24000
WHISPER_SR = 16000


def resample_24k_to_16k(pcm: np.ndarray) -> np.ndarray:
    if pcm.size == 0:
        return pcm
    return resample_poly(pcm, 2, 3)  # 24000 * 2/3 = 16000


class STTService:
    def __init__(self, model_name: str, device: str, compute_type: str, silence_s: float, min_silence_s: float,
                 partial_interval_s: float = 0.6, partial_commit_s: float = 2.0):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.silence_s = silence_s
        self.min_silence_s = min_silence_s
        self.partial_interval_s = partial_interval_s  # live feedback cadence while speaking
        self.partial_commit_s = partial_commit_s  # how old a word must be before it is "settled"
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)

    async def _transcribe(self, pcm16: np.ndarray, beam_size: int, word_timestamps: bool):
        """Run whisper in a worker thread (CTranslate2 releases the GIL during
        inference, so the event loop keeps receiving audio meanwhile)."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.model.transcribe(
                pcm16,
                language="pt",
                beam_size=beam_size,
                temperature=0.0,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": int(self.min_silence_s * 1000)},
                word_timestamps=word_timestamps,
            ),
        )

    def _seg_words(self, segments) -> list[str]:
        words: list[str] = []
        for seg in segments:
            if getattr(seg, "words", None):
                for w in seg.words:
                    words.append(w.word)
            else:
                words.extend(seg.text.split())
        return [w.strip() for w in words if w.strip()]

    async def handle(self, ws: websockets.WebSocketServerProtocol) -> None:
        # audio for the current utterance (resampled to 16k)
        audio16: list[np.ndarray] = []
        last_audio_at = time.time()
        last_speech_at = time.time()  # last frame with actual speech energy
        last_partial_at = 0.0  # throttle live partials to partial_interval_s
        finalizing = False  # guards against double-finalize
        shown_words: list[str] = []  # words already streamed as partials

        async def transcribe_and_send() -> None:
            """Transcribe the complete utterance and emit ALL its words at once,
            then signal the bridge with UtteranceEnd so it can answer."""
            nonlocal finalizing
            if finalizing or not audio16:
                return
            finalizing = True
            try:
                pcm16 = np.concatenate(audio16) if len(audio16) > 1 else audio16[0]
                if pcm16.size < WHISPER_SR * 0.25:  # <0.25s: skip (avoid hallucinations)
                    audio16.clear()
                    shown_words.clear()
                    return
                # skip pure digital silence / near-zero audio (whisper hallucinates)
                rms = float(np.sqrt((pcm16**2).mean())) if pcm16.size else 0.0
                if rms < 0.0005:
                    audio16.clear()
                    shown_words.clear()
                    return
                t0 = time.time()
                segments, _info = await self._transcribe(pcm16, beam_size=5, word_timestamps=True)
                clean_words = self._seg_words(segments)
                for wtxt in clean_words:
                    await ws.send(msgpack.packb({"type": "Word", "text": wtxt}, use_bin_type=True))
                print(f"[stt] utterance {pcm16.size / WHISPER_SR:.1f}s -> {len(clean_words)} words "
                      f"({time.time() - t0:.2f}s): {clean_words!r}", flush=True)
                audio16.clear()
                shown_words.clear()
                # tell the bridge the utterance is complete so it can answer now
                await ws.send(msgpack.packb({"type": "UtteranceEnd"}, use_bin_type=True))
                print("[stt] sent UtteranceEnd to bridge", flush=True)
            finally:
                finalizing = False

        async def send_partial() -> None:
            """Live feedback while speaking: transcribe the FULL utterance so far
            and emit only words whose audio is already "committed" (old enough
            that whisper won't revise them). Words accumulate; the final
            transcription on silence replaces everything with the accurate text.

            Whisper re-decodes the growing buffer differently each time, so a
            plain prefix-diff duplicates/revises already-shown words. By using
            word timestamps and a safety margin we only ever ADD settled words."""
            nonlocal last_partial_at
            if finalizing or not audio16:
                return
            pcm16 = np.concatenate(audio16) if len(audio16) > 1 else audio16[0]
            if pcm16.size < WHISPER_SR * 0.5:  # <0.5s: not worth a partial
                return
            try:
                segments, _info = await self._transcribe(pcm16, beam_size=1, word_timestamps=True)
            except Exception:
                return
            last_partial_at = time.time()
            # words whose end is well before the current buffer end are settled;
            # the newest ~2s tail is still being re-decoded and may change
            buf_end = pcm16.size / WHISPER_SR
            commit_cut = buf_end - self.partial_commit_s
            settled: list[str] = []
            for seg in segments:
                if not getattr(seg, "words", None):
                    continue
                for w in seg.words:
                    if getattr(w, "end", None) is not None and float(w.end) <= commit_cut:
                        t = str(w.word).strip()
                        if t:
                            settled.append(t)
            if not settled:
                return
            # the browser REPLACES the bubble with this text, and settled words
            # only grow as audio ages, so sending the full settled text each time
            # naturally accumulates without duplication. Whisper revising early
            # words just means the displayed text updates in place.
            partial = " ".join(settled)
            shown_words[:] = settled
            try:
                await ws.send(msgpack.packb({"type": "PartialText", "text": partial}, use_bin_type=True))
            except Exception:
                return

        async def flusher():
            """Stream live partials while the user speaks; finalize the utterance
            once the user pauses (silence gap)."""
            while True:
                await asyncio.sleep(0.2)
                try:
                    if not audio16:
                        continue
                    if finalizing:
                        continue
                    if (time.time() - last_speech_at) < self.silence_s:
                        # speaking: live partial feedback
                        if (time.time() - last_partial_at) >= self.partial_interval_s:
                            await send_partial()
                    else:
                        # user paused: finalize the utterance (block repeat)
                        await transcribe_and_send()
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    print(f"[stt] flusher err: {type(e).__name__}: {e}", flush=True)

        flusher_task = asyncio.create_task(flusher())
        try:
            async for raw in ws:
                if isinstance(raw, str):
                    continue
                try:
                    msg = msgpack.unpackb(raw, raw=False)
                except Exception:
                    continue
                if not isinstance(msg, dict):
                    continue
                t = msg.get("type")
                if t == "Audio":
                    pcm24 = np.asarray(msg.get("pcm", []), dtype=np.float32)
                    if pcm24.size:
                        audio16.append(resample_24k_to_16k(pcm24))
                        last_audio_at = time.time()
                        # only count frames with real speech energy towards the
                        # "user is still talking" clock; ambient/silent frames
                        # stream continuously while the mic is open and must NOT
                        # reset the silence timer
                        rms = float(np.sqrt((pcm24**2).mean()))
                        if rms >= 0.002:
                            last_speech_at = time.time()
                elif t == "Eos":
                    # explicit turn boundary: transcribe and reset
                    await transcribe_and_send()
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            flusher_task.cancel()
            audio16.clear()


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=31607)
    p.add_argument("--model", default="medium", help="faster-whisper model (tiny/base/small/medium)")
    p.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    p.add_argument("--compute-type", default="float16")
    p.add_argument("--silence-s", type=float, default=0.6,
                   help="silence gap that ends an utterance (turn boundary)")
    p.add_argument("--min-silence-s", type=float, default=0.3)
    p.add_argument("--partial-commit-s", type=float, default=2.0,
                   help="words older than this are streamed as settled partials")
    p.add_argument("--partial-interval-s", type=float, default=0.6,
                   help="cadence of live partial transcripts")
    args = p.parse_args()

    svc = STTService(args.model, args.device, args.compute_type, args.silence_s, args.min_silence_s, partial_interval_s=args.partial_interval_s, partial_commit_s=args.partial_commit_s)
    print(f"[stt] faster-whisper '{args.model}' ready on :{args.port} (pt, silence {args.silence_s}s)", flush=True)
    async with websockets.serve(svc.handle, host="", port=args.port, max_size=8 << 20):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())