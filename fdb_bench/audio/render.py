"""Render a session's assistant audio into a benchmark ``output.wav``.

FDB's metric scripts VAD ``output.wav`` and compare its speech intervals with
``input.wav``'s, so the two files must share a time origin, and its own reference
outputs are padded to the input's length. This renderer therefore writes a
16 kHz mono WAV *the same length as the input*, with each assistant turn placed
at the offset the session recorded; anything that would run past the end of the
input is clipped there, and the clipped amount is recorded rather than hidden.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

OUT_SR = 16000


def resample_24k_to_16k(audio: np.ndarray) -> np.ndarray:
    from scipy.signal import resample_poly

    if audio.size == 0:
        return audio
    return resample_poly(audio, 2, 3).astype(np.float32)


def render_output(res, input_duration_s: float, *, sr: int = OUT_SR) -> tuple[np.ndarray, dict]:
    """Build ``output.wav`` samples plus a small render report."""
    n = int(round(input_duration_s * sr))
    out = np.zeros(max(n, 0), dtype=np.float32)
    placed = clipped = 0
    for turn in res.turns:
        audio = resample_24k_to_16k(turn.audio_slice())
        if audio.size == 0:
            continue
        start = int(round(turn.start * sr))
        if start >= out.size:
            clipped += audio.size / sr
            continue
        end = min(start + audio.size, out.size)
        take = end - start
        out[start:end] += audio[:take]
        placed += take / sr
        clipped += (audio.size - take) / sr
    return out, {
        "placed_speech_s": round(placed, 3),
        "clipped_speech_s": round(clipped, 3),
        "turns": len(res.turns),
    }


def write_wav(path: str | Path, samples: np.ndarray, *, sr: int = OUT_SR) -> None:
    import soundfile as sf

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(samples, -1.0, 1.0), sr, subtype="PCM_16")
