"""Render a session's assistant audio into a benchmark ``output.wav``.

FDB's metric scripts VAD ``output.wav`` and compare its speech intervals with
``input.wav``'s, so the two files must share a time origin. Its own reference
outputs are exactly as long as the input, because the reference systems answer
inside that window; ours can be late by design -- the deployed turn end needs a
0.6 s silence, the ASR cadence adds up to 0.6 s, and the model's own compute comes
on top -- so a response can begin *after* the input ends. Clipping there would
render it as silence, which is indistinguishable from a model that chose to stay
quiet, and the two are different behaviours.

``tail_s`` therefore extends the WAV past the end of the input (default 4 s) so a
late answer still lands, and the trace records both readings: what was audible
inside the input window and what fell beyond it. Scoring the tail-inclusive audio
answers "did the model take the turn"; ``speech_within_input_s`` from the trace
answers "would a fixed-length session have heard it".
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


def render_output(res, input_duration_s: float, *, sr: int = OUT_SR,
                  tail_s: float = 4.0) -> tuple[np.ndarray, dict]:
    """Build ``output.wav`` samples plus a small render report."""
    n = int(round((input_duration_s + max(0.0, tail_s)) * sr))
    out = np.zeros(max(n, 0), dtype=np.float32)
    placed = clipped = 0
    placed_within = always_beyond = 0
    for turn in res.turns:
        audio = resample_24k_to_16k(turn.audio_slice())
        if audio.size == 0:
            continue
        start = int(round(turn.start * sr))
        # audible only inside the input window, however long the tail is
        win = int(round(input_duration_s * sr))
        if start >= out.size:
            clipped += audio.size / sr
            if start >= win:
                always_beyond += audio.size / sr
            continue
        end = min(start + audio.size, out.size)
        take = end - start
        out[start:end] += audio[:take]
        placed += take / sr
        placed_within += max(0, min(end, win) - start) / sr
        clipped += (audio.size - take) / sr
        if start >= win:
            always_beyond += take / sr
    return out, {
        "placed_speech_s": round(placed, 3),
        "placed_within_input_s": round(placed_within, 3),
        "speech_beyond_input_s": round(always_beyond, 3),
        "clipped_speech_s": round(clipped, 3),
        "tail_s": round(tail_s, 3),
        "turns": len(res.turns),
    }


def write_wav(path: str | Path, samples: np.ndarray, *, sr: int = OUT_SR) -> None:
    import soundfile as sf

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(samples, -1.0, 1.0), sr, subtype="PCM_16")
