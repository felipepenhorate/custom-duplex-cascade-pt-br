"""Pitch-preserving time stretch, without librosa/numba.

The pt-BR audio set has to keep the English timeline: every GT annotation (pause
intervals, annotated turn ends, interruption spans) is expressed in seconds of the
original audio, and the whole point of the rebuilt set is that those annotations
transfer verbatim. So each translated span is stretched to fill exactly the slot
its English counterpart occupied.

``librosa.effects.time_stretch`` would do it but needs numba, and the numba wheel
available here refuses the installed numpy (`Numba needs NumPy 2.2 or less`). This
is WSOLA instead -- waveform-similarity overlap-add: fixed synthesis hop, analysis
hop scaled by the stretch factor, and a small per-frame search that picks the
analysis offset whose waveform best continues what has already been written. No
pitch shift (unlike resampling) and no dependencies.

``tests/test_stretch.py`` checks the property that matters: a pure tone stretched
by 1.5x keeps its frequency, which resampling would not.
"""

from __future__ import annotations

import numpy as np


def wsola(x: np.ndarray, rate: float, *, sr: int = 24000, frame_ms: float = 40.0,
          hop_ms: float = 10.0, search_ms: float = 15.0,
          deadband: float = 0.02) -> np.ndarray:
    """Stretch ``x`` by ``rate`` (>1 shortens, <1 lengthens) keeping pitch.

    The result is exactly ``round(len(x)/rate)`` samples long, which is what lets
    a segment be dropped into its original slot with no drift.
    """
    x = np.ascontiguousarray(np.asarray(x, dtype=np.float32).reshape(-1))
    n_out = int(round(x.size / rate)) if rate > 0 else x.size
    if x.size == 0 or n_out <= 0:
        return np.zeros(max(0, n_out), dtype=np.float32)
    if abs(rate - 1.0) <= deadband:
        # nothing to do: pad or trim to the exact target
        return (np.pad(x, (0, n_out - x.size)) if x.size < n_out else x[:n_out])

    L = max(4, int(round(frame_ms * sr / 1000.0)) | 1)
    Hs = max(1, int(round(hop_ms * sr / 1000.0)))
    Ha = max(1, int(round(Hs * rate)))
    S = max(1, int(round(search_ms * sr / 1000.0)))
    if Hs >= L:
        raise ValueError("hop must be smaller than the frame")

    win = np.hanning(L).astype(np.float32)
    xp = np.pad(x, (L, L + 2 * S))
    out = np.zeros(n_out + 2 * L, dtype=np.float32)
    wsum = np.zeros_like(out)

    prev_tail: np.ndarray | None = None
    ana = L                     # analysis position, in padded coordinates
    syn = 0                     # synthesis position, in output coordinates
    while syn < n_out and ana + L < xp.size:
        if prev_tail is None or prev_tail.size == 0:
            pos = ana
        else:
            lo = max(0, ana - S)
            hi = min(xp.size - L, ana + S)
            pos, best = ana, -np.inf
            for cand in range(lo, hi + 1):
                c = float(np.dot(xp[cand:cand + prev_tail.size], prev_tail))
                if c > best:
                    best, pos = c, cand
        out[syn:syn + L] += xp[pos:pos + L] * win
        wsum[syn:syn + L] += win
        # what the next frame will overlap in the output: match against that
        prev_tail = out[syn + Hs:syn + L].copy()
        ana = pos + Ha
        syn += Hs

    m = wsum > 1e-6
    out[m] /= wsum[m]
    y = out[:n_out]
    if y.size < n_out:
        y = np.pad(y, (0, n_out - y.size))
    return np.clip(y, -1.0, 1.0).astype(np.float32)


def fit_to_slot(audio: np.ndarray, slot_s: float, *, sr: int = 24000,
                cap: float = 2.0, min_rate: float = 0.75
                ) -> tuple[np.ndarray, float, bool, bool, bool]:
    """Put ``audio`` in ``slot_s`` seconds, never damaging the timeline.

    Returns ``(audio, rate, clamped, padded, truncated)`` with exactly
    ``slot_s * sr`` samples.

    The first pt-BR build stretched every span to fill its slot in *both*
    directions, and heavy expansion is what destroyed it: at rates near 0.3-0.5
    WSOLA stops resynthesising speech and drones on a repeated pitch period, which
    whisper then transcribes as a run of one character ("Ooooo...", "Peeee...").
    Measured over the 727 rebuilt scenarios, every degraded file was one with
    expansion (rate < 0.75) and every file that only ever compressed was clean.

    So the policy is one-sided: compress up to ``cap`` (and truncate at the slot
    end rather than overflow into a pause), expand at most to ``min_rate``, and
    below that leave the span short and pad silence. The annotations still hold --
    the pause begins when the silence begins; the span simply ends earlier than
    its English counterpart.
    """
    if audio.size == 0 or slot_s <= 0:
        return audio, 1.0, False, False, False
    target = int(round(slot_s * sr))
    rate = (audio.size / sr) / slot_s
    clamped = padded = truncated = False
    if rate > cap:                     # far too long: compress all we allow
        rate, clamped = cap, True
    elif rate < min_rate:              # far too short: do NOT drone, pad instead
        rate, clamped, padded = 1.0, True, True
    y = wsola(audio, rate, sr=sr)
    if y.size < target:
        y = np.pad(y, (0, target - y.size))
        padded = padded or rate < 0.999
    elif y.size > target:
        y, truncated = y[:target], True
    return y.astype(np.float32), rate, clamped, padded, truncated
