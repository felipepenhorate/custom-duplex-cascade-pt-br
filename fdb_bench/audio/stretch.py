"""Pitch-preserving time stretch, without librosa/numba.

The pt-BR audio set has to keep the English timeline, so each translated span is
stretched to its slot. This began as a hand-written WSOLA and that was a mistake:
measured on pocket-tts speech it transcribes perfectly at rate 1.0 and becomes a
100 Hz buzz at 1.2 (`Ta-ra-ra-ra...`), i.e. the overlap-add repeats the synthesis
hop. It passed a pure-tone test -- which is exactly why it got shipped: a tone
survives a broken OLA, speech does not.

The phase vocoder here is torchaudio's, which needs no numba (librosa's would,
and the numba wheel refuses the installed numpy). Verified on real synthesised
speech: at rates 1.2 / 1.4 / 0.85 / 0.6 the transcript stays intact; the only
change is the trailing word being cut by the shorter window.

``fit_to_slot`` is one-sided on purpose. Compress up to ``cap`` and truncate at the
slot end rather than overflow a pause; expand at most to ``min_rate``; below that
leave the span short and pad silence.
"""

from __future__ import annotations

import numpy as np

N_FFT = 1024
HOP = 256


def time_stretch(x: np.ndarray, rate: float, *, sr: int = 24000,
                 deadband: float = 0.02) -> np.ndarray:
    """Stretch ``x`` by ``rate`` (>1 shortens, <1 lengthens) keeping pitch.

    Returns exactly ``round(len(x)/rate)`` samples.
    """
    x = np.ascontiguousarray(np.asarray(x, dtype=np.float32).reshape(-1))
    if x.size == 0:
        return x
    n_out = int(round(x.size / rate)) if rate > 0 else x.size
    if abs(rate - 1.0) <= deadband:
        return (np.pad(x, (0, n_out - x.size)) if x.size < n_out else x[:n_out])

    import torch
    import torchaudio

    t = torch.from_numpy(x)
    win = torch.hann_window(N_FFT)
    spec = torch.stft(t, n_fft=N_FFT, hop_length=HOP, win_length=N_FFT,
                      window=win, return_complex=True)
    phase_advance = torch.linspace(0, np.pi * HOP, N_FFT // 2 + 1).view(1, -1, 1)
    moved = torchaudio.functional.phase_vocoder(spec, rate=rate,
                                                phase_advance=phase_advance)
    y = torch.istft(moved, n_fft=N_FFT, hop_length=HOP, win_length=N_FFT,
                    window=win).numpy().astype(np.float32)
    if y.size < n_out:
        y = np.pad(y, (0, n_out - y.size))
    else:
        y = y[:n_out]
    return np.clip(y, -1.0, 1.0).astype(np.float32)


def fit_to_slot(audio: np.ndarray, slot_s: float, *, sr: int = 24000,
                cap: float = 2.0, min_rate: float = 0.75
                ) -> tuple[np.ndarray, float, bool, bool, bool]:
    """Put ``audio`` in ``slot_s`` seconds, never damaging the timeline.

    Returns ``(audio, rate, clamped, padded, truncated)`` with exactly
    ``slot_s * sr`` samples. See the module docstring for why the policy is
    one-sided.
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
    y = time_stretch(audio, rate, sr=sr)
    if y.size < target:
        y = np.pad(y, (0, target - y.size))
        padded = padded or rate < 0.999
    elif y.size > target:
        y, truncated = y[:target], True
    return y.astype(np.float32), rate, clamped, padded, truncated
