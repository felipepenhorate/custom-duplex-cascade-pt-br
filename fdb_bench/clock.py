"""Virtual speech clock: turns pt-BR text into the seconds FDB metrics need.

Full-Duplex-Bench's TOR/latency thresholds are expressed in seconds of audio
(1 s speech duration, 3 s "formal speaking" segment) and in words (> 3 words).
These models make every turn-taking decision in *text* (LLM → TTS), so a
simulator needs a clock to place those decisions on an audio-like timeline.

Model
-----
``SpeechClock`` converts a piece of spoken text into a duration as

    duration(text) = max(MIN, n_words(text) / words_per_second)

with one calibrated constant, ``words_per_second`` (WPS), defaulting to the
measured **3.25**. Calibration (``python -m fdb_bench.calibrate_clock``)
synthesizes scenario utterances with ``pocket_tts`` — the same TTS the live
bridge uses — and times them:

    ==========  ======  =========  ============  ==========
    lang        mean    median     trimmed mean  chosen
    ==========  ======  =========  ============  ==========
    pt          4.34    3.69       3.66          3.33
    en          3.36    3.28       3.34          3.17
    ==========  ======  =========  ============  ==========

``chosen`` is half the trimmed mean, half the 3.0 words/s anchor implied by
FDB's own released ASR timestamps (~180 wpm). The two languages land within 5%
of each other, so the bench uses a single constant for both. Note the measured
*rate rises with utterance length* — fixed TTS overhead amortizes over more
words — and very short utterances occasionally come back truncated, which is
why the robust trimmed mean is used rather than the raw mean.

Reproduce with::

    python -m fdb_bench.calibrate_clock --n 24 --lang pt
    python -m fdb_bench.calibrate_clock --n 24 --lang en \
        --out fdb_bench/clock_calibration_en.json

(``pocket_tts`` samples with ``temp=0.7``, so repeat runs vary by ~±0.2 w/s;
the trimmed estimator keeps that jitter out of the constant.) Because one
constant cannot be simultaneously right for every utterance, sweep it with
``--words-per-second`` to check that no conclusion flips.

Everything downstream (``latency_s``, TOR duration gates, backchannel ``freq``)
derives from this one constant, so ``--words-per-second`` gives a clean
sensitivity sweep. ``simulate.py`` additionally records unit-free
``latency_chunks`` (how many user micro-turns the model sat through before
taking the floor), which is invariant to the clock by construction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# A single short word ("ah", "né") still takes ~0.35 s to say; anything shorter
# reads as instantaneous, which breaks the backchannel duration gates.
MIN_SPEECH_S = 0.35

# Backchannel segments: FDB treats a model segment as a *backchannel* (rather
# than a formal turn) when it is <= 3 s and <= 3 words.
BACKCHANNEL_MAX_S = 3.0
BACKCHANNEL_MAX_WORDS = 3

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def words(text: str) -> list[str]:
    """Word tokens: Unicode letters only (pt-BR accents survive, punctuation does not)."""
    return _WORD_RE.findall(_PUNCT_RE.sub(" ", text.lower()))


def n_words(text: str) -> int:
    return len(words(text))


@dataclass
class SpeechClock:
    """Text → duration, with a single calibrated rate constant."""

    words_per_second: float = 3.25

    def duration(self, text: str) -> float:
        """Seconds needed to say ``text``."""
        n = n_words(text)
        if n == 0:
            return 0.0
        return max(MIN_SPEECH_S, n / self.words_per_second)

    def is_backchannel(self, text: str, duration: float | None = None) -> bool:
        """FDB's acoustic definition of a backchannel segment.

        ``eval_backchannel.py``: a segment counts as formal speaking when it
        exceeds 3 s, or when it carries more than 3 words.
        """
        dur = self.duration(text) if duration is None else duration
        return dur <= BACKCHANNEL_MAX_S and n_words(text) <= BACKCHANNEL_MAX_WORDS

    def is_formal_turn(self, text: str, duration: float | None = None) -> bool:
        """FDB's definition of "the model took a formal turn".

        ``eval_pause_handling.py`` / ``eval_smooth_turn_taking.py``:
        TOR = 1 when the model's output spans >= 1 s, or when it is shorter but
        carries more than 3 words; otherwise the output counts as a non-takeover
        (a backchannel / hesitation).
        """
        dur = self.duration(text) if duration is None else duration
        if dur < 1.0:
            return n_words(text) > BACKCHANNEL_MAX_WORDS
        return True
