"""User-side chunking: reproduce the *training* input distribution.

Why this module exists
----------------------
The duplex models never see whole sentences. ``DuplexChunker.user_chunks`` in
``*/data/build_duplex_dataset.py`` delivers each user turn as a stream of tiny
pieces: it splits the turn into sentences, tokenizes each sentence, then emits
consecutive chunks of ``rng.randint(1, user_chunk_max)`` **tokens**
(``user_chunk_max = 7``). Measured over 3000 training records:

    user chunks n=197106  mean=2.19 words  median=2  p90=3  max=7
    100% of chunks are <= 7 words

The first version of this harness fed whole sentences (10-25 words) as if they
were ASR partials — a 5-10x shift on the model's primary input, and the single
largest fidelity bug in the bench. On that input the distill model produces
degenerate, rambling continuations and looks far worse than it is; once the
stream matches training, it behaves (see ``PROMPTS.md`` and ``README.md``).

``TrainingChunker`` is a faithful port of the training algorithm and needs the
real tokenizer, because the cut points are token-level (so pieces can end
mid-word, exactly as in training, and carry the same leading-space artefacts).
``word_chunker`` is a dependency-free approximation with a matched length
distribution, used when no tokenizer is available (scenario unit tests).
"""

from __future__ import annotations

import random
import re
from typing import Callable, Protocol

# From DuplexChunker.__init__ (both sft/ and distill/ copies).
USER_CHUNK_MAX = 7

_SENT_RE = re.compile(r"(?<=[.!?…])\s+")
_TERMINAL_RE = re.compile(r"[.!?…]\s*$")


class Chunker(Protocol):
    """``(text, rng) -> list[str]`` — one user turn into ASR-like pieces."""

    def __call__(self, text: str, rng: random.Random) -> list[str]: ...


def _clean(text: str) -> str:
    """Same normalisation the training data builder applies."""
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> list[str]:
    """``DuplexChunker._split_sentences``, verbatim."""
    parts = _SENT_RE.split(_clean(text))
    return [p.strip() for p in parts if p.strip()]


class TrainingChunker:
    """Faithful port of ``DuplexChunker.user_chunks``.

    Token-level, so it reproduces the training stream exactly: pieces of 1-7
    tokens, cut mid-sentence and even mid-word, with the leading-space artefacts
    the model was trained on.
    """

    def __init__(self, tokenizer, user_chunk_max: int = USER_CHUNK_MAX) -> None:
        self.tok = tokenizer
        self.user_chunk_max = int(user_chunk_max)

    def __call__(self, text: str, rng: random.Random) -> list[str]:
        chunks: list[str] = []
        for sent in split_sentences(text):
            ids = self.tok.encode(sent, add_special_tokens=False)
            pos = 0
            while pos < len(ids):
                # identical expression to the training code, including the
                # "remaining tail becomes one chunk" branch
                if pos + self.user_chunk_max < len(ids):
                    n = rng.randint(1, self.user_chunk_max)
                else:
                    n = len(ids) - pos
                pieces = self.tok.convert_ids_to_tokens(ids[pos:pos + n])
                chunk = self._detokenize(ids[pos:pos + n], pieces)
                if chunk:
                    chunks.append(chunk)
                pos += n
        return chunks or [text]

    def _detokenize(self, ids: list[int], pieces: list[str]) -> str:
        """Decode one piece the way the training data records it.

        The training items are built with ``DuplexChunker.display`` ->
        ``tok.decode(ids, skip_special_tokens=False)``, which is why the stream
        carries leading spaces. We keep that: the whole point is to feed the
        model the shape it was trained on.
        """
        return self.tok.decode(ids, skip_special_tokens=False,
                               clean_up_tokenization_spaces=False)


def word_chunker(text: str, rng: random.Random) -> list[str]:
    """Tokenizer-free approximation with a matched length distribution.

    Draws 1-3 words per piece (mean ~2, matching the measured 2.19) with an
    occasional tail up to 7, so scenario tests exercise the same shape without
    pulling in transformers. Prefer :class:`TrainingChunker` for real runs.
    """
    words = _clean(text).split()
    if not words:
        return [text]
    out: list[str] = []
    i = 0
    while i < len(words):
        remaining = len(words) - i
        n = min(rng.choice([1, 1, 2, 2, 2, 3, 3, 5, 7]), remaining)
        out.append(" ".join(words[i:i + n]))
        i += n
    return out


def ends_sentence(text: str) -> bool:
    """True when this piece ends a sentence (used for the backchannel GT)."""
    return bool(_TERMINAL_RE.search(text))


def default_chunker() -> Callable[[str, random.Random], list[str]]:
    return word_chunker
