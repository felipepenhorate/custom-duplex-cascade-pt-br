"""Assistant text -> real speech, with the deployed TTS settings.

Uses ``pocket-tts`` exactly as the shipped voice service does
(``sft/services/tts_service.py``): ``TTSModel.load_model(language=...)``,
``get_state_for_audio_prompt(<catalog voice>)``, then
``generate_audio_stream`` per text chunk, 24 kHz mono.

Two consequences worth stating, because they replace the text-domain harness's
word-rate clock:

* the duration of an assistant turn is now **measured from the audio that was
  actually produced**, so every FDB duration gate (formal turn at >= 1 s or
  > 3 words, backchannel at <= 3 s and <= 3 words) is applied to real speech
  rather than to a word count times a constant;
* the same text does not always produce the same number of seconds --
  ``pocket-tts`` samples at ``temp=0.7``, which is what the product does -- so
  the audio bench is not bit-reproducible. ``--tts-temp 0.0`` makes it so; the
  default keeps the shipped behaviour.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

import numpy as np

TTS_SR = 24000


def filter_period_for_tts(text: str) -> str:
    """pt-BR-safe period stripping, copied from the bridge.

    Strips only sentence-final periods so abbreviations and decimals survive
    (``1.5``, ``10h30``). The shipped bridge applies it on the way to TTS.
    """
    return re.sub(r"(?<=\S)\.(?=\s|$)", "", text)


def strip_for_speech(text: str) -> str:
    """Make model output speakable: drop markdown, keep the words.

    The duplex models emit markdown ("**TutorMe** - ...") because they are chat
    models first; a voice pipeline reads that as noise. Only the *audio* path is
    cleaned -- the text that reaches the relevance judge is the model's own
    output, untouched, so the transcript the benchmark scores is still what the
    model actually produced.
    """
    t = re.sub(r"^\s*[-*•]\s+", "", text, flags=re.M)   # bullets
    t = re.sub(r"^\s*#{1,6}\s+", "", t, flags=re.M)     # headings
    t = t.replace("**", " ").replace("__", " ")
    t = re.sub(r"(?<!\w)[*_`]+|[*_`]+(?!\w)", " ", t)   # stray emphasis marks
    t = re.sub(r"^\s*\d+[.)]\s+", "", t, flags=re.M)    # numbered lists
    t = re.sub(r"[ \t]{2,}", " ", t)
    return t.strip()


def split_for_tts(text: str, max_words: int = 30) -> list[str]:
    """Cut text into pocket-tts-sized pieces.

    ``pocket-tts`` warns and *skips words* when a generation call carries more
    than 50 of its tokens, and the shipped bridge never hits that limit because it
    forwards the LLM's text stream chunk by chunk. This harness hands over whole
    turns, so it splits on sentence boundaries first and only then on words.
    """
    text = (text or "").strip()
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?…])\s+|\n+", text)
    chunks: list[str] = []
    cur: list[str] = []
    cur_n = 0
    for sent in sentences:
        words = sent.split()
        if not words:
            continue
        if len(words) > max_words:
            if cur:
                chunks.append(" ".join(cur))
                cur, cur_n = [], 0
            for i in range(0, len(words), max_words):
                chunks.append(" ".join(words[i:i + max_words]))
            continue
        if cur_n + len(words) > max_words and cur:
            chunks.append(" ".join(cur))
            cur, cur_n = [], 0
        cur.extend(words)
        cur_n += len(words)
    if cur:
        chunks.append(" ".join(cur))
    return [c for c in chunks if c.strip()]


@dataclass
class SpokenChunk:
    text: str
    audio: np.ndarray  # float32 mono @ 24 kHz
    duration_s: float
    compute_s: float


class AssistantVoice:
    """pocket-tts behind a plain synchronous API."""

    def __init__(
        self,
        language: str = "english",
        device: str = "cuda",
        quantize: bool = False,
        voice: str | None = None,
        temp: float = 0.7,
    ) -> None:
        import torch
        from pocket_tts import TTSModel
        from pocket_tts.default_parameters import get_default_voice_for_language

        self.language = language
        self.temp = float(temp)
        self.model = TTSModel.load_model(language=language, temp=self.temp) if temp != 0.7 \
            else TTSModel.load_model(language=language)
        self.model.to(device)
        self.model.eval()
        self.voice = voice or get_default_voice_for_language(language)
        self.init_state = self.model.get_state_for_audio_prompt(self.voice)
        self.n_chunks = 0
        self.total_audio_s = 0.0
        self.total_compute_s = 0.0
        self._torch = torch

    def speak(self, text: str, *, tts_safe: bool | None = None) -> SpokenChunk:
        """Synthesize one assistant text chunk. Never raises.

        ``tts_safe`` defaults to the bridge's rule being applied for Portuguese
        only: the period stripping in ``filter_period_for_tts`` is pt-BR-specific
        (it exists to stop fragmentation on pt abbreviations), and the English
        voice is left the same as the clock calibration heard it.
        """
        text = (text or "").strip()
        if tts_safe is None:
            tts_safe = self.language == "portuguese"
        if tts_safe:
            text = filter_period_for_tts(text).strip()
        text = strip_for_speech(text)
        pieces_text = split_for_tts(text)
        if not pieces_text:
            return SpokenChunk("", np.zeros(0, dtype=np.float32), 0.0, 0.0)
        t0 = time.monotonic()
        audio_parts: list[np.ndarray] = []
        for piece in pieces_text:
            try:
                for chunk in self.model.generate_audio_stream(self.init_state, piece):
                    if isinstance(chunk, self._torch.Tensor):
                        chunk = chunk.detach().cpu().float().numpy()
                    audio_parts.append(np.asarray(chunk, dtype=np.float32).reshape(-1))
            except Exception:  # noqa: BLE001 - keep the scenario alive
                continue
        audio = np.concatenate(audio_parts) if audio_parts else np.zeros(0, dtype=np.float32)
        compute = time.monotonic() - t0
        dur = audio.size / TTS_SR
        self.n_chunks += 1
        self.total_audio_s += dur
        self.total_compute_s += compute
        return SpokenChunk(text, audio, dur, compute)
