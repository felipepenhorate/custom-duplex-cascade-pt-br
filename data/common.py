"""Shared datums for the DuplexCascade-v2-PT data pipeline.

Special-token vocabulary (exact paper names, `<| |>` delimited like the
original inference server), loss weights from the paper §4.1, pt-BR
backchannel lexicon, and small text helpers.
"""

from __future__ import annotations

import re
import unicodedata

# ---------------------------------------------------------------------------
# Conversational special tokens (paper §3.2 + §4.1; aligned with server.py v2)
# ---------------------------------------------------------------------------
TOKEN = {
    "no_voice": "<|no voice|>",
    "user_is_speaking": "<|user is speaking|>",
    "user_finish_speaking": "<|user finish speaking|>",
    "user_is_thinking": "<|user is thinking|>",
    "user_interruption": "<|user interruption|>",
    "user_backchannel": "<|user backchannel|>",
    "system_backchannel": "<|system backchannel|>",
}

SPECIAL_TOKENS = list(TOKEN.values())

# Loss weights (paper §4.1):  <user is speaking>=1, <user finish
# speaking>=10, <user interruption>=5, <user backchannel>=2,
# <user is thinking>=1, <system backchannel>=3.
TOKEN_WEIGHT = {
    "user_is_speaking": 1,
    "user_finish_speaking": 10,
    "user_interruption": 5,
    "user_backchannel": 2,
    "user_is_thinking": 1,
    "system_backchannel": 3,
}

# pt-BR conversational backchannels used to simulate the "user backchannel"
# phenomenon (paper §3.3.2, English examples "yes"/"okay").
PT_BACKCHANNELS = [
    "aham",
    "uhum",
    "tá",
    "sim",
    "certo",
    "ok",
    "okay",
    "entendi",
    "é",
    "claro",
    "pois é",
    "né",
    "sei",
    "hmm",
    "ah tá",
]

# Marker used to supervise <system backchannel> on user turns (paper: <BC/>
# inserted by Qwen2-72B-Instruct after sentence-level units).
BC_MARKER = "<BC/>"

# ChatML framing used by the inference server (server.py prepare_llm_tokens:
# <|im_start|>user\n ... <|im_end|>\n and <|im_start|>assistant\n ...).
IM_START_USER = "<|im_start|>user\n"
IM_START_ASSISTANT = "<|im_start|>assistant\n"
IM_END = "<|im_end|>"
IM_END_NL = "<|im_end|>\n"

# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

_PT_STOPWORDS = frozenset(
    "de a o que e do da em um para é com não uma os no se na por mais as dos"
    " como mas foi ao ele das tem à seu sua ou ser quando muito há nos já está"
    " eu você pra tá né a gente meu minha esse essa isso isto este esta aquilo"
    " aquela onde como porque então também sobre entre depois antes durante"
    " mesmo cada outro outra pouco muito menos mais ainda assim também até"
    " atrás frente lado aqui ali lá agora ontem hoje amanhã sempre nunca"
    " alguém alguem ninguém todos todas tudo nada algo qualquer cada próprio"
    " deve pode deveria poderia quer quero vamos vou ir fazer feito diz disse"
    " falar falou gosto gosta acho acha sei sabe viu olha espera calma "
    " obrigado obrigada por favor desculpa".split()
)

_EN_STOPWORDS = frozenset(
    "the a an and or but of to in for on with from by at as is are was were"
    " be been being have has had do does did will would can could should"
    " shall may might must i you he she it we they them his her its our your"
    " their this that these those what which who whom whose when where why"
    " how not no yes so if then than because about into over under again"
    " further once here there all any both each few more most other some"
    " such only own same too very just also get got make made way people"
    " know knew think thought say said speak spoke".split()
)

_ACCENT_CHARS = set("áàâãäéèêëíìîïóòôõöúùûüçñ")


def remove_accents(text: str) -> str:
    return "".join(
        c
        for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )


def norm_text(text: str) -> str:
    """Collapse whitespace/punctuation-space and lowercase, keeping accents."""
    t = re.sub(r"\s+", " ", text)
    t = re.sub(r"\s+([.,!?;:])", r"\1", t)
    return t.strip().lower()


def is_portuguese(text: str, min_words: int = 8) -> bool:
    """Heuristic pt-BR detection via stopword ratio + accent evidence."""
    words = re.findall(r"[a-záàâãäéèêëíìîïóòôõöúùûüçñ']+", remove_accents(text).lower())
    words = [w for w in words if len(w) > 1]
    if len(words) < min_words:
        return False
    pt = sum(1 for w in words if w in _PT_STOPWORDS)
    en = sum(1 for w in words if w in _EN_STOPWORDS)
    if pt - en < 4 and pt < en:
        return False
    if pt == 0:
        return False
    accents = sum(1 for ch in text if ch in _ACCENT_CHARS)
    if accents == 0 and pt < 6:
        return False
    return True


_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+|\n+")


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENT_SPLIT.split(text)]
    return [p for p in parts if p]


def sha256_dedup_key(text: str) -> str:
    import hashlib

    return hashlib.sha256(norm_text(text).encode("utf-8")).hexdigest()