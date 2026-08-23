"""Duplex-trigger system prompt + ChatML prompt builder for the SDFT teacher.

M1 — the teacher (frozen base model, Qwen/Qwen3-4B-Instruct-2507) generates the
assistant content that becomes the "demonstration" of the duplex protocol. The
system message frames the model as a DuplexCascade full-duplex voice assistant.

Because the base model has NOT been fine-tuned on the micro-turn protocol, its
generations will not (yet) follow it faithfully — that is expected and by design
(SPEC §4.2). We only use the teacher's *content*; the deterministic duplex builder
(data/build_duplex_dataset.py) places the special tags afterwards.

The content-only variant (default) asks for the bare speech so we never have to
strip tags the teacher occasionally emits. The `--allow-tags` variant lets the
teacher attempt the full protocol (used for the "ICL assumption" ablation of the
trigger prompt, SPEC §8).
"""

from __future__ import annotations

TRIGGER_VERSION = "v2"

TRIGGER_SYSTEM = (
    "Você é um assistente de voz full-duplex do sistema DuplexCascade. "
    "Numa conversa duplex, sistema e usuário falam em micro-turnos curtos e "
    "naturais, com sobreposição, como numa conversa de voz real. "
    "Você gerencia a tomada de turno com tags especiais: "
    "<|user is speaking|> (usuário está falando), "
    "<|user finish speaking|> (usuário terminou de falar), "
    "<|user is thinking|> (usuário está pensando), "
    "<|user interruption|> (usuário interrompeu), "
    "<|user backchannel|> (usuário faz um backchannel), "
    "<|system backchannel|> (backchannel do sistema) e "
    "<|no voice|> (silêncio)."
)

TRIGGER_SYSTEM_CONTENT_ONLY = TRIGGER_SYSTEM + (
    " Responda de forma natural e completa, em português brasileiro coloquial, "
    "como uma fala falada em voz alta. Dê uma resposta coesa que responda de "
    "fato ao que o usuário perguntou, geralmente em uma a três frases curtas. "
    "Responda apenas com a fala — não imprima nenhuma tag especial. "
    "Não se repita: diga cada informação uma única vez."
)

TRIGGER_SYSTEM_ALLOW_TAGS = TRIGGER_SYSTEM + (
    " Responda de forma natural e completa, em português brasileiro coloquial, "
    "como uma fala falada em voz alta. Siga o protocolo duplex e use as tags "
    "especiais quando fizer sentido. Dê uma resposta coesa que responda de fato "
    "ao que o usuário perguntou, geralmente em uma a três frases curtas. "
    "Não se repita: diga cada informação uma única vez."
)


def trigger_system(allow_tags: bool = False) -> str:
    return TRIGGER_SYSTEM_ALLOW_TAGS if allow_tags else TRIGGER_SYSTEM_CONTENT_ONLY


def build_teacher_messages(history: list[dict], allow_tags: bool = False) -> list[dict]:
    """ChatML messages for one assistant turn of the teacher.

    `history` is the conversation so far (user + already-generated assistant
    turns, ending with the user turn that this reply answers). A system message
    carrying the duplex-cascade trigger is prepended.
    """
    return [
        {"role": "system", "content": trigger_system(allow_tags)},
        *history,
    ]


def strip_duplex_tags(text: str) -> str:
    """Remove any duplex special tokens the teacher may have emitted.

    The content-only prompt should prevent them, but `allow_tags` mode and
    off-protocol generations occasionally leak tags into the text. The
    deterministic builder re-places tags anyway, so we drop any that appear.
    """
    from data.common import SPECIAL_TOKENS

    out = text
    for tok in SPECIAL_TOKENS:
        out = out.replace(tok, "")
    return out