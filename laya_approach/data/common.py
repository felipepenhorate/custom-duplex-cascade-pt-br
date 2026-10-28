"""Shared schema for the Laya duplex companion.

The companion is a typed decision model, not a causal LM.  A row is the Laya
native format::

    {"state": {...}, "questions": {"action": question}, "gold": {"action": tag}}

The state deliberately exposes the *stream event* (ASR partial, finalized
chunk, or silence) rather than forcing the encoder to infer it from the last
word.  This is the fix for the false ``user_finish_speaking`` floors measured
in ``results_audio*``: ASR partials very often end in ``. ? ! …``, so terminal
punctuation alone is not a turn-end signal.

Labels are the protocol tag names from ``mtp_like/data/common.py``, plus
``no_tag`` (emit nothing).
"""

from __future__ import annotations

import re
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# Protocol vocabulary
# ---------------------------------------------------------------------------

NO_TAG = "no_tag"

# The tags this companion emits.  ``no_voice`` is an input-only marker and is
# never a label.  The promptable-controller tags from mtp_like v6 are omitted:
# this approach is about stream-native turn taking, not rule following.
ACTION_KEYS = [
    NO_TAG,
    "user_is_speaking",
    "user_finish_speaking",
    "user_interruption",
    "user_backchannel",
    "user_is_thinking",
    "system_backchannel",
]

PROTOCOL_TAGS = [key for key in ACTION_KEYS if key != NO_TAG]

NO_VOICE = "<|no voice|>"

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

# Terminal punctuation we care about.  Quotes/brackets after the mark are
# stripped before the test, because ASR writes things like ``"sério?"``.
_TERMINAL = ".?!…"
_TRAILING = "\"'”’»)]}"

# ---------------------------------------------------------------------------
# Typed question
# ---------------------------------------------------------------------------

ACTION_CRITERIA = {
    NO_TAG: (
        "Nada deve ser emitido. O sistema continua no estado atual; por "
        "exemplo, o usuário está em silêncio enquanto o sistema fala."
    ),
    "user_is_speaking": (
        "O usuário ainda está produzindo o turno. Espere. Use esta ação para "
        "qualquer evento parcial (partial), mesmo que o texto termine em "
        "pontuação, e para pausas no meio do turno."
    ),
    "user_finish_speaking": (
        "O turno do usuário foi finalizado (evento final com ends_turn=true). "
        "Pare de esperar e comece a resposta agora. A pontuação ajuda, mas o "
        "sinal decisivo é o fim do turno."
    ),
    "user_interruption": (
        "O usuário começou a falar conteúdo novo enquanto o sistema ainda "
        "estava falando. Pare a fala do sistema e devolva o turno."
    ),
    "user_backchannel": (
        "O usuário fez apenas um reconhecimento curto (por exemplo 'aham', "
        "'tá', 'entendi') enquanto o sistema falava. Não é interrupção; "
        "continue falando."
    ),
    "user_is_thinking": (
        "O usuário está em silêncio para pensar ou o sistema aguarda o "
        "próximo turno. Espere sem falar."
    ),
    "system_backchannel": (
        "O sistema deve emitir um backchannel curto de escuta. Use com "
        "moderação, apenas quando a fala do usuário pedir esse sinal."
    ),
}

ACTION_QUESTION = {
    "type": "choice",
    "instructions": (
        "Você controla a tomada de turno de um assistente de voz em "
        "português brasileiro. Escolha a próxima ação de protocolo depois do "
        "evento mais recente. O evento informa se o texto é um parcial do "
        "ASR, um trecho finalizado, ou silêncio; 'ends_turn' diz se o turno "
        "do usuário terminou. Pontuação terminal NÃO significa fim de turno: "
        "parciais do ASR frequentemente terminam em '.', '?' ou '!'. Se o "
        "sistema está falando, um reconhecimento curto é backchannel; "
        "conteúdo novo é interrupção."
    ),
    "criteria": ACTION_CRITERIA,
}


def action_question() -> dict:
    """A fresh copy, so callers cannot mutate the shared schema."""
    return {
        "type": "choice",
        "instructions": ACTION_QUESTION["instructions"],
        "criteria": dict(ACTION_CRITERIA),
    }


# ---------------------------------------------------------------------------
# State construction
# ---------------------------------------------------------------------------

_RECENT_ITEMS = 6
_RECENT_CHARS = 170
# One event is normally an ASR piece (a few words).  Cap it so a caller that
# hands over a whole utterance cannot push the fixed head/question out of the
# 640-token window; the cap is far above any training chunk.
_EVENT_CHARS = 300
_WS = re.compile(r"\s+")


def clean_text(text: Any) -> str:
    return _WS.sub(" ", str(text or "")).strip()


def terminal_punctuation(text: str) -> str:
    """The terminal mark of ``text`` after stripping closing quotes/brackets."""
    clean = clean_text(text).rstrip(_TRAILING)
    return clean[-1] if clean and clean[-1] in _TERMINAL else ""


def _normalise_kind(kind: Any, ends_turn: Any = None, text: str = "") -> str:
    kind = str(kind or "").lower()
    if kind in {"silence", "partial", "final"}:
        return kind
    if kind in {"chunk", "speech", "words", ""}:
        return "final" if bool(ends_turn) else ("partial" if clean_text(text) else "silence")
    return kind


def normalise_event(event: dict) -> dict:
    text = clean_text(event.get("text", ""))
    kind = _normalise_kind(event.get("kind"), event.get("ends_turn"), text)
    if kind == "silence":
        text = ""
    text = text[:_EVENT_CHARS]
    return {
        "kind": kind,
        "ends_turn": bool(event.get("ends_turn", kind == "final")),
        "text": text,
        "system_speaking": bool(event.get("system_speaking", False)),
        "stream": str(event.get("stream") or "text"),
        "user_turn_state": event.get("user_turn_state"),
    }


def normalise_history_item(item: dict) -> dict | None:
    role = str(item.get("role", "")).lower()
    if role not in {"user", "assistant"}:
        return None
    text = clean_text(item.get("text", ""))
    kind = _normalise_kind(item.get("kind"), item.get("ends_turn"), text)
    if role == "user" and kind == "silence":
        return {"role": "user", "kind": "silence"}
    if not text:
        return None
    out = {"role": role, "text": text[:_RECENT_CHARS]}
    if role == "user":
        out["kind"] = kind
        out["ends_turn"] = bool(item.get("ends_turn", kind == "final"))
    return out


def _last_user_state(history: list[dict]) -> str:
    """What the turn state was immediately before the current event."""
    for i in range(len(history) - 1, -1, -1):
        item = history[i]
        if item["role"] != "user" or item.get("kind") == "silence":
            continue
        after = history[i + 1 :]
        if any(x["role"] == "assistant" for x in after):
            return "answered"
        if item.get("ends_turn"):
            return "awaiting_answer"
        return "in_turn"
    return "none"


def build_state(
    event: dict,
    history: Iterable[dict] = (),
    *,
    language: str = "pt-BR",
) -> dict:
    """Build the compact decision state from one event plus prior items.

    ``event`` is the newest item (the one the model must act on); ``history``
    is everything before it, in chronological order.  In the runtime adapter
    this is exactly ``items[-1]`` and ``items[:-1]``.
    """
    ev = normalise_event(event)
    recent = [x for x in (normalise_history_item(i) for i in history) if x][-_RECENT_ITEMS:]
    prior_turn = _last_user_state(recent)
    kind, text = ev["kind"], ev["text"]
    mark = terminal_punctuation(text)

    if kind == "silence":
        if prior_turn == "awaiting_answer":
            turn_state = "complete_awaiting_answer"
        elif prior_turn == "in_turn":
            turn_state = "in_turn"
        elif prior_turn == "answered":
            turn_state = "answered"
        else:
            turn_state = "none"
    elif kind == "final" and ev["ends_turn"]:
        turn_state = "complete_now"
    elif kind == "final":
        turn_state = "ended_without_final_signal"
    else:
        turn_state = "in_turn"

    state = {
        "language": language,
        "stream": ev.get("stream") or "text",
        "event": {
            "kind": kind,
            "ends_turn": ev["ends_turn"],
            "text": text,
            "has_words": bool(text),
            "word_count": len(text.split()),
            "terminal_punctuation": bool(mark),
            "punctuation": mark,
            "question": text.rstrip(_TRAILING).endswith("?"),
            "system_speaking": ev["system_speaking"],
            "user_turn_state": turn_state,
        },
        "recent": recent,
    }
    return state


def item_event(item: dict) -> dict:
    """Translate one runtime item into the ``event`` shape used by training."""
    role = str(item.get("role", "")).lower()
    text = clean_text(item.get("text", ""))
    if role == "assistant":
        return {
            "kind": "assistant",
            "ends_turn": False,
            "text": text,
            "system_speaking": True,
        }
    if text == NO_VOICE:
        return {
            "kind": "silence",
            "ends_turn": False,
            "text": "",
            "system_speaking": bool(item.get("system_speaking", False)),
        }
    kind = item.get("kind") or item.get("event_kind")
    if kind == "chunk":
        kind = None
    if kind is None:
        kind = "final" if item.get("ends_turn") else "partial"
    return {
        "kind": kind,
        "ends_turn": bool(item.get("ends_turn", kind == "final")),
        "text": text,
        "system_speaking": bool(item.get("system_speaking", False)),
    }


def state_from_items(
    items: list[dict], *, language: str = "pt-BR", stream: str = "text"
) -> tuple[dict, dict]:
    """Runtime helper: ``(state, event)`` for the newest item in ``items``."""
    if not items:
        event = {"kind": "silence", "ends_turn": False, "text": "",
                 "system_speaking": False, "stream": stream}
        return build_state(event, (), language=language), event
    event = item_event(items[-1])
    event["stream"] = stream
    return build_state(event, items[:-1], language=language), event


# ---------------------------------------------------------------------------
# Decision policy
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLDS = {
    NO_TAG: 0.0,
    "user_is_speaking": 0.35,
    "user_finish_speaking": 0.45,
    "user_interruption": 0.50,
    "user_backchannel": 0.55,
    "user_is_thinking": 0.45,
    "system_backchannel": 0.65,
}


def finish_is_allowed(event: dict, state: dict) -> bool:
    """Guard the answer trigger; this is the runtime analogue of the data fix.

    ``user_finish_speaking`` is allowed only when the stream says the turn is
    over (a final event with ``ends_turn``), or when silence follows a turn
    that was already complete.  A partial with punctuation never qualifies.
    """
    if event.get("system_speaking"):
        # Taking the floor while the system is still audible starts a second
        # stream on top of the first.  The correct action is interruption (or
        # a backchannel), not finish.
        return False
    if bool(event.get("ends_turn")):
        return True
    if event.get("kind") != "silence":
        return False
    turn = state.get("event", {}).get("user_turn_state")
    return turn == "complete_awaiting_answer"


def resolve_action(
    probs: dict[str, float],
    event: dict,
    state: dict,
    thresholds: dict[str, float] | None = None,
) -> tuple[str, dict[str, float]]:
    """Constrained argmax: the best tag that passes thresholds and guards.

    Falling back to the next-best tag (rather than collapsing straight to
    ``no_tag``) matters for the exact ambiguity the audio traces exposed: on a
    partial ending in ``?`` the model may rank ``user_finish_speaking`` first
    with ``user_is_speaking`` a close second.  The guard vetoes finish, and the
    correct action is to keep listening -- which is the second tag, not
    nothing.  ``no_tag`` is the answer only when no admissible tag clears its
    threshold.
    """
    th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    for tag, prob in ranked:
        if tag == NO_TAG:
            # ``no_tag`` is the fallback, never a candidate that outranks a
            # confident tag; if it is genuinely on top, nothing should fire.
            return NO_TAG, probs
        if prob < th.get(tag, 0.5):
            continue
        if tag == "user_finish_speaking" and not finish_is_allowed(event, state):
            continue
        if tag == "user_interruption" and not event.get("system_speaking"):
            continue
        if tag == "user_backchannel" and not event.get("system_speaking"):
            continue
        return tag, probs
    return NO_TAG, probs


# ---------------------------------------------------------------------------
# Evaluation buckets (shared by the data audit, eval and probe scripts)
# ---------------------------------------------------------------------------

def event_bucket(event: dict, state: dict) -> str:
    kind = event.get("kind")
    mark = bool(state["event"]["terminal_punctuation"])
    speaking = bool(event.get("system_speaking"))
    if kind == "silence":
        turn = state["event"]["user_turn_state"]
        if speaking:
            return "silence_during_system"
        if turn == "complete_awaiting_answer":
            return "silence_after_final"
        if turn == "in_turn":
            return "silence_mid_turn"
        return "silence_other"
    if kind == "partial":
        if speaking:
            return "partial_during_system_punct" if mark else "partial_during_system"
        return "partial_terminal_punct" if mark else "partial_no_punct"
    if kind == "final":
        if speaking:
            return "final_during_system"
        if not state["event"]["ends_turn"]:
            # The ASR finalized a piece, but the stream says the turn goes on
            # (a long utterance is delivered in several final pieces, and the
            # audio harness marks every non-last one ends_turn=False).
            return "final_mid_turn"
        if not state["event"]["text"]:
            return "empty_final"
        return "final_with_punct" if mark else "final_no_punct"
    return "other"
