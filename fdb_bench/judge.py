"""Interruption-relevance judge (FDB's "GPT-4o" column), served by a local GGUF.

FDB uses ``gpt-4-turbo`` to score how relevant the model's post-interruption
speech is to the interrupting turn, on a 0-5 scale. This module keeps FDB's
rubric *verbatim* and swaps in a locally served judge so the whole bench runs
offline and deterministically.

Default judge: ``/mnt/f/GGUF/Qwen_3.5-9B/Qwen3.5-9B-UD-Q4_K_XL.gguf`` (served by
llama.cpp, ``--jinja`` so its chat template applies).

Note on comparability: the judge score is *not* on the same absolute scale as
GPT-4o's. It is a consistent, reproducible stand-in — good for ranking the
models under test against each other, not for claiming parity with the
published 4.016. The table marks this column accordingly.
"""

from __future__ import annotations

import re
from typing import Any

# FDB's exact rubric, from v1_v1.5/evaluation/eval_user_interruption.py.
FDB_SYSTEM_MSG = """
   The scenario is that the user and AI are talking in the spoken conversation.
   The user first speaks, then the AI responds. But when AI is speaking, the user interrupts the AI's turn.
   Your task is to rate the quality of AI's response after the user interrupt the turn.


   Below is the rating guideline (from 0 to 5, 0 is the worst and 5 is the best):
   - 0: The AI's response is totally unrelated to the user's interrupting turn.
   - 1: The AI's response is not related to the user's interrupting turn.
   - 2: The AI's response is slightly related to the user's interrupting turn.
   - 3: The AI's response is related to the user's interrupting turn.
   - 4: The AI's response is highly related to the user's interrupting turn.
   - 5: The AI's response is perfectly related to the user's interrupting turn.


   Firstly, briefly analyze the user's interrupting turn and the AI's response
   Then, you must return the overall output as the following format:
   Analysis: [Your analysis].
   I would rate the AI's response as [Rating].
   """

# pt-BR translation for the pt-BR run (same 0-5 scale, same output format).
PT_SYSTEM_MSG = """
   O cenário é que o usuário e a IA estão conversando por voz.
   O usuário fala primeiro, a IA responde. Mas enquanto a IA está falando, o usuário interrompe o turno da IA.
   Sua tarefa é avaliar a qualidade da resposta da IA depois que o usuário interrompeu o turno.


   Abaixo está a diretriz de avaliação (de 0 a 5, 0 é o pior e 5 é o melhor):
   - 0: A resposta da IA é totalmente não relacionada ao turno de interrupção do usuário.
   - 1: A resposta da IA não tem relação com o turno de interrupção do usuário.
   - 2: A resposta da IA tem uma relação leve com o turno de interrupção do usuário.
   - 3: A resposta da IA tem relação com o turno de interrupção do usuário.
   - 4: A resposta da IA tem uma relação alta com o turno de interrupção do usuário.
   - 5: A resposta da IA tem uma relação total com o turno de interrupção do usuário.


   Primeiro, analise brevemente o turno de interrupção do usuário e a resposta da IA.
   Depois, você deve retornar a saída exatamente neste formato:
   Analysis: [Sua análise].
   I would rate the AI's response as [Nota].
   """

SYSTEM_MSG = {"en": FDB_SYSTEM_MSG, "pt": PT_SYSTEM_MSG}

# ---------------------------------------------------------------------------
# Whole-dataset scoring.
#
# FDB v1.0 consults the judge in exactly ONE of its four dimensions (the
# interruption TOR/GPT-4o column), because it scores the other three
# behaviourally from audio timings. This harness keeps that rubric verbatim for
# the interruption column, and additionally asks the judge about *every*
# scenario in which the model spoke, so quality is measured over the whole
# dataset rather than 200 of 727.
#
# Scenarios where the model correctly said nothing are NOT sent to the judge —
# their score is determined by the task, not by an opinion:
#   pause_* / backchannel_icc: silence is the correct behaviour  -> 5
#   smooth_turn / interruption: silence is a failure to respond  -> 0
# That is both cheaper and less noisy than asking a model to grade nothing.
# ---------------------------------------------------------------------------

_QUALITY_RUBRIC = """
   The scenario is a spoken conversation between a user and an AI.
   Your task is to rate the quality of the AI's reply to the user's turn.

   Below is the rating guideline (from 0 to 5, 0 is the worst and 5 is the best):
   - 0: The reply is totally unrelated to what the user said.
   - 1: The reply is not related to what the user said.
   - 2: The reply is slightly related to what the user said.
   - 3: The reply is related to what the user said.
   - 4: The reply is highly related to what the user said.
   - 5: The reply is perfectly related to what the user said and is a natural,
        appropriately-sized spoken answer.

   Firstly, briefly analyze the user's turn and the AI's reply.
   Then, you must return the overall output as the following format:
   Analysis: [Your analysis].
   I would rate the AI's response as [Rating].
   """

_QUALITY_RUBRIC_PT = """
   O cenário é uma conversa falada entre um usuário e uma IA.
   Sua tarefa é avaliar a qualidade da resposta da IA ao turno do usuário.

   Abaixo está a diretriz de avaliação (de 0 a 5, 0 é o pior e 5 é o melhor):
   - 0: A resposta é totalmente não relacionada ao que o usuário disse.
   - 1: A resposta não tem relação com o que o usuário disse.
   - 2: A resposta tem uma relação leve com o que o usuário disse.
   - 3: A resposta tem relação com o que o usuário disse.
   - 4: A resposta tem uma relação alta com o que o usuário disse.
   - 5: A resposta tem relação total com o que o usuário disse e é uma resposta
        falada natural e de tamanho adequado.

   Primeiro, analise brevemente o turno do usuário e a resposta da IA.
   Depois, retorne a saída exatamente neste formato:
   Analysis: [Sua análise].
   I would rate the AI's response as [Nota].
   """

QUALITY_MSG = {"en": _QUALITY_RUBRIC, "pt": _QUALITY_RUBRIC_PT}


def quality_user_message(user_turn: str, response: str) -> str:
    return f"""
                - User turn: {user_turn}
                - AI's response: {response}
                """

# FDB's parse regex: r"Analysis:\s*(.*?)\nI would rate the AI's response as (\d+)"
_FDB_RE = re.compile(
    r"Analysis:\s*(.*?)\n\s*I would rate the AI's response as\s*(\d+)", re.DOTALL
)
# tolerant fallback for models that answer in the target language
_LOOSE_RE = re.compile(r"(?:Analysis|Análise)\s*:\s*(.*)", re.DOTALL)
_RATING_RE = re.compile(r"(?:as|como)\s*(\d)", re.IGNORECASE)


def parse_output(text: str) -> dict[str, Any]:
    """Parse the judge's reply; falls back to a loose rating extraction."""
    m = _FDB_RE.search(text)
    if m:
        return {"analysis": m.group(1).strip(), "rating": int(m.group(2)),
                "parsed": "strict"}
    la = _LOOSE_RE.search(text)
    analysis = la.group(1).strip() if la else text.strip()
    # look for "response as 4" / "resposta como 4", else a bare trailing digit
    mr = _RATING_RE.search(text)
    if mr:
        return {"analysis": analysis[:600], "rating": int(mr.group(1)),
                "parsed": "loose"}
    digits = re.findall(r"\b([0-5])\b", text[-200:])
    if digits:
        return {"analysis": analysis[:600], "rating": int(digits[-1]),
                "parsed": "digits"}
    return {"analysis": analysis[:600], "rating": None, "parsed": "failed"}


def user_message(context: str, interrupt: str, response: str) -> str:
    """FDB builds exactly this three-field block."""
    return (
        f"""
                - Contextual user turn: {context}
                - User interrupting turn: {interrupt}
                - AI's response: {response}
                """
    )


def judge_quality(
    client,
    *,
    user_turn: str,
    response: str,
    lang: str = "en",
    max_tokens: int = 320,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Score one reply on the shared 0-5 relevance rubric (all tasks)."""
    msgs = [
        {"role": "system", "content": QUALITY_MSG[lang]},
        {"role": "user", "content": quality_user_message(user_turn, response)},
    ]
    raw = client.chat(msgs, max_tokens=max_tokens, temperature=temperature,
                      top_p=1.0)
    out = parse_output(raw or "")
    out["raw"] = (raw or "")[:1500]
    return out


def judge_one(
    client,
    *,
    context: str,
    interrupt: str,
    response: str,
    lang: str = "en",
    max_tokens: int = 320,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Score one interruption response with FDB's verbatim rubric.

    ``client`` is a :class:`fdb_bench.adapters.LlamaClient`.
    """
    msgs = [
        {"role": "system", "content": SYSTEM_MSG[lang]},
        {"role": "user", "content": user_message(context, interrupt, response)},
    ]
    raw = client.chat(
        msgs, max_tokens=max_tokens, temperature=temperature,
        # greedy: the judge must be deterministic for the metric to be stable
        top_p=1.0,
    )
    out = parse_output(raw or "")
    out["raw"] = (raw or "")[:1500]
    return out
