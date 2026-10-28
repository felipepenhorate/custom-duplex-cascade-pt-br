"""Build Laya-native duplex decision data from DuplexCascade-PT-BR-V0.

Why this generator exists instead of reusing ``mtp_like``'s prepped Qwen
records: Laya does not read a ChatML token stream.  It reads one typed state
and answers one typed choice.  The state keeps the *event* (partial / final /
silence), the turn-end flag, terminal punctuation and recent context.  That is
what lets the model learn the distinction the audio traces exposed:

    partial ASR piece ending in "?"  ->  keep waiting
    finalized turn ending in "?"     ->  answer now

The generator also emits the exact hard pairs as separate rows:

* a finalized chunk -> ``user_finish_speaking``
* the same text, marked partial -> ``user_is_speaking``
* the same finalized chunk with its punctuation stripped -> finish
* silence after a finalized chunk -> finish
* silence mid-turn -> speaking
* user content while the system is speaking -> interruption
* a short acknowledgement while the system is speaking -> backchannel
* silence while the system is speaking -> no_tag

Run from the repository root::

    python -m laya_approach.data.prepare_laya_dataset \
        --out-dir laya_approach/data/generated \
        --train-dialogues 5000 --val-dialogues 400 --test-dialogues 400
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from .common import (
    ACTION_KEYS,
    NO_TAG,
    NO_VOICE,
    PT_BACKCHANNELS,
    action_question,
    build_state,
    clean_text,
    event_bucket,
    terminal_punctuation,
)

REPO_ID = "lumierenoir/DuplexCascade-PT-BR-V0"
DIALOGUES_FILE = "data/dialogues_combined.jsonl"

_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+")
_WS = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------------

def resolve_dialogues(path: str | Path | None) -> Path:
    if path:
        p = Path(path).expanduser()
        if not p.is_file():
            raise FileNotFoundError(p)
        return p
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(REPO_ID, DIALOGUES_FILE, repo_type="model", token=False)
    )


def load_dialogues(path: Path) -> list[dict]:
    out = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            msgs = [
                {"role": m.get("role"), "text": clean_text(m.get("content", ""))}
                for m in rec.get("messages", [])
                if m.get("role") in {"user", "assistant"} and clean_text(m.get("content", ""))
            ]
            if len(msgs) < 4:
                continue
            rec["_messages"] = msgs
            out.append(rec)
    return out


def split_dialogues(
    dialogues: list[dict],
    *,
    seed: int,
    n_train: int,
    n_val: int,
    n_test: int,
) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    order = list(dialogues)
    rng.shuffle(order)
    if n_test > 0:
        test = order[:n_test]
    else:
        test = []
    order = order[n_test:]
    if n_val > 0:
        val = order[:n_val]
    else:
        val = []
    order = order[n_val:]
    train = order[:n_train] if n_train > 0 else order
    return {"train": train, "val": val, "test": test}


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def sentence_split(text: str) -> list[str]:
    parts = [clean_text(p) for p in _SENT_SPLIT.split(clean_text(text)) if clean_text(p)]
    return parts or ([clean_text(text)] if clean_text(text) else [])


def _chunk_sentence(
    sentence: str, rng: random.Random, *, target_words: int | None = None
) -> list[str]:
    words = sentence.split()
    if not words:
        return []
    if target_words:
        # Audio micro-turns: keep pieces near ``micro_turn_s * wps`` words.
        # A two-word piece is ~0.6 s at 3.25 w/s; the occasional one/three-word
        # piece mirrors ASR settle windows and keeps boundaries from being a
        # perfectly regular clock.
        sizes = [1, 1, 2, 2, 2, 2, 3]
    else:
        sizes = [1, 2, 2, 3, 3, 3, 4, 4, 5, 6]
    out, i = [], 0
    while i < len(words):
        n = min(rng.choice(sizes), len(words) - i)
        out.append(" ".join(words[i : i + n]))
        i += n
    return out


def chunk_user_turn(
    text: str,
    rng: random.Random,
    *,
    micro_turn_s: float | None = None,
    words_per_second: float = 3.25,
) -> list[str]:
    """ASR-like pieces, preserving sentence-final punctuation.

    ``micro_turn_s`` switches from the old 1--6 word pieces to the audio
    cadence: roughly ``micro_turn_s * words_per_second`` words per piece.
    """
    target = None
    if micro_turn_s:
        target = max(1, int(round(micro_turn_s * words_per_second)))
    chunks: list[str] = []
    for sentence in sentence_split(text):
        chunks.extend(_chunk_sentence(sentence, rng, target_words=target))
    return chunks


def strip_terminal(text: str) -> str:
    clean = clean_text(text)
    while clean and clean[-1] in ".?!…,;:":
        clean = clean[:-1].rstrip()
    return clean


def add_punctuation(text: str, rng: random.Random) -> str:
    clean = strip_terminal(text)
    if not clean:
        return text
    return clean + rng.choice([".", "?", "!", "…"])


def _prefix_words(text: str, rng: random.Random) -> str:
    words = clean_text(text).split()
    if len(words) <= 4:
        return clean_text(text)
    # Stop at the first sentence when possible; otherwise take 45-70%.
    sentences = sentence_split(text)
    if len(sentences) > 1:
        return sentences[0]
    keep = max(4, int(len(words) * rng.uniform(0.45, 0.7)))
    return " ".join(words[:keep])


# ---------------------------------------------------------------------------
# Reservoir sampling
# ---------------------------------------------------------------------------

class Reservoir:
    def __init__(self, cap: int, rng: random.Random):
        self.cap = max(1, int(cap))
        self.rng = rng
        self.items: list[dict] = []
        self.seen = 0

    def add(self, item: dict) -> None:
        self.seen += 1
        if len(self.items) < self.cap:
            self.items.append(item)
            return
        j = self.rng.randrange(self.seen)
        if j < self.cap:
            self.items[j] = item


DEFAULT_TRAIN_TARGETS = {
    "user_is_speaking": 16000,
    "user_finish_speaking": 12000,
    "user_interruption": 6500,
    "user_backchannel": 4500,
    "user_is_thinking": 4500,
    "no_tag": 6500,
    "system_backchannel": 600,
}

# Per-label reservoir headroom.  A label with a single bucket (thinking,
# backchannel, system_backchannel) needs the full quota in that bucket or the
# cap silently becomes the label's ceiling.  Labels spread over several buckets
# stay below 1.0 so one artificial bucket cannot dominate the class.
CAP_MULTIPLIER = {
    "user_is_speaking": 0.55,
    "user_finish_speaking": 0.55,
    "user_interruption": 0.55,
    "user_backchannel": 1.0,
    "user_is_thinking": 1.0,
    "no_tag": 1.0,
    "system_backchannel": 1.0,
}


def scaled_targets(scale: float, multiplier: float = 1.0) -> dict[str, int]:
    return {
        k: max(1, int(v * scale * multiplier))
        for k, v in DEFAULT_TRAIN_TARGETS.items()
    }


class DecisionPool:
    """One reservoir per (label, bucket), sampled to a class quota at the end."""

    def __init__(self, targets: dict[str, int], rng: random.Random):
        self.targets = targets
        self.rng = rng
        self.pools: dict[tuple[str, str], Reservoir] = {}
        self.counts = Counter()
        self.bucket_counts = Counter()

    def add(self, state: dict, event: dict, label: str, meta: dict) -> None:
        bucket = event_bucket(event, state)
        key = (label, bucket)
        if key not in self.pools:
            # Enough headroom for six or seven punctuation/context buckets.
            cap = max(
                80,
                int(
                    self.targets.get(label, 1000)
                    * CAP_MULTIPLIER.get(label, 0.55)
                ),
            )
            self.pools[key] = Reservoir(cap, self.rng)
        row = {
            "state": state,
            "questions": {"action": action_question()},
            "gold": {"action": label},
            "_meta": {**meta, "label": label, "bucket": bucket},
        }
        self.pools[key].add(row)
        self.counts[label] += 1
        self.bucket_counts[bucket] += 1

    def sample(self, *, max_total: int | None = None) -> list[dict]:
        by_label: dict[str, list[dict]] = defaultdict(list)
        for (label, _bucket), pool in self.pools.items():
            by_label[label].extend(pool.items)
        out: list[dict] = []
        for label in ACTION_KEYS:
            target = self.targets.get(label, 0)
            pool = by_label.get(label, [])
            if target <= 0 or not pool:
                continue
            take = min(target, len(pool))
            picked = self.rng.sample(pool, take)
            out.extend(picked)
        if max_total and len(out) > max_total:
            out = self.rng.sample(out, max_total)
        self.rng.shuffle(out)
        return out


# ---------------------------------------------------------------------------
# Dialogue -> decisions
# ---------------------------------------------------------------------------

class DialogueGenerator:
    def __init__(
        self,
        sink: DecisionPool,
        split: str,
        rng: random.Random,
        *,
        micro_turn_s: float | None = None,
        words_per_second: float = 3.25,
        vad_final_prob: float = 0.0,
    ):
        self.sink = sink
        self.split = split
        self.rng = rng
        self.dialogue_id = ""
        self.micro_turn_s = micro_turn_s
        self.words_per_second = words_per_second
        self.vad_final_prob = vad_final_prob

    def emit(
        self,
        event: dict,
        label: str,
        history: list[dict],
        *,
        bucket_kind: str,
    ) -> dict:
        state = build_state(event, history)
        self.sink.add(
            state,
            event,
            label,
            {
                "dialogue_id": self.dialogue_id,
                "split": self.split,
                "source": bucket_kind,
            },
        )
        return state

    def _backchannel(self) -> str:
        return self.rng.choice(PT_BACKCHANNELS)

    def generate(self, dialogue: dict) -> None:
        self.dialogue_id = str(dialogue.get("id", "?"))
        msgs = dialogue["_messages"]
        history: list[dict] = []
        pending_interruption = False

        for idx in range(0, len(msgs) - 1):
            user_msg = msgs[idx]
            asst_msg = msgs[idx + 1]
            if user_msg["role"] != "user" or asst_msg["role"] != "assistant":
                continue
            user_text = user_msg["text"]
            asst_text = asst_msg["text"]
            chunks = chunk_user_turn(
                user_text, self.rng,
                micro_turn_s=self.micro_turn_s,
                words_per_second=self.words_per_second,
            )
            if not chunks:
                continue
            if pending_interruption:
                # A barge-in can arrive as one finalized piece, not just as a
                # stream of partials.  Train that shape explicitly: it is the
                # abrupt "pera" that the mtp_like companion answered instead of
                # yielding.  The same words as a *final* while the system speaks
                # must be an interruption, not a fresh answer.
                self.emit(
                    {
                        "kind": "final",
                        "ends_turn": True,
                        "text": " ".join(chunks),
                        "system_speaking": True,
                    },
                    "user_interruption",
                    history,
                    bucket_kind="barge_in_final",
                )
                # Keep the split so the interruption and the later finalized
                # turn are distinct events when the user keeps talking.
                if len(chunks) == 1:
                    words = chunks[0].split()
                    if len(words) >= 2:
                        cut = max(1, len(words) // 2)
                        chunks = [" ".join(words[:cut]), " ".join(words[cut:])]
                    else:
                        pending_interruption = False

            system_speaking = pending_interruption
            for ci, chunk in enumerate(chunks):
                is_last = ci == len(chunks) - 1
                kind = "final" if is_last else "partial"
                event = {
                    "kind": kind,
                    "ends_turn": is_last,
                    "text": chunk,
                    "system_speaking": system_speaking and ci == 0,
                }
                if pending_interruption and ci == 0:
                    label = "user_interruption"
                elif is_last:
                    label = "user_finish_speaking"
                else:
                    label = "user_is_speaking"
                self.emit(event, label, history, bucket_kind="main")

                # Hard negative 1: the same text may arrive as a partial that
                # already contains terminal punctuation.  It is still mid-turn.
                if not is_last:
                    # Hard negative 3: the ASR *finalized* a piece mid-turn.
                    # stt.py delivers a long utterance as several final pieces
                    # and marks only the last one ends_turn=True.  Same words,
                    # same text shape as a real final, but the turn goes on.
                    if not event["system_speaking"] and self.rng.random() < 0.5:
                        self.emit(
                            {**event, "kind": "final", "ends_turn": False},
                            "user_is_speaking",
                            history,
                            bucket_kind="finalized_mid_turn",
                        )
                    if (not event["system_speaking"]
                            and not terminal_punctuation(chunk)
                            and self.rng.random() < 0.55):
                        punct = {
                            **event,
                            "text": add_punctuation(chunk, self.rng),
                        }
                        self.emit(
                            punct, "user_is_speaking", history,
                            bucket_kind="partial_punctuation_negative",
                        )
                    # A sentence-final chunk can occur mid-turn (the user is
                    # continuing after a full sentence).  It is already in the
                    # generator's sentence-aware chunking; duplicate a few to
                    # make the bucket large enough for evaluation.
                    if (not event["system_speaking"]
                            and terminal_punctuation(chunk)
                            and self.rng.random() < 0.35):
                        self.emit(
                            event, "user_is_speaking", history,
                            bucket_kind="sentence_final_mid_turn",
                        )
                else:
                    # Hard negative 2: premature partial of the finalized text.
                    # This is the exact audio failure: a complete-looking ASR
                    # phrase arrives before the turn-end event.
                    if self.rng.random() < 0.9:
                        self.emit(
                            {**event, "kind": "partial", "ends_turn": False},
                            "user_is_speaking",
                            history,
                            bucket_kind="premature_final_partial",
                        )
                    # Hard positive: a finalized turn that has no punctuation.
                    # Earlier companions waited forever on this shape.
                    stripped = strip_terminal(chunk)
                    if stripped != chunk:
                        self.emit(
                            {**event, "text": stripped},
                            "user_finish_speaking",
                            history,
                            bucket_kind="final_without_punctuation",
                        )

                history.append(
                    {
                        "role": "user",
                        "text": chunk,
                        "kind": kind,
                        "ends_turn": is_last,
                    }
                )
                if pending_interruption and ci == 0:
                    pending_interruption = False
                    system_speaking = False

                # Audio ASR hard family: the VAD finalizes the current
                # utterance after a short silence, even though the user has not
                # finished the turn.  The final carries ends_turn=True and may
                # be empty (all words were already delivered as partials), but
                # the correct action is to keep listening.  This is the shape
                # that the text-only corpus lacked and that made the audio
                # benchmark answer through pauses.
                vad_pause = False
                if (not is_last and not event["system_speaking"]
                        and self.rng.random() < self.vad_final_prob):
                    vad_pause = True
                    tail = "" if self.rng.random() < 0.5 else chunk
                    self.emit(
                        {
                            "kind": "final",
                            "ends_turn": True,
                            "text": tail,
                            "system_speaking": False,
                        },
                        "user_is_speaking",
                        history,
                        bucket_kind="vad_final_pause",
                    )
                    if tail:
                        history.append(
                            {
                                "role": "user",
                                "text": tail,
                                "kind": "final",
                                "ends_turn": True,
                            }
                        )
                    silence = {
                        "kind": "silence",
                        "ends_turn": False,
                        "text": "",
                        "system_speaking": False,
                    }
                    self.emit(
                        silence, "user_is_speaking", history,
                        bucket_kind="vad_final_pause_silence",
                    )
                    history.append({"role": "user", "kind": "silence"})

                # A pause inside the user turn: the user is still speaking.
                if not is_last and not vad_pause and self.rng.random() < 0.18:
                    silence = {
                        "kind": "silence",
                        "ends_turn": False,
                        "text": "",
                        "system_speaking": False,
                    }
                    self.emit(
                        silence, "user_is_speaking", history,
                        bucket_kind="pause_mid_turn",
                    )
                    history.append({"role": "user", "kind": "silence"})

                # A rare system backchannel during a user turn.  It is the
                # beta protocol tag; keep it sparse because it masks the
                # listener and can hurt audio overlap metrics.
                if not is_last and self.rng.random() < 0.012:
                    self.emit(
                        event, "system_backchannel", history,
                        bucket_kind="system_backchannel",
                    )

            # The audio stream delivers the final chunk as a *positional tail*.
            # When the last partial already contained every word, the turn end
            # arrives as an empty final event.  It must still start the answer;
            # this is the shape a text-only turn-end signal would miss.
            hist_empty = [dict(x) for x in history]
            for item in reversed(hist_empty):
                if item.get("role") == "user" and item.get("text") == chunks[-1]:
                    item["kind"] = "partial"
                    item["ends_turn"] = False
                    break
            self.emit(
                {"kind": "final", "ends_turn": True, "text": "",
                 "system_speaking": False},
                "user_finish_speaking",
                hist_empty,
                bucket_kind="empty_final",
            )

            # After the finalized chunk: the exact minimal silence query.
            final_event = {
                "kind": "final",
                "ends_turn": True,
                "text": chunks[-1],
                "system_speaking": False,
            }
            if self.rng.random() < 0.65:
                silence = {
                    "kind": "silence",
                    "ends_turn": False,
                    "text": "",
                    "system_speaking": False,
                }
                self.emit(
                    silence, "user_finish_speaking", history,
                    bucket_kind="silence_after_final",
                )

            # The assistant replies.  Sometimes the next user turn interrupts
            # it; the interruption training records are created at the start
            # of the next loop iteration, but the open assistant prefix and the
            # while-speaking negatives belong here.
            next_user = msgs[idx + 2] if idx + 2 < len(msgs) else None
            can_interrupt = (
                next_user is not None
                and next_user["role"] == "user"
                and len(chunk_user_turn(
                    next_user["text"], random.Random(0),
                    micro_turn_s=self.micro_turn_s,
                    words_per_second=self.words_per_second,
                )) >= 2
            )
            interrupt_next = (
                can_interrupt
                and self.rng.random() < 0.28
                and len(asst_text.split()) >= 6
            )
            if interrupt_next:
                prefix = _prefix_words(asst_text, self.rng)
                history.append({"role": "assistant", "text": prefix})
                # Silence while the system is speaking -> no_tag.
                for _ in range(self.rng.randint(1, 2)):
                    silence = {
                        "kind": "silence",
                        "ends_turn": False,
                        "text": "",
                        "system_speaking": True,
                    }
                    self.emit(
                        silence, NO_TAG, history,
                        bucket_kind="silence_during_system",
                    )
                    history.append({"role": "user", "kind": "silence"})
                # A short acknowledgement is not an interruption.  The ASR
                # may close it (ends_turn=True) or leave it open; both shapes
                # must stay a backchannel rather than a fresh answer.
                if self.rng.random() < 0.5:
                    bc_text = self._backchannel()
                    for bc_ends in (False, True):
                        bc = {
                            "kind": "final",
                            "ends_turn": bc_ends,
                            "text": bc_text,
                            "system_speaking": True,
                        }
                        self.emit(
                            bc, "user_backchannel", history,
                            bucket_kind="backchannel_during_system",
                        )
                    history.append(
                        {
                            "role": "user",
                            "text": bc_text,
                            "kind": "final",
                            "ends_turn": False,
                        }
                    )
                pending_interruption = True
            else:
                history.append({"role": "assistant", "text": asst_text})
                pending_interruption = False
                # After the answer the user is usually quiet/composing.
                if self.rng.random() < 0.45:
                    silence = {
                        "kind": "silence",
                        "ends_turn": False,
                        "text": "",
                        "system_speaking": False,
                    }
                    # The system has finished; silence is the user composing
                    # the next turn.  FDB's probe expects user_is_thinking, and
                    # a random no_tag here would make identical states carry
                    # two different labels.
                    self.emit(
                        silence, "user_is_thinking", history,
                        bucket_kind="silence_after_system",
                    )
                    history.append({"role": "user", "kind": "silence"})

            # Keep the context bounded.  State builder trims for the model;
            # this only keeps generation memory flat on long dialogues.
            if len(history) > 60:
                history = history[-40:]


# ---------------------------------------------------------------------------
# Targeted probes
# ---------------------------------------------------------------------------

def build_probes() -> list[dict]:
    """Hand-written diagnostics for the failure modes that motivated Laya."""
    assistant = [
        {"role": "assistant", "text": "Claro, eu posso te ajudar com isso."},
    ]
    assistant_plus_user = assistant + [
        {"role": "user", "text": "Então eu queria saber", "kind": "partial", "ends_turn": False}
    ]
    final_history = [
        {"role": "user", "text": "Eu preciso resolver uma coisa hoje.", "kind": "final", "ends_turn": True},
    ]
    rows: list[dict] = []

    def add(group: str, event: dict, label: str, history: list[dict], note: str) -> None:
        state = build_state(event, history)
        rows.append(
            {
                "state": state,
                "questions": {"action": action_question()},
                "gold": {"action": label},
                "_meta": {
                    "probe_group": group,
                    "label": label,
                    "bucket": event_bucket(event, state),
                    "source": "probe",
                    "note": note,
                },
            }
        )

    partial_texts = [
        "Você pode me dizer qual é o",
        "Eu queria saber se o pagamento",
        "E o prazo de entrega, como",
        "Me explica melhor essa parte",
        "Tô pensando em mudar de plano",
        "Qual é o horário de",
    ]
    for i, text in enumerate(partial_texts):
        punct = text + ("?" if i % 2 == 0 else ".")
        add(
            "partial_terminal_punctuation",
            {"kind": "partial", "ends_turn": False, "text": punct, "system_speaking": False},
            "user_is_speaking",
            [],
            "partial ASR piece ending in punctuation stays mid-turn",
        )
        add(
            "partial_no_punctuation",
            {"kind": "partial", "ends_turn": False, "text": text, "system_speaking": False},
            "user_is_speaking",
            [],
            "partial without punctuation is still mid-turn",
        )
        add(
            "final_terminal_punctuation",
            {"kind": "final", "ends_turn": True, "text": punct, "system_speaking": False},
            "user_finish_speaking",
            [],
            "finalized turn with punctuation",
        )
        add(
            "final_no_punctuation",
            {"kind": "final", "ends_turn": True, "text": strip_terminal(punct), "system_speaking": False},
            "user_finish_speaking",
            [],
            "finalized turn without punctuation must still answer",
        )

    for text in ["Qual é o seu nome?", "Você pode repetir?", "Entendi.", "E agora?"]:
        add(
            "silence_after_final",
            {"kind": "silence", "ends_turn": False, "text": "", "system_speaking": False},
            "user_finish_speaking",
            [{"role": "user", "text": text, "kind": "final", "ends_turn": True}],
            "silence after a complete turn starts the answer",
        )
        add(
            "silence_mid_turn",
            {"kind": "silence", "ends_turn": False, "text": "", "system_speaking": False},
            "user_is_speaking",
            [{"role": "user", "text": strip_terminal(text), "kind": "partial", "ends_turn": False}],
            "silence mid-turn keeps waiting",
        )
        add(
            "partial_during_system",
            {"kind": "partial", "ends_turn": False, "text": text, "system_speaking": True},
            "user_interruption",
            assistant,
            "partial words over system speech are a barge-in",
        )
        add(
            "final_during_system",
            {"kind": "final", "ends_turn": True, "text": text, "system_speaking": True},
            "user_interruption",
            assistant,
            "a finalized phrase over system speech is still a barge-in, not a fresh answer",
        )
        add(
            "silence_during_system",
            {"kind": "silence", "ends_turn": False, "text": "", "system_speaking": True},
            NO_TAG,
            assistant,
            "silence during system speech keeps talking",
        )

    for text in ["Qual é o seu nome?", "E o prazo de entrega?"]:
        add(
            "empty_final_after_partial",
            {"kind": "final", "ends_turn": True, "text": "", "system_speaking": False},
            "user_finish_speaking",
            [{"role": "user", "text": strip_terminal(text), "kind": "partial", "ends_turn": False}],
            "the ASR tail is empty because the partials already carried every word",
        )

    for text, prior in (
        ("Primeiro eu preciso", "Então, deixa eu ver."),
        ("e o prazo?", "Primeiro eu preciso"),
    ):
        add(
            "final_without_turn_end",
            {"kind": "final", "ends_turn": False, "text": text, "system_speaking": False},
            "user_is_speaking",
            [{"role": "user", "text": prior, "kind": "final", "ends_turn": False}],
            "the ASR finalized this piece but the stream has not ended the turn",
        )

    for bc in PT_BACKCHANNELS[:8]:
        add(
            "backchannel_during_system",
            {"kind": "final", "ends_turn": False, "text": bc, "system_speaking": True},
            "user_backchannel",
            assistant,
            "short acknowledgement over system speech is not an interruption",
        )
        add(
            "backchannel_partial_during_system",
            {"kind": "partial", "ends_turn": False, "text": bc, "system_speaking": True},
            "user_backchannel",
            assistant,
            "partial acknowledgement is still a backchannel",
        )
        add(
            "backchannel_closed_during_system",
            {"kind": "final", "ends_turn": True, "text": bc, "system_speaking": True},
            "user_backchannel",
            assistant,
            "a closed acknowledgement is still not a fresh answer",
        )

    for text in ["Certo, obrigado.", "Tá bom.", "Entendi, valeu."]:
        add(
            "silence_after_system",
            {"kind": "silence", "ends_turn": False, "text": "", "system_speaking": False},
            "user_is_thinking",
            assistant,
            "silence after an answer waits for the next user turn",
        )

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _write(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def _audit(rows: list[dict]) -> dict:
    labels = Counter(r["gold"]["action"] for r in rows)
    buckets = Counter(r["_meta"].get("bucket", "?") for r in rows)
    sources = Counter(r["_meta"].get("source", "?") for r in rows)
    return {
        "n": len(rows),
        "labels": dict(labels),
        "buckets": dict(buckets),
        "sources": dict(sources),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dialogues", default=None)
    ap.add_argument("--out-dir", default="laya_approach/data/generated")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--train-dialogues", type=int, default=5000,
                    help="0 = all train dialogues after val/test holdout")
    ap.add_argument("--val-dialogues", type=int, default=400)
    ap.add_argument("--test-dialogues", type=int, default=400)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="scale the per-class training quotas")
    ap.add_argument("--max-total", type=int, default=0)
    ap.add_argument("--micro-turn-s", type=float, default=0.0,
                    help="cut user pieces at this audio cadence (0 = legacy "
                         "1-6 word pieces); 0.6 matches fdb_bench/audio/stt.py")
    ap.add_argument("--words-per-second", type=float, default=3.25,
                    help="speech rate used to convert micro-turn seconds to words")
    ap.add_argument("--vad-final-prob", type=float, default=0.0,
                    help="probability of a VAD-style final at an intra-turn "
                         "pause, labeled wait (the audio ASR hard family)")
    ap.add_argument("--respect-split", action="store_true",
                    help="use _pilot.split from the input instead of re-splitting; "
                         "used for the tag-stratified pilot")
    args = ap.parse_args()

    path = resolve_dialogues(args.dialogues)
    print(f"[laya-data] dialogues: {path}")
    dialogues = load_dialogues(path)
    print(f"[laya-data] usable dialogues: {len(dialogues):,}")
    splits = split_dialogues(
        dialogues,
        seed=args.seed,
        n_train=args.train_dialogues,
        n_val=args.val_dialogues,
        n_test=args.test_dialogues,
    )
    if args.respect_split:
        grouped: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
        for dialogue in dialogues:
            split = str(dialogue.get("_pilot", {}).get("split", "train"))
            if split in grouped:
                grouped[split].append(dialogue)
        splits = grouped
    for name, rows in splits.items():
        print(f"[laya-data] {name}: {len(rows):,} dialogues")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audits = {}

    for name, ds in splits.items():
        if not ds:
            continue
        rng = random.Random(args.seed + {"train": 0, "val": 1, "test": 2}[name])
        quota_scale = args.scale if name == "train" else max(0.18, args.scale * 0.18)
        targets = scaled_targets(quota_scale)
        pool = DecisionPool(targets, rng)
        gen = DialogueGenerator(
            pool, name, rng,
            micro_turn_s=args.micro_turn_s or None,
            words_per_second=args.words_per_second,
            vad_final_prob=args.vad_final_prob,
        )
        for i, dlg in enumerate(ds):
            gen.generate(dlg)
            if (i + 1) % 500 == 0:
                print(
                    f"[laya-data] {name}: {i + 1:,}/{len(ds):,} dialogues, "
                    f"{sum(pool.counts.values()):,} candidate decisions",
                    flush=True,
                )
        max_total = None
        if name == "train" and args.max_total > 0:
            max_total = args.max_total
        rows = pool.sample(max_total=max_total)
        n = _write(out_dir / f"{name}.jsonl", rows)
        audits[name] = {"requested": targets, **_audit(rows)}
        print(f"[laya-data] wrote {n:,} rows -> {out_dir / f'{name}.jsonl'}")
        print(f"[laya-data] {name} labels: {audits[name]['labels']}")

    probes = build_probes()
    n = _write(out_dir / "probes.jsonl", probes)
    audits["probes"] = _audit(probes)
    print(f"[laya-data] wrote {n:,} targeted probes -> {out_dir / 'probes.jsonl'}")

    # A deterministic content hash makes the audit traceable from the model
    # config without committing the generated JSONL (it is gitignored).
    h = hashlib.sha256()
    for name in ("train", "val", "test", "probes"):
        p = out_dir / f"{name}.jsonl"
        if p.is_file():
            h.update(p.read_bytes())
    audits["sha256"] = h.hexdigest()
    (out_dir / "audit.json").write_text(
        json.dumps(audits, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[laya-data] audit -> {out_dir / 'audit.json'}")


if __name__ == "__main__":
    main()
