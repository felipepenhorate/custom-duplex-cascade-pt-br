"""M1 — policy-companion dataset preparation (target stream over the sft
duplex data).

Turns the sft duplex micro-turn items (output of
../sft/data/build_duplex_dataset.py, e.g. ../sft/data/duplex_train_continue.jsonl)
into training sequences for the MTP-like companion (SPEC 5):

  * MAIN record — ONE sequence per dialogue: the full duplex stream
    rendered in ChatML (user items as <|im_start|>user\\n ... <|im_end|>\\n,
    assistant items as <|im_start|>assistant\\n ... <|im_end|>), with the
    tag tokens and <|no voice|> inside the content — exactly the context
    shape the runtime orchestrator maintains.
  * LABELS only at ITEM BOUNDARIES (the final token of each item): the
    target is the duplex tag the assistant emits next (SPEC 5.2 table), or
    the label-only <|no tag|> token when nothing fires (all assistant
    items, and user items answered by plain text). All other positions are
    -100. This is plain next-token CE training with masked labels — no
    custom trainer needed.
  * SAS (streaming-aware supervision) records — one short record per
    sampled partial prefix of every multi-token user content chunk: the
    context ends mid-chunk, exactly like a runtime query after ASR words
    arrive. Partial prefixes are labeled <|user is speaking|> (words still
    arriving), except partials of an interrupting chunk which are labeled
    <|user interruption|>; only the COMPLETE final chunk of a turn is
    labeled <|user finish speaking|> (in the main record).

  * BARGE-IN (controller) records — the companion is the system controller:
    it must learn that the user can barge in while the assistant is
    mid-utterance. For randomly chosen points inside every system turn we
    render the assistant's current item as an OPEN item (partial content,
    no <|im_end|> — it is still being spoken) followed by the user's next
    real words, labeled <|user interruption|> (plus a partial-prefix
    variant: the first words of the barge-in). Balanced negatives:
    open assistant + <|no voice|> -> no_tag (silence means KEEP TALKING),
    open assistant + backchannel word -> <|user backchannel|> (a short
    acknowledgment is NOT a barge-in).

  * SILENCE-SEMANTICS records — the runtime queries the companion on ASR
    silence (inserting <|no voice|>); the data must teach what silence
    means in each state. For every user content chunk we add
    chunk + <|no voice|> records:
      * non-final chunk + <|no voice|>      -> <|user is speaking|>
        (user paused mid-speech, still talking)
      * FINAL chunk + <|no voice|>          -> <|user finish speaking|>
        (user finished + went silent: START ANSWERING — the key runtime
        behavior the sft data never shows, since finish always follows the
        last chunk directly)
      * partial FINAL chunk + <|no voice|>  -> <|user is speaking|>
        (user paused before completing the sentence)

  * loss_weight column: per-record loss multiplier so the rare but critical
    controller patterns dominate the gradient: barge-in silence negatives
    (open assistant + <|no voice|> -> no_tag) x3, silence-semantics
    finish records x3, barge-in interruption positives x1.5.

Class symbols: the 6 tag tokens (+<|no voice|> only in the input) plus
<|no tag|> as the emit-nothing target.

Usage:
  python training/prep_policy_dataset.py \\
      --duplex ../sft/data/duplex_train_continue.jsonl \\
      --out /mnt/f/duplex_cascade_runs/mtp_like/prepped_policy --tokenizer Qwen/Qwen3.5-0.8B
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.common import (
    IM_END,
    IM_END_NL,
    IM_START_ASSISTANT,
    IM_START_SYSTEM,
    IM_START_USER,
    NO_TAG,
    PT_BACKCHANNELS,
    ROLE_ASSISTANT,
    ROLE_SYSTEM,
    ROLE_USER,
    SPECIAL_TOKENS,
    TOKEN,
    TOKEN_WEIGHT,
)

NO_TAG_WEIGHT = 0.3  # EMPTY dominates; keep its loss contribution small

# per-record loss multipliers for the critical controller patterns
RECORD_WEIGHT = {
    "barge": {"no_tag": 3.0, "user_interruption": 1.5},
    "silence": {"user_finish_speaking": 3.0},
    "rule": {"system_take_floor": 5.0, "system_handover": 5.0},
}


def record_weight(kind: str, label: str) -> float:
    return RECORD_WEIGHT.get(kind, {}).get(label, 1.0)

_HEADER = {ROLE_USER: IM_START_USER, ROLE_ASSISTANT: IM_START_ASSISTANT,
           ROLE_SYSTEM: IM_START_SYSTEM}
_TAIL = {ROLE_USER: IM_END_NL, ROLE_ASSISTANT: IM_END, ROLE_SYSTEM: IM_END_NL}

# generic instruction mixed into ~50% of the regular records (v6): teaches
# the companion that instructions exist and that a generic one means
# "behave as usual" (the negatives for the system tags)
GENERIC_INSTRUCTION = "Gerencie a conversa normalmente, com turnos naturais."

# prompt-rule templates for the synthetic rule scenarios (v6)
RULES = [
    ("cpf_letters",
     "No CPF, apenas números são permitidos. Se o cliente disser uma letra em vez de um número, interrompa e avise que apenas números são aceitos.",
     ["a", "b", "c", "d", "e", "f", "g"],
     "system_take_floor"),
    ("sensitive_word",
     "Se o cliente mencionar a palavra atrasado, interrompa e peça desculpas pelo atraso.",
     ["atrasado", "atrasada", "está atrasado", "chegou atrasado"],
     "system_take_floor"),
    ("repetition",
     "Se o cliente repetir a mesma palavra três vezes seguidas, interrompa educadamente.",
     ["mesmo mesmo mesmo", "não não não", "sim sim sim", "ok ok ok"],
     "system_take_floor"),
    ("investments",
     "Se o cliente falar sobre investimentos, avise que isso está com o gerente e que ele será chamado.",
     ["investimentos", "quero investir", "aplicações", "renda fixa"],
     "system_handover"),
    ("complaint",
     "Se o cliente mencionar uma reclamação, informe que ela será registrada e tratada.",
     ["reclamação", "reclamei", "estou insatisfeito", "quero reclamar"],
     "system_handover"),
]


def load_tokenizer(name: str) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(name, use_fast=True, token=False)
    n_added = tok.add_special_tokens(
        {"additional_special_tokens": SPECIAL_TOKENS + [NO_TAG]}
    )
    print(f"[prep] added {n_added} special tokens; vocab now {len(tok)}")
    for key, text in list(TOKEN.items()) + [("no_tag", NO_TAG)]:
        ids = tok.encode(text, add_special_tokens=False)
        assert len(ids) == 1, f"{text!r} not single-id: {ids}"
    return tok


def render_segment(tok, header_ids, tail_ids, content_ids: list[int]) -> list[int]:
    return header_ids + content_ids + tail_ids


class DialogueEncoder:
    """Encodes one duplex dialogue into its boundary-target stream.

    segments[i]   = token list of the ChatML segment for item i
    targets[i]    = class key ("no_tag" or a TOKEN key) for the boundary
                    after item i (the label of the segment's final token)
    """

    def __init__(self, tok):
        self.tok = tok
        self.header_ids = {
            r: tok.encode(_HEADER[r], add_special_tokens=False) for r in _HEADER
        }
        self.tail_ids = {r: tok.encode(_TAIL[r], add_special_tokens=False) for r in _TAIL}
        self.content_cache: dict[int, list[int]] = {}

    def new_dialogue(self) -> None:
        """Clear the content cache: it is keyed by item INDEX, which repeats
        across dialogues. Without this, dialogues after the first one get
        the wrong text as input (correct labels, corrupted content)."""
        self.content_cache.clear()

    def content(self, item: dict, idx: int) -> list[int]:
        if idx not in self.content_cache:
            self.content_cache[idx] = self.tok.encode(
                item["text"], add_special_tokens=False
            )
        return self.content_cache[idx]

    def segment(self, item: dict, idx: int) -> list[int]:
        role = item["role"]
        return render_segment(
            self.tok,
            self.header_ids[role],
            self.tail_ids[role],
            self.content(item, idx),
        )

    def encode(self, items: list[dict]) -> tuple[list[list[int]], list[str]]:
        prev = None
        for it in items:
            assert it["role"] != prev, f"non-alternating roles in dialogue"
            assert it["role"] in _HEADER, f"unknown role {it['role']!r}"
            prev = it["role"]
        segments = [self.segment(it, i) for i, it in enumerate(items)]
        targets: list[str] = []
        for i, it in enumerate(items):
            if it["role"] == ROLE_ASSISTANT:
                targets.append("no_tag")
                continue
            nxt = items[i + 1] if i + 1 < len(items) else None
            sp = nxt.get("special") if nxt is not None else None
            targets.append(sp if sp in TOKEN else "no_tag")
        return segments, targets


def front_truncate(seq: list[int], seg_lens: list[int], max_seq: int) -> list[int]:
    """Drop whole segments from the front until len(seq) <= max_seq."""
    if len(seq) <= max_seq:
        return seq
    keep = len(seq)
    for i, sl in enumerate(seg_lens):
        if keep - sl >= max_seq:
            keep -= sl
        else:
            break
    return seq[len(seq) - keep :]


def to_record(
    seq: list[int],
    boundaries: list[tuple[int, str]],
    tag_ids: dict[str, int],
    dialogue_id: str,
    kind: str,
    loss_weight: float = 1.0,
    prefix_seg: list[int] | None = None,
) -> dict | None:
    if prefix_seg:
        seq = prefix_seg + seq
        boundaries = [(p + len(prefix_seg), key) for p, key in boundaries]
    labels = [-100] * len(seq)
    kept = 0
    for pos, key in boundaries:
        if pos < len(seq):
            labels[pos] = tag_ids[key]
            kept += 1
    if kept == 0:
        return None
    return {
        "input_ids": torch.tensor(seq, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.ones(len(seq), dtype=torch.long),
        "dialogue_id": dialogue_id,
        "kind": kind,
        "loss_weight": loss_weight,
    }


def build_barge_in_records(
    enc: DialogueEncoder,
    items: list[dict],
    segments: list[list[int]],
    max_seq: int,
    rng: random.Random,
    tag_ids: dict[str, int],
    dialogue_id: str,
    n_barge: int,
    n_bc: int,
    n_neg: int,
) -> list[dict]:
    """Controller records: the user barges in while the assistant is
    mid-utterance.

    For randomly chosen assistant items inside the system turns (plain-text
    continuation chunks or the <user finish speaking> item), we render the
    current assistant item as an OPEN partial item (no <|im_end|> — speech
    in progress) and append:
      * the user's next real words        -> <|user interruption|>
      * a partial prefix of those words   -> <|user interruption|>
      * <|no voice|>                      -> no_tag (keep talking)
      * a sampled backchannel word        -> <|user backchannel|>
    """
    records: list[dict] = []

    def user_content_after(idx: int) -> list[int] | None:
        for j in range(idx + 1, len(items)):
            if items[j]["role"] != ROLE_USER:
                continue
            if items[j]["text"].strip() != TOKEN["no_voice"]:
                return enc.content(items[j], j)
        return None

    def render_open_assistant(idx: int) -> list[int] | None:
        it = items[idx]
        content = enc.content(it, idx)
        lo = 2 if it.get("special") is not None else 1  # keep the tag token
        if len(content) <= lo + 1:
            return None
        p = rng.randint(lo, len(content) - 1)
        return enc.header_ids[ROLE_ASSISTANT] + content[:p]

    candidates = [
        i
        for i, it in enumerate(items)
        if it["role"] == ROLE_ASSISTANT
        and (it.get("special") is None or it.get("special") == "user_finish_speaking")
        and len(enc.content(it, i)) >= 3
    ]
    if not candidates:
        return []
    chosen = rng.sample(candidates, min(len(candidates), max(1, n_barge + n_bc + n_neg)))

    all_user_chunks = [
        enc.content(items[i], i)
        for i, it in enumerate(items)
        if it["role"] == ROLE_USER and it["text"].strip() != TOKEN["no_voice"]
    ]
    bc_words = [enc.tok.encode(w, add_special_tokens=False) for w in PT_BACKCHANNELS]
    no_voice_ids = enc.tok.encode(TOKEN["no_voice"], add_special_tokens=False)
    header_u, tail_u = enc.header_ids[ROLE_USER], enc.tail_ids[ROLE_USER]

    for idx in chosen:
        open_seg = render_open_assistant(idx)
        if open_seg is None:
            continue
        barge_text = user_content_after(idx) or (
            rng.choice(all_user_chunks) if all_user_chunks else None
        )
        prefix = [t for seg in segments[:idx] for t in seg]

        def emit(user_seg: list[int], label: str, partial_p: int | None = None) -> None:
            u = header_u + (user_seg[:partial_p] if partial_p else user_seg) + tail_u
            seq = prefix + open_seg + u
            if len(seq) > max_seq:
                seq = seq[-max_seq:]  # tail truncation keeps the label at the end
            rec = to_record(seq, [(len(seq) - 1, label)], tag_ids, dialogue_id, "barge")
            if rec is not None:
                records.append(rec)

        if barge_text is not None and n_barge > 0:
            emit(barge_text, "user_interruption")
            if len(barge_text) > 1:  # the FIRST words of the barge-in already
                emit(barge_text, "user_interruption",
                     partial_p=rng.randint(1, len(barge_text) - 1))
        if n_bc > 0:
            emit(rng.choice(bc_words), "user_backchannel")
        if n_neg > 0:
            emit(no_voice_ids, "no_tag")

    return records


def build_silence_records(
    enc: DialogueEncoder,
    items: list[dict],
    segments: list[list[int]],
    targets: list[str],
    max_seq: int,
    rng: random.Random,
    tag_ids: dict[str, int],
    dialogue_id: str,
) -> list[dict]:
    """Silence-semantics records: user chunk + <|no voice|> -> what silence
    means at that point (SPEC 5.3b):
      * non-final chunk  -> <|user is speaking|>  (paused mid-speech)
      * FINAL chunk      -> <|user finish speaking|> (finished + silent)
      * partial final    -> <|user is speaking|>  (paused pre-completion)
    """
    records: list[dict] = []
    header_u, tail_u = enc.header_ids[ROLE_USER], enc.tail_ids[ROLE_USER]
    no_voice_ids = enc.tok.encode(TOKEN["no_voice"], add_special_tokens=False)
    for i, it in enumerate(items):
        if it["role"] != ROLE_USER:
            continue
        content = enc.content(it, i)
        tgt = targets[i]
        if tgt not in ("user_is_speaking", "user_finish_speaking"):
            continue
        # prefix EXCLUDES the chunk itself (segments[:i]); the chunk is
        # added once by emit — [..before.., u chunk, u no voice|]
        prefix = [t for seg in segments[:i] for t in seg]

        def emit(chunk: list[int], label: str) -> None:
            seq = prefix + header_u + chunk + tail_u + header_u + no_voice_ids + tail_u
            if len(seq) > max_seq:
                seq = seq[-max_seq:]
            rec = to_record(
                seq, [(len(seq) - 1, label)], tag_ids, dialogue_id, "silence",
                record_weight("silence", label),
            )
            if rec is not None:
                records.append(rec)

        def emit_minimal(chunk: list[int], label: str, weight: float) -> None:
            """[u chunk, u no voice|] — the exact runtime query shape for a
            short single-batch utterance (rare in the data: only 16 records
            at n=2; the model otherwise never fires finish there)."""
            seq = header_u + chunk + tail_u + header_u + no_voice_ids + tail_u
            rec = to_record(
                seq, [(len(seq) - 1, label)], tag_ids, dialogue_id, "silence",
                weight,
            )
            if rec is not None:
                records.append(rec)

        if tgt == "user_finish_speaking":
            emit(content, "user_finish_speaking")
            emit_minimal(content, "user_finish_speaking", weight=5.0)
            if len(content) > 1:  # paused before completing the sentence
                emit(content[: rng.randint(1, len(content) - 1)], "user_is_speaking")
        else:  # mid-speech pause
            emit(content, "user_is_speaking")
            emit_minimal(content, "user_is_speaking", weight=1.0)
    return records


def build_rule_records(
    enc: DialogueEncoder,
    items: list[dict],
    segments: list[list[int]],
    targets: list[str],
    max_seq: int,
    rng: random.Random,
    tag_ids: dict[str, int],
    dialogue_id: str,
) -> list[dict]:
    """Synthetic prompt-rule scenarios (v6): the companion learns to READ a
    system instruction in its context and fire <|system take floor|> /
    <|system handover|> when the rule's trigger appears in the user's words.
    Also builds negatives: rule + normal chunk -> the normal tag (no system
    fire), so the model doesn't barge spuriously."""
    records: list[dict] = []
    rule = rng.choice(RULES)
    key, instruction, triggers, tag = rule
    sys_seg = (
        enc.header_ids[ROLE_SYSTEM]
        + enc.tok.encode(instruction, add_special_tokens=False)
        + enc.tail_ids[ROLE_SYSTEM]
    )
    trigger_ids = enc.tok.encode(rng.choice(triggers), add_special_tokens=False)
    # the runtime letter chunk arrives with its comma ("a,"). The model
    # must learn the NUMBER-vs-LETTER CONCEPT, not memorize words: train on
    # the FULL alphabet so an unseen letter still fires, and a RICH number
    # set (all digits + tens/hundreds) so any number word stays calm.
    CPF_DIGIT_WORDS = [
        "zero", "um", "dois", "três", "quatro", "cinco", "seis", "sete",
        "oito", "nove", "dez", "onze", "doze", "vinte", "trinta", "cem",
        "mil",
    ]
    CPF_LETTERS = [
        "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
        "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
    ]
    if key == "cpf_letters":
        trigger_ids = enc.tok.encode(
            rng.choice([L + "," for L in CPF_LETTERS]),
            add_special_tokens=False,
        )
    no_voice_ids = enc.tok.encode(TOKEN["no_voice"], add_special_tokens=False)
    h_u, t_u = enc.header_ids[ROLE_USER], enc.tail_ids[ROLE_USER]
    h_a, t_a = enc.header_ids[ROLE_ASSISTANT], enc.tail_ids[ROLE_ASSISTANT]

    # prefix: for the CPF rule, synthetic digit chunks in the REAL dictation
    # shape (variable count and order, RICH number words, so the model
    # learns "any number word is fine" instead of hardcoding); otherwise a
    # random slice of a real dialogue (user chunk + its assistant reply)
    if key == "cpf_letters":
        prefix_ids: list[int] = []
        n_digits = rng.randint(1, 6)
        digits = rng.sample(CPF_DIGIT_WORDS, n_digits)
        for num in digits:
            prefix_ids += h_u + enc.tok.encode(num + ",", add_special_tokens=False) + t_u
            prefix_ids += h_a + [tag_ids["user_is_speaking"]] + t_a
    else:
        user_ids = [
            i
            for i, it in enumerate(items)
            if it["role"] == ROLE_USER
            and it["text"].strip() != TOKEN["no_voice"]
        ]
        if not user_ids:
            return records
        i = rng.choice(user_ids)
        end = min(i + 2, len(items))  # chunk + its assistant reply
        prefix_ids = [t for seg in segments[:end] for t in seg]

    def emit(chunk_ids: list[int], label: str, weight: float) -> None:
        seq = sys_seg + prefix_ids + h_u + chunk_ids + t_u
        if len(seq) <= max_seq:
            rec = to_record(seq, [(len(seq) - 1, label)], tag_ids, dialogue_id,
                            "rule", weight)
            if rec is not None:
                records.append(rec)
        # silence-query variant: [.., trigger, no voice|] -> same tag
        seq2 = seq + h_u + no_voice_ids + t_u
        if len(seq2) <= max_seq:
            rec2 = to_record(seq2, [(len(seq2) - 1, label)], tag_ids,
                             dialogue_id, "rule", weight)
            if rec2 is not None:
                records.append(rec2)

    # positive: the trigger chunk fires the rule's tag
    emit(trigger_ids, tag, 5.0)

    # negative: a normal dialogue chunk with the rule present -> normal tag
    if key != "cpf_letters":
        j = rng.choice(user_ids)
        tgt = targets[j]
        label = tgt if tgt in ("user_is_speaking", "user_finish_speaking") \
            else "user_is_speaking"
        seq3 = sys_seg + [t for seg in segments[: j + 1] for t in seg]
        if len(seq3) <= max_seq:
            rec3 = to_record(seq3, [(len(seq3) - 1, label)], tag_ids,
                             dialogue_id, "rule", 1.0)
            if rec3 is not None:
                records.append(rec3)
    else:
        # CPF negatives: digit chunks + a NON-number, NON-letter word (e.g.
        # "meu", "cpf", "é") must NOT fire the rule — only letters do
        for w in ["meu", "cpf", "é", "o", "número", "espera"]:
            seq4 = sys_seg + prefix_ids + h_u + \
                enc.tok.encode(w + ",", add_special_tokens=False) + t_u
            if len(seq4) <= max_seq:
                rec4 = to_record(seq4, [(len(seq4) - 1, "user_is_speaking")],
                                 tag_ids, dialogue_id, "rule", 1.0)
                if rec4 is not None:
                    records.append(rec4)
    return records


def build_main_record(
    enc: DialogueEncoder,
    items: list[dict],
    max_seq: int,
    tag_ids: dict[str, int],
    dialogue_id: str,
) -> dict | None:
    segments, targets = enc.encode(items)
    seq: list[int] = []
    boundaries: list[tuple[int, str]] = []
    seg_lens: list[int] = []
    for seg, tgt in zip(segments, targets):
        pos = len(seq) + len(seg) - 1
        seq.extend(seg)
        seg_lens.append(len(seg))
        boundaries.append((pos, tgt))
    seq = front_truncate(seq, seg_lens, max_seq)
    if len(seq) <= 0:
        return None
    return to_record(seq, boundaries, tag_ids, dialogue_id, "main")


def build_sas_records(
    enc: DialogueEncoder,
    items: list[dict],
    segments: list[list[int]],
    targets: list[str],
    max_seq: int,
    rng: random.Random,
    sas_per_chunk: int,
    tag_ids: dict[str, int],
    dialogue_id: str,
) -> list[dict]:
    """Partial-prefix records of every multi-token user content chunk."""
    records: list[dict] = []
    header = enc.header_ids[ROLE_USER]
    tail = enc.tail_ids[ROLE_USER]
    for i, it in enumerate(items):
        if it["role"] != ROLE_USER:
            continue
        content = enc.content(it, i)
        tgt = targets[i]
        if len(content) < 2 or tgt == "no_tag":
            continue  # single-token (e.g. <|no voice|>) or unsupervized chunk
        n = min(sas_per_chunk, len(content) - 1)
        for p in sorted(rng.sample(range(1, len(content)), n)):
            partial = render_segment(enc.tok, header, tail, content[:p])
            seq = [t for seg in segments[:i] for t in seg] + partial
            seg_lens = [len(seg) for seg in segments[:i]] + [len(partial)]
            seq = front_truncate(seq, seg_lens, max_seq)
            if len(seq) <= 0:
                continue
            label = "user_interruption" if tgt == "user_interruption" else "user_is_speaking"
            rec = to_record(
                seq, [(len(seq) - 1, label)], tag_ids, dialogue_id, "sas"
            )
            if rec is not None:
                records.append(rec)
    return records


def prepend_instruction(rec: dict, sys_seg: list[int]) -> dict:
    """Prepend a system instruction item; all boundary positions shift."""
    shift = len(sys_seg)
    return {
        "input_ids": torch.cat([torch.tensor(sys_seg, dtype=torch.long),
                                rec["input_ids"]]),
        "labels": torch.cat([torch.full((shift,), -100, dtype=torch.long),
                             rec["labels"]]),
        "attention_mask": torch.cat(
            [torch.ones(shift, dtype=torch.long), rec["attention_mask"]]),
        "dialogue_id": rec["dialogue_id"],
        "kind": rec["kind"],
        "loss_weight": rec["loss_weight"],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--duplex", default="../sft/data/duplex_train_continue.jsonl")
    p.add_argument("--out", default="/mnt/f/duplex_cascade_runs/mtp_like/prepped_policy")
    p.add_argument("--tokenizer", default="Qwen/Qwen3.5-0.8B")
    p.add_argument("--max-seq", type=int, default=4096)
    p.add_argument("--eval-frac", type=float, default=0.005)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--sas-per-chunk", type=int, default=1,
                   help="partial prefixes per user chunk (0 disables SAS)")
    p.add_argument("--barge-in-per-dialogue", type=int, default=2,
                   help="mid-utterance barge-in positives per dialogue (0 disables)")
    p.add_argument("--barge-in-bc", type=int, default=1,
                   help="open-assistant backchannel negatives per dialogue")
    p.add_argument("--barge-in-neg", type=int, default=1,
                   help="open-assistant <no voice> negatives per dialogue")
    p.add_argument("--with-rules", type=int, default=1,
                   help="prompt-rule scenarios (v6) per dialogue (0 disables)")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    tok = load_tokenizer(args.tokenizer)
    tag_ids: dict[str, int] = {}
    for key, text in TOKEN.items():
        tag_ids[key] = tok.encode(text, add_special_tokens=False)[0]
    tag_ids["no_tag"] = tok.encode(NO_TAG, add_special_tokens=False)[0]

    enc = DialogueEncoder(tok)
    rng = random.Random(args.seed + 1)
    sys_generic = (
        enc.header_ids[ROLE_SYSTEM]
        + tok.encode(GENERIC_INSTRUCTION, add_special_tokens=False)
        + enc.tail_ids[ROLE_SYSTEM]
    )

    stats: Counter[str] = Counter()
    skipped = 0
    n_records = 0
    n_main = 0
    n_sas = 0
    n_barge = 0
    n_silence = 0
    n_rule = 0
    n_instr = 0

    out_dir = Path(args.out)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir.with_suffix(".parquet")
    schema = pa.schema(
        [
            pa.field("input_ids", pa.list_(pa.int64())),
            pa.field("labels", pa.list_(pa.int64())),
            pa.field("attention_mask", pa.list_(pa.int64())),
            pa.field("dialogue_id", pa.string()),
            pa.field("kind", pa.string()),
            pa.field("loss_weight", pa.float64()),
        ]
    )
    writer = pq.ParquetWriter(parquet_path, schema)
    batch_rows: list[dict] = []

    def flush() -> None:
        nonlocal batch_rows
        if not batch_rows:
            return
        rows_out = []
        for r in batch_rows:
            rows_out.append(
                {
                    "input_ids": r["input_ids"].tolist(),
                    "labels": r["labels"].tolist(),
                    "attention_mask": r["attention_mask"].tolist(),
                    "dialogue_id": r["dialogue_id"],
                    "kind": r["kind"],
                    "loss_weight": float(r["loss_weight"]),
                }
            )
        writer.write_table(pa.Table.from_pylist(rows_out))
        batch_rows = []

    with Path(args.duplex).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            items = rec["items"]
            enc.new_dialogue()
            if args.limit and n_records >= args.limit:
                break

            def add(r: dict | None) -> None:
                nonlocal n_records, n_instr
                if r is None:
                    return
                if r["kind"] != "rule" and rng.random() < 0.5:
                    # instruction-conditioned mix (rule records already carry
                    # their own specific instruction)
                    r = prepend_instruction(r, sys_generic)
                    n_instr += 1
                batch_rows.append(r)
                n_records += 1

            main = build_main_record(enc, items, args.max_seq, tag_ids, rec.get("id", ""))
            if main is None:
                skipped += 1
                continue
            add(main)
            n_main += 1
            segments, targets = enc.encode(items)
            if args.sas_per_chunk > 0:
                for sas in build_sas_records(
                    enc, items, segments, targets, args.max_seq, rng,
                    args.sas_per_chunk, tag_ids, rec.get("id", ""),
                ):
                    add(sas)
                    n_sas += 1
            if args.barge_in_per_dialogue + args.barge_in_bc + args.barge_in_neg > 0:
                for barg in build_barge_in_records(
                    enc, items, segments, args.max_seq, rng, tag_ids,
                    rec.get("id", ""),
                    args.barge_in_per_dialogue, args.barge_in_bc, args.barge_in_neg,
                ):
                    add(barg)
                    n_barge += 1
            for sil in build_silence_records(
                enc, items, segments, targets, args.max_seq, rng, tag_ids,
                rec.get("id", ""),
            ):
                add(sil)
                n_silence += 1
            if args.with_rules:
                for rul in build_rule_records(
                    enc, items, segments, targets, args.max_seq, rng,
                    tag_ids, rec.get("id", ""),
                ):
                    add(rul)
                    n_rule += 1
            for lab in main["labels"].tolist():
                if lab != -100:
                    key = next((k for k, v in tag_ids.items() if v == lab), "?")
                    stats[key] += 1
            if len(batch_rows) >= 25000:
                flush()

    flush()
    writer.close()
    del writer

    ds = Dataset.from_parquet(str(parquet_path))
    n = len(ds)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed)).tolist()
    n_eval = int(n * args.eval_frac) if args.eval_frac > 0 else 0
    eval_idx, train_idx = idx[:n_eval], idx[n_eval:]
    split = DatasetDict(
        {
            "train": ds.select(train_idx),
            "eval": ds.select(eval_idx) if n_eval else ds.select([]),
        }
    )
    split.save_to_disk(args.out)
    parquet_path.unlink(missing_ok=True)

    example = ds.select([0])[0] if n else None
    lens = torch.tensor(
        [len(r["input_ids"]) for r in ds.select(range(min(5000, n)))],
        dtype=torch.float,
    )
    print(f"\n[prep] {n} records (main {n_main}, sas {n_sas}, barge {n_barge}, "
          f"silence {n_silence}, rule {n_rule}, instr-mixed {n_instr}; "
          f"train {len(train_idx)}, eval {len(eval_idx)}), {skipped} skipped")
    print(f"[prep] mean seq len {lens.mean():.0f} (sample 5000), "
          f"masked share N/A (streamed)")
    print("[prep] boundary target counts (loss positions, main records):")
    for k in ["no_tag"] + list(TOKEN_WEIGHT):
        print(f"   {k:28s} {stats.get(k, 0):8d}")

    if example is not None:
        print("\n[prep] example main record (boundaries marked with |):")
        ids = example["input_ids"]
        labs = example["labels"]
        out = []
        for tid, lab in zip(ids, labs):
            out.append(tok.decode([tid], skip_special_tokens=False))
            if lab != -100:
                out[-1] += " |"
        print("   " + "".join(out)[:800])


if __name__ == "__main__":
    main()