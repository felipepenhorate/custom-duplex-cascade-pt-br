"""M1 — duplex micro-turn construction (stage 2 of the data pipeline).

Port of DuplexCascade paper §3.3 "Dynamic Construction of Duplex Training
Data" to pt-BR: splits user/system long turns into token-level micro-turns
(user 1-7 tokens, system fixed 10) and simulates the 6 interaction
phenomena:

  1. randomized micro-turn length         (§3.3.2)
  2. natural pauses    (p=0.10, 1-5x <no voice> -> <user is speaking>)
  3. user interruption (p=0.30 per system turn -> <user interruption>)
  4. user backchannels (p=0.01 per system micro-turn boundary)
  5. system backchannels (<BC/> markers -> <system backchannel>, beta variant)
  6. user thinking     (1-20x <no voice> -> <user is thinking>)

Micro-turn content is stored as canonical token ids (never re-encoded), so
sub-word chunk boundaries survive exactly into the training stream. Loss
supervision follows the paper: next-token prediction is applied ONLY on
system micro-turns (user tokens are masked in training/prep_dataset.py), and
special tokens carry their weighted loss (§4.1 weights in data/common.py).

Messages strictly alternate user/assistant (ChatML-compatible). Each user
message gets exactly one assistant reply (a single micro-turn).

Usage:
  python data/build_duplex_dataset.py --dialogues data/dialogues_pt.jsonl \\
      --out data/duplex_train.jsonl [--with-system-backchannel]
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.common import PT_BACKCHANNELS, SPECIAL_TOKENS, TOKEN

ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"

SPECIAL_WEIGHT = {
    "user_is_speaking": 1.0,
    "user_finish_speaking": 10.0,
    "user_interruption": 5.0,
    "user_backchannel": 2.0,
    "user_is_thinking": 1.0,
    "system_backchannel": 3.0,
}

_WS_SPLIT = re.compile(r"\s+")


def _clean(text: str) -> str:
    return _WS_SPLIT.sub(" ", text).strip()


class DuplexChunker:
    """Token-level micro-turn chunker, faithful to paper §3.3.2."""

    def __init__(self, tokenizer, system_chunk_len: int = 10, user_chunk_max: int = 7,
                 system_chunk_max: int = 48):
        self.tok = tokenizer
        self.system_chunk_len = system_chunk_len
        self.user_chunk_max = user_chunk_max
        self.system_chunk_max = system_chunk_max
        # special tokens must be registered so their encodings are single ids
        self.tok.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
        self.special_id: dict[str, int] = {}
        for key in TOKEN:
            ids = self.tok.encode(TOKEN[key], add_special_tokens=False)
            assert len(ids) == 1, f"special token not single-id: {TOKEN[key]} -> {ids}"
            self.special_id[key] = ids[0]

    def _split_sentences(self, text: str) -> list[str]:
        parts = re.split(r"(?<=[.!?…])\s+", text)
        return [p.strip() for p in parts if p.strip()]

    def user_chunks(self, text: str, rng: random.Random) -> list[list[int]]:
        chunks: list[list[int]] = []
        for sent in self._split_sentences(_clean(text)):
            ids = self.tok.encode(sent, add_special_tokens=False)
            pos = 0
            while pos < len(ids):
                n = min(rng.randint(1, self.user_chunk_max), len(ids) - pos) if pos + self.user_chunk_max < len(ids) else len(ids) - pos
                chunks.append(ids[pos : pos + n])
                pos += n
        return chunks

    def system_chunks(self, text: str, rng: random.Random | None = None) -> list[list[int]]:
        """Chunk an assistant turn into micro-turns of VARIABLE length.

        The original duplex pipeline fixed every system micro-turn at 10 tokens,
        which taught the model to always speak in ~10-token bursts (the observed
        "small utterances" behavior). To let the model learn BOTH short and long
        utterances we keep a mix: short 8-14 token micro-turns (preserves the
        turn-taking pattern) and longer chunks up to system_chunk_max tokens that
        cover whole phrases/sentences.

        Chunk size is drawn randomly when an rng is supplied (data generation);
        otherwise a fixed small chunk is used (deterministic / eval path)."""
        rng = rng or random.Random(0)
        chunks: list[list[int]] = []
        for sent in self._split_sentences(_clean(text)):
            ids = self.tok.encode(sent, add_special_tokens=False)
            pos = 0
            while pos < len(ids):
                # draw a chunk size in [system_chunk_len, system_chunk_max],
                # biased toward the short end so most micro-turns stay natural
                lo = self.system_chunk_len
                hi = min(self.system_chunk_max, len(ids) - pos)
                if hi <= lo:
                    n = hi
                else:
                    # geometric-ish: mostly small, occasionally large
                    n = max(lo, min(hi, rng.randint(lo, hi) + rng.randint(0, 4)))
                n = min(n, len(ids) - pos)
                chunks.append(ids[pos : pos + n])
                pos += n
        return chunks

    def display(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False).strip()


def build_duplex_items(
    dialogue: dict,
    rng: random.Random,
    chunker: DuplexChunker,
    *,
    pause_prob: float = 0.10,
    pause_max: int = 5,
    interrupt_prob: float = 0.30,
    user_bc_prob: float = 0.01,
    with_system_bc: bool = True,
    system_bc_prob: float = 0.20,
    thinking_max: int = 20,
) -> list[dict]:
    """Convert one dialogue to the duplex micro-turn item list.

    Walk: for every user turn push its chunked messages (each chunk gets
    exactly one assistant reply); the following system turn delivers its
    content chunkwise after <user finish speaking>, separated by <no voice>
    user messages; <user thinking> pairs close each completed system turn.
    On interruption (p=0.30) the remainder of the system turn is abandoned
    and the NEXT user turn is emitted with <user interruption> on its first
    chunk; its own reply is then handled by the ordinary system-turn path.
    """
    tok_ids = chunker.special_id
    items: list[dict] = []

    def push_user(ids: list[int]) -> None:
        items.append(
            {
                "role": ROLE_USER,
                "token_ids": ids,
                "text": chunker.display(ids),
                "special": None,
                "weight": 0.0,
            }
        )

    def push_assistant(ids: list[int], special: str | None = None) -> None:
        weight = 1.0 if special is None else SPECIAL_WEIGHT.get(special, 1.0)
        items.append(
            {
                "role": ROLE_ASSISTANT,
                "token_ids": ids,
                "text": chunker.display(ids),
                "special": special,
                "weight": weight,
            }
        )

    def no_voice() -> list[int]:
        return [tok_ids["no_voice"]]

    def single(key: str, extra: list[int] | None = None) -> list[int]:
        ids = [tok_ids[key]]
        if extra:
            ids.extend(extra)
        return ids

    def thinking_pairs() -> None:
        for _ in range(rng.randint(1, thinking_max)):
            push_user(no_voice())
            push_assistant(single("user_is_thinking"), "user_is_thinking")

    turns = [m["content"].strip() for m in dialogue["messages"]]
    roles = [m["role"] for m in dialogue["messages"]]

    def next_user_idx(idx: int) -> int | None:
        while idx < len(turns) and roles[idx] != ROLE_USER:
            idx += 1
        return idx if idx < len(turns) else None

    def user_phase(uid: int, interrupter: bool, pre_chunks: list[list[int]] | None = None) -> None:
        """Push the micro-turns of user turn `uid`: every chunk is paired
        with its reply, except the last chunk (the system turn answers it
        right afterwards). `interrupter` relabels the turn's chunks with
        <user interruption> (first) / <user is speaking> (middle), per the
        paper's interruption simulation; such turns get no pauses.
        `pre_chunks` is the chunking already drawn by the interruption
        decision (system_phase) — reusing it keeps the rng stream single-
        pass per dialogue."""
        if interrupter:
            chunks = pre_chunks or chunker.user_chunks(turns[uid], rng)
        else:
            chunks = chunker.user_chunks(turns[uid], rng)
        bc_hits: set[int] = set()
        if with_system_bc and not interrupter and len(chunks) > 1:
            for ci in range(len(chunks) - 1):
                if chunker.display(chunks[ci]).endswith(("!", "?", ".", "…")) and rng.random() < system_bc_prob:
                    bc_hits.add(ci)
        for ci, ids in enumerate(chunks):
            if interrupter and ci == 0:
                push_user(ids)
                push_assistant(single("user_interruption"), "user_interruption")
            elif ci in bc_hits:
                push_user(ids)
                push_assistant(single("system_backchannel"), "system_backchannel")
            elif ci < len(chunks) - 1:
                push_user(ids)
                push_assistant(single("user_is_speaking"), "user_is_speaking")
                if not interrupter and rng.random() < pause_prob:
                    for _ in range(rng.randint(1, pause_max)):
                        push_user(no_voice())
                        push_assistant(single("user_is_speaking"), "user_is_speaking")
            else:
                push_user(ids)  # answered by the system turn that follows

    def system_phase(sid: int) -> tuple[bool, int, list[list[int]] | None]:
        """Deliver system turn `sid` chunkwise after <user finish speaking>,
        interrupted with p=0.30 (the remainder is abandoned: the walk
        continues with the next user turn, relabeled as interrupter).
        A completed turn closes with 1-20x <user is thinking>.

        An interruption requires a MULTI-chunk next question: with a
        single-chunk interrupter the stream would degenerate into two
        consecutive assistant messages (<user interruption> answered the
        only user chunk; <user finish speaking> would follow it), which
        breaks strict alternation — such turns run to completion instead.

        Returns (interrupted?, next turn index, question chunks drawn for
        the interruption decision, so the caller can reuse them)."""
        s_chunks = chunker.system_chunks(turns[sid], rng)

        q_uid = next_user_idx(sid + 1)
        q_chunks = chunker.user_chunks(turns[q_uid], rng) if q_uid is not None else []
        can_interrupt = len(q_chunks) > 1
        interrupted = can_interrupt and len(s_chunks) > 1 and rng.random() < interrupt_prob
        cut = rng.randint(1, len(s_chunks) - 1) if interrupted else len(s_chunks)

        push_assistant(single("user_finish_speaking", s_chunks[0]), "user_finish_speaking")
        for si in range(1, cut):
            if rng.random() < user_bc_prob:
                push_user(chunker.tok.encode(rng.choice(PT_BACKCHANNELS), add_special_tokens=False))
                push_assistant(single("user_backchannel", s_chunks[si]), "user_backchannel")
            else:
                push_user(no_voice())
                push_assistant(s_chunks[si], None)

        if interrupted:
            return True, sid + 1, q_chunks
        thinking_pairs()
        return False, sid + 1, None

    idx = 0
    interrupter = False
    pre_chunks: list[list[int]] | None = None
    while True:
        uid = next_user_idx(idx)
        if uid is None:
            break
        sid = uid + 1
        if sid >= len(turns) or roles[sid] != ROLE_ASSISTANT:
            # defensive: user turn with no assistant reply — a strict duplex
            # stream pairs every user micro-turn with a reply, so a dangling
            # final user turn is dropped entirely (keeps alternation)
            idx = sid
            continue
        user_phase(uid, interrupter, pre_chunks)
        interrupter = False
        interrupted, nxt, pre_chunks = system_phase(sid)
        interrupter = interrupted  # next user turn is the interrupting question
        idx = nxt

    return items


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dialogues", default="data/dialogues_pt.jsonl")
    p.add_argument("--out", default="data/duplex_train.jsonl")
    p.add_argument("--tokenizer", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--system-chunk-min", type=int, default=10,
                   help="min assistant micro-turn length in tokens (orig: 10)")
    p.add_argument("--system-chunk-max", type=int, default=48,
                   help="max assistant micro-turn length; larger = longer utterances")
    p.add_argument("--pause-prob", type=float, default=0.10)
    p.add_argument("--pause-max", type=int, default=5)
    p.add_argument("--interrupt-prob", type=float, default=0.30)
    p.add_argument("--user-bc-prob", type=float, default=0.01)
    p.add_argument("--system-bc-prob", type=float, default=0.20)
    p.add_argument("--thinking-max", type=int, default=20)
    p.add_argument("--with-system-backchannel", action="store_true",
                   help="train the beta variant with <system backchannel> supervision")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    chunker = DuplexChunker(tokenizer, system_chunk_len=args.system_chunk_min,
                            user_chunk_max=7, system_chunk_max=args.system_chunk_max)
    rng = random.Random(args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats: Counter[str] = Counter()
    n_total = 0
    n_written = 0
    with out_path.open("w", encoding="utf-8") as f:
        with Path(args.dialogues).open("r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                n_total += 1
                rec = json.loads(line)
                items = build_duplex_items(
                    rec,
                    rng,
                    chunker,
                    pause_prob=args.pause_prob,
                    pause_max=args.pause_max,
                    interrupt_prob=args.interrupt_prob,
                    user_bc_prob=args.user_bc_prob,
                    with_system_bc=args.with_system_backchannel,
                    system_bc_prob=args.system_bc_prob,
                    thinking_max=args.thinking_max,
                )
                if not items:
                    continue
                for it in items:
                    if it["role"] == ROLE_ASSISTANT and it.get("special"):
                        stats[it["special"]] += 1
                stats["total_items"] += len(items)
                # invariant: strictly alternating roles
                prev = None
                for it in items:
                    assert it["role"] != prev, f"non-alternating in {rec.get('id')}"
                    prev = it["role"]
                out = {
                    "id": rec.get("id", f"x{n_written}"),
                    "scenario": rec.get("scenario", {}),
                    "items": items,
                    "meta": {"variant": "beta" if args.with_system_backchannel else "base"},
                }
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
                f.flush()
                n_written += 1
                if args.limit and n_total >= args.limit:
                    break

    print(f"[duplex] {n_written} duplex dialogues written from {n_total} source (-> {out_path})")
    print("[duplex] assistant special-token supervision counts (per token):")
    total = stats["total_items"]
    for k in list(SPECIAL_WEIGHT) + ["total_items"]:
        v = stats.get(k, 0)
        frac = f" ({v / max(total, 1):6.1%})" if k != "total_items" else ""
        print(f"   {k:28s} {v:8d}{frac}")
    print(f"   total items: {total}")


if __name__ == "__main__":
    main()