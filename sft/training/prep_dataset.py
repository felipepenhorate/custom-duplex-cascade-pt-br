"""M1 — tokenization + label preparation (stage 3 of the data pipeline).

Turns duplex micro-turn items (data/build_duplex_dataset.py output) into
ChatML-formatted training sequences with:

  * input_ids  : <bos> <|im_start|>user\n ... <|im_end|>\n <|im_start|>assistant\n ...
  * labels     : next-token targets; USER-token positions are -100 (loss is
                 applied only on system micro-turns, paper §3.3.1); the final
                 position (no next token) is also -100.
  * weights    : per-token loss weight (1.0 for text, §4.1 weight for special
                 tokens, 0.0 on masked positions).

Sequences that exceed --max-seq are truncated from the front (keeps the
tail where all supervision lives). Output is an HF Dataset (arrow) split
into train/eval, saved to disk.

Usage:
  python training/prep_dataset.py --duplex data/duplex_train.jsonl \\
      --out data/prepped --tokenizer Qwen/Qwen3-4B-Instruct-2507
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

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
    IM_START_USER,
    SPECIAL_TOKENS,
    TOKEN,
    TOKEN_WEIGHT,
)

ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"


def special_token_id(tokenizer, text: str) -> int | None:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None


def encode_items(tokenizer, items: list[dict], max_seq: int) -> dict | None:
    """Encode one duplex dialogue into input_ids/labels/weights.

    Returns None when the dialogue cannot be represented (e.g., malformed
    alternation after front-truncation).
    """
    header_user = tokenizer.encode(IM_START_USER, add_special_tokens=False)
    header_assistant = tokenizer.encode(IM_START_ASSISTANT, add_special_tokens=False)
    im_end = tokenizer.encode(IM_END, add_special_tokens=False)
    im_end_nl = tokenizer.encode(IM_END_NL, add_special_tokens=False)

    # token-id -> loss weight for assistant-side special tokens
    id_weight: dict[int, float] = {}
    for key in TOKEN_WEIGHT:
        if key == "no_voice":
            continue
        tid = special_token_id(tokenizer, TOKEN[key])
        if tid is not None:
            id_weight[tid] = TOKEN_WEIGHT[key]

    seq: list[int] = []
    seq_weight: list[float] = []  # weight of the token AT the same index
    seq_supervised: list[bool] = []  # is position i a supervised predictor?
    boundaries: list[int] = [0]  # index of every message start (for truncation)

    if tokenizer.bos_token_id is not None:
        seq.append(int(tokenizer.bos_token_id))
        seq_weight.append(0.0)
        seq_supervised.append(False)

    for item in items:
        role = item["role"]
        if role == ROLE_USER:
            if "token_ids" in item:
                content_ids: list[int] = list(item["token_ids"])
            else:
                content_ids = tokenizer.encode(item["text"], add_special_tokens=False)
            seg = header_user + content_ids + im_end_nl
            seg_weight = [0.0] * len(seg)
            seg_sup = [False] * len(seg)
        else:
            if "token_ids" in item:
                content_ids = list(item["token_ids"])
            else:
                content_ids = tokenizer.encode(item["text"], add_special_tokens=False)
            seg = header_assistant + content_ids + im_end
            seg_weight = [1.0] * len(seg)
            for pos, tid in enumerate(seg):
                if tid in id_weight:
                    seg_weight[pos] = id_weight[tid]
            seg_sup = [True] * len(seg)

        boundaries.append(len(seq) + len(seg))
        seq.extend(seg)
        seq_weight.extend(seg_weight)
        seq_supervised.extend(seg_sup)

    # front-truncation to max_seq: drop whole messages from the front,
    # choosing the first message boundary >= len(seq) - max_seq.
    if len(seq) > max_seq:
        cut = len(seq) - max_seq
        for b in boundaries:
            if b >= cut:
                cut = b
                break
        else:
            cut = len(seq)
        seq = seq[cut:]
        seq_weight = seq_weight[cut:]
        seq_supervised = seq_supervised[cut:]

    if not any(seq_supervised):
        return None

    # labels: next-token target; -100 where not supervised or at the tail
    labels = [-100] * len(seq)
    for i in range(len(seq) - 1):
        if seq_supervised[i]:
            labels[i] = seq[i + 1]

    # weights apply to the *predicted* token (seq[i+1]) at position i
    weights = [0.0] * len(seq)
    for i in range(len(seq) - 1):
        if labels[i] != -100:
            weights[i] = seq_weight[i + 1] if seq_supervised[i + 1] else 0.0

    return {
        "input_ids": torch.tensor(seq, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "weights": torch.tensor(weights, dtype=torch.float32),
        "attention_mask": torch.ones(len(seq), dtype=torch.long),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--duplex", default="data/duplex_train.jsonl")
    p.add_argument("--out", default="data/prepped")
    p.add_argument("--tokenizer", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--max-seq", type=int, default=4096)
    p.add_argument("--eval-frac", type=float, default=0.005)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    n_added = tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    print(f"[prep] added {n_added} special tokens; vocab now {len(tokenizer)}")

    rows: list[dict] = []
    stats: Counter[str] = Counter()
    skipped = 0

    with Path(args.duplex).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            enc = encode_items(tokenizer, rec["items"], max_seq=args.max_seq)
            if enc is None:
                skipped += 1
                continue
            for it in rec["items"]:
                if it["role"] == ROLE_ASSISTANT and it.get("special"):
                    stats[it["special"]] += 1
            enc["id"] = rec.get("id", "")
            rows.append(enc)
            if args.limit and len(rows) >= args.limit:
                break

    ds = Dataset.from_list(rows)
    n = len(ds)

    # eval split: first `eval_frac` share (seeded ordering for reproducibility)
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

    lens = torch.tensor([len(r["input_ids"]) for r in rows], dtype=torch.float)
    masked = torch.tensor([(r["labels"] == -100).float().mean().item() for r in rows])
    print(f"[prep] {n} sequences (train {len(train_idx)}, eval {len(eval_idx)}), "
          f"{skipped} skipped/oversized")
    print(f"[prep] mean seq len {lens.mean():.0f}, max {lens.max():.0f}, "
          f"masked(user+headers) share {masked.mean():.2%}")
    print("[prep] assistant special-token supervision counts:")
    for k, v in stats.most_common():
        print(f"   {k:28s} {v:8d}")


if __name__ == "__main__":
    main()