"""M2 — tokenization + label preparation for DuplexCascade-Distill.

Fork of duplex_cascade/training/prep_dataset.py extended for the SDFT loss:

  * duplex rows (from data/build_teacher_duplex.py, same items schema):
      input_ids / labels / weights / attention_mask as before, plus
      `is_anchor=False`. Loss supervision is split in the trainer into
      TAG positions (label is a duplex special token -> weighted CE) and
      CONTENT positions (everything else supervised -> forward-KL to the
      frozen base model). User-side + header positions stay masked (-100).
  * anchor rows (from data/build_anchor_prompts.py, D2 variant):
      `is_anchor=True`; the whole ChatML prompt (system + user) is
      supervised as CONTENT so the trainer KLs the student back to the base
      distribution on general prompts (function calling, QA) -> LoRA is
      pinned to ~0 there. No special tokens ever appear in anchor rows.
  * artifact masking (SDFT "Learned Artifacts" fix): with
      --mask-first-tokens N, the first N content tokens after each
      <|user finish speaking|> tag are NOT supervised, so the student never
      copies teacher surface artifacts (e.g. "Entendi..." prefixed output).

The trainer (training/train_distill.py) does NOT need is_anchor for the loss
(anchor rows simply have is_special=False everywhere -> all-KL), but the column
is kept for bookkeeping/analysis.

Usage:
  python training/prep_dataset.py --duplex data/teacher_duplex_train.jsonl \\
      [--anchors data/anchor_prompts.jsonl] \\
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

FINISH_SPECIAL = "user_finish_speaking"


def special_token_id(tokenizer, text: str) -> int | None:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None


def encode_items(
    tokenizer,
    items: list[dict],
    max_seq: int,
    mask_first_tokens: int = 2,
) -> dict | None:
    """Encode one duplex dialogue into input_ids/labels/weights/is_anchor.

    Same as the original, plus:
      * `is_anchor` = False,
      * artifact masking: in every <|user finish speaking|> assistant item,
        the first `mask_first_tokens` CONTENT tokens of the response are left
        unsupervised (SDFT fix). The tag itself is never the masked target —
        masking position 0 of the finish item only drops the CE on the first
        content token (the tag is predicted by the PREVIOUS position).
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
    seq_weight: list[float] = []
    seq_supervised: list[bool] = []
    boundaries: list[int] = [0]

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
            if item.get("special") == FINISH_SPECIAL and mask_first_tokens > 0:
                # mask the first `mask_first_tokens` content tokens of the
                # response: positions 0..mask-1 of this item (position 0
                # predicts content[0]; the tag itself is predicted earlier).
                n = min(mask_first_tokens, len(seg) - 1)
                seg_sup[:n] = [False] * n

        boundaries.append(len(seq) + len(seg))
        seq.extend(seg)
        seq_weight.extend(seg_weight)
        seq_supervised.extend(seg_sup)

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

    labels = [-100] * len(seq)
    for i in range(len(seq) - 1):
        if seq_supervised[i]:
            labels[i] = seq[i + 1]

    weights = [0.0] * len(seq)
    for i in range(len(seq) - 1):
        if labels[i] != -100:
            weights[i] = seq_weight[i + 1] if seq_supervised[i + 1] else 0.0

    return {
        "input_ids": torch.tensor(seq, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "weights": torch.tensor(weights, dtype=torch.float32),
        "attention_mask": torch.ones(len(seq), dtype=torch.long),
        "is_anchor": torch.zeros(len(seq), dtype=torch.bool),
    }


def encode_anchor(tokenizer, rec: dict, max_seq: int) -> dict | None:
    """Encode one anchor prompt (system + user) as a fully-supervised sequence.

    The whole prompt is CONTENT supervision (is_anchor=True, no tags), so the
    trainer applies forward-KL to the frozen base model on every position.
    """
    header_user = tokenizer.encode(IM_START_USER, add_special_tokens=False)
    im_end_nl = tokenizer.encode(IM_END_NL, add_special_tokens=False)
    header_assistant = tokenizer.encode(IM_START_ASSISTANT, add_special_tokens=False)

    system_text = str(rec.get("system") or "").strip()
    user_text = str(rec.get("user") or "").strip()
    if not user_text:
        return None

    seg: list[int] = []
    if system_text:
        seg += tokenizer.encode(IM_START_USER, add_special_tokens=False) + \
            tokenizer.encode(system_text, add_special_tokens=False) + \
            tokenizer.encode(IM_END_NL, add_special_tokens=False)
    seg += header_user + tokenizer.encode(user_text, add_special_tokens=False) + \
        im_end_nl + header_assistant

    if len(seg) > max_seq:
        seg = seg[-max_seq:]

    seq = list(seg)
    seq_supervised = [True] * len(seq)
    labels = [-100] * len(seq)
    for i in range(len(seq) - 1):
        if seq_supervised[i]:
            labels[i] = seq[i + 1]

    return {
        "input_ids": torch.tensor(seq, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "weights": torch.ones(len(seq), dtype=torch.float32),
        "attention_mask": torch.ones(len(seq), dtype=torch.long),
        "is_anchor": torch.ones(len(seq), dtype=torch.bool),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--duplex", default="data/teacher_duplex_train.jsonl")
    p.add_argument("--anchors", default=None, help="anchor prompts JSONL (D2 variant)")
    p.add_argument("--out", default="data/prepped")
    p.add_argument("--tokenizer", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--max-seq", type=int, default=4096)
    p.add_argument("--eval-frac", type=float, default=0.005)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--mask-first-tokens", type=int, default=2,
                   help="artifact masking: leave first N content tokens after each "
                        "<|user finish speaking|> unsupervised (SDFT fix)")
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
            enc = encode_items(tokenizer, rec["items"], max_seq=args.max_seq,
                               mask_first_tokens=args.mask_first_tokens)
            if enc is None:
                skipped += 1
                continue
            for it in rec["items"]:
                if it["role"] == ROLE_ASSISTANT and it.get("special"):
                    stats[it["special"]] += 1
            enc["id"] = rec.get("id", "")
            enc["kind"] = "duplex"
            rows.append(enc)
            if args.limit and len(rows) >= args.limit:
                break

    if args.anchors:
        n_anchor = 0
        with Path(args.anchors).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                enc = encode_anchor(tokenizer, rec, max_seq=args.max_seq)
                if enc is None:
                    skipped += 1
                    continue
                enc["id"] = rec.get("id", f"anchor-{n_anchor}")
                enc["kind"] = "anchor"
                rows.append(enc)
                n_anchor += 1
        print(f"[prep] {n_anchor} anchor rows appended")

    ds = Dataset.from_list(rows)
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

    lens = torch.tensor([len(r["input_ids"]) for r in rows], dtype=torch.float)
    masked = torch.tensor([(r["labels"] == -100).float().mean().item() for r in rows])
    anchors = sum(1 for r in rows if r["is_anchor"].any())
    print(f"[prep] {n} sequences (train {len(train_idx)}, eval {len(eval_idx)}), "
          f"{skipped} skipped/oversized, {anchors} anchor rows")
    print(f"[prep] mean seq len {lens.mean():.0f}, max {lens.max():.0f}, "
          f"masked(user+headers+artifact) share {masked.mean():.2%}")
    print("[prep] assistant special-token supervision counts:")
    for k, v in stats.most_common():
        print(f"   {k:28s} {v:8d}")


if __name__ == "__main__":
    main()