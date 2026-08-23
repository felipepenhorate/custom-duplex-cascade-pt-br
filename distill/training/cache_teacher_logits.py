"""M2 (SPEC §5.3) — offline teacher top-k logit cache.

Precomputes, for every position of every prepped sequence, the frozen base
model's top-k next-token ids + logits. Training (train_distill.py
--teacher-cache) then skips the second forward pass entirely: peak VRAM drops
~2x, steps speed up (~SFT rate) and the allocator fragmentation seen in the
on-the-fly path (16 GB ceiling + two full [B,T,V] logit forwards) disappears.

The teacher is the SAME distribution the on-the-fly trainer used: base model
minus adapters. We load the base in bf16 (closest to the M1 content
distribution) with the same tokenizer (7 duplex specials appended, embeddings
resized, special rows Gaussian-init sigma=0.02) so the tag ids in the input
are valid ids for the forward.

Output layout (--out):
  dataset/           train+eval prepped dataset, each row gains `cache_idx`
  topk_ids.pkl       list[np.int32 [T_i, K]]   (global index == cache_idx)
  topk_logits.pkl    list[np.float16 [T_i, K]]

Usage:
  python training/cache_teacher_logits.py \\
      --dataset /mnt/f/duplex_cascade_runs/distill/prepped \\
      --out /mnt/f/duplex_cascade_runs/distill/prepped_cached
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.common import SPECIAL_TOKENS


def lazy_init_special_embeddings(model, tokenizer, sigma: float = 0.02) -> None:
    emb = model.get_input_embeddings()
    n = len(SPECIAL_TOKENS)
    start = emb.weight.size(0) - n
    with torch.no_grad():
        emb.weight.data[start:] = torch.randn_like(emb.weight.data[start:]) * sigma
    if model.config.tie_word_embeddings:
        model.get_output_embeddings().weight.data[start:] = emb.weight.data[start:]
    print(f"[cache] special embeddings (last {n} rows) gaussian-init sigma={sigma}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="/mnt/f/duplex_cascade_runs/distill/prepped")
    p.add_argument("--out", default="/mnt/f/duplex_cascade_runs/distill/prepped_cached")
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--top-k", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--limit", type=int, default=None, help="cap rows per split (test)")
    args = p.parse_args()

    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ds = load_from_disk(args.dataset)
    print(f"[cache] dataset: train={len(ds['train'])} eval={len(ds['eval'])}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    n_added = tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    print(f"[cache] added {n_added} special tokens; vocab = {len(tokenizer)}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.resize_token_embeddings(len(tokenizer))
    torch.manual_seed(1234)  # reproducible special-embedding init (matches eval)
    lazy_init_special_embeddings(model, tokenizer)
    model.eval()

    K = args.top_k
    ids_list: list[np.ndarray] = []
    logits_list: list[np.ndarray] = []
    global_idx = 0

    def cache_split(split_name: str, split) -> None:
        nonlocal global_idx
        rows = split.select(range(min(len(split), args.limit))) if args.limit else split
        n = len(rows)
        start = time.time()
        with torch.no_grad():
            for b0 in range(0, n, args.batch_size):
                idxs = range(b0, min(b0 + args.batch_size, n))
                batch = rows.select(idxs)
                lengths = [len(r["input_ids"]) for r in batch]
                max_t = max(lengths)
                input_ids = torch.full(
                    (len(batch), max_t), 0, dtype=torch.long
                )
                for bi, r in enumerate(batch):
                    input_ids[bi, : lengths[bi]] = torch.as_tensor(r["input_ids"])
                input_ids = input_ids.to(model.device)
                attn = torch.ones_like(input_ids)
                logits = model(input_ids=input_ids, attention_mask=attn).logits
                for bi, r in enumerate(batch):
                    T = lengths[bi]
                    vals, ids = logits[bi, :T].topk(K, dim=-1)  # [T,K]
                    ids_list.append(ids.cpu().numpy().astype(np.int32))
                    logits_list.append(
                        vals.float().cpu().numpy().astype(np.float16)
                    )
                    global_idx += 1
                del logits, input_ids
        print(f"[cache] {split_name}: {n} rows in {time.time() - start:.1f}s", flush=True)

    cache_split("train", ds["train"])
    cache_split("eval", ds["eval"])

    # attach a global cache index to every cached row (train then eval)
    cache_ids = list(range(global_idx))
    n_train_cached = len(ds["train"].select(range(min(len(ds["train"]), args.limit)))) if args.limit else len(ds["train"])
    n_eval_cached = len(ds["eval"].select(range(min(len(ds["eval"]), args.limit)))) if args.limit else len(ds["eval"])
    train_src = ds["train"].select(range(n_train_cached))
    eval_src = ds["eval"].select(range(n_eval_cached))
    new_train = train_src.add_column("cache_idx", cache_ids[:n_train_cached])
    new_eval = eval_src.add_column("cache_idx", cache_ids[n_train_cached:n_train_cached + n_eval_cached])
    from datasets import DatasetDict

    new_ds = DatasetDict({"train": new_train, "eval": new_eval})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    new_ds.save_to_disk(str(out / "dataset"))
    with (out / "topk_ids.pkl").open("wb") as f:
        pickle.dump(ids_list, f)
    with (out / "topk_logits.pkl").open("wb") as f:
        pickle.dump(logits_list, f)
    print(f"[cache] wrote {global_idx} rows -> {out}")


if __name__ == "__main__":
    main()