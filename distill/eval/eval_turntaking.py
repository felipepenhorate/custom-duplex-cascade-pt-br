"""M3 — duplex turn-taking + base-distribution drift eval (SPEC §6 M3).

For a given bf16 model, scores the held-out duplex eval split (prepped at
max-seq 1024) using the offline teacher cache:

  * tag_accuracy   — argmax == supervised duplex special token at tag positions
  * tag_acc_by_tok — per-special-token accuracy (the protocol is learned?)
  * kl_drift       — mean per-token forward-KL (student ∥ frozen base) on
                     content positions (the "everything else stays the same" metric)

Reference points:
  * base   — tag_accuracy ≈ 0 (doesn't know the protocol), kl_drift ≈ 0 (it IS the
             base), content logits vs the bf16 cache are tiny.
  * distill— high tag accuracy + LOW kl_drift  (learned tags without drifting).
  * sft    — high tag accuracy + HIGH kl_drift (learned tags by drifting = the
             regression the project fixes).

Usage:
  python eval/eval_turntaking.py --model <path> --name base \\
      --prepped /mnt/f/duplex_cascade_runs/distill/prepped_1024_cached/dataset \\
      --cache-dir /mnt/f/duplex_cascade_runs/distill/prepped_1024_cached
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.common import SPECIAL_TOKENS  # noqa: E402


def lazy_init_special_embeddings(model, tokenizer, sigma: float = 0.02) -> None:
    emb = model.get_input_embeddings()
    n = len(SPECIAL_TOKENS)
    start = emb.weight.size(0) - n
    with torch.no_grad():
        emb.weight.data[start:] = torch.randn_like(emb.weight.data[start:]) * sigma
    if model.config.tie_word_embeddings:
        model.get_output_embeddings().weight.data[start:] = emb.weight.data[start:]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--name", default="model")
    p.add_argument("--prepped",
                   default="/mnt/f/duplex_cascade_runs/distill/prepped_1024_cached/dataset")
    p.add_argument("--cache-dir",
                   default="/mnt/f/duplex_cascade_runs/distill/prepped_1024_cached")
    p.add_argument("--top-k", type=int, default=32)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    ds = load_from_disk(args.prepped)
    eval_rows = ds["eval"]
    if args.limit:
        eval_rows = eval_rows.select(range(min(args.limit, len(eval_rows))))

    with (Path(args.cache_dir) / "topk_ids.pkl").open("rb") as f:
        ids_l = pickle.load(f)
    with (Path(args.cache_dir) / "topk_logits.pkl").open("rb") as f:
        logits_l = pickle.load(f)

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=False)
    n_added = tok.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    if n_added > 0:
        model.resize_token_embeddings(len(tok))
        torch.manual_seed(1234)  # match the cache generator's special-init
        lazy_init_special_embeddings(model, tok)
    model.eval()
    special_ids = {
        tid for s in SPECIAL_TOKENS
        if len(ids := tok.encode(s, add_special_tokens=False)) == 1
        for tid in ids
    }
    id_to_name = {
        tok.encode(s, add_special_tokens=False)[0]: s
        for s in SPECIAL_TOKENS
        if len(tok.encode(s, add_special_tokens=False)) == 1
    }
    print(f"[{args.name}] loaded {args.model} (bf16); special ids {sorted(special_ids)}", file=sys.stderr)

    n_tag = 0
    n_tag_correct = 0
    tag_by_tok: dict[int, list[bool]] = {}
    kl_sum = 0.0
    n_content = 0

    with torch.no_grad():
        for r in eval_rows:
            input_ids = torch.tensor(r["input_ids"], dtype=torch.long).unsqueeze(0).to(model.device)
            labels = torch.as_tensor(r["labels"], dtype=torch.long)
            ci = int(r["cache_idx"])
            t_ids = torch.as_tensor(ids_l[ci]).to(model.device)
            t_vals = torch.as_tensor(logits_l[ci]).to(model.device).to(torch.bfloat16)
            T = input_ids.size(1)
            logits = model(input_ids).logits[0, :T]  # [T, V]

            flat = logits.reshape(-1, logits.size(-1))
            lse = torch.logsumexp(flat, dim=-1)
            preds = flat.argmax(dim=-1)

            for t in range(T - 1):
                lab = int(labels[t])
                if lab == -100:
                    continue
                if lab in special_ids:
                    n_tag += 1
                    ok = int(preds[t].item() == lab)
                    n_tag_correct += ok
                    tag_by_tok.setdefault(lab, []).append(bool(ok))
                else:
                    # content KL vs cached teacher top-k
                    k = min(args.top_k, t_ids.shape[1])
                    tid_k = t_ids[t, :k]
                    tval_k = t_vals[t, :k]
                    p = torch.softmax(tval_k, dim=-1)
                    sval = flat[t].gather(-1, tid_k)
                    logq = sval - lse[t]
                    kl = -(p * logq).sum()
                    kl_sum += kl.item()
                    n_content += 1

    tag_acc = n_tag_correct / max(n_tag, 1)
    kl_drift = kl_sum / max(n_content, 1)

    summary = {
        "name": args.name,
        "tag_accuracy": round(tag_acc, 4),
        "n_tag_positions": n_tag,
        "kl_drift": round(kl_drift, 4),
        "n_content_positions": n_content,
        "tag_acc_by_token": {
            id_to_name.get(k, str(k)): round(sum(v) / len(v), 3)
            for k, v in sorted(tag_by_tok.items())
        },
    }
    print(json_dumps(summary))
    print(f"[{args.name}] done in {(time.time() - t0) / 60:.1f} min", file=sys.stderr)


def json_dumps(obj) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()