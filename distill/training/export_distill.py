"""Export the trained DuplexCascade-Distill adapter into server/GGUF-ready artifacts.

Same recipe as duplex_cascade/training/export_model.py (base loaded in BF16, NOT
4-bit, so the LoRA merge yields a clean bf16 model) but uses the modern unsloth
save API `unsloth_save_pretrained_merged` (unsloth >= 2026.8).

Writes:
  * merged_bf16/            - full bf16 merged model + tokenizer (GGUF-ready)
  * train_cfg.json          - the config the server expects (+ distill hyper-params)

Vocab handling: Qwen3-4B-Instruct-2507 ships model rows 151936 / tokenizer 151669;
the adapter's tokenizer is 151676 (151669 + 7 duplex specials). Load the adapter's
tokenizer and resize_token_embeddings BEFORE PeftModel.from_pretrained, and save
the tokenizer alongside the merged model.

Usage:
  python training/export_distill.py --adapter /mnt/f/duplex_cascade_runs/distill/full/adapter \\
      --out /mnt/f/duplex_cascade_runs/distill/full/export \\
      --kl-weight 1.0 --teacher-temperature 1.0 --top-k 32
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from unsloth import FastLanguageModel
    from unsloth.save import unsloth_save_pretrained_merged
except ImportError:  # pragma: no cover
    print("[export] run from the unsloth venv: /home/penhfel/unsloth_uv/bin/python")
    sys.exit(1)

from transformers import AutoTokenizer
from peft import PeftModel

from data.common import SPECIAL_TOKENS, TOKEN_WEIGHT


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", default="/mnt/f/duplex_cascade_runs/distill/full/adapter")
    p.add_argument("--base-model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--out", default="/mnt/f/duplex_cascade_runs/distill/full/export")
    p.add_argument("--max-seq-length", type=int, default=4096)
    # distill hyper-params recorded in train_cfg.json (informational)
    p.add_argument("--kl-weight", type=float, default=1.0)
    p.add_argument("--teacher-temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=32)
    args = p.parse_args()

    t0 = time.time()
    adapter_dir = Path(args.adapter)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(adapter_dir, use_fast=True)
    print(f"[export] adapter tokenizer: {len(tok)} tokens (specials {len(SPECIAL_TOKENS)})")

    model, _ = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        dtype="bfloat16",
        load_in_4bit=False,
        trust_remote_code=False,
    )
    base_rows = model.get_input_embeddings().weight.size(0)
    model.resize_token_embeddings(len(tok))
    print(f"[export] base embed rows {base_rows} -> {len(tok)} (resize before adapter load)")

    model = PeftModel.from_pretrained(model, adapter_dir)
    print(f"[export] adapter '{model.active_adapter}' loaded")

    merged_dir = out_dir / "merged_bf16"
    merged_dir.mkdir(parents=True, exist_ok=True)

    unsloth_save_pretrained_merged(
        model, str(merged_dir), tokenizer=tok, save_method="merged_16bit"
    )
    # tokenizer must be saved WITH the model (converter / server need it)
    tok.save_pretrained(merged_dir)
    print(f"[export] bf16 merged model + tokenizer -> {merged_dir}")

    train_cfg = {
        "model": {"name": f"file://{merged_dir}", "trust_remote_code": False},
        "tokenizer": f"file://{merged_dir}",
        "special_tokens": SPECIAL_TOKENS,
        "token_weights": TOKEN_WEIGHT,
        "distill": {
            "kl_weight": args.kl_weight,
            "teacher_temperature": args.teacher_temperature,
            "top_k": args.top_k,
            "teacher": "frozen-base (offline top-k cache)",
        },
    }
    (out_dir / "train_cfg.json").write_text(
        json.dumps(train_cfg, ensure_ascii=False, indent=2)
    )
    print(f"[export] train_cfg.json -> {out_dir / 'train_cfg.json'}")
    print(f"[export] done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()