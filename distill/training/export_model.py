"""Export the trained duplex adapter into server-ready artifacts (SPEC 6.5).

Loads the QLoRA adapter produced by training/train_qlora.py and writes:
  * merged_bf16/           - full bf16 merged model + tokenizer (GGUF-ready)
  * model_state.safetensors - flat state dict (DuplexCascade/server.py path)
  * train_cfg.json         - the config DuplexCascade/server.py expects

The base is loaded in BF16 (not 4-bit) so the LoRA merge is a clean bf16 model;
the earlier 4-bit merge kept quantized bnb weights and tripped transformers'
tied-weight / weight-conversion checks. The adapter's LoRA deltas are dtype-
independent, so merging onto bf16 works.

Vocab handling (verified against the smoke run): Qwen3-4B-Instruct-2507 ships
model rows 151936 but a 151669-token tokenizer. The adapter's own tokenizer is
151676 (151669 + 7 duplex specials). Loading therefore MUST use the adapter's
tokenizer and resize_token_embeddings before PeftModel.from_pretrained, and the
tokenizer MUST be saved alongside the merged model.

Usage:
  python training/export_model.py --adapter /mnt/f/duplex_cascade_runs/full/adapter \\
      --out /mnt/f/duplex_cascade_runs/full/export
  python training/export_gguf.py --merged /mnt/f/duplex_cascade_runs/full/export/merged_bf16
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
except ImportError:  # pragma: no cover
    print("[export] run from the unsloth venv: /home/penhfel/unsloth_uv/bin/python")
    sys.exit(1)

import torch
from transformers import AutoTokenizer
from peft import PeftModel

from data.common import SPECIAL_TOKENS, TOKEN_WEIGHT


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", default="/mnt/f/duplex_cascade_runs/full/adapter")
    p.add_argument("--base-model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--out", default="/mnt/f/duplex_cascade_runs/full/export")
    p.add_argument("--max-seq-length", type=int, default=4096)
    args = p.parse_args()

    t0 = time.time()
    adapter_dir = Path(args.adapter)
    out_dir = Path(args.out)

    # adapter's own tokenizer is authoritative (151676 with the 7 specials)
    tok = AutoTokenizer.from_pretrained(adapter_dir, use_fast=True)
    print(f"[export] adapter tokenizer: {len(tok)} tokens (specials {len(SPECIAL_TOKENS)})")

    # Load the base in BF16 (NOT 4-bit) so the LoRA merge yields a clean bf16
    # model that converts to GGUF. The adapter's LoRA deltas are dtype-
    # independent. (The earlier 4-bit merge path kept quantized bnb weights and
    # tripped transformers' tied-weight / weight-conversion checks.)
    model, _ = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        dtype=torch.bfloat16,
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

    mm = model.merge_and_unload()
    mm.save_pretrained(merged_dir)
    # tokenizer must be saved WITH the model (the converter / server need it)
    tok.save_pretrained(merged_dir)
    print(f"[export] bf16 merged model + tokenizer -> {merged_dir}")

    # server.py load path: a flat `model_state.safetensors` next to train_cfg.json
    from safetensors.torch import save_file

    sd = {k: v.contiguous().to("cpu") for k, v in mm.state_dict().items()}
    save_file(sd, str(out_dir / "model_state.safetensors"))
    print(f"[export] model_state.safetensors -> {out_dir / 'model_state.safetensors'}")

    train_cfg = {
        "model": {"name": f"file://{merged_dir}", "trust_remote_code": False},
        "tokenizer": f"file://{merged_dir}",
        "special_tokens": SPECIAL_TOKENS,
        "token_weights": TOKEN_WEIGHT,
        "gguf": {
            "q4_k_m": str(out_dir / "gguf" / "DuplexCascade-PT-q4_k_m.gguf"),
            "bf16": str(out_dir / "gguf" / "DuplexCascade-PT-bf16.gguf"),
        },
    }
    (out_dir / "train_cfg.json").write_text(
        json.dumps(train_cfg, ensure_ascii=False, indent=2)
    )
    print(f"[export] train_cfg.json -> {out_dir / 'train_cfg.json'}")
    print(f"[export] done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()