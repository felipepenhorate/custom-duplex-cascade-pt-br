"""Export the trained policy-companion LoRA adapter to a standalone bf16
model (SPEC 7 / sft LEARNINGS #4).

Loads the base Qwen3.5-0.8B text-only in bf16 (NOT 4-bit), resizes to the
adapter's tokenizer, attaches the adapter via PeftModel, merges and unloads,
and saves the merged model + tokenizer + policy_cfg.json — the artifact the
runtime sidecar (policy/duplex_policy.py) loads.

Usage:
  python training/export_policy.py \\
      --adapter /mnt/f/duplex_cascade_runs/mtp_like/runs/final/adapter \\
      --out /mnt/f/duplex_cascade_runs/mtp_like/runs/final/merged
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", default="/mnt/f/duplex_cascade_runs/mtp_like/runs/final/adapter")
    p.add_argument("--out", default="/mnt/f/duplex_cascade_runs/mtp_like/runs/final/merged")
    p.add_argument("--model", default=None,
                   help="base model (default: read from adapter policy_cfg.json)")
    args = p.parse_args()

    adapter_dir = Path(args.adapter)
    cfg = json.loads((adapter_dir / "policy_cfg.json").read_text())
    base = args.model or cfg["base_model"]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[export] base {base} in bf16 (merge onto bf16, not 4-bit)")
    model = AutoModelForCausalLM.from_pretrained(
        base, dtype=torch.bfloat16, device_map="cuda", token=False,
    )
    tok = AutoTokenizer.from_pretrained(str(adapter_dir), token=False)
    model.resize_token_embeddings(len(tok))
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model = model.merge_and_unload()
    model.save_pretrained(str(out_dir), safe_serialization=True)
    tok.save_pretrained(str(out_dir))  # REQUIRED (sft LEARNINGS #3)
    (out_dir / "policy_cfg.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2)
    )
    print(f"[export] merged bf16 + tokenizer + policy_cfg.json -> {out_dir}")


if __name__ == "__main__":
    main()