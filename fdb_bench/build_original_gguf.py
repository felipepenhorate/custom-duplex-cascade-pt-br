#!/usr/bin/env python3
"""Merge the upstream sbintuitions/DuplexCascade checkpoint into a servable GGUF.

What the checkpoint actually is
-------------------------------
``model_state.safetensors`` (17.4 GB, a real safetensors file despite the
``.safetensors``-via-``torch.save`` naming confusion) holds **453 tensors**:

* 337 full ``Qwen2-7B-Instruct`` weights (keys prefixed ``llm.base_model.model.``),
* 56 ``lora_A`` + 56 ``lora_B`` (r=16, alpha=32 -> scale 2.0) on ``q_proj``/
  ``v_proj`` for all 28 layers — matching the released ``train_cfg.json``
  (``use_lora: qv``, ``lora_r: 16``, ``lora_alpha: 32``),
* 2 ``modules_to_save.default`` tensors: the trained input/output embeddings,
* 2 ``original_module`` copies (PEFT bookkeeping, unused).

So it is a complete model, and the merge is deterministic. Done **streaming and
in place**: the base is loaded once in bf16 on CPU (15.2 GB) and each weight is
copied straight from the mmapped file, with the LoRA delta computed on the fly
(``B @ A`` is at most 3584x16, so nothing large is ever materialised). Loading a
full state dict or calling ``merge_and_unload()`` would need ~33 GB of RAM
against the 31 GB available here.

Then: ``save_pretrained`` -> ``convert_hf_to_gguf.py --outtype bf16`` ->
``llama-quantize q4_k_m`` (4-bit, matching how the three pt-BR variants are
served), and the intermediates are removed.

Run:  /home/penhfel/unsloth_uv/bin/python -m fdb_bench.build_original_gguf
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
from pathlib import Path

ORIG = Path("/mnt/f/duplex_cascade_runs/original_dc")
BASE = "Qwen/Qwen2-7B-Instruct"
LLAMA_DIR = Path("/home/penhfel/llama.cpp")
PREFIX = "llm.base_model.model."
LORA_SCALE = 2.0          # alpha 32 / r 16, from the released train_cfg.json


def merge(orig_dir: Path, out_dir: Path) -> None:
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    sd = orig_dir / "model_state.safetensors"
    if not sd.is_file():
        raise SystemExit(f"missing {sd} — download it first (see README)")

    # The upstream tokenizer carries the 6 duplex tokens on top of Qwen2's vocab
    # (151646 + 6 = 151652), which is exactly the embedding size in the
    # checkpoint — so the vocab must be resized BEFORE the weights are copied.
    tok = AutoTokenizer.from_pretrained(str(orig_dir / "tokenizer"))
    print(f"tokenizer vocab: {len(tok)}")

    with safe_open(str(sd), framework="pt", device="cpu") as f:
        keys = set(f.keys())

        def take(name: str):
            return f.get_tensor(PREFIX + name)

        cfg = AutoConfig.from_pretrained(BASE)
        model = AutoModelForCausalLM.from_pretrained(
            BASE, dtype=torch.bfloat16, low_cpu_mem_usage=True, config=cfg)
        model.resize_token_embeddings(len(tok))
        print(f"base loaded; vocab resized to {len(tok)}")

        n_copied = n_lora = 0
        with torch.no_grad():
            for name, param in model.named_parameters():
                # Embeddings and the head live under modules_to_save in the PEFT
                # checkpoint, where the parameter path loses its trailing
                # ``.weight``: ``model.embed_tokens.weight`` becomes
                # ``model.embed_tokens.modules_to_save.default.weight``. Inserting
                # after the *module* path, not appending to the parameter name.
                stem = name[: -len(".weight")] if name.endswith(".weight") else name
                cands = [f"{stem}.modules_to_save.default.weight", name]
                hit = next((c for c in cands if PREFIX + c in keys), None)
                if hit is None:
                    print(f"  WARNING: no key for {name}")
                    continue
                param.copy_(take(hit).to(param.dtype))
                n_copied += 1

            # add the LoRA deltas to q_proj / v_proj
            for i in range(model.config.num_hidden_layers):
                for proj in ("q_proj", "v_proj"):
                    stem = (f"model.layers.{i}.self_attn.{proj}")
                    ka = f"{PREFIX}{stem}.lora_A.default.weight"
                    kb = f"{PREFIX}{stem}.lora_B.default.weight"
                    if ka not in keys or kb not in keys:
                        continue
                    a = f.get_tensor(ka).float()          # [r, in]
                    b = f.get_tensor(kb).float()          # [out, r]
                    delta = (b @ a) * LORA_SCALE
                    mod = model.get_submodule(stem)
                    mod.weight.add_(delta.to(mod.weight.dtype))
                    n_lora += 1
        print(f"copied {n_copied} base tensors, applied {n_lora} LoRA deltas")

        out_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(out_dir), safe_serialization=True)
        tok.save_pretrained(str(out_dir))
        print(f"merged model saved -> {out_dir}")
        del model
        gc.collect()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orig-dir", type=Path, default=ORIG)
    ap.add_argument("--keep-intermediates", action="store_true")
    ap.add_argument("--skip-merge", action="store_true")
    ap.add_argument("--skip-convert", action="store_true")
    args = ap.parse_args()

    merged = args.orig_dir / "merged_bf16"
    gguf_dir = args.orig_dir / "gguf"
    gguf_dir.mkdir(parents=True, exist_ok=True)
    bf16_gguf = gguf_dir / "DuplexCascade-Original-bf16.gguf"
    q4 = gguf_dir / "DuplexCascade-Original-q4_k_m.gguf"

    if not args.skip_merge:
        merge(args.orig_dir, merged)

    if not args.skip_convert:
        print("converting to GGUF (bf16) ...", flush=True)
        subprocess.run([sys.executable, str(LLAMA_DIR / "convert_hf_to_gguf.py"),
                        str(merged), "--outfile", str(bf16_gguf),
                        "--outtype", "bf16"],
                       check=True, cwd=str(LLAMA_DIR))

        print("quantizing to q4_k_m ...", flush=True)
        subprocess.run([str(LLAMA_DIR / "build/bin/llama-quantize"),
                        str(bf16_gguf), str(q4), "q4_k_m", "8"], check=True)

        if not args.keep_intermediates:
            bf16_gguf.unlink(missing_ok=True)
            subprocess.run(["rm", "-rf", str(merged)], check=False)

    print(json.dumps({
        "gguf": str(q4),
        "exists": q4.is_file(),
        "size_gb": round(q4.stat().st_size / 1e9, 2) if q4.is_file() else None,
        "serving_hint": f"llama-server -m {q4} -sp",
    }, indent=2))


if __name__ == "__main__":
    main()
