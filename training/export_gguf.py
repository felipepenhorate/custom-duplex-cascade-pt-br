"""Convert the bf16 merged duplex model to a quantized GGUF for llama.cpp.

Uses the local llama.cpp toolchain directly (the same checkout that serves the
model): `convert_hf_to_gguf.py` (HF -> bf16 GGUF) then `llama-quantize`
(bf16 -> q4_k_m / q5_k_m / q8_0 / ...).

Note: the Qwen3-4B-Instruct-2507 tokenizer hash must be registered in
`llama.cpp/conversion/base.py` (get_vocab_base_pre) so the BPE pre-tokenizer is
recognized as `qwen2`; see the `4f53cda1...` entry added for DuplexCascade-PT.

Prereq: training/export_model.py --merged produced the bf16 merged model dir
(including tokenizer.json; the bf16 merge also needs `tok.save_pretrained`).

Usage:
  python training/export_gguf.py --merged /mnt/f/duplex_cascade_runs/full/export/merged_bf16 \\
      --llamacpp ~/llama.cpp --out /mnt/f/duplex_cascade_runs/full/export/gguf --quant q4_k_m
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

QUANTS = ["q4_k_m", "q5_k_m", "q8_0", "f16", "q4_0", "q5_0", "q6_k"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--merged", default="/mnt/f/duplex_cascade_runs/full/export/merged_bf16")
    p.add_argument("--llamacpp", default=str(Path.home() / "llama.cpp"))
    p.add_argument("--out", default="/mnt/f/duplex_cascade_runs/full/export/gguf")
    p.add_argument("--quant", default="q4_k_m", choices=QUANTS)
    p.add_argument("--outtype", default="bf16", choices=["bf16", "f16", "f32"])
    p.add_argument("--keep-bf16", action="store_true", help="also keep the bf16 GGUF")
    args = p.parse_args()

    merged = Path(args.merged)
    llcpp = Path(args.llamacpp)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    converter = llcpp / "convert_hf_to_gguf.py"
    quantizer = llcpp / "build/bin/llama-quantize"
    for tool in (converter, quantizer):
        if not tool.exists():
            print(f"[gguf] missing tool: {tool} (build llama-quantize: cmake --build {llcpp}/build --target llama-quantize)")
            sys.exit(1)

    tag = "DuplexCascade-PT"
    bf16_gguf = out_dir / f"{tag}-{args.outtype}.gguf"
    quant_gguf = out_dir / f"{tag}-{args.quant}.gguf"

    t0 = time.time()
    print(f"[gguf] [1/2] HF -> {args.outtype} GGUF")
    subprocess.run(
        [sys.executable, str(converter), str(merged),
         "--outfile", str(bf16_gguf), "--outtype", args.outtype],
        check=True,
    )
    if args.quant == "f16":
        # f16 requested: no second quantization step
        print(f"[gguf] done in {(time.time() - t0) / 60:.1f} min -> {bf16_gguf}")
        return

    print(f"[gguf] [2/2] {args.outtype} -> {args.quant}")
    subprocess.run(
        [str(quantizer), str(bf16_gguf), str(quant_gguf), args.quant],
        check=True,
    )
    if not args.keep_bf16:
        bf16_gguf.unlink(missing_ok=True)
    print(f"[gguf] done in {(time.time() - t0) / 60:.1f} min -> {quant_gguf}")


if __name__ == "__main__":
    main()