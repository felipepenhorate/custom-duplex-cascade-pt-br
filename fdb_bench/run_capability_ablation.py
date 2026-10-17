"""Ablate the anchoring strength lambda on capability retention.

The combined article's claim for self-distillation is that it keeps function
calling and instruction following at base level while SFT does not. That claim
rests on a single setting of the anchoring weight ``lambda``. This script varies
it and measures the two capabilities that regress under plain SFT.

Arms
----
``lambda=0`` is the plain-SFT baseline and ``lambda=1.0`` at 2000 steps is the
shipped model, both already evaluated in ``distill/logs/eval/``. The three
*sweep* adapters are the interesting ones: they were trained at a matched budget
(300 steps) at ``lambda`` = 0.1 / 0.3 / 1.0, so comparing them isolates the
anchoring weight from training length. The shipped 2000-step model is reported
alongside as the full-budget point.

Reuses ``distill/eval/eval_capabilities.py`` verbatim for prompts, decoding and
scoring (``fc_correct`` / ``general_correct`` / ``extract_json``) so the numbers
are comparable to the committed ones.

Run:  /home/penhfel/unsloth_uv/bin/python -m fdb_bench.run_capability_ablation
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

RUNS = Path("/mnt/f/duplex_cascade_runs")
BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"

# name -> adapter dir (LoRA) or merged path (full model)
SWEEP_ARMS = {
    "lambda_0.1": RUNS / "distill/sweep/lambda_0.1/adapter",
    "lambda_0.3": RUNS / "distill/sweep/lambda_0.3/adapter",
    "lambda_1.0_sweep": RUNS / "distill/sweep/lambda_1.0/adapter",
}
REFERENCE_ARMS = {
    "base": BASE_MODEL,
    "sft": RUNS / "continue/export/merged_bf16",
    "distill": RUNS / "distill/full/export/merged_bf16",
}


def _eval_module():
    """Import the distill capability eval to reuse its prompts and scoring."""
    spec_path = Path(__file__).resolve().parent.parent / "distill" / "eval" / "eval_capabilities.py"
    import importlib.util
    spec = importlib.util.spec_from_file_location("eval_capabilities", spec_path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"cannot import {spec_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["eval_capabilities"] = mod
    spec.loader.exec_module(mod)
    return mod


def load_model(path: Path, device: str = "cuda"):
    """Load a merged bf16 model, or base + LoRA adapter for an adapter dir."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if (path / "adapter_config.json").is_file():
        from peft import PeftModel
        tok = AutoTokenizer.from_pretrained(str(path), use_fast=True)
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, dtype=torch.bfloat16, device_map="cpu")
        # the adapter trained embed_tokens/lm_head for 7 added tokens
        base.resize_token_embeddings(len(tok))
        model = PeftModel.from_pretrained(base, str(path))
        model = model.merge_and_unload()
        model = model.to(device).eval()
        return model, tok
    tok = AutoTokenizer.from_pretrained(str(path), use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(path), dtype=torch.bfloat16, device_map=device)
    return model.eval(), tok


def run_arm(name: str, path: Path, ev, *, max_tokens: int, device: str) -> dict:
    import torch
    from pathlib import Path as _P

    fc_path = _P(__file__).resolve().parent.parent / "distill/data/eval_function_calling.jsonl"
    gen_path = _P(__file__).resolve().parent.parent / "distill/data/eval_general.jsonl"

    model, tok = load_model(path, device=device)
    t0 = time.time()

    fc_rows = [json.loads(l) for l in fc_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    fc_ok = 0
    fc_by_tool: dict[str, list[bool]] = {}
    for r in fc_rows:
        tool = r["tool"]
        msgs = [{"role": "system", "content": "Você é um assistente com acesso a ferramentas."},
                {"role": "user", "content": r["user"]}]
        prompt = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, tools=[tool],
            chat_template_kwargs={"enable_thinking": False})
        enc = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_tokens, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        call = ev.extract_json(text)
        ok = ev.fc_correct(call, tool["function"], r.get("expected", {}))
        fc_ok += ok
        fc_by_tool.setdefault(tool["function"]["name"], []).append(ok)

    gen_rows = [json.loads(l) for l in gen_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    gen_ok = 0
    for r in gen_rows:
        # the general set carries its own system prompt per row; using it (rather
        # than a fixed one) is what keeps these numbers comparable to the
        # committed distill/logs/eval/capabilities_*.json
        msgs = [{"role": "system", "content": r.get("system", "Você é um assistente útil.")},
                {"role": "user", "content": r["user"]}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                        chat_template_kwargs={"enable_thinking": False})
        enc = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_tokens, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        gen_ok += ev.general_correct(text, r)

    del model
    torch.cuda.empty_cache()

    return {
        "name": name, "model": str(path),
        "function_calling_accuracy": fc_ok / len(fc_rows),
        "general_accuracy": gen_ok / len(gen_rows),
        "n_function_calling": len(fc_rows), "n_general": len(gen_rows),
        "fc_by_tool": {k: sum(v) / len(v) for k, v in sorted(fc_by_tool.items())},
        "max_tokens": max_tokens, "wall_s": round(time.time() - t0, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path,
                    default=Path("results_ablation/capability"))
    ap.add_argument("--arms", default="sweep",
                    choices=["sweep", "sweep+refs"],
                    help="'sweep' runs the three lambda adapters; 'sweep+refs' "
                         "also re-runs base/sft/distill (slow)")
    ap.add_argument("--max-tokens", type=int, default=800,
                    help="800 matches the committed distill/logs/eval runs")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    ev = _eval_module()
    arms = dict(SWEEP_ARMS)
    if args.arms == "sweep+refs":
        arms = {**REFERENCE_ARMS, **arms}

    args.out.mkdir(parents=True, exist_ok=True)
    for name, path in arms.items():
        print(f"=== {name} ({path}) ===", flush=True)
        try:
            res = run_arm(name, Path(path), ev, max_tokens=args.max_tokens,
                          device=args.device)
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED: {type(e).__name__}: {e}")
            continue
        (args.out / f"{name}.json").write_text(
            json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  FC {res['function_calling_accuracy']:.3f}  "
              f"general {res['general_accuracy']:.3f}  "
              f"({res['wall_s']}s)", flush=True)


if __name__ == "__main__":
    main()
