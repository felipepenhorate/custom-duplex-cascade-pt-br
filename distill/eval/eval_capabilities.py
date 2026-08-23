"""M3 — capability-retention eval for DuplexCascade-Distill (SPEC §6 M3).

Runs a bf16 model over:
  * a pt-BR function-calling set (native Qwen tool-calling via the chat
    template) — the reported regression; scored by tool-name + argument match.
  * a small pt-BR general set (QA + instruction-following), scored heuristically.

Designed to compare three models with identical prompts:
  * base   — Qwen/Qwen3-4B-Instruct-2507 (the capability reference)
  * distill— this project's merged model (distill/full/export/merged_bf16)
  * sft    — DuplexCascade-PT SFT baseline (continue/export/merged_bf16)

The key result: does DISTILL keep base-level function-calling accuracy where the
plain SFT model regresses?

Usage:
  python eval/eval_capabilities.py --model <path> --name base \\
      [--function-calling data/eval_function_calling.jsonl] \\
      [--general data/eval_general.jsonl] [--max-tokens 200]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def norm(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", s).strip().lower()


def extract_json(text: str) -> dict | None:
    """Pull the first balanced JSON object (with a 'name' key) out of a response."""
    m = re.search(r"<tool_call>(.*?)</tool_call>", text, re.S)
    if m:
        text = m.group(1)
    i, n = 0, len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth, j = 0, i
        while j < n:
            c = text[j]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[i : j + 1])
                    except Exception:
                        break
                    if isinstance(obj, dict) and obj.get("name"):
                        return obj
                    break
            j += 1
        i = j + 1
    return None


def _norm_val(v):
    """Normalize an argument value for comparison. Dates/times compare by
    digits only (2026-09-10 vs 2026/09/10; 15:00 vs 15h -> "1500")."""
    if isinstance(v, (int, float)):
        return str(v)
    s = str(v)
    if re.search(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", s) or re.match(r"\d{1,2}\s*[:h]", s):
        return re.sub(r"\D", "", s)
    return norm(s)


def fc_correct(call: dict | None, tool_fn: dict, expected: dict) -> bool:
    if call is None:
        return False
    if norm(call.get("name") or "") != norm(expected["name"]):
        return False
    args = call.get("arguments")
    if not isinstance(args, dict):
        return False
    required = tool_fn.get("parameters", {}).get("required", [])
    if not all(r in args for r in required):
        return False
    for k, v in expected["arguments"].items():
        if v == "":
            continue  # unconstrained (e.g. CEP not given in the request)
        if k not in args:
            return False
        nv, nmv = _norm_val(v), _norm_val(args[k])
        if isinstance(v, (int, float)):
            try:
                if float(nmv) != float(v):
                    return False
            except (TypeError, ValueError):
                return False
        else:
            if nv not in nmv and nmv not in nv:
                return False
    return True


def general_correct(text: str, rec: dict) -> bool:
    expected = rec.get("expected", "")
    mode = rec.get("match", "contains")
    t = norm(text)
    if mode == "exact_norm":
        return t == norm(expected)
    if mode == "count3":
        items = [x for x in re.split(r"[,;\n•-]| e ", text) if norm(x)]
        return len(items) == 3
    # contains / contains_norm
    n = norm(expected)
    return n in t if n else False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--name", default="model")
    p.add_argument("--function-calling", default="data/eval_function_calling.jsonl")
    p.add_argument("--general", default="data/eval_general.jsonl")
    p.add_argument("--max-tokens", type=int, default=800)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    print(f"[{args.name}] loaded {args.model} (bf16)", file=sys.stderr)

    # ---------------- function calling ----------------
    fc_rows = [json.loads(l) for l in Path(args.function_calling).open() if l.strip()]
    if args.limit:
        fc_rows = fc_rows[: args.limit]
    fc_correct_n = 0
    fc_by_name: dict[str, list[bool]] = {}
    fc_failures: list[tuple[str, str]] = []
    for r in fc_rows:
        tool = r["tool"]
        messages = [
            {"role": "system", "content": "Você é um assistente com acesso a ferramentas."},
            {"role": "user", "content": r["user"]},
        ]
        prompt = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            tools=[tool],
            chat_template_kwargs={"enable_thinking": False},
        )
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=args.max_tokens, do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
        call = extract_json(gen)
        ok = fc_correct(call, tool["function"], r["expected"])
        fc_correct_n += int(ok)
        fc_by_name.setdefault(tool["function"]["name"], []).append(ok)
        if not ok:
            fc_failures.append((r["id"], gen.strip()[:120].replace("\n", " ")))
    fc_acc = fc_correct_n / len(fc_rows)

    # ---------------- general ----------------
    gen_rows = [json.loads(l) for l in Path(args.general).open() if l.strip()]
    gen_correct_n = 0
    gen_failures: list[tuple[str, str]] = []
    for r in gen_rows:
        messages = [
            {"role": "system", "content": r.get("system", "Você é um assistente útil.")},
            {"role": "user", "content": r["user"]},
        ]
        prompt = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": False},
        )
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=args.max_tokens, do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        ok = general_correct(gen, r)
        gen_correct_n += int(ok)
        if not ok:
            gen_failures.append((r["id"], gen.strip()[:120].replace("\n", " ")))

    gen_acc = gen_correct_n / len(gen_rows)

    summary = {
        "name": args.name,
        "model": args.model,
        "function_calling_accuracy": round(fc_acc, 4),
        "function_calling_by_tool": {k: round(sum(v) / len(v), 3) for k, v in fc_by_name.items()},
        "general_accuracy": round(gen_acc, 4),
        "n_function_calling": len(fc_rows),
        "n_general": len(gen_rows),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if fc_failures:
        print(f"[{args.name}] FC failures ({len(fc_failures)}):", file=sys.stderr)
        for fid, g in fc_failures[:6]:
            print(f"   {fid}: {g!r}", file=sys.stderr)
    if gen_failures:
        print(f"[{args.name}] GEN failures ({len(gen_failures)}):", file=sys.stderr)
        for fid, g in gen_failures[:6]:
            print(f"   {fid}: {g!r}", file=sys.stderr)
    print(f"[{args.name}] done in {(time.time() - t0) / 60:.1f} min", file=sys.stderr)


if __name__ == "__main__":
    main()