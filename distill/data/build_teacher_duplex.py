"""M1 — teacher (SDFT) duplex content generation + duplex post-processing.

Stage 1 of the DuplexCascade-Distill data pipeline (SPEC §4.2 / §6 M1).

For each source dialogue (user turns from the existing validated pt-BR corpus,
e.g. duplex_cascade/data/dialogues_pt.jsonl) the FROZEN base model — prompted
with the duplex-cascade trigger system message (data/teacher_prompts.py) —
generates every assistant turn, conditioned on the history built so far. The
teacher's generations will not faithfully follow the micro-turn protocol (the
base model was never fine-tuned on it); that is expected. We keep only the
teacher's *content*; the deterministic duplex builder
(data/build_duplex_dataset.py) places the special tags afterwards, producing the
aligned "demonstration" sequence that M2 will distill the tags from and anchor
the content distribution to.

Two generation backends:
  * llama.cpp OpenAI-compatible API (--api-base)   — parallel, cheap, quantized
  * in-process transformers forward (--in-process) — exact base model, serial

Output JSONL, one record per dialogue:
  {id, src_id, scenario, messages (teacher-rewritten), teacher{...}, items
   (duplex micro-turn items, schema of build_duplex_dataset.py), meta}

Usage:
  python data/build_teacher_duplex.py --dialogues <src.jsonl> \\
      --out data/teacher_duplex_train.jsonl --api-base http://127.0.0.1:8080/v1 \\
      --workers 4 [--limit 2000]
  python data/build_teacher_duplex.py --dialogues <src.jsonl> \\
      --out data/teacher_duplex_train.jsonl --in-process \\
      --model Qwen/Qwen3-4B-Instruct-2507 [--limit 2000]
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import requests
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.build_duplex_dataset import DuplexChunker, build_duplex_items  # noqa: E402
from data.teacher_prompts import TRIGGER_VERSION, build_teacher_messages, strip_duplex_tags  # noqa: E402

ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"

_WS_SPLIT = re.compile(r"\s+")
_ROLE_PREFIX = re.compile(r"^\s*(?:Assistente|Usu[aá]rio|Assistant|User)\s*:\s*")


def clean_response(text: str) -> str:
    """Normalize a teacher generation into bare assistant content."""
    text = strip_duplex_tags(text or "")
    text = _ROLE_PREFIX.sub("", text)
    return _WS_SPLIT.sub(" ", text).strip()


def validate_response(text: str) -> bool:
    if not text:
        return False
    w = len(text.split())
    return 1 <= w <= 180


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
class LlamaCppClient:
    """Thin OpenAI-compatible client (mirrors duplex_cascade build_dialogues)."""

    def __init__(self, api_base: str, model: str | None = None, timeout: int = 600):
        self.api_base = api_base.rstrip("/")
        self.timeout = timeout
        self.model = model or self._detect_model()

    def _detect_model(self) -> str:
        r = requests.get(f"{self.api_base}/models", timeout=30)
        r.raise_for_status()
        ids = [m["id"] for m in r.json().get("data", [])]
        if not ids:
            raise RuntimeError(f"no model served at {self.api_base}/models")
        if len(ids) > 1:
            print(f"[teacher] multiple models served, using first: {ids[0]}")
        return ids[0]

    def complete(
        self,
        messages: list[dict],
        temperature: float,
        top_p: float,
        max_tokens: int,
        max_retries: int = 3,
    ) -> str:
        payload = {
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        last_err: Exception | None = None
        for attempt in range(max_retries):
            try:
                r = requests.post(
                    f"{self.api_base}/chat/completions",
                    json=payload,
                    timeout=self.timeout,
                )
                if r.status_code in (429, 503):  # slot busy: backoff + retry
                    time.sleep(1 + attempt * 2)
                    continue
                r.raise_for_status()
                msg = r.json()["choices"][0]["message"]
                return msg.get("content") or ""
            except requests.RequestException as e:
                last_err = e
                time.sleep(1 + attempt * 2)
        raise RuntimeError(f"llama.cpp API failed after {max_retries} tries: {last_err}")


class InProcessTeacher:
    """Frozen base model generating in-process (exact bf16 teacher).

    Serial on purpose: a single 16 GB 4080 cannot serve this alongside training,
    and GPU generation does not parallelize across threads meaningfully.
    """

    def __init__(self, model_name: str, max_new_tokens: int):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    def complete(
        self,
        messages: list[dict],
        temperature: float,
        top_p: float,
        max_tokens: int,
        max_retries: int = 1,
    ) -> str:
        tok = self.tokenizer
        prompt = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": False},
        )
        inputs = tok(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        gen_kwargs = dict(
            max_new_tokens=max_tokens or self.max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tok.eos_token_id,
            use_cache=True,
        )
        with self.torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return tok.decode(new_tokens, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Teacher rewriting
# ---------------------------------------------------------------------------
def rewrite_dialogue(
    dialogue: dict,
    backend,
    rng: random.Random,
    *,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    allow_tags: bool,
    max_retries: int,
    fallback_source: bool,
) -> dict:
    """Regenerate every assistant turn with the teacher; keep user turns."""
    src_messages = dialogue["messages"]
    history: list[dict] = []
    rewrites = 0
    fallbacks = 0
    for m in src_messages:
        if m["role"] == ROLE_USER:
            history.append(m)
            continue
        # assistant turn -> teacher generation conditioned on history
        response: str | None = None
        for attempt in range(max_retries + 1):
            try:
                text = backend.complete(
                    build_teacher_messages(history, allow_tags=allow_tags),
                    temperature,
                    top_p,
                    max_new_tokens,
                )
                cleaned = clean_response(text)
                if validate_response(cleaned):
                    response = cleaned
                    break
            except Exception:
                if attempt >= max_retries:
                    raise
        if response is None and fallback_source:
            response = _WS_SPLIT.sub(" ", m["content"]).strip()
            fallbacks += 1
        if response is None:
            return None
        rewrites += 1
        history.append({"role": ROLE_ASSISTANT, "content": response})

    return {
        "messages": history,
        "rewrites": rewrites,
        "fallbacks": fallbacks,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dialogues", default="data/dialogues_pt.jsonl",
                   help="source pt-BR dialogues (user turns + original assistant text)")
    p.add_argument("--out", default="data/teacher_duplex_train.jsonl")
    p.add_argument("--api-base", default=None,
                   help="llama.cpp OpenAI-compatible endpoint; mutually exclusive with --in-process")
    p.add_argument("--in-process", action="store_true",
                   help="generate with the exact bf16 base model in-process")
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507",
                   help="base model (--in-process load / API model id)")
    p.add_argument("--tokenizer", default="Qwen/Qwen3-4B-Instruct-2507",
                   help="tokenizer for the duplex chunker")
    p.add_argument("--workers", type=int, default=4,
                   help="concurrent dialogues (API mode only)")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-new-tokens", type=int, default=160)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--allow-tags", action="store_true",
                   help="let the teacher attempt the full tag protocol (ICL ablation, SPEC §8)")
    p.add_argument("--no-fallback-source", action="store_true",
                   help="do NOT fall back to the source assistant text when generation fails")
    p.add_argument("--limit", type=int, default=None, help="cap source dialogues")
    # duplex builder passthrough (see build_duplex_dataset.py)
    p.add_argument("--system-chunk-min", type=int, default=10)
    p.add_argument("--system-chunk-max", type=int, default=48)
    p.add_argument("--pause-prob", type=float, default=0.10)
    p.add_argument("--pause-max", type=int, default=5)
    p.add_argument("--interrupt-prob", type=float, default=0.30)
    p.add_argument("--user-bc-prob", type=float, default=0.01)
    p.add_argument("--system-bc-prob", type=float, default=0.20)
    p.add_argument("--thinking-max", type=int, default=20)
    p.add_argument("--with-system-backchannel", action="store_true")
    args = p.parse_args()

    if bool(args.api_base) == bool(args.in_process):
        p.error("choose exactly one backend: --api-base <url> or --in-process")

    if args.in_process:
        backend = InProcessTeacher(args.model, args.max_new_tokens)
        model_name = args.model
        generator = "in-process-bf16"
        api_base = None
    else:
        backend = LlamaCppClient(args.api_base, args.model)
        model_name = backend.model
        generator = "llama.cpp-openai"
        api_base = backend.api_base
    print(f"[teacher] backend: {generator} ({model_name})", flush=True)

    # tokenizer + duplex chunker (always in-process, small)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    chunker = DuplexChunker(
        tokenizer,
        system_chunk_len=args.system_chunk_min,
        user_chunk_max=7,
        system_chunk_max=args.system_chunk_max,
    )
    rng = random.Random(args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # resume: skip source ids already processed
    written_ids: set[str] = set()
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    written_ids.add(json.loads(line)["src_id"])
                except Exception:
                    pass
    print(f"[teacher] resuming with {len(written_ids)} dialogues already done", flush=True)

    with out_path.open("a", encoding="utf-8") as fout:
        with Path(args.dialogues).open("r", encoding="utf-8") as fin:
            stats: Counter[str] = Counter()
            lock = threading.Lock()

            def process(dialogue: dict) -> dict | None:
                out = rewrite_dialogue(
                    dialogue,
                    backend,
                    rng,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                    allow_tags=args.allow_tags,
                    max_retries=args.max_retries,
                    fallback_source=not args.no_fallback_source,
                )
                if out is None:
                    return None
                items = build_duplex_items(
                    {"messages": out["messages"]},
                    rng,
                    chunker,
                    pause_prob=args.pause_prob,
                    pause_max=args.pause_max,
                    interrupt_prob=args.interrupt_prob,
                    user_bc_prob=args.user_bc_prob,
                    with_system_bc=args.with_system_backchannel,
                    system_bc_prob=args.system_bc_prob,
                    thinking_max=args.thinking_max,
                )
                if not items:
                    return None
                # invariant: strictly alternating roles
                prev = None
                for it in items:
                    if it["role"] == prev:
                        return None
                    prev = it["role"]
                rec = {
                    "id": f"distill-{dialogue.get('id', 'dlg')}",
                    "src_id": dialogue.get("id", ""),
                    "scenario": dialogue.get("scenario", {}),
                    "messages": out["messages"],
                    "teacher": {
                        "model": model_name,
                        "generator": generator,
                        "api_base": api_base,
                        "trigger_version": TRIGGER_VERSION,
                        "allow_tags": args.allow_tags,
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                    },
                    "items": items,
                    "meta": {
                        "variant": "beta" if args.with_system_backchannel else "base",
                        "fallbacks": out["fallbacks"],
                        "rewrites": out["rewrites"],
                    },
                }
                return rec

            def emit(dialogue: dict, rec: dict | None) -> None:
                with lock:
                    src_id = dialogue.get("id", "")
                    if rec is None:
                        stats["rejected"] += 1
                        return
                    for it in rec["items"]:
                        if it["role"] == ROLE_ASSISTANT and it.get("special"):
                            stats[it["special"]] += 1
                    stats["items"] += len(rec["items"])
                    stats["fallbacks"] += rec["meta"]["fallbacks"]
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fout.flush()
                    stats["accepted"] += 1
                    written_ids.add(src_id)

            # iterate source dialogues, skipping already-written ones
            def source_iter():
                n = 0
                with Path(args.dialogues).open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        rec = json.loads(line)
                        if rec.get("id", "") in written_ids:
                            continue
                        if args.limit and n >= args.limit:
                            break
                        n += 1
                        yield rec

            start = time.time()
            if args.in_process:
                for dialogue in source_iter():
                    emit(dialogue, process(dialogue))
            else:
                inflight: dict = {}
                attempts = 0
                srcs = list(source_iter())

                def launch(dlg) -> None:
                    nonlocal attempts
                    inflight[ex.submit(process, dlg)] = dlg
                    attempts += 1

                with ThreadPoolExecutor(max_workers=args.workers) as ex:
                    with tqdm(total=len(srcs), desc="teacher", unit="d") as bar:
                        for dlg in srcs:
                            while len(inflight) >= args.workers:
                                done, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
                                for fut in done:
                                    d = inflight.pop(fut)
                                    try:
                                        emit(d, fut.result())
                                    except Exception as e:
                                        print(f"[teacher] worker error: {e}", flush=True)
                            launch(dlg)
                        while inflight:
                            done, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
                            for fut in done:
                                d = inflight.pop(fut)
                                try:
                                    emit(d, fut.result())
                                except Exception as e:
                                    print(f"[teacher] worker error: {e}", flush=True)
                        bar.n = stats["accepted"]
                        bar.refresh()

            elapsed = time.time() - start
            print(
                f"[teacher] done: {stats['accepted']} accepted, {stats['rejected']} rejected "
                f"in {elapsed/60:.1f} min ({stats['accepted']/max(elapsed,1e-9)*60:.1f} dlg/min) "
                f"-> {out_path}",
                flush=True,
            )
            print("[teacher] assistant special-token supervision counts (per token):")
            total = stats["items"]
            for k in ["user_is_speaking", "user_finish_speaking", "user_interruption",
                      "user_backchannel", "user_is_thinking", "system_backchannel"]:
                v = stats.get(k, 0)
                frac = f" ({v / max(total, 1):6.1%})"
                print(f"   {k:28s} {v:8d}{frac}")
            print(f"   total items: {total}, source-fallback assistant turns: {stats['fallbacks']}")


if __name__ == "__main__":
    main()