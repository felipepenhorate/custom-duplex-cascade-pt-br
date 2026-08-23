"""M1-long - pt-BR dialogue generator with LONGER assistant turns.

Same pipeline as data/build_dialogues.py (Gemma 4 E4B served through llama.cpp
/ OpenAI-compatible API), but the generation prompts ask for a MIX of response
lengths: most assistant turns stay short and colloquial, while a fraction are
longer, multi-sentence, didactic replies. The fine-tuned model currently only
ever answers in 1-2 short sentences because the M1 data forced short replies
("respostas curtas do Assistente. Nada de textos longos"); adding longer
examples lets the model learn to answer with both short and long utterances.

Usage:
  python data/build_long_dialogues.py --n-dialogues 5000 \
      --api-base http://127.0.0.1:8081/v1 \
      --out data/dialogues_long_pt.jsonl

  # ~20% of dialogues are "long-heavy" (assistant gives 2-4 sentence answers),
  # the rest keep the original short style so both behaviors are preserved.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path

import requests
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.build_dialogues import SCENARIOS, LlamaCppClient, parse_dialogue
from data.common import is_portuguese, sha256_dedup_key

# ---------------------------------------------------------------------------
# Generation styles. The short style mirrors the original M1 hint; the long
# style explicitly asks for detailed, multi-sentence assistant replies.
# ---------------------------------------------------------------------------
STYLE_SHORT = (
    "Use português brasileiro coloquial, com naturalidade de uma conversa real: "
    "frases curtas, gírias leves, hesitações ocasionais e respostas curtas do "
    "Assistente. Nada de textos longos, listas ou cabeçalhos."
)

STYLE_LONG = (
    "Use português brasileiro coloquial e natural. As falas do Usuário são "
    "curtas e informais, mas o Assistente responde de forma mais completa: "
    "duas a quatro frases por resposta, explicando com calma, dando exemplos "
    "e detalhes úteis, sempre em tom de conversa (nada de listas ou "
    "cabeçalhos)."
)

STYLE_MIXED = (
    "Use português brasileiro coloquial e natural. As falas do Usuário são "
    "curtas e informais. O Assistente varia o tamanho das respostas: algumas "
    "curtas e diretas, outras mais longas, com duas a quatro frases dando "
    "explicações e exemplos."
)

_FORMAT_RULES = (
    "Cada fala deve ocupar exatamente uma linha, começando com 'Usuário:' ou "
    "'Assistente:', nesse formato nítido e alternado. Nada além das falas. "
    "Nenhum conteúdo fora desse formato."
)

PROMPT_SYSTEM = (
    "Você é um roteirista de diálogos em português brasileiro. Você cria "
    "conversas realistas de chat entre um Usuário e um Assistente. "
    "Responda direto, sem raciocinar em voz alta."
)

PROMPT_USER = (
    "Crie uma conversa de chat em português brasileiro com exatamente {n_turns} "
    "falas ({n_usr} do Usuário e {n_asst} do Assistente) sobre este assunto: {topic}.\n"
    "Contexto da conversa: {context}.\n"
    "{dialect}\n{format}"
)

# list of (style_name, dialect_hint, weight) - higher weight = picked more often
STYLES: list[tuple[str, str, int]] = [
    ("short", STYLE_SHORT, 6),
    ("long", STYLE_LONG, 2),
    ("mixed", STYLE_MIXED, 2),
]


def build_prompt(topic: str, context: str, n_turns: int, rng: random.Random) -> list[dict]:
    style, dialect, _w = rng.choices(STYLES, weights=[w for _, _, w in STYLES])[0]
    n_usr = (n_turns + 1) // 2
    n_asst = n_turns - n_usr
    user_msg = PROMPT_USER.format(
        n_turns=n_turns,
        n_usr=n_usr,
        n_asst=n_asst,
        topic=topic,
        context=context,
        dialect=dialect,
        format=_FORMAT_RULES,
    )
    return [{"role": "system", "content": PROMPT_SYSTEM}, {"role": "user", "content": user_msg}]


def validate_dialogue(lines: list[dict]) -> bool:
    if not lines:
        return False
    if lines[-1]["role"] != "assistant":
        return False
    joined_user = " ".join(x["content"] for x in lines if x["role"] == "user")
    joined_asst = " ".join(x["content"] for x in lines if x["role"] == "assistant")
    for text in (joined_user, joined_asst):
        if not is_portuguese(text):
            return False
    # allow longer assistant turns (up to 220 words) than the short-style
    # pipeline (180), so genuine multi-sentence replies survive validation
    for line in lines:
        w = len(line["content"].split())
        if w < 1 or w > 220:
            return False
    counts: dict[str, int] = {}
    for line in lines:
        key = line["content"].lower()
        counts[key] = counts.get(key, 0) + 1
        if counts[key] >= 3:
            return False
    return True


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--api-base", default="http://127.0.0.1:8081/v1",
                   help="Gemma llama.cpp OpenAI-compatible endpoint")
    p.add_argument("--model", default=None,
                   help="served model id (default: first model from /v1/models)")
    p.add_argument("--n-dialogues", type=int, default=5000)
    p.add_argument("--workers", type=int, default=4,
                   help="concurrent requests (needs --parallel slots on the server)")
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-new-tokens", type=int, default=1600,
                   help="more budget than the short generator: longer replies")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--out", default="data/dialogues_long_pt.jsonl")
    p.add_argument("--max-retries", type=int, default=3, help="retries per row on validation failure")
    args = p.parse_args()

    rng = random.Random(args.seed)
    client = LlamaCppClient(args.api_base, args.model)
    print(f"[long] served model: {client.model}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    seen_keys: set[str] = set()
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                written += 1
                try:
                    seen_keys.add(json.loads(line)["dedup_key"])
                except Exception:
                    pass
    needed = max(0, args.n_dialogues - written)
    print(f"[long] resuming with {written} done; generating {needed} more", flush=True)

    def sample_meta() -> dict:
        topic, context = rng.choice(SCENARIOS)
        return {"topic": topic, "context": context, "n_turns": rng.randint(6, 16)}

    def generate_one(meta: dict) -> dict | None:
        for attempt in range(args.max_retries + 1):
            prompt = build_prompt(meta["topic"], meta["context"], meta["n_turns"], rng)
            text = client.complete(
                prompt, args.temperature, args.top_p, args.max_new_tokens
            )
            lines = parse_dialogue(text)
            if lines is not None and validate_dialogue(lines):
                return lines
            time.sleep(0.5 * attempt)
        return None

    stats: Counter[str] = Counter()
    lock = threading.Lock()
    accepted = 0
    rejected = 0

    def emit(rec: dict | None, meta: dict) -> None:
        nonlocal accepted, rejected
        with lock:
            if rec is None:
                stats["rejected_raw"] += 1
                rejected += 1
                return
            key = sha256_dedup_key(" ".join(x["content"] for x in rec))
            if key in seen_keys:
                stats["rejected_dup"] += 1
                rejected += 1
                return
            seen_keys.add(key)
            accepted += 1
            asst_words = [len(x["content"].split()) for x in rec if x["role"] == "assistant"]
            out = {
                "id": f"pt-long-{accepted:07d}",
                "dedup_key": key,
                "scenario": {"topic": meta["topic"], "context": meta["context"]},
                "messages": rec,
                "meta": {
                    "model": client.model,
                    "generator": "llama.cpp-openai",
                    "api_base": client.api_base,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "n_turns": len(rec),
                    "max_assistant_words": max(asst_words) if asst_words else 0,
                },
            }
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(out, ensure_ascii=False) + "\n")

    start = time.time()
    inflight: dict[Future, dict] = {}
    attempts = 0

    def launch() -> None:
        nonlocal attempts
        meta = sample_meta()
        inflight[ex.submit(generate_one, meta)] = meta
        attempts += 1

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        with tqdm(total=needed, desc="long-dialogues", unit="d") as bar:
            exhausted = False
            while accepted < needed and not exhausted:
                while len(inflight) < args.workers:
                    launch()
                done, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
                for fut in done:
                    meta = inflight.pop(fut)
                    rec = None
                    try:
                        rec = fut.result()
                    except Exception as e:
                        rec = None
                        print(f"[long] worker error: {e}", flush=True)
                    if rec is None:
                        if attempts < needed * 20:
                            launch()
                        else:
                            exhausted = True
                            print(f"[long] attempt cap hit at {accepted}/{needed}", flush=True)
                    else:
                        emit(rec, meta)
                bar.n = accepted
                bar.refresh()

    elapsed = time.time() - start
    print(
        f"[long] done: {accepted} accepted, {rejected} rejected in {elapsed/60:.1f} min "
        f"({accepted/max(elapsed, 1e-9)*60:.1f} dialogues/min) -> {out_path}",
        flush=True,
    )
    print(f"[long] breakdown: {dict(stats)}")


if __name__ == "__main__":
    main()