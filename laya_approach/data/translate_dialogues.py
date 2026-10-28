"""Translate the selected pilot dialogues to English with local Qwen3.5-9B.

Reads ``selected_dialogues.jsonl`` (id, scenario, messages, _pilot), translates
only the message contents, and writes the same schema with ``messages[*].content``
in English.  Resume-safe: already translated ids are skipped.

The 9B server is expected on :8099 with ``--jinja``.  Long dialogues are split
into consecutive windows so a request stays inside the 2048-token slots.

    python -m laya_approach.data.translate_dialogues \
        --in laya_approach/data/duplexcascade_pilot/selected_dialogues.jsonl \
        --out laya_approach/data/duplexcascade_pilot/dialogues_en.jsonl \
        --workers 4
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SYSTEM = (
    "You are a professional translator. Translate every message content from "
    "Brazilian Portuguese to natural conversational English. Keep the same "
    "roles, order and number of messages. Preserve names, numbers and the "
    "informal register. Return ONLY a JSON object of the form "
    '{"messages":[{"role":"user|assistant","content":"..."}]}. '
    "Do not add explanations, markdown or extra keys."
)


def _ask(messages: list[dict], api: str, max_tokens: int = 1200) -> str:
    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps({"messages": messages},
                                                   ensure_ascii=False)},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"{api}/chat/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as response:
        obj = json.loads(response.read())
    return ((obj.get("choices") or [{}])[0].get("message") or {}).get("content") or ""


def _parse(text: str) -> list[dict] | None:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    msgs = obj.get("messages")
    if not isinstance(msgs, list):
        return None
    out = []
    for m in msgs:
        if not isinstance(m, dict) or m.get("role") not in ("user", "assistant"):
            return None
        content = str(m.get("content") or "").strip()
        if not content:
            return None
        out.append({"role": m["role"], "content": content})
    return out


def _estimate(messages: list[dict]) -> int:
    return sum(max(8, len(m["content"]) // 3) for m in messages) + 200


def translate_one(
    record: dict,
    api: str,
    *,
    batch_messages: int = 6,
    batch_tokens: int = 1200,
    max_tokens: int = 1600,
) -> list[dict]:
    messages = [{"role": m["role"], "content": m["content"]}
                for m in record["messages"]]
    batches: list[list[dict]] = []
    current: list[dict] = []
    for m in messages:
        if current and (len(current) >= batch_messages
                        or _estimate(current + [m]) > batch_tokens):
            batches.append(current)
            current = []
        current.append(m)
    if current:
        batches.append(current)

    translated: list[dict] = []
    for batch in batches:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                raw = _ask(batch, api,
                           max_tokens=min(max_tokens, 300 + len(batch) * 90))
                parsed = _parse(raw)
                if parsed is not None and len(parsed) == len(batch):
                    translated.extend(parsed)
                    break
                last_error = ValueError(f"bad translation ({len(parsed or [])} vs {len(batch)})")
            except Exception as exc:  # noqa: BLE001
                last_error = exc
            time.sleep(1.0 + attempt)
        else:
            raise RuntimeError(f"translation failed for {record['id']}: {last_error}")
    return translated


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp",
                    default="laya_approach/data/duplexcascade_pilot/selected_dialogues.jsonl")
    ap.add_argument("--out", dest="out",
                    default="laya_approach/data/duplexcascade_pilot/dialogues_en.jsonl")
    ap.add_argument("--api", default="http://127.0.0.1:8099/v1")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--batch-messages", type=int, default=4,
                    help="max consecutive messages per translation request")
    ap.add_argument("--batch-tokens", type=int, default=700,
                    help="estimated prompt-token ceiling per request")
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    records = [json.loads(line) for line in Path(args.inp).open(encoding="utf-8")
               if line.strip()]
    if args.limit:
        records = records[: args.limit]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    done: set[str] = set()
    if out.is_file():
        for line in out.open(encoding="utf-8"):
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass
    todo = [r for r in records if r["id"] not in done]
    print(f"[translate] {len(records)} records, {len(done)} cached, {len(todo)} to do")

    failures = out.parent / "translation_failures.jsonl"
    started = time.time()
    n_done = 0
    with out.open("a", encoding="utf-8") as fh, failures.open("a", encoding="utf-8") as fh_err:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    translate_one, rec, args.api,
                    batch_messages=args.batch_messages,
                    batch_tokens=args.batch_tokens,
                    max_tokens=args.max_tokens,
                ): rec
                for rec in todo
            }
            for future in as_completed(futures):
                rec = futures[future]
                try:
                    messages = future.result()
                except Exception as exc:  # noqa: BLE001
                    fh_err.write(json.dumps({"id": rec["id"], "error": str(exc)},
                                            ensure_ascii=False) + "\n")
                    fh_err.flush()
                    print(f"[translate] FAIL {rec['id']}: {exc}", flush=True)
                    continue
                out_rec = {
                    "id": rec["id"],
                    "dedup_key": rec.get("dedup_key"),
                    "scenario": rec.get("scenario"),
                    "meta": rec.get("meta"),
                    "_pilot": rec.get("_pilot"),
                    "_translation": {
                        "model": "Qwen3.5-9B-Q4_K_M",
                        "prompt": "literal-v1",
                        "lang": "en",
                    },
                    "messages": [{"role": m["role"], "content": m["content"]}
                                 for m in messages],
                }
                fh.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                fh.flush()
                n_done += 1
                if n_done % 25 == 0 or n_done == 1:
                    rate = (time.time() - started) / n_done
                    remaining = rate * (len(todo) - n_done)
                    print(f"[translate] {n_done}/{len(todo)} "
                          f"({rate:.2f}s/item, eta {remaining/60:.1f} min)", flush=True)

    print(f"[translate] wrote {n_done} -> {out}")


if __name__ == "__main__":
    main()
