"""Phase 2: score the runs with the local judge GGUF.

Two scopes:

``--scope interruption``
    FDB v1.0's design: the judge is consulted **only** for the user-interruption
    dimension, using the paper's verbatim rubric. That is the column comparable
    to the published "GPT-4o" one.

``--scope all`` (default)
    Additionally scores **every scenario in which the model spoke**, on a shared
    0-5 relevance rubric, so quality is measured over the whole dataset rather
    than 200 of 727 scenarios. The interruption column keeps FDB's rubric.

Scenarios where the model correctly said nothing are not sent to the judge —
their score follows from the task, not from an opinion:

    pause_* / backchannel_icc   silence is correct      -> 5
    smooth_turn / interruption  silence is a failure    -> 0

That is cheaper and less noisy than asking a model to grade an empty string.

Outputs ``judged.jsonl`` (one record per interruption scenario, FDB column) and
``quality.jsonl`` (one record per scored scenario) plus ``judge_meta.json``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .adapters import LlamaClient
from .judge import judge_one, judge_quality

DEFAULT_JUDGE_GGUF = Path(
    "/mnt/f/GGUF/Qwen_3.5-9B/Qwen3.5-9B-UD-Q4_K_XL.gguf"
)

# tasks where the model is *supposed* to stay quiet
SILENCE_IS_CORRECT = ("pause_synthetic", "pause_candor", "backchannel_icc")


def _speech(row: dict) -> str:
    """Everything the model said in this scenario, as one string."""
    return " ".join(m["text"] for m in row["model_turns"] if m["text"].strip()).strip()


def _user_turn(row: dict) -> str:
    """The user's side of the exchange, for the relevance rubric."""
    parts = [e["text"] for e in row["user_events"]
             if e["kind"] == "chunk" and e["text"].strip()
             and e["text"].strip() != "<|no voice|>"]
    if row.get("context"):
        return f"{row['context']} || {row.get('interrupt','')}".strip(" |")
    return " ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, type=Path,
                    help="a run_inference.py --out directory")
    ap.add_argument("--api-base", default="http://127.0.0.1:8099/v1",
                    help="llama-server serving the judge GGUF")
    ap.add_argument("--lang", default="pt", choices=["pt", "en"])
    ap.add_argument("--scope", default="all", choices=["all", "interruption"])
    ap.add_argument("--judge-gguf", type=Path, default=DEFAULT_JUDGE_GGUF)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=320)
    args = ap.parse_args()

    raw_path = args.results / "raw.jsonl"
    if not raw_path.is_file():
        raise SystemExit(f"no raw.jsonl in {args.results}")
    rows = [json.loads(l) for l in raw_path.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    interr = [r for r in rows if r["task"] == "interruption_synthetic"]
    if args.limit:
        interr = interr[: args.limit]

    meta = {}
    mp = args.results / "meta.json"
    if mp.is_file():
        meta = json.loads(mp.read_text())
    print(f"chunker that produced this run: {meta.get('chunker', '<unrecorded>')}")
    print(f"judge GGUF: {args.judge_gguf}")
    print(f"judge API:  {args.api_base}   scope: {args.scope}")

    missing = [r["sid"] for r in interr
               if not (r.get("context") or r.get("interrupt"))]
    if missing:
        raise SystemExit(
            f"{len(missing)} rows carry no context/interrupt text (e.g. {missing[0]}).\n"
            "Those rows predate the fix that stores the judge inputs on each row;\n"
            "re-run inference for this model so the judge scores the right scenario."
        )

    client = LlamaClient(args.api_base, timeout=180.0)
    t0 = time.time()

    # ---- 1. FDB's interruption column (verbatim rubric) -------------------
    n_rated = n_skipped = n_failed = 0
    with (args.results / "judged.jsonl").open("w", encoding="utf-8") as fh:
        for r in interr:
            rec = {"sid": r["sid"], "model": r["model"], "lang": args.lang,
                   "rated": False, "rating": None, "rubric": "fdb_interruption",
                   "context": r.get("context", ""), "interrupt": r.get("interrupt", "")}
            if not (r.get("took_floor_after_interrupt")
                    and (r.get("post_interrupt_text") or "").strip()):
                rec["skipped"] = "model did not take the floor after the barge-in"
                n_skipped += 1
            else:
                try:
                    v = judge_one(client, context=rec["context"],
                                  interrupt=rec["interrupt"],
                                  response=r["post_interrupt_text"],
                                  lang=args.lang, max_tokens=args.max_tokens)
                    rec.update({"rating": v.get("rating"), "analysis": v.get("analysis"),
                                "parsed": v.get("parsed"), "raw": v.get("raw"),
                                "response": r["post_interrupt_text"][:600],
                                "rated": v.get("rating") is not None})
                    n_rated += rec["rated"]
                    n_failed += not rec["rated"]
                except Exception as e:  # noqa: BLE001
                    rec["error"] = f"{type(e).__name__}: {e}"
                    n_failed += 1
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
    print(f"interruption: {len(interr)} rows, {n_rated} rated, "
          f"{n_skipped} skipped, {n_failed} failed "
          f"({(time.time()-t0)/max(1,len(interr)):.2f}s/item)", flush=True)

    # ---- 2. whole-dataset quality (all tasks) ----------------------------
    quality_summary = {"n": 0, "judged": 0, "silence_correct": 0, "silence_failed": 0,
                       "judge_failed": 0}
    if args.scope == "all":
        t1 = time.time()
        with (args.results / "quality.jsonl").open("w", encoding="utf-8") as fh:
            for r in rows:
                if r["task"] == "interruption_synthetic" and args.limit:
                    pass
                speech = _speech(r)
                rec = {"sid": r["sid"], "task": r["task"], "model": r["model"],
                       "lang": args.lang, "rubric": "quality_relevance",
                       "spoke": bool(speech)}
                if not speech:
                    # no speech: the score is determined by the task, not a judge
                    if r["task"] in SILENCE_IS_CORRECT:
                        rec.update({"rating": 5, "rated": True,
                                    "source": "rule:silence_correct"})
                        quality_summary["silence_correct"] += 1
                    else:
                        rec.update({"rating": 0, "rated": True,
                                    "source": "rule:silence_failed"})
                        quality_summary["silence_failed"] += 1
                else:
                    try:
                        v = judge_quality(client, user_turn=_user_turn(r),
                                          response=speech, lang=args.lang,
                                          max_tokens=args.max_tokens)
                        rec.update({"rating": v.get("rating"),
                                    "analysis": v.get("analysis"),
                                    "parsed": v.get("parsed"), "raw": v.get("raw"),
                                    "response": speech[:600],
                                    "rated": v.get("rating") is not None,
                                    "source": "judge"})
                        quality_summary["judged"] += rec["rated"]
                        quality_summary["judge_failed"] += not rec["rated"]
                    except Exception as e:  # noqa: BLE001
                        rec["error"] = f"{type(e).__name__}: {e}"
                        quality_summary["judge_failed"] += 1
                quality_summary["n"] += 1
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if quality_summary["n"] % 50 == 0:
                    fh.flush()
                    el = time.time() - t1
                    print(f"  quality {quality_summary['n']}/{len(rows)} "
                          f"({el/quality_summary['n']:.2f}s/item)", flush=True)
        print(f"quality: {json.dumps(quality_summary)}", flush=True)

    summary = {"n": len(interr), "rated": n_rated, "skipped": n_skipped,
               "failed": n_failed, "wall_s": round(time.time() - t0, 1),
               "judge_gguf": str(args.judge_gguf), "lang": args.lang,
               "scope": args.scope, "quality": quality_summary,
               "rubric": "FDB eval_user_interruption.py (verbatim for en)"}
    (args.results / "judge_meta.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
