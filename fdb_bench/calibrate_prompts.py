"""Measure which system prompt actually activates each variant's duplex mode.

The question this answers
-------------------------
A duplex model can only be scored on turn-taking in a configuration that makes
it *behave* as a duplex model. Two failure modes look identical in a results
table but are completely different findings:

* the model cannot do duplex turn-taking, or
* the prompt it was handed told it not to (the distill bridge prompt says
  "não imprima nenhuma tag especial" — it forbids the very tags being measured).

Running one arbitrary prompt and reporting the score conflates the two. So this
script measures every candidate prompt for every variant on the same probes and
picks with numbers instead of taste.

What is measured
----------------
For a fixed sample of scenarios, every decision point is scored against the
scenario's own gold label (``gold_ends_turn`` — does this user chunk end the
user's turn?), giving:

* **finish precision / recall / F1** — how well the model emits
  ``<|user finish speaking|>`` exactly when the turn ends;
* **pause-stayed-silent** — fraction of silence ticks where it correctly did
  *not* take the floor;
* **content sanity** on the utterances it does produce: empty rate, non-Latin
  script rate, repeat rate, and mean length.

A candidate "activates duplex mode" when it emits the tags at all (recall > 0)
and the content it produces is clean. Candidates that fail either are reported
with the numbers that disqualified them.

Usage::

    # serve the model's GGUF, then:
    python -m fdb_bench.calibrate_prompts --model distill --lang pt
    python -m fdb_bench.calibrate_prompts --model all --lang pt --out PROMPTS.md

Requires a llama-server for the model under test (``--api-base``), because the
whole point is to observe the real generation behaviour.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import unicodedata
from pathlib import Path
from typing import Any

from . import metrics as M
from . import protocol as P
from . import scenarios as S
from .adapters import PROMPT_CANDIDATES, LlamaClient, build_adapter
from .clock import SpeechClock
from .simulate import DuplexSimulator

# A scenario sample that covers all four decision shapes without taking hours.
DEFAULT_SAMPLE = {"smooth_turn": 14, "pause_synthetic": 14, "pause_candor": 14,
                  "backchannel_icc": 6, "interruption_synthetic": 10}

# Function words that are strong evidence of Portuguese/English prose.
_PT_MARKERS = {"que", "não", "para", "com", "uma", "você", "isso", "mais",
               "como", "então", "também", "está", "muito", "aqui", "porque"}
_EN_MARKERS = {"the", "and", "you", "that", "with", "for", "have", "this",
               "what", "would", "about", "there", "from", "your"}


def _nonlatin_ratio(text: str) -> float:
    """Fraction of letters outside the Latin script (catches Korean/CJK slips)."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    bad = [c for c in letters
           if "LATIN" not in unicodedata.name(c, "")]
    return len(bad) / len(letters)


def _content_quality(texts: list[str], lang: str) -> dict[str, Any]:
    """Cheap, explicit sanity checks on generated speech."""
    if not texts:
        return {"n": 0, "empty_rate": 1.0, "nonlatin_rate": 0.0,
                "repeat_rate": 0.0, "marker_rate": 0.0, "mean_words": 0.0}
    markers = _PT_MARKERS if lang == "pt" else _EN_MARKERS
    empty = sum(1 for t in texts if not t.strip())
    nonlatin = statistics.mean(_nonlatin_ratio(t) for t in texts)
    # a chunk that repeats verbatim is the degeneration these fine-tunes fall into
    repeats = sum(1 for a, b in zip(texts, texts[1:])
                  if a.strip() and a.strip() == b.strip())
    marker = sum(1 for t in texts
                 if markers & {w.strip(".,!?…").lower() for w in t.split()})
    return {
        "n": len(texts),
        "empty_rate": empty / len(texts),
        "nonlatin_rate": nonlatin,
        "repeat_rate": repeats / max(1, len(texts) - 1),
        "marker_rate": marker / len(texts),
        "mean_words": statistics.mean(len(t.split()) for t in texts),
    }


def run_candidate(
    model: str, candidate: str, lang: str, *, api_base: str,
    big_api_base: str | None, sample: dict[str, int], seed: int, wps: float,
    device: str = "cuda", collect_texts: int = 40, chunker=None,
) -> dict[str, Any]:
    """Run the whole sample under one prompt candidate and score it."""
    adapter = build_adapter(
        model, api_base=api_base, big_api_base=big_api_base, lang=lang,
        prompt_candidate=candidate, device=device,
    )
    sim = DuplexSimulator(adapter, SpeechClock(wps))
    if chunker is None:
        from .run_inference import build_chunker
        chunker = build_chunker()
    results = []
    for task, n in sample.items():
        for sc in S.build_scenarios(task, limit=n, seed=seed, lang=lang,
                                    chunker=chunker):
            results.append(sim.run(sc))

    acc = M.tag_decision_accuracy(results)
    texts: list[str] = []
    for r in results:
        for m in r.model_turns:
            if m.kind == "speech" and m.text.strip():
                texts.append(m.text)
    texts = texts[:collect_texts]
    quality = _content_quality(texts, lang)

    errors = [r.error for r in results if r.error]
    return {
        "model": model,
        "candidate": candidate,
        "lang": lang,
        "system_prompt_len": len(adapter.system_prompt or ""),
        "system_prompt_head": (adapter.system_prompt or "<none>")[:70],
        "n_scenarios": len(results),
        "n_errors": len(errors),
        **{k: v for k, v in acc.items()},
        **{f"content_{k}": v for k, v in quality.items()},
        "floor_takings": sum(len(r.model_turns) for r in results),
    }


def verdict(row: dict[str, Any]) -> str:
    """One-word read-out of whether a candidate activates duplex mode."""
    rec = row.get("finish_recall")
    prec = row.get("finish_precision")
    if row["n_errors"]:
        return "BROKEN"
    if not rec or (isinstance(rec, float) and rec != rec) or rec <= 0.05:
        return "NO-TAGS"
    if row.get("content_nonlatin_rate", 0) > 0.02:
        return "DEGENERATE"
    if row.get("content_empty_rate", 1) > 0.5:
        return "NO-SPEECH"
    if prec is not None and not isinstance(prec, float):
        return "PARTIAL"
    if prec is not None and prec == prec and prec < 0.5:
        return "OVER-EAGER"
    return "OK"


def markdown(rows: list[dict[str, Any]], lang: str) -> str:
    head = ("| Model | Candidate | prompt len | finish P | finish R | finish F1 | "
            "chunk acc | pause silent | non-Latin | repeats | empty | verdict |")
    lines = [f"### {lang} — prompt candidates\n", head, "|" + "---|" * 12]
    for r in rows:
        lines.append(
            f"| {r['model']} | `{r['candidate']}` | {r['system_prompt_len']} | "
            f"{_n(r['finish_precision'])} | {_n(r['finish_recall'])} | "
            f"{_n(r['finish_f1'])} | {_n(r['chunk_accuracy'])} | "
            f"{_n(r['pause_stayed_silent'])} | "
            f"{_n(r['content_nonlatin_rate'])} | {_n(r['content_repeat_rate'])} | "
            f"{_n(r['content_empty_rate'])} | **{verdict(r)}** |"
        )
    return "\n".join(lines)


def _n(x: Any) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return "-" if x != x else f"{x:.3f}"
    return str(x)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="all",
                    help="a model-registry key, or 'all'")
    ap.add_argument("--lang", default="pt", choices=["pt", "en"])
    ap.add_argument("--api-base", default="http://127.0.0.1:8098/v1")
    ap.add_argument("--big-api-base", default=None)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--words-per-second", type=float, default=3.25)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    models = (sorted(PROMPT_CANDIDATES) if args.model == "all" else [args.model])
    if len(models) > 1:
        ap.error("serve one model at a time: pass --model <key> (start a "
                 "llama-server for it and point --api-base at it)")

    rows: list[dict[str, Any]] = []
    for model in models:
        for cand in PROMPT_CANDIDATES[model]:
            print(f"\n=== {model} / {cand} ===", flush=True)
            try:
                row = run_candidate(
                    model, cand, args.lang, api_base=args.api_base,
                    big_api_base=args.big_api_base, sample=DEFAULT_SAMPLE,
                    seed=args.seed, wps=args.words_per_second, device=args.device,
                )
            except Exception as e:  # noqa: BLE001
                print(f"  FAILED: {type(e).__name__}: {e}")
                row = {"model": model, "candidate": cand, "lang": args.lang,
                       "n_errors": -1, "system_prompt_len": 0,
                       "system_prompt_head": "", "finish_precision": None,
                       "finish_recall": None, "finish_f1": None,
                       "chunk_accuracy": None, "pause_stayed_silent": None,
                       "content_nonlatin_rate": None, "content_repeat_rate": None,
                       "content_empty_rate": None, "floor_takings": 0,
                       "n_scenarios": 0}
                row["verdict"] = "FAILED"
            row["verdict"] = verdict(row)
            print(f"  {json.dumps({k: v for k, v in row.items() if k not in ('system_prompt_head',)}, default=str)}")
            rows.append(row)

    md = markdown(rows, args.lang)
    print("\n" + md)
    out = args.out or Path(f"fdb_bench/prompt_calibration_{args.lang}.json")
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}")
    md_path = Path(f"fdb_bench/prompt_calibration_{args.lang}.md")
    md_path.write_text(
        f"# Duplex prompt calibration ({args.lang})\n\n"
        "Measured by `python -m fdb_bench.calibrate_prompts`. Each row is one\n"
        "system-prompt candidate run over the same scenario sample; the verdict\n"
        "column says whether it activates duplex mode (see "
        "`calibrate_prompts.verdict`).\n\n" + md + "\n",
        encoding="utf-8",
    )
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
