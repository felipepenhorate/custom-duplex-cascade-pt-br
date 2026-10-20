"""Which language did the model answer in -- judged by the local model.

`output_language.py` classifies a reply from discriminative function words and
diacritics. That is cheap and it is blind to the cases that matter here: a
reply that code-switches halfway, a short reply with no function words at all,
and a reply that quotes English inside Portuguese. This module asks the local
Qwen3.5-9B instead -- the same model that answers the benchmark's interruption
rubric -- and reports:

* the share of assistant turns answered in Portuguese,
* the share of scenarios whose answers are in Portuguese (the "amount of
  examples answered in Portuguese"),
* the agreement with the heuristic, as a sanity check.

The judge is asked to classify, never to translate, and to answer one line per
input line, so a run costs a few hundred requests instead of one per turn.

    python -m fdb_bench.audio.language_judge \\
        --run "EN:sft=results_audio/decision/sft" \\
        --run "PT:sft=results_audio_pt/decision/sft"
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

BATCH = 10

SYSTEM = (
    "You classify the language of a text. You never translate and you never "
    "rewrite. You answer only in the requested format, with nothing else."
)

INSTRUCTION = (
    "For each numbered line below, decide the language of that line's text.\n"
    "Answer 'yes' if the text is written (mostly) in Brazilian Portuguese.\n"
    "Answer 'no' if the text is in English or in any other language.\n"
    "Answer exactly one line per input line, in the form 'N: yes' or 'N: no', "
    "with the same N. Do not write anything else."
)

_VERDICT = re.compile(r"^\s*(\d+)\s*[:.)-]\s*(yes|no|sim|n[aã]o|true|false)\b",
                      re.IGNORECASE)


def _turns(root: Path) -> list[dict]:
    """Every assistant turn of a run, with its scenario key."""
    out = []
    for td in sorted(root.glob("*/*/trace.json")):
        d = json.loads(td.read_text())
        sid = d.get("sid")
        subset = d.get("subset")
        for i, t in enumerate(d.get("turns") or []):
            txt = (t.get("text") or "").strip()
            if txt:
                out.append({"key": f"{subset}/{sid}", "turn": i, "text": txt})
    return out


def judge(turns: list[dict], api_base: str, *, batch: int = BATCH) -> dict:
    """Classify each turn's language with the local model."""
    from .score import LocalJudgeShim

    client = LocalJudgeShim(api_base)
    verdicts: dict[int, str] = {}
    retries = 0
    for start in range(0, len(turns), batch):
        chunk = turns[start:start + batch]
        listing = "\n".join(f"{i + 1}: {t['text']}" for i, t in enumerate(chunk))
        try:
            raw = client.ask([
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"{INSTRUCTION}\n\n{listing}"},
            ])
        except Exception:  # noqa: BLE001 - a failed batch is retried below
            raw = ""
        got: dict[int, str] = {}
        for line in (raw or "").splitlines():
            m = _VERDICT.match(line)
            if not m:
                continue
            n = int(m.group(1))
            if 1 <= n <= len(chunk):
                v = m.group(2).lower()
                got[n] = "pt" if v in ("yes", "sim", "true") else "en"
        missing = [i for i in range(len(chunk)) if (i + 1) not in got]
        if missing:                       # one retry, one turn per request
            retries += len(missing)
            for i in missing:
                try:
                    raw1 = client.ask([
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content":
                            f"{INSTRUCTION}\n\n1: {chunk[i]['text']}"},
                    ])
                except Exception:  # noqa: BLE001
                    raw1 = ""
                m = _VERDICT.search(raw1 or "")
                got[i + 1] = ("pt" if m and m.group(2).lower() in ("yes", "sim", "true")
                              else "en" if m else "unclear")
        for i in range(len(chunk)):
            verdicts[start + i] = got.get(i + 1, "unclear")
    return {"verdicts": verdicts, "judge_calls": client.calls, "retries": retries}


def summarise(turns: list[dict], verdicts: dict[int, str]) -> dict:
    from .output_language import classify

    per_scenario: dict[str, Counter] = {}
    counts = Counter()
    agree = Counter()
    n_words_pt = n_words = 0
    for i, t in enumerate(turns):
        v = verdicts.get(i, "unclear")
        counts[v] += 1
        per_scenario.setdefault(t["key"], Counter())[v] += 1
        w = len(t["text"].split())
        n_words += w
        if v == "pt":
            n_words_pt += w
        h = classify(t["text"])
        agree[(v, h)] += 1
    scen_pt = sum(1 for c in per_scenario.values() if c["pt"] > c["en"])
    scen_any = sum(1 for c in per_scenario.values() if c["pt"] > 0)
    return {
        "n_turns": len(turns),
        "n_scenarios": len(per_scenario),
        "turns_pt": counts["pt"],
        "turns_en": counts["en"],
        "turns_unclear": counts["unclear"],
        "share_turns_pt": counts["pt"] / max(1, len(turns)),
        "share_scenarios_pt": scen_pt / max(1, len(per_scenario)),
        "share_scenarios_any_pt": scen_any / max(1, len(per_scenario)),
        "share_words_pt": n_words_pt / max(1, n_words),
        "heuristic_agreement": {
            f"{a}/{b}": c for (a, b), c in sorted(agree.items())
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True,
                    help="LABEL=RESULTS_DIR (repeatable); the results dir is the "
                         "one holding <subset>/<sid>/trace.json")
    ap.add_argument("--judge-api-base", default="http://127.0.0.1:8099/v1")
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--out", type=Path, default=None,
                    help="summary file (default: <results dir>/language_llm.json "
                         "per run, plus the printed table)")
    args = ap.parse_args()

    rows = []
    for spec in args.run:
        label, _, path = spec.partition("=")
        root = Path(path)
        turns = _turns(root)
        if not turns:
            print(f"{label}: no traces under {root}")
            continue
        res = judge(turns, args.judge_api_base, batch=args.batch)
        summary = summarise(turns, res["verdicts"])
        summary.update({"label": label, "results": str(root),
                        "judge": "Qwen3.5-9B (local llama-server)",
                        "judge_calls": res["judge_calls"],
                        "judge_retries": res["retries"]})
        (root / "language_llm.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
        rows.append(summary)
        print(f"{label:22s} turns {summary['n_turns']:5d}  "
              f"pt {summary['share_turns_pt']:6.1%}  "
              f"scenarios pt {summary['share_scenarios_pt']:6.1%}  "
              f"(any pt {summary['share_scenarios_any_pt']:6.1%})  "
              f"words pt {summary['share_words_pt']:6.1%}  "
              f"calls {summary['judge_calls']}", flush=True)
    if args.out:
        args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=1),
                            encoding="utf-8")


if __name__ == "__main__":
    main()
