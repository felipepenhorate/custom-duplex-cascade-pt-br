"""Cross-language and cross-variant comparison over finished FDB runs.

``run_metrics.py`` produces one table per language. This produces the two views
that answer the questions the bench exists for:

* **pt vs en** — how much of each variant's turn-taking survives when the
  content is out-of-distribution (all three were fine-tuned pt-BR only), i.e.
  cross-lingual transfer of the *protocol*;
* **variant vs variant** — which training method (SFT / self-distillation /
  policy companion) produces the best turn-taking, and *why*, read off the tag
  histograms.

    python -m fdb_bench.compare --results results
    python -m fdb_bench.compare --results results --markdown > comparison.md
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

# columns of the summary.json rows that form the FDB table
METRICS = [
    ("pause_synthetic_tor", "Pause TOR↓ synth", True),
    ("pause_candor_tor", "Pause TOR↓ candor", True),
    ("bc_tor", "BC TOR↓", True),
    ("bc_freq", "BC Freq↑", False),
    ("bc_jsd", "BC JSD↓", True),
    ("smooth_tor", "Smooth TOR↑", False),
    ("smooth_latency_s", "Floor lat↓", True),
    ("intr_tor", "Intr TOR↑", False),
    ("intr_rating", "Intr rating↑", False),
    ("intr_yield_compute_s", "Stop lat↓", True),
    ("intr_response_compute_s", "Resp lat↓", True),
    ("ata", "ATA↑", False),
]

# tags that matter for explaining a score
KEY_TAGS = [
    "user_finish_speaking",
    "user_is_speaking",
    "user_is_thinking",
    "user_interruption",
    "user_backchannel",
    "system_backchannel",
    "no_tag",
]


def _fmt(x: Any, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def load_summaries(results: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """``{lang: {model: row}}`` from every ``<lang>/summary.json``."""
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for d in sorted(results.iterdir()):
        sj = d / "summary.json"
        if not sj.is_file():
            continue
        data = json.loads(sj.read_text(encoding="utf-8"))
        out[d.name] = {r["model"]: r for r in data.get("rows", [])}
        out[d.name]["__lang__"] = {  # type: ignore[assignment]
            "words_per_second": data.get("words_per_second"),
            "reference": data.get("reference", []),
        }
    return out


def language_table(summaries: dict[str, dict[str, dict[str, Any]]]) -> str:
    """One row per (model, lang), for eyeballing the pt→en change."""
    langs = sorted(summaries)
    models = sorted({m for L in summaries.values() for m in L if m != "__lang__"})
    head = "| Model | Lang | " + " | ".join(n for _, n, _ in METRICS) + " |"
    lines = [head, "|" + "---|" * (len(METRICS) + 2)]
    for model in models:
        for lang in langs:
            row = summaries.get(lang, {}).get(model)
            if not row:
                continue
            cells = [_fmt(row.get(k)) for k, _, _ in METRICS]
            lines.append(f"| {row.get('label', model)} | {lang} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def delta_table(summaries: dict[str, dict[str, dict[str, Any]]]) -> str:
    """pt → en deltas, oriented so a positive delta is always *better*."""
    if "pt" not in summaries or "en" not in summaries:
        return "_need both pt and en summaries_"
    head = "| Model | " + " | ".join(n + " Δ(en−pt)" for _, n, _ in METRICS) + " |"
    lines = [head, "|" + "---|" * (len(METRICS) + 1)]
    models = sorted(set(summaries["pt"]) & set(summaries["en"]) - {"__lang__"})
    for model in models:
        p = summaries["pt"][model]
        e = summaries["en"][model]
        cells = []
        for key, _, lower_is_better in METRICS:
            pv, ev = p.get(key), e.get(key)
            if pv is None or ev is None or _nan(pv) or _nan(ev):
                cells.append("-")
                continue
            d = ev - pv
            if lower_is_better:
                d = -d          # so + always means an improvement
            cells.append(f"{d:+.3f}")
        lines.append(f"| {p.get('label', model)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _nan(x: Any) -> bool:
    return isinstance(x, float) and math.isnan(x)


def tag_table(summaries: dict[str, dict[str, dict[str, Any]]]) -> str:
    """Share of decisions spent on each protocol tag — the *why* behind a row."""
    langs = sorted(summaries)
    head = ("| Model | Lang | decisions | " + " | ".join(t + " %" for t in KEY_TAGS) + " |")
    lines = [head, "|" + "---|" * (len(KEY_TAGS) + 3)]
    for lang in langs:
        for model, row in sorted(summaries[lang].items()):
            if model == "__lang__":
                continue
            hist = row.get("tag_hist") or {}
            total = sum(hist.values()) or 1
            frac = [hist.get(t, 0) / total * 100 for t in KEY_TAGS]
            cells = " | ".join(f"{f:.1f}" for f in frac)
            lines.append(f"| {row.get('label', model)} | {lang} | {total} | {cells} |")
    return "\n".join(lines)


def headline(summaries: dict[str, dict[str, dict[str, Any]]]) -> str:
    """A short prose read-out of the run, for the top of a report."""
    langs = sorted(summaries)
    out: list[str] = []
    for lang in langs:
        rows = {m: r for m, r in summaries[lang].items() if m != "__lang__"}
        if not rows:
            continue
        best = max(rows.values(), key=lambda r: r.get("ata") or -1)
        out.append(
            f"**{lang}** — best ATA: {best.get('label', '?')} "
            f"({_fmt(best.get('ata'))}); "
            + "; ".join(
                f"{r.get('label', m)}: ATA {_fmt(r.get('ata'))}, "
                f"pause {_fmt(r.get('pause_synthetic_tor'))}/{_fmt(r.get('pause_candor_tor'))}, "
                f"smooth {_fmt(r.get('smooth_tor'))}, intr {_fmt(r.get('intr_tor'))}, "
                f"rating {_fmt(r.get('intr_rating'))}"
                for m, r in sorted(rows.items())
            )
        )
    return "\n\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--markdown", action="store_true",
                    help="print a full markdown report instead of the summary")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    summaries = load_summaries(args.results)
    if not summaries:
        raise SystemExit(f"no <lang>/summary.json under {args.results} — run run_metrics.py first")

    parts = []
    if args.markdown:
        parts.append(
            "# DuplexCascade pt-BR — Full-Duplex-Bench comparison\n\n"
            "Text-domain FDB v1.0 harness. See `fdb_bench/README.md` for what is\n"
            "and is not faithful, and `fdb_bench/REFERENCE.md` for the published\n"
            "numbers. Within a table, our rows are directly comparable to each\n"
            "other; the paper row is a range check, not a baseline.\n"
        )
        parts.append("## Headline\n\n" + headline(summaries))
        parts.append("## All runs\n\n" + language_table(summaries))
        parts.append(
            "## Cross-lingual transfer (pt → en)\n\n"
            "Deltas are oriented so **+ is always better**. English is\n"
            "out-of-distribution for all three variants (they were fine-tuned\n"
            "on pt-BR duplex data only), so this is protocol transfer, not\n"
            "language skill.\n\n" + delta_table(summaries)
        )
        parts.append(
            "## Protocol diagnostics — where the decisions go\n\n"
            "Share of decisions spent on each tag. A model with a low pause TOR\n"
            "but a high `user_is_speaking` share is genuinely waiting; one with\n"
            "many `no_tag` or `user_interruption` emissions is misfiring.\n\n"
            + tag_table(summaries)
        )
        text = "\n\n".join(parts) + "\n"
    else:
        text = headline(summaries) + "\n\n" + language_table(summaries) + "\n"

    print(text)
    if args.out:
        args.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
