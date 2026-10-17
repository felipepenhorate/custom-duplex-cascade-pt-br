"""Phase 3: aggregate raw timelines into the FDB table.

Reads every ``results/<lang>/<model>/{raw,judged}.jsonl`` and emits
``results/<lang>/summary.json`` plus a markdown table laid out like the
published FDB Table 1, with the published DuplexCascade row alongside for
reference.

    python -m fdb_bench.run_metrics --results results/pt
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from . import metrics as M
from .clock import SpeechClock
from .metrics import mean as _mean
from .simulate import ModelTurn, SimResult, StreamEvent


def load_run(d: Path) -> tuple[dict[str, SimResult], dict[str, Any]]:
    raw = d / "raw.jsonl"
    if not raw.is_file():
        raise SystemExit(f"missing {raw}")
    res: dict[str, SimResult] = {}
    for line in raw.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        res[o["sid"]] = SimResult(
            sid=o["sid"], task=o["task"], lang=o["lang"], model=o["model"],
            user_events=[StreamEvent(**e) for e in o["user_events"]],
            model_turns=[ModelTurn(**m) for m in o["model_turns"]],
            tags=o["tags"],
            gt_backchannel_times=o.get("gt_backchannel_times", []),
            context=o.get("context", ""), interrupt=o.get("interrupt", ""),
            interrupt_time=o.get("interrupt_time"),
            interrupt_end=o.get("interrupt_end"),
            post_interrupt_text=o.get("post_interrupt_text", ""),
            took_floor_after_interrupt=o.get("took_floor_after_interrupt", False),
            yield_time=o.get("yield_time"),
            response_compute_s=o.get("response_compute_s", 0.0),
            # end_time drives the backchannel frequency (per second) and the
            # JSD window grid: leaving it at the 0.0 default made total_s
            # collapse to its 1e-9 guard.
            end_time=o.get("end_time", 0.0),
            floor_compute_s=o.get("floor_compute_s", 0.0),
            model_words_at_interrupt=o.get("model_words_at_interrupt", 0),
            model_speaking_at_interrupt=o.get("model_speaking_at_interrupt", False),
            yield_compute_s=o.get("yield_compute_s", 0.0),
            decisions=o.get("decisions", []),
            empty_floor_takings=o.get("empty_floor_takings", 0),
            error=o.get("error"),
            n_decisions=o.get("n_decisions", 0),
            compute_s=o.get("compute_s", 0.0),
        )
    meta: dict[str, Any] = {}
    mp = d / "meta.json"
    if mp.is_file():
        meta = json.loads(mp.read_text())
    return res, meta


def load_quality(d: Path) -> dict[str, dict[str, Any]]:
    """Whole-dataset judged records (``quality.jsonl``), keyed by sid."""
    p = d / "quality.jsonl"
    if not p.is_file():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            o = json.loads(line)
            out[o["sid"]] = o
    return out


def load_judged(d: Path) -> dict[str, dict[str, Any]]:
    p = d / "judged.jsonl"
    if not p.is_file():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        out[o["sid"]] = o
    return out


def summarise(d: Path, clock: SpeechClock) -> dict[str, Any]:
    results, meta = load_run(d)
    judged = load_judged(d)
    by_task: dict[str, list[SimResult]] = {}
    for r in results.values():
        by_task.setdefault(r.task, []).append(r)

    row: dict[str, Any] = {
        "model": meta.get("model", d.name),
        # The prompt configuration goes on the label: the variants are prompt
        # sensitive and two rows for the same model are only comparable once the
        # configuration is visible. See PROMPTS.md.
        "label": f"{meta.get('label', d.name)} [{meta.get('prompt_candidate', '?')}]",
        "prompt_candidate": meta.get("prompt_candidate", "?"),
        "lang": meta.get("lang", "?"),
        "prime_mode": meta.get("prime_mode", "?"),
        "words_per_second": meta.get("words_per_second", clock.words_per_second),
        "n_scenarios": len(results),
        "n_errors": sum(1 for r in results.values() if r.error),
        "empty_floor_takings": sum(r.empty_floor_takings for r in results.values()),
        "s_per_scenario": round(meta.get("wall_s", 0.0) / max(1, len(results)), 3),
    }

    ps = M.pause_handling(by_task.get("pause_synthetic", []), clock)
    pc = M.pause_handling(by_task.get("pause_candor", []), clock)
    bc = M.backchannel(by_task.get("backchannel_icc", []), clock)
    st = M.smooth_turn_taking(by_task.get("smooth_turn", []), clock)
    it = M.user_interruption_tor_latency(by_task.get("interruption_synthetic", []))
    jr = M.aggregate_interruption_ratings(
        {k: v for k, v in judged.items()}
    )
    qual = M.aggregate_quality_scores(load_quality(d))

    row.update({
        "pause_synthetic_tor": ps["tor"], "pause_synthetic_n": ps["n"],
        "pause_candor_tor": pc["tor"], "pause_candor_n": pc["n"],
        "bc_tor": bc["tor"], "bc_freq": bc["freq"], "bc_jsd": bc["jsd"], "bc_n": bc["n"],
        "smooth_tor": st["tor"], "smooth_latency_chunks": st["latency_chunks"],
        "smooth_latency_s": _mean([r.floor_compute_s for r in by_task.get("smooth_turn", [])
                                   if not r.error and r.floor_compute_s > 0]),
        "smooth_latency_n": sum(1 for r in by_task.get("smooth_turn", [])
                                if not r.error and r.floor_compute_s > 0),
        "smooth_n": st["n"],
        "intr_tor": it["tor"], "intr_n_harness_unscored": it["n_harness_unscored"],
        "intr_response_compute_s": it["response_compute_s"],
        "intr_yield_compute_s": it["yield_compute_s"],
        "intr_n": it["n"],
        "intr_rating": jr["judge_rating"], "intr_rating_all": jr["judge_rating_all"],
        "intr_judge_n": jr["judge_n"],
        # tag histogram: how the model spent its decisions
        "tag_hist": _tag_hist(results),
    })
    row.update(M.tag_decision_accuracy(list(results.values())))
    row["quality"] = qual["quality"]
    row["quality_n"] = qual["quality_n"]
    row["quality_n_all"] = qual["quality_n_all"]
    row["quality_from_judge"] = qual["quality_from_judge"]
    row["quality_by_task"] = qual["quality_by_task"]
    row["ata"] = M.averaged_turn_taking_accuracy(row)
    return row


def _tag_hist(results: dict[str, SimResult]) -> dict[str, int]:
    h: dict[str, int] = {}
    for r in results.values():
        for t in r.tags:
            h[t] = h.get(t, 0) + 1
    return dict(sorted(h.items(), key=lambda kv: -kv[1]))


def _fmt(x: Any, nd: int = 3) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        if math.isnan(x):
            return "-"
        return f"{x:.{nd}f}"
    return str(x)


HEADER = (
    "| Model | Lang | Pause TOR↓ (synth) | Pause TOR↓ (candor) | BC TOR↓ | "
    "BC Freq↑ | BC JSD↓ | Smooth TOR↑ | Floor lat (s)↓ | Intr TOR↑ | "
    "Intr rating↑ | Stop lat (s)↓ | Resp lat (s)↓ | Quality↑ | **ATA**↑ |"
)
SEP = "|" + "---|" * 15

_LABELS = {
    "smooth_latency_s": "floor_compute_s",
    "intr_latency_s": "yield_compute_s",
}


def _row(cols: list[str], published: bool = False) -> str:
    return "| " + " | ".join(cols) + " |"


def table(rows: list[dict[str, Any]], reference: list[dict[str, Any]] | None = None) -> str:
    """Render the FDB-style table: published reference rows, then ours."""
    lines = [HEADER, SEP]
    for r in (reference or []):
        # a published row has no *measured* latencies; its two latency columns
        # come from the paper and are printed in the only slots they fit
        lines.append(_row([
            f"{r['label']} _(published, EN audio)_", "en",
            _fmt(r["pause_synthetic_tor"]), _fmt(r["pause_candor_tor"]),
            _fmt(r["bc_tor"]), _fmt(r["bc_freq"]), _fmt(r["bc_jsd"]),
            _fmt(r["smooth_tor"]), _fmt(r["smooth_latency_s"]),
            _fmt(r["intr_tor"]), _fmt(r["intr_rating"]),
            _fmt(r["intr_latency_s"]), "-", "-",
            f"**{_fmt(r['ata'])}**",
        ]))
    for r in rows:
        lines.append(_row([
            r["label"], r["lang"],
            _fmt(r.get("pause_synthetic_tor")), _fmt(r.get("pause_candor_tor")),
            _fmt(r.get("bc_tor")), _fmt(r.get("bc_freq")), _fmt(r.get("bc_jsd")),
            _fmt(r.get("smooth_tor")), _fmt(r.get("smooth_latency_s")),
            _fmt(r.get("intr_tor")), _fmt(r.get("intr_rating")),
            _fmt(r.get("intr_yield_compute_s")),
            _fmt(r.get("intr_response_compute_s")),
            _fmt(r.get("quality")),
            f"**{_fmt(r.get('ata'))}**",
        ]))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True, type=Path,
                    help="results/<lang> directory containing per-model subdirs")
    ap.add_argument("--words-per-second", type=float, default=3.25)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    clock = SpeechClock(words_per_second=args.words_per_second)
    runs = sorted(d for d in args.results.iterdir() if (d / "raw.jsonl").is_file())
    if not runs:
        raise SystemExit(f"no model runs with raw.jsonl under {args.results}")

    rows = [summarise(d, clock) for d in runs]

    ref = reference_rows()
    md = "# Full-Duplex-Bench (pt-BR / en, text-domain)\n\n"
    md += (f"Scenarios per subset follow FDB v1.0 sizes; clock = "
           f"{args.words_per_second} words/s.\n\n")
    md += table(rows, ref) + "\n"
    md += "\n## Per-model detail\n\n```json\n"
    md += json.dumps(rows, indent=2, ensure_ascii=False) + "\n```\n"

    out = args.out or (args.results / "summary.md")
    out.write_text(md, encoding="utf-8")
    (args.results / "summary.json").write_text(
        json.dumps({"rows": rows, "reference": ref,
                    "words_per_second": args.words_per_second}, indent=2,
                   ensure_ascii=False),
        encoding="utf-8",
    )
    print(md)


def reference_rows() -> list[dict[str, Any]]:
    from .reference import published_rows
    return published_rows()


if __name__ == "__main__":
    main()
