"""Roll the audio runs up into one comparison table.

Reads ``results_audio/<mode>/<model>/metrics.json`` (written by ``run_audio.py``)
and writes ``results_audio/<mode>/COMPARISON.md`` next to them. Re-runnable, so
the table can be refreshed when another model finishes.

The published row is Table 1 of the DuplexCascade paper
(``fdb_bench/REFERENCE.md``), reproduced here for reference only: it is English
audio scored by GPT-4o, and the judge column here is a local Qwen3.5-9B, so the
comparison is about *ranges and directions*, not equality.

Run:  python -m fdb_bench.audio.compare_audio
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..metrics import averaged_turn_taking_accuracy

PUBLISHED = {
    "pause_synth": 0.058, "pause_candor": 0.222, "bc_tor": 0.218,
    "bc_freq": 0.009, "bc_jsd": 0.949, "smooth_tor": 0.832, "smooth_lat": 1.724,
    "intr_tor": 0.955, "intr_rel": 4.016, "intr_lat": 1.225, "ata": 0.858,
}

# subset -> (official metric key, column)
FIELDS = {
    "candor_pause_handling": ("tor", "pause_candor"),
    "synthetic_pause_handling": ("tor", "pause_synth"),
    "icc_backchannel": ("tor", "bc_tor"),
    "candor_turn_taking": ("tor", "smooth_tor"),
    "synthetic_user_interruption": ("tor", "intr_tor"),
}
def fmt(v) -> str:
    if v is None:
        return "—"
    return f"{v:.3f}" if isinstance(v, float) else str(v)


LABELS = {
    "original_dc": "Original (upstream, EN)",
    "sft": "SFT (pt-BR)",
    "distill": "Distill (pt-BR)",
    "mtp_like": "MTP companion (pt-BR)",
    "base": "Base (control)",
}


TOR_COLS = [("pause_synth", "pause_synth"), ("pause_candor", "pause_candor"),
            ("bc_tor", "bc_tor"), ("smooth_tor", "smooth_tor"),
            ("intr_tor", "intr_tor")]

SUBSET_OF_TOR = {"pause_synth": "synthetic_pause_handling",
                 "pause_candor": "candor_pause_handling",
                 "bc_tor": "icc_backchannel",
                 "smooth_tor": "candor_turn_taking",
                 "intr_tor": "synthetic_user_interruption"}
SUBSET_SIZE = {"synthetic_pause_handling": 137, "candor_pause_handling": 216,
               "icc_backchannel": 55, "candor_turn_taking": 119,
               "synthetic_user_interruption": 200}


def partial_audio_tor(results_dir: Path, model: str) -> dict:
    """Audio-truth take-over rate for a model whose run has not finished.

    Reads the traces of the subsets that are complete and reports the formal-turn
    rate measured from the rendered audio. Labelled as partial in the table: it is
    the same rule the official evaluator applies, but measured from the rendered
    audio rather than from parakeet's transcript of it.
    """
    out: dict = {"_partial": True}
    for col, subset in SUBSET_OF_TOR.items():
        d = results_dir / model / subset
        traces = sorted(d.glob("*/trace.json")) if d.is_dir() else []
        if len(traces) < SUBSET_SIZE[subset]:
            out[col] = None
            continue
        rows = [json.loads(t.read_text(encoding="utf-8")) for t in traces]
        out[col] = round(sum(1 for r in rows if r["formal_turn_by_audio"]) / len(rows), 3)
        out.setdefault("_done", 0)
        out["_done"] = out["_done"] + len(rows)
    out["n"] = out.get("_done", 0)
    return out


def ata_audio(off: dict) -> float:
    """ATA with the project's own convention (``metrics.averaged_turn_taking_accuracy``)."""
    try:
        return round(averaged_turn_taking_accuracy({
            "pause_synthetic_tor": off["synthetic_pause_handling"]["tor"],
            "pause_candor_tor": off["candor_pause_handling"]["tor"],
            "smooth_tor": off["candor_turn_taking"]["tor"],
            "intr_tor": off["synthetic_user_interruption"]["tor"],
        }), 3)
    except Exception:  # noqa: BLE001 - an unfinished run has no ATA
        return float("nan")


def text_rows(path: Path) -> dict[str, dict]:
    """Rows from the text-domain bench (results/<lang>/summary.json)."""
    if not path.is_file():
        return {}
    doc = json.loads(path.read_text(encoding="utf-8"))
    out: dict = {}
    for r in doc.get("rows", []):
        out[r["model"]] = {
            "pause_synth": r.get("pause_synthetic_tor"),
            "pause_candor": r.get("pause_candor_tor"),
            "bc_tor": r.get("bc_tor"),
            "smooth_tor": r.get("smooth_tor"),
            "intr_tor": r.get("intr_tor"),
            "intr_rel": r.get("intr_rating_all"),
            "ata": r.get("ata"),
            "n": r.get("n_scenarios"),
        }
    return out


def tor_section(rows: dict[str, dict], text: dict[str, dict],
                partials: dict[str, dict]) -> list[str]:
    L = ["## Take-over rates, everything tested\n",
         "The same five columns the benchmark reports TOR for. Arrow = the good "
         "direction; pause and backchannel rate a model that speaks too eagerly, "
         "smooth turn-taking and interruption rate a model that answers.\n",
         "| run | model | pause_synth ↓ | pause_candor ↓ | bc_tor ↓ | smooth_tor ↑ | intr_tor ↑ | ATA ↑ |",
         "|---|---|---|---|---|---|---|---|",
         "| *published* | *DuplexCascade (EN audio, GPT-4o)* | *0.058* | *0.222* | *0.218* | *0.832* | *0.955* | *0.858* |"]
    for m, r in rows.items():
        off = r.get("_official") or {}
        ata_v = ata_audio(off) if off else None
        L.append(f"| audio (real WAVs, official evaluators) | {LABELS.get(m, m)} | "
                 + " | ".join(fmt(r.get(c)) for c, _ in TOR_COLS)
                 + f" | {fmt(ata_v)} |")
    for m, r in partials.items():
        L.append(f"| audio (in progress, audio-truth reading) | {LABELS.get(m, m)} "
                 f"({r['n']}/727 scenarios) | "
                 + " | ".join(fmt(r.get(c)) for c, _ in TOR_COLS) + " | — |")
    for m, r in text.items():
        L.append(f"| text (synthetic stimuli, en) | {LABELS.get(m, m)} | "
                 + " | ".join(fmt(r.get(c)) for c, _ in TOR_COLS)
                 + f" | {fmt(r.get('ata'))} |")
    L.append("")
    L.append("Reading the two runs side by side: the text bench feeds clean "
             "transcripts and knows where turns end, so it flatters every model; "
             "the audio bench feeds ASR output and a real silence gate, which is "
             "where the over-eager pt-BR models separate from the upstream one.\n")
    return L


def collect(mode_dir: Path) -> dict[str, dict]:
    out = {}
    for d in sorted(mode_dir.iterdir()):
        mp = d / "metrics.json"
        if not mp.is_file():
            continue
        doc = json.loads(mp.read_text(encoding="utf-8"))
        off = doc.get("official_task_metrics", {})
        row = {col: off.get(sub, {}).get(key) for sub, (key, col) in FIELDS.items()}
        bc = off.get("icc_backchannel", {})
        row["bc_freq"] = bc.get("freq_mean")
        row["bc_jsd"] = bc.get("jsd_mean")
        row["smooth_lat"] = off.get("candor_turn_taking", {}).get("latency_s")
        it = off.get("synthetic_user_interruption", {})
        row["intr_rel"] = it.get("relevance")
        row["intr_lat"] = it.get("latency_s")
        roll = doc.get("audio_rollup", {})
        row["spoke_but_late_only"] = round(
            sum(v.get("spoke_but_late_only_rate", 0) * v.get("n", 0) for v in roll.values())
            / max(1, sum(v.get("n", 0) for v in roll.values())), 4)
        row["mean_llm_s"] = round(
            sum(v.get("mean_compute_s", 0) * v.get("n", 0) for v in roll.values())
            / max(1, sum(v.get("n", 0) for v in roll.values())), 2)
        row["mean_tts_s"] = round(
            sum(v.get("mean_tts_s", 0) * v.get("n", 0) for v in roll.values())
            / max(1, sum(v.get("n", 0) for v in roll.values())), 2)
        row["n"] = sum(v.get("n", 0) for v in roll.values())
        row["_official"] = off
        out[d.name] = row
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=Path("results_audio"))
    ap.add_argument("--mode", default="decision")
    args = ap.parse_args()

    rows = collect(args.results / args.mode)
    if not rows:
        raise SystemExit(f"no metrics.json under {args.results / args.mode}")

    cols = ["pause_synth", "pause_candor", "bc_tor", "bc_freq", "bc_jsd",
            "smooth_tor", "smooth_lat", "intr_tor", "intr_rel", "intr_lat"]
    arrow = {"pause_synth": "↓", "pause_candor": "↓", "bc_tor": "↓", "bc_freq": "↑",
             "bc_jsd": "↓", "smooth_tor": "↑", "smooth_lat": "↓", "intr_tor": "↑",
             "intr_rel": "↑", "intr_lat": "↓"}

    L: list[str] = []
    L.append("# Audio Full-Duplex-Bench: real FDB v1.0 audio\n")
    L.append("Produced by `fdb_bench/audio/` (design and caveats in "
             "`fdb_bench/audio/README.md`). Every number comes from the "
             "benchmark's own `eval_*.py` reading `output.wav`/`output.json`, "
             "where `output.json` is produced by `nvidia/parakeet-tdt-0.6b-v2` "
             "exactly as upstream's `get_transcript/asr.py` does. "
             f"{next(iter(rows.values()))['n']} scenarios per model.\n")
    L.append("## Official task metrics\n")
    head = "| model | " + " | ".join(f"{c} {arrow[c]}" for c in cols) + " |"
    L.append(head)
    L.append("|" + "---|" * (len(cols) + 1))
    L.append("| *published DuplexCascade (EN audio, GPT-4o)* | "
             + " | ".join(f"*{fmt(PUBLISHED[c])}*" for c in cols) + " |")
    for name, row in rows.items():
        L.append(f"| {LABELS.get(name, name)} | "
                 + " | ".join(fmt(row.get(c)) for c in cols) + " |")
    L.append("")

    L.append("## Where the pipeline's time goes, and when a reply arrives too late\n")
    L.append("| model | mean LLM s | mean pocket-tts s | replies audible only past the "
             "end of the input |")
    L.append("|---|---|---|---|")
    for name, row in rows.items():
        L.append(f"| {LABELS.get(name, name)} | {row['mean_llm_s']} | "
                 f"{row['mean_tts_s']} | {row['spoke_but_late_only']:.3f} |")
    L.append("")
    L.append("The last column is the pipeline's own latency made visible: a reply "
             "that begins after the input ends is audible only because the render "
             "carries a 4 s tail. FDB's own outputs are exactly input-length, so "
             "those replies would be scored as silence there — the trace keeps both "
             "readings (`spoke` vs `audible_within_input`).\n")

    partials = {}
    for d in sorted((args.results / args.mode).iterdir()):
        if d.is_dir() and not (d / "metrics.json").is_file():
            pr = partial_audio_tor(args.results / args.mode, d.name)
            if pr.get("n"):
                partials[d.name] = pr
    text = text_rows(Path("results/en/summary.json"))
    L = L[:3] + tor_section(rows, text, partials) + L[3:]

    out = args.results / args.mode / "COMPARISON.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
