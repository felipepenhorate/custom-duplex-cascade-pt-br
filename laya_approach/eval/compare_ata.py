"""Compare Averaged Turn-Taking Accuracy (ATA) across solutions.

The text table already carries ATA.  The audio harness reports raw task TORs, so
this script applies the same orientation and smooth-turn gate used by
``fdb_bench.metrics.averaged_turn_taking_accuracy``:

    ATA = mean(1 - pause_synth, 1 - pause_candor, smooth, interruption)

with the guard that a model below 0.5 smooth TOR collapses to its smooth score
(a model that never takes the floor must not collect pause credit).

    python -m laya_approach.eval.compare_ata \
        --results custom-duplex-cascade-pt-br/results \
        --audio-pt laya_approach/data/fdb_audio_results/laya_v7_pt/metrics_audio_judged.json \
        --audio-en /tmp/opencode/en_audio_results/decision/laya_companion_v7/metrics_audio_judged.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def audio_ata(path: Path) -> float:
    data = json.loads(path.read_text(encoding="utf-8"))
    m = data.get("official_task_metrics") or data
    try:
        pause_syn = float(m["synthetic_pause_handling"]["tor"])
        pause_can = float(m["candor_pause_handling"]["tor"])
        smooth = float(m["candor_turn_taking"]["tor"])
        intr = float(m["synthetic_user_interruption"]["tor"])
    except (KeyError, TypeError, ValueError):
        return float("nan")
    if smooth < 0.5:
        return smooth
    return _mean([1.0 - pause_syn, 1.0 - pause_can, smooth, intr])


def text_rows(results: Path, lang: str) -> list[dict]:
    path = results / lang / "summary.json"
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("rows") or []


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=Path("custom-duplex-cascade-pt-br/results"))
    ap.add_argument("--audio-pt", type=Path, default=None)
    ap.add_argument("--audio-en", type=Path, default=None)
    ap.add_argument("--audio-pt-name", default="laya_v7")
    ap.add_argument("--audio-en-name", default="laya_v7")
    ap.add_argument("--mtp-audio-pt", type=Path,
                    default=Path("custom-duplex-cascade-pt-br/results_audio_pt/decision/mtp_like/metrics_whisper.json"))
    ap.add_argument("--mtp-audio-en", type=Path,
                    default=Path("custom-duplex-cascade-pt-br/results_audio/decision/mtp_like/metrics_whisper.json"))
    args = ap.parse_args()

    print("## Text FDB — Averaged Turn-Taking Accuracy\n")
    for lang in ("pt", "en"):
        rows = text_rows(args.results, lang)
        if not rows:
            continue
        print(f"### {lang}\n")
        print("| model | ATA ↑ | smooth TOR | pause TOR (s/c) | intr TOR |")
        print("|---|---:|---:|---:|---:|")
        for row in sorted(rows, key=lambda r: -(r.get("ata") or 0.0)):
            ata = row.get("ata")
            if ata is None or (isinstance(ata, float) and ata != ata):
                continue
            print(
                f"| {row.get('model')} | **{ata:.4f}** | "
                f"{row.get('smooth_tor', float('nan')):.3f} | "
                f"{row.get('pause_synthetic_tor', float('nan')):.3f} / "
                f"{row.get('pause_candor_tor', float('nan')):.3f} | "
                f"{row.get('intr_tor', float('nan')):.3f} |"
            )
        print()

    print("## Audio FDB — ATA from task TORs\n")
    print("| run | ATA ↑ | pause synth ↓ | pause candor ↓ | smooth ↑ | intr ↑ |")
    print("|---|---:|---:|---:|---:|---:|")
    entries: list[tuple[str, Path]] = []
    if args.audio_pt:
        entries.append((f"{args.audio_pt_name} / pt", args.audio_pt))
    if args.audio_en:
        entries.append((f"{args.audio_en_name} / en", args.audio_en))
    if args.mtp_audio_pt.is_file():
        entries.append(("mtp_like / pt", args.mtp_audio_pt))
    if args.mtp_audio_en.is_file():
        entries.append(("mtp_like / en", args.mtp_audio_en))
    for name, path in entries:
        data = json.loads(path.read_text(encoding="utf-8"))
        m = data.get("official_task_metrics") or data
        ata = audio_ata(path)
        print(
            f"| {name} | **{ata:.4f}** | "
            f"{m.get('synthetic_pause_handling', {}).get('tor', float('nan')):.3f} | "
            f"{m.get('candor_pause_handling', {}).get('tor', float('nan')):.3f} | "
            f"{m.get('candor_turn_taking', {}).get('tor', float('nan')):.3f} | "
            f"{m.get('synthetic_user_interruption', {}).get('tor', float('nan')):.3f} |"
        )


if __name__ == "__main__":
    main()
