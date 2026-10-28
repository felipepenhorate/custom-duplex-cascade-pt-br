"""Check that the local scenario generator reproduces the published PT text set.

The text-domain harness builds its 727 scenarios in code (``fdb_bench/scenarios``).
The published ``lumierenoir/synthetic-full-duplex-bench-pt-br`` ``pt_text``
configuration is exactly those scenarios, so a run is only comparable to the
published set if the generated event streams match it.

The match is exact with the **word chunker**.  The published set was built with
it (its pieces are word-level, which is also what a live ASR emits and what
Laya was trained on), not with the Qwen-token-level ``TrainingChunker``.

    python -m laya_approach.eval.check_fdb_parity \
        --parquet laya_approach/data/fdb_pt_text/data/pt_text/train-00000-of-00001.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "custom-duplex-cascade-pt-br"))

from fdb_bench import scenarios as S  # noqa: E402
from fdb_bench.chunking import word_chunker  # noqa: E402

FIELDS = ("kind", "text", "ends_turn", "is_backchannel", "ends_sentence", "after_turn_end")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--parquet",
        default="laya_approach/data/fdb_pt_text/data/pt_text/train-00000-of-00001.parquet",
    )
    args = ap.parse_args()

    import pyarrow.parquet as pq

    rows = pq.read_table(args.parquet).to_pylist()
    published = {(row["task"], row["sid"]): row for row in rows}
    checked = mismatch = missing = 0
    for task in S.TASKS:
        for scenario in S.build_scenarios(task, lang="pt", chunker=word_chunker):
            row = published.get((task, scenario.sid))
            if row is None:
                missing += 1
                continue
            expected = [{key: event[key] for key in FIELDS} for event in row["events"]]
            got = [
                {
                    "kind": event.kind,
                    "text": event.text,
                    "ends_turn": event.ends_turn,
                    "is_backchannel": event.is_backchannel,
                    "ends_sentence": event.ends_sentence,
                    "after_turn_end": event.after_turn_end,
                }
                for event in scenario.events
            ]
            checked += 1
            if expected != got:
                mismatch += 1
                if mismatch <= 3:
                    print(f"MISMATCH {task}/{scenario.sid}")
                    for a, b in zip(expected, got):
                        if a != b:
                            print(f"  published {a}\n  generated {b}")
                            break
    print(
        f"[fdb-parity] published={len(rows)} checked={checked} "
        f"missing={missing} mismatch={mismatch}"
    )
    if mismatch or missing or checked != len(rows):
        raise SystemExit("local scenarios do not reproduce the published pt_text set")
    print("[fdb-parity] OK: the published pt_text set is reproduced exactly")


if __name__ == "__main__":
    main()
