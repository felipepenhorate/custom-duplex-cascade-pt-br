"""Build a permissive training mix: MIT text data + MIT synthetic FDB audio.

The public v7 release is CC BY-NC 4.0 because it uses CANDOR/ICC annotations.
This filter removes every row derived from those subsets, keeping:

* all text rows (source dialogues and the 9B-translated pilot are MIT);
* audio-trace rows from the two **synthetic** FDB subsets
  (`synthetic_pause_handling`, `synthetic_user_interruption`, MIT).

The result can be used to train a model released under Apache-2.0 or MIT,
inheriting only the Apache-2.0 Laya base.  It is intentionally narrower than the
research model: no CANDOR smooth-turn, no CANDOR pause, no ICC backchannel.

    python -m laya_approach.data.filter_commercial \
        --src data/generated_v7_mix --out data/generated_v7_commercial
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

COMMERCIAL_AUDIO_SUBSETS = {
    "synthetic_pause_handling",
    "synthetic_user_interruption",
}


def keep(row: dict, keep_subsets: set[str] | None = None) -> bool:
    meta = row.get("_meta") or {}
    if meta.get("subset") in (keep_subsets or COMMERCIAL_AUDIO_SUBSETS):
        return True
    # Every text row comes from DuplexCascade-PT-BR-V0 (MIT) or the translated
    # pilot of that corpus.  Audio rows without a whitelisted subset are
    # CANDOR/ICC-derived and must be dropped.
    return (row.get("state") or {}).get("stream") == "text"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, type=Path,
                    help="directory with train/val/test/probes.jsonl")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--keep-audio", nargs="*", default=sorted(COMMERCIAL_AUDIO_SUBSETS),
                    help="synthetic audio subsets to keep (default: both MIT sets)")
    args = ap.parse_args()

    keep_subsets = set(args.keep_audio)
    args.out.mkdir(parents=True, exist_ok=True)

    audit: dict[str, dict] = {}
    for split in ("train", "val", "test"):
        src = args.src / f"{split}.jsonl"
        if not src.is_file():
            raise SystemExit(f"missing {src}")
        rows = []
        dropped = Counter()
        for line in src.open(encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            if keep(row, keep_subsets):
                row["state"].setdefault("stream", "text")
                rows.append(row)
            else:
                meta = row.get("_meta") or {}
                dropped[meta.get("subset") or meta.get("source") or "unknown"] += 1
        path = args.out / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        streams = Counter((r.get("state") or {}).get("stream", "text") for r in rows)
        labels = Counter(r["gold"]["action"] for r in rows)
        audit[split] = {
            "rows": len(rows),
            "streams": dict(streams),
            "labels": dict(labels),
            "dropped": dict(dropped),
        }
        print(f"[commercial] {split}: {len(rows)} rows, streams={dict(streams)}, "
              f"dropped={dict(dropped)}")
        print(f"[commercial] {split} labels: {dict(labels)}")

    # Text probes are MIT; keep them as a boundary check.
    probes = args.src / "probes.jsonl"
    if probes.is_file():
        rows = [json.loads(line) for line in probes.open(encoding="utf-8") if line.strip()]
        rows = [r for r in rows
                if keep(r, keep_subsets)
                or (r.get("state") or {}).get("stream") != "audio"]
        with (args.out / "probes.jsonl").open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[commercial] probes: {len(rows)} rows")

    (args.out / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[commercial] wrote {args.out}")


if __name__ == "__main__":
    main()
