"""Select a tag-stratified pilot from the generated decision corpus.

The dialogue sampler in ``prepare_laya_dataset`` is uniform over dialogues, but
the labels a dialogue produces are not: a few conversations carry most of the
``system_backchannel``, ``user_backchannel`` and barge-in rows.  A plain random
2k subset can therefore miss a tag family entirely.  This script reconstructs
the same dialogue split, maps each dialogue to the tags/buckets its decisions
produce, and greedily picks dialogues that cover the rare families while still
filling the common ones.

Output keeps the original dialogue objects and adds ``_pilot`` with the tag
counts, so the translated corpus is auditable.

    python -m laya_approach.data.select_pilot_dialogues \
        --generated-dir laya_approach/data/generated_v2 \
        --out laya_approach/data/duplexcascade_pilot/selected_dialogues.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from laya_approach.data.prepare_laya_dataset import (
    load_dialogues,
    split_dialogues,
)


def load_tags(generated_dir: Path, split: str) -> dict[str, dict]:
    rows = [
        json.loads(line)
        for line in (generated_dir / f"{split}.jsonl").open(encoding="utf-8")
        if line.strip()
    ]
    out: dict[str, dict] = defaultdict(lambda: {"labels": Counter(),
                                                 "buckets": Counter(),
                                                 "rows": 0})
    for row in rows:
        did = row["_meta"]["dialogue_id"]
        out[did]["labels"][row["gold"]["action"]] += 1
        out[did]["buckets"][row["_meta"].get("bucket", "?")] += 1
        out[did]["rows"] += 1
    return out


def select(
    dialogues: list[dict],
    tags: dict[str, dict],
    n: int,
    *,
    rng: random.Random,
) -> list[dict]:
    candidates = [d for d in dialogues if d["id"] in tags]
    if len(candidates) <= n:
        return candidates

    full_labels = Counter()
    full_buckets = Counter()
    for d in candidates:
        full_labels.update(tags[d["id"]]["labels"])
        full_buckets.update(tags[d["id"]]["buckets"])
    frac = n / len(candidates)
    target_labels = {k: max(1, round(v * frac)) for k, v in full_labels.items()}
    target_buckets = {k: max(1, round(v * frac)) for k, v in full_buckets.items()}
    weight_labels = {k: 1.0 / (1.0 + v) for k, v in target_labels.items()}
    weight_buckets = {k: 1.0 / (1.0 + v) for k, v in target_buckets.items()}

    selected: list[dict] = []
    chosen: set[str] = set()
    got_labels = Counter()
    got_buckets = Counter()

    for _ in range(n):
        best = None
        best_score = -1.0
        for d in candidates:
            did = d["id"]
            if did in chosen:
                continue
            t = tags[did]
            score = 0.0
            for k, v in t["labels"].items():
                rem = max(0, target_labels.get(k, 0) - got_labels[k])
                if rem:
                    score += min(v, rem) * weight_labels.get(k, 1.0)
            for k, v in t["buckets"].items():
                rem = max(0, target_buckets.get(k, 0) - got_buckets[k])
                if rem:
                    score += min(v, rem) * weight_buckets.get(k, 1.0)
            # A small nudge keeps ties from always going to the earliest id.
            score += rng.random() * 1e-6
            if score > best_score:
                best_score = score
                best = d
        if best is None:
            break
        selected.append(best)
        chosen.add(best["id"])
        got_labels.update(tags[best["id"]]["labels"])
        got_buckets.update(tags[best["id"]]["buckets"])

    # Fill with unrepresented dialogues if the coverage pass stopped early.
    if len(selected) < n:
        rest = [d for d in candidates if d["id"] not in chosen]
        selected.extend(rest[: n - len(selected)])

    print(f"  selected {len(selected)}/{len(candidates)} dialogues")
    for label in sorted(full_labels):
        print(f"    {label:24s} {got_labels[label]:6d} / {full_labels[label]:6d} "
              f"({got_labels[label] / max(1, full_labels[label]):.1%})")
    return selected


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dialogues", default=None)
    ap.add_argument("--generated-dir", default="laya_approach/data/generated_v2")
    ap.add_argument("--out", default="laya_approach/data/duplexcascade_pilot/selected_dialogues.jsonl")
    ap.add_argument("--train", type=int, default=2000)
    ap.add_argument("--val", type=int, default=200)
    ap.add_argument("--test", type=int, default=200)
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    from laya_approach.data.prepare_laya_dataset import resolve_dialogues

    dialogues = load_dialogues(resolve_dialogues(args.dialogues))
    splits = split_dialogues(
        dialogues, seed=args.seed, n_train=5000, n_val=400, n_test=400
    )
    generated = Path(args.generated_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with out.open("w", encoding="utf-8") as fh:
        for split, n in (("train", args.train), ("val", args.val), ("test", args.test)):
            print(f"[pilot] {split}: target {n}")
            tags = load_tags(generated, split)
            picked = select(
                splits[split], tags, n, rng=random.Random(args.seed + len(split))
            )
            for i, dialogue in enumerate(picked):
                t = tags[dialogue["id"]]
                record = dict(dialogue)
                record["_pilot"] = {
                    "split": split,
                    "labels": dict(t["labels"]),
                    "buckets": dict(t["buckets"]),
                    "n_rows": t["rows"],
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                total += 1
    print(f"[pilot] wrote {total} dialogues -> {out}")


if __name__ == "__main__":
    main()
