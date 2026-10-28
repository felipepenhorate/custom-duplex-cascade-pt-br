"""Evaluate a trained Laya companion.

Reports per-label precision/recall/F1, the punctuation failure buckets that
motivated the approach, and the hand-written probes.  Both the raw argmax and
the guarded decision (thresholds + turn-end guard) are reported, because the
deployed policy applies the guards.

    python -m laya_approach.eval.evaluate_laya \
        --model laya_approach/runs/laya_companion_v1/merged \
        --data-dir laya_approach/data/generated --split test
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import unsloth  # noqa: F401
import numpy as np
import torch
from unsloth import FastDecisionModel
from unsloth.models.decision import _decision_logits
from unsloth.models._decision_common import (
    _internal,
    _option_keys,
    _served_temperatures,
)

from laya_approach.data.common import (
    ACTION_KEYS,
    NO_TAG,
    event_bucket,
    resolve_action,
)


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    for row in rows:
        row["state"].setdefault("stream", "text")
    return rows


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def label_metrics(gold: list[str], pred: list[str]) -> dict:
    labels = [x for x in ACTION_KEYS if x in set(gold) | set(pred)]
    out = {}
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(gold, pred))
        fp = sum(g != label and p == label for g, p in zip(gold, pred))
        fn = sum(g == label and p != label for g, p in zip(gold, pred))
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        out[label] = {
            "support": sum(g == label for g in gold),
            "predicted": sum(p == label for p in pred),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
    return out


def confusion(gold: list[str], pred: list[str]) -> dict:
    matrix: dict[str, Counter] = defaultdict(Counter)
    for g, p in zip(gold, pred):
        matrix[g][p] += 1
    return {g: dict(c) for g, c in sorted(matrix.items())}


def bucket_metrics(rows: list[dict], pred: list[str], guarded: list[str]) -> dict:
    buckets: dict[str, dict] = {}
    for row, raw, guard in zip(rows, pred, guarded):
        bucket = row["_meta"].get("bucket", "?")
        entry = buckets.setdefault(
            bucket, {"n": 0, "correct_raw": 0, "correct_guarded": 0,
                     "gold": Counter(), "pred_raw": Counter(), "pred_guarded": Counter()}
        )
        gold = row["gold"]["action"]
        entry["n"] += 1
        entry["gold"][gold] += 1
        entry["pred_raw"][raw] += 1
        entry["pred_guarded"][guard] += 1
        entry["correct_raw"] += int(raw == gold)
        entry["correct_guarded"] += int(guard == gold)
    for bucket, entry in buckets.items():
        entry["accuracy_raw"] = round(safe_div(entry["correct_raw"], entry["n"]), 4)
        entry["accuracy_guarded"] = round(safe_div(entry["correct_guarded"], entry["n"]), 4)
    return dict(sorted(buckets.items()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--subfolder", default=None)
    ap.add_argument("--data-dir", default="laya_approach/data/generated")
    ap.add_argument("--split", default="test", choices=("train", "val", "test", "probes"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--guarded", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--thresholds", default=None,
                    help="JSON file with per-tag overrides")
    ap.add_argument("--show-probes", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    rows = load_rows(data_dir / f"{args.split}.jsonl")
    if args.max_rows:
        rows = rows[: args.max_rows]

    kwargs = {"max_seq_length": 640, "dtype": torch.float16}
    if args.subfolder:
        kwargs["subfolder"] = args.subfolder
    model, tokenizer = FastDecisionModel.from_pretrained(args.model, **kwargs)
    thresholds = _load_thresholds(args.thresholds)
    if not thresholds:
        # Mirror the runtime: LayaPolicy reads thresholds from the checkpoint.
        thresholds = _load_thresholds(str(Path(args.model) / "policy_cfg.json"))
    items, report = FastDecisionModel.build_dataset(rows, tokenizer, model)
    if report["skipped"]:
        raise SystemExit(f"skipped decisions: {report['reason']}")
    print(f"[laya-eval] {args.split}: {len(rows):,} rows, {report}")

    logits, items = _decision_logits(model, tokenizer, items, args.batch_size)
    temps = _served_temperatures(model.decision_config, logits, items)

    raw_pred: list[str] = []
    guarded_pred: list[str] = []
    probs_by_row: dict[int, dict[str, float]] = {}
    for z, item, temp in zip(logits, items, temps):
        row = rows[item["row"]]
        keys = _option_keys(_internal(row["questions"]["action"]))
        probs = torch.softmax(z / temp, -1).tolist()
        probs = dict(zip(keys, probs))
        state = row["state"]
        event = state["event"]
        raw = max(probs, key=probs.__getitem__)
        guard, _ = resolve_action(probs, event, state, thresholds)
        raw_pred.append(raw)
        guarded_pred.append(guard)
        probs_by_row[item["row"]] = probs

    gold = [row["gold"]["action"] for row in rows]
    result = {
        "model": args.model,
        "split": args.split,
        "n": len(rows),
        "build_report": report,
        "accuracy_raw": round(safe_div(sum(g == p for g, p in zip(gold, raw_pred)), len(rows)), 4),
        "accuracy_guarded": round(
            safe_div(sum(g == p for g, p in zip(gold, guarded_pred)), len(rows)), 4
        ),
        "labels": label_metrics(gold, raw_pred),
        "labels_guarded": label_metrics(gold, guarded_pred),
        "confusion_raw": confusion(gold, raw_pred),
        "buckets": bucket_metrics(rows, raw_pred, guarded_pred),
    }

    # The specific failure the approach exists to remove.
    result["critical"] = critical_rates(rows, raw_pred, guarded_pred)

    print(f"[laya-eval] accuracy raw={result['accuracy_raw']:.4f} "
          f"guarded={result['accuracy_guarded']:.4f}")
    print("[laya-eval] critical:")
    for name, value in result["critical"].items():
        print(f"  {name:58s} {value:.4f}")

    if args.show_probes and args.split == "probes":
        result["probe_rows"] = []
        for i, row in enumerate(rows):
            probs = probs_by_row.get(i, {})
            top = sorted(probs.items(), key=lambda kv: -kv[1])[:3]
            result["probe_rows"].append(
                {
                    "note": row["_meta"].get("note"),
                    "bucket": row["_meta"].get("bucket"),
                    "gold": gold[i],
                    "pred": raw_pred[i],
                    "guarded": guarded_pred[i],
                    "top": [[k, round(v, 4)] for k, v in top],
                }
            )
        print("[laya-eval] probes:")
        for probe in result["probe_rows"]:
            mark = "OK " if probe["pred"] == probe["gold"] else "MISS"
            top = ", ".join(f"{k}={v:.2f}" for k, v in probe["top"])
            print(f"  {mark} {probe['note']}: gold={probe['gold']} pred={probe['pred']} ({top})")

    out = Path(args.out) if args.out else (
        data_dir.parent.parent / "runs" / f"eval_{args.split}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[laya-eval] wrote {out}")


def _load_thresholds(path: str | None) -> dict[str, float]:
    if not path:
        return {}
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    # Accept the three shapes this project writes: a flat {tag: threshold}
    # file, a tuning report ({"recommended": ...}), and a merged checkpoint's
    # policy_cfg.json ({"thresholds": ...}).
    for key in ("recommended", "thresholds"):
        if isinstance(cfg.get(key), dict):
            cfg = cfg[key]
            break
    return {str(tag): float(value) for tag, value in cfg.items()
            if isinstance(value, (int, float))}


def _bucket_rate(
    rows: list[dict],
    pred: list[str],
    bucket: str,
    label: str,
    over: str = "bucket",
) -> float:
    """Rate of ``pred == label`` inside one bucket.

    ``over="bucket"`` divides by every row in the bucket: that is the false
    positive/finish rate.  ``over="gold"`` divides only by the rows whose gold
    label is ``label``: that is recall.  Buckets such as ``final_during_system``
    mix legitimate backchannels with barge-ins, so dividing by bucket size there
    under-reports interruption recall by half.
    """
    indices = [i for i, row in enumerate(rows) if row["_meta"].get("bucket") == bucket]
    if over == "gold":
        indices = [i for i in indices if rows[i]["gold"]["action"] == label]
    if not indices:
        return 0.0
    return round(safe_div(sum(pred[i] == label for i in indices), len(indices)), 4)


# (name, bucket, expected label, denominator).  Raw and guarded are both
# reported: the raw rate is what the model learned, the guarded rate is what the
# deployed policy does, and the gap between them is exactly the work the
# turn-end guard does.
CRITICAL = (
    ("partial_punct_false_finish", "partial_terminal_punct", "user_finish_speaking", "bucket"),
    ("final_mid_turn_false_finish", "final_mid_turn", "user_finish_speaking", "bucket"),
    ("final_punct_finish_recall", "final_with_punct", "user_finish_speaking", "gold"),
    ("final_no_punct_finish_recall", "final_no_punct", "user_finish_speaking", "gold"),
    ("empty_final_finish_recall", "empty_final", "user_finish_speaking", "gold"),
    ("silence_after_final_finish_recall", "silence_after_final", "user_finish_speaking", "gold"),
    ("partial_during_system_interruption_recall", "partial_during_system", "user_interruption", "gold"),
    ("partial_punct_during_system_interruption_recall", "partial_during_system_punct", "user_interruption", "gold"),
    ("final_during_system_interruption_recall", "final_during_system", "user_interruption", "gold"),
    ("backchannel_recall", "backchannel_during_system", "user_backchannel", "gold"),
    ("silence_during_system_no_tag_recall", "silence_during_system", NO_TAG, "gold"),
)


def critical_rates(rows: list[dict], raw: list[str], guarded: list[str]) -> dict[str, float]:
    present = {row["_meta"].get("bucket") for row in rows}
    out: dict[str, float] = {}
    for name, bucket, label, over in CRITICAL:
        if bucket not in present:
            continue
        out[f"{name}_raw"] = _bucket_rate(rows, raw, bucket, label, over)
        out[f"{name}_guarded"] = _bucket_rate(rows, guarded, bucket, label, over)
    return out


if __name__ == "__main__":
    main()
