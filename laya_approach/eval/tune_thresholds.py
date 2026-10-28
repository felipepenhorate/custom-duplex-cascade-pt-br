"""Tune per-tag decision thresholds on a validation split.

The deployed policy takes the argmax tag and then applies a threshold plus the
turn-end guards.  This script sweeps one threshold at a time and reports the
F1-optimal value, with a separate high-precision operating point for
``user_finish_speaking`` (a false finish is the expensive error).

    python -m laya_approach.eval.tune_thresholds \
        --model laya_approach/runs/laya_companion_v1/merged \
        --data-dir laya_approach/data/generated --split val
"""

from __future__ import annotations

import argparse
import json
import random
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
    DEFAULT_THRESHOLDS,
    NO_TAG,
    resolve_action,
)


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    for row in rows:
        row["state"].setdefault("stream", "text")
    return rows


def prf(gold: list[str], pred: list[str], label: str) -> tuple[float, float, float]:
    tp = sum(g == label and p == label for g, p in zip(gold, pred))
    fp = sum(g != label and p == label for g, p in zip(gold, pred))
    fn = sum(g == label and p != label for g, p in zip(gold, pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data-dir", default="laya_approach/data/generated")
    ap.add_argument("--split", default="val", choices=("train", "val", "test"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-rows", type=int, default=0,
                    help="evaluate only the first N rows (smoke tests)")
    ap.add_argument("--write-cfg", action="store_true",
                    help="write the F1-optimal set into <model>/policy_cfg.json")
    ap.add_argument("--min-gain", type=float, default=5e-4,
                    help="minimum validation-accuracy gain needed to change a "
                         "default threshold (guards against noise)")
    args = ap.parse_args()

    rows = load_rows(Path(args.data_dir) / f"{args.split}.jsonl")
    if args.max_rows:
        rng = random.Random(3407)
        rows = rng.sample(rows, min(args.max_rows, len(rows)))
    model, tokenizer = FastDecisionModel.from_pretrained(
        args.model, max_seq_length=640, dtype=torch.float16
    )
    items, report = FastDecisionModel.build_dataset(rows, tokenizer, model)
    if report["skipped"]:
        raise SystemExit(report["reason"])
    logits, items = _decision_logits(model, tokenizer, items, args.batch_size)
    temps = _served_temperatures(model.decision_config, logits, items)

    probs: list[dict[str, float]] = []
    for z, item, temp in zip(logits, items, temps):
        row = rows[item["row"]]
        keys = _option_keys(_internal(row["questions"]["action"]))
        probs.append(dict(zip(keys, torch.softmax(z / temp, -1).tolist())))

    gold = [row["gold"]["action"] for row in rows]
    grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
    out: dict = {"split": args.split, "n": len(rows), "grid": grid.tolist(), "tags": {}}
    buckets = [row["_meta"].get("bucket", "?") for row in rows]
    final_buckets = ("final_with_punct", "final_no_punct", "empty_final")
    final_idx = [i for i, b in enumerate(buckets) if b in final_buckets]
    partial_punct_idx = [i for i, b in enumerate(buckets) if b == "partial_terminal_punct"]
    silence_mid_idx = [i for i, b in enumerate(buckets) if b == "silence_mid_turn"]

    for label in ACTION_KEYS:
        if label == NO_TAG:
            continue
        sweeps = []
        for threshold in grid:
            pred = [
                resolve_action(
                    prob, row["state"]["event"], row["state"],
                    {label: float(threshold)},
                )[0]
                for prob, row in zip(probs, rows)
            ]
            precision, recall, f1 = prf(gold, pred, label)
            entry = {
                "threshold": float(threshold),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
            }
            if label == "user_finish_speaking":
                # The trade-off that matters: answering finalized turns versus
                # taking the floor on an ASR partial or an intra-turn pause.
                entry["final_turn_recall"] = round(
                    sum(pred[i] == label for i in final_idx) / max(1, len(final_idx)), 4
                )
                entry["false_partial_punct"] = round(
                    sum(pred[i] == label for i in partial_punct_idx)
                    / max(1, len(partial_punct_idx)), 4
                )
                entry["false_silence_mid_turn"] = round(
                    sum(pred[i] == label for i in silence_mid_idx)
                    / max(1, len(silence_mid_idx)), 4
                )
            sweeps.append(entry)
        best_f1 = max(
            sweeps,
            # F1 ties are the norm once a tag is saturated; prefer the
            # threshold closest to the calibrated default instead of the
            # arbitrary low edge of the grid.
            key=lambda x: (
                x["f1"],
                x["precision"],
                -abs(x["threshold"] - DEFAULT_THRESHOLDS[label]),
            ),
        )
        high_precision = next(
            (x for x in sweeps if x["precision"] >= 0.98 and x["recall"] > 0),
            None,
        )
        out["tags"][label] = {
            "default": DEFAULT_THRESHOLDS[label],
            "best_f1": best_f1,
            "high_precision": high_precision,
            "sweeps": sweeps,
        }
        extra = ""
        if label == "user_finish_speaking":
            extra = (
                f"  final_recall={best_f1['final_turn_recall']:.3f}"
                f" false_partial={best_f1['false_partial_punct']:.3f}"
                f" false_mid_silence={best_f1['false_silence_mid_turn']:.3f}"
            )
        print(f"{label:24s} default={DEFAULT_THRESHOLDS[label]:.2f}  "
              f"best_f1={best_f1['threshold']:.2f} "
              f"(P={best_f1['precision']:.3f} R={best_f1['recall']:.3f} "
              f"F1={best_f1['f1']:.3f}){extra}")

    # Greedy combined pass: start from the calibrated defaults and adopt a
    # tag's F1-optimal threshold only when the whole validation accuracy
    # improves.  Per-tag F1 ignores how a veto interacts with the fallback
    # chain (a raised threshold can turn a correct ``user_is_speaking`` into
    # ``no_tag``), so the tag sweep alone is not a safe recommendation.
    def predict_with(thresholds: dict[str, float]) -> list[str]:
        return [
            resolve_action(prob, row["state"]["event"], row["state"], thresholds)[0]
            for prob, row in zip(probs, rows)
        ]

    default_pred = predict_with({})
    tuned = dict(DEFAULT_THRESHOLDS)
    best_accuracy = sum(g == p for g, p in zip(gold, default_pred)) / len(rows)
    for label, cfg in out["tags"].items():
        candidate = cfg["best_f1"]["threshold"]
        if candidate == tuned[label]:
            continue
        trial = {**tuned, label: float(candidate)}
        trial_pred = predict_with(trial)
        trial_accuracy = sum(g == p for g, p in zip(gold, trial_pred)) / len(rows)
        if trial_accuracy > best_accuracy + args.min_gain:
            tuned = trial
            best_accuracy = trial_accuracy
    for label in ACTION_KEYS:
        if label != NO_TAG and label not in out["tags"]:
            tuned[label] = DEFAULT_THRESHOLDS[label]

    for name, thresholds in (("default", {}), ("tuned", tuned)):
        pred = predict_with(thresholds)
        out[f"accuracy_{name}"] = round(
            sum(g == p for g, p in zip(gold, pred)) / len(rows), 4
        )
        out[f"labels_{name}"] = {
            label: dict(zip(("precision", "recall", "f1"), map(
                lambda v: round(v, 4), prf(gold, pred, label)
            )))
            for label in ACTION_KEYS
        }

    out_path = Path(args.out) if args.out else (
        Path(args.model).parent / f"thresholds_{args.split}.json"
    )
    out["recommended"] = tuned
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.write_cfg:
        cfg_path = Path(args.model) / "policy_cfg.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
        cfg["thresholds"] = tuned
        cfg["thresholds_source"] = {
            "split": args.split,
            "n": len(rows),
            "accuracy_default": out["accuracy_default"],
            "accuracy_tuned": out["accuracy_tuned"],
            "sweep": out_path.name if out_path.parent == Path(args.model) else str(out_path),
        }
        cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote thresholds into {cfg_path}")
    print(f"accuracy default={out['accuracy_default']:.4f} "
          f"tuned={out['accuracy_tuned']:.4f}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
