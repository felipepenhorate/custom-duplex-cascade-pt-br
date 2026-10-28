"""Fine-tune Laya multilingual as the duplex companion.

Laya is a typed decision model, not a causal LM: each row is a state plus one
``choice`` question, and the model scores the protocol tags as options.  The
companion therefore trains with ``FastDecisionModel`` / ``DecisionTrainer``,
not with ``FastLanguageModel``.

Typical run (from the repository root, inside ``.venv-unsloth``)::

    python -m laya_approach.training.train_laya_companion \
        --data-dir laya_approach/data/generated \
        --out-dir laya_approach/runs/laya_companion_v1 \
        --max-train-rows 30000

The merged checkpoint is written to ``<out-dir>/merged`` and is directly
loadable with ``FastDecisionModel.from_pretrained``.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import unsloth  # noqa: F401  (must be imported before transformers)
import torch
from transformers import TrainingArguments
from unsloth import FastDecisionModel
from unsloth.models.decision import DecisionTrainer

from laya_approach.data.common import ACTION_KEYS, NO_TAG, NO_VOICE

DEFAULT_MODEL = "convaiinnovations/laya"
DEFAULT_SUBFOLDER = "multilingual"
DEFAULT_DATA = "laya_approach/data/generated"
DEFAULT_OUT = "laya_approach/runs/laya_companion_v1"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                row = json.loads(line)
                row["state"].setdefault("stream", "text")
                rows.append(row)
    return rows


def label_counts(rows: list[dict]) -> dict[str, int]:
    return dict(Counter(row["gold"]["action"] for row in rows))


def stratified_subsample(
    rows: list[dict],
    n: int,
    *,
    seed: int = 3407,
) -> list[dict]:
    """Take ``n`` rows while preserving the label quotas (largest remainder)."""
    if n <= 0 or n >= len(rows):
        return list(rows)
    by_label: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_label[row["gold"]["action"]].append(row)
    total = len(rows)
    exact = {label: len(items) * n / total for label, items in by_label.items()}
    take = {label: int(math.floor(value)) for label, value in exact.items()}
    left = n - sum(take.values())
    for label in sorted(exact, key=lambda k: exact[k] - take[k], reverse=True)[:left]:
        take[label] += 1
    rng = random.Random(seed)
    out: list[dict] = []
    for label, items in by_label.items():
        k = min(take[label], len(items))
        out.extend(rng.sample(items, k))
    rng.shuffle(out)
    return out


def describe(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "labels": label_counts(rows),
        "buckets": dict(Counter(r["_meta"].get("bucket", "?") for r in rows)),
        "sources": dict(Counter(r["_meta"].get("source", "?") for r in rows)),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def training_arguments(args: argparse.Namespace, output_dir: Path) -> TrainingArguments:
    kwargs = dict(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps if args.warmup_steps > 0 else args.warmup_ratio,
        lr_scheduler_type="cosine",
        weight_decay=args.weight_decay,
        max_grad_norm=1.0,
        logging_steps=args.log_every,
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        fp16=True,
        bf16=False,
        optim="adamw_torch",
        dataloader_num_workers=args.workers,
        dataloader_pin_memory=True,
        seed=args.seed,
        data_seed=args.seed,
    )
    if args.max_steps > 0:
        kwargs["warmup_steps"] = min(
            args.warmup_steps or max(1, int(args.max_steps * args.warmup_ratio)),
            max(1, args.max_steps // 20),
        )
    return TrainingArguments(**kwargs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=DEFAULT_DATA)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--subfolder", default=DEFAULT_SUBFOLDER)
    ap.add_argument("--max-seq-length", type=int, default=640)
    ap.add_argument("--max-train-rows", type=int, default=30000,
                    help="stratified cap; 0 uses every training row")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--head-lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--warmup-steps", type=int, default=0)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--calibrate", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--save-merged", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    train_rows = load_rows(data_dir / "train.jsonl")
    val_rows = load_rows(data_dir / "val.jsonl")
    if args.max_train_rows:
        train_rows = stratified_subsample(train_rows, args.max_train_rows, seed=args.seed)
    print(f"[laya-train] train rows: {len(train_rows):,}  val rows: {len(val_rows):,}")
    print(f"[laya-train] train labels: {label_counts(train_rows)}")

    print(f"[laya-train] loading {args.model} ({args.subfolder}) ...", flush=True)
    model, tokenizer = FastDecisionModel.from_pretrained(
        args.model,
        subfolder=args.subfolder,
        max_seq_length=args.max_seq_length,
        dtype=torch.float16,
    )
    cfg = model.decision_config
    print(f"[laya-train] max_len={cfg['max_len']} head_max_len={cfg['head_max_len']}")

    train_items, train_report = FastDecisionModel.build_dataset(train_rows, tokenizer, model)
    val_items, val_report = FastDecisionModel.build_dataset(val_rows, tokenizer, model)
    print(f"[laya-train] train decisions: {train_report}")
    print(f"[laya-train] val decisions:   {val_report}")
    if train_report["skipped"] or val_report["skipped"]:
        raise SystemExit("refusing to train with skipped decisions")
    if train_report["truncated"] or val_report["truncated"]:
        raise SystemExit("refusing to train with truncated states")

    model = FastDecisionModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules="all-linear",
        use_gradient_checkpointing="unsloth",
    )
    model = FastDecisionModel.for_training(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[laya-train] trainable parameters: {trainable / 1e6:.1f} M")

    targs = training_arguments(args, out_dir)
    trainer = DecisionTrainer(
        model=model,
        args=targs,
        train_dataset=train_items,
        eval_dataset=val_items,
        tokenizer=tokenizer,
        head_learning_rate=args.head_lr,
    )
    print("[laya-train] starting training", flush=True)
    train_started = time.time()
    trainer.train()
    train_seconds = time.time() - train_started
    print(f"[laya-train] training took {train_seconds / 60:.1f} min", flush=True)
    log_history = list(trainer.state.log_history)

    metrics: dict = {}
    if args.calibrate:
        print("[laya-train] calibrating on val ...", flush=True)
        metrics["val"] = FastDecisionModel.calibrate(model, tokenizer, val_items)
        print(f"[laya-train] val (calibrated): {metrics['val']}")

    metrics["train_seconds"] = round(train_seconds, 1)
    metrics["wall_seconds"] = round(time.time() - started, 1)
    metrics["trainable_params"] = trainable
    metrics["log_history"] = log_history
    metrics["train_report"] = train_report
    metrics["val_report"] = val_report
    metrics["train_data"] = describe(train_rows)
    metrics["val_data"] = describe(val_rows)
    metrics["labels"] = ACTION_KEYS
    metrics["no_tag"] = NO_TAG
    metrics["no_voice"] = NO_VOICE
    metrics["thresholds"] = {
        "source": "laya_approach.data.common.DEFAULT_THRESHOLDS",
    }
    metrics["model"] = args.model
    metrics["subfolder"] = args.subfolder
    metrics["max_seq_length"] = args.max_seq_length
    metrics["lora"] = {
        "r": args.lora_r,
        "alpha": args.lora_alpha,
        "dropout": args.lora_dropout,
    }
    metrics["args"] = vars(args)
    (out_dir / "train_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if args.save_merged:
        merged = out_dir / "merged"
        print(f"[laya-train] saving merged checkpoint -> {merged}", flush=True)
        model.save_pretrained_merged(str(merged), tokenizer)
        # The companion loads thresholds and the tag vocabulary from beside the
        # weights; keep them with the checkpoint so a policy cannot drift from
        # the data it was trained on.
        (merged / "policy_cfg.json").write_text(
            json.dumps(
                {
                    "tag_vocab": ACTION_KEYS,
                    "no_tag": NO_TAG,
                    "no_voice": NO_VOICE,
                    "max_len": model.decision_config.get("max_len"),
                    "head_max_len": model.decision_config.get("head_max_len"),
                    "fine_tuned": True,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[laya-train] done in {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
