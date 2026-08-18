"""M2 - QLoRA fine-tuning of the duplex micro-turn model (DuplexCascade-PT).

Trains the pt-BR duplex model on the prepped dataset produced by
training/prep_dataset.py using Unsloth (4-bit QLoRA, gradient checkpointing).
Loss follows the paper (SPEC 6.3): next-token CE is computed ONLY on system
micro-turn positions (user tokens are labels=-100) and each predicted token
carries its per-special-token weight from data/common.TOKEN_WEIGHT
(<user finish speaking>=10, <user interruption>=5, ...).

The 7 conversational special tokens are appended to the tokenizer, their
embedding rows Gaussian-initialized (sigma=0.02) and fine-tuned via
embedding_learning_rate. Adapter + optional 16-bit merged export + the
train_cfg.json expected by DuplexCascade/server.py are saved to --out-dir.

Usage (smoke -> full):
  python training/train_qlora.py --dataset data/prepped_smoke --max-steps 100 \
      --per-device-batch-size 2 --grad-accum 8 --out-dir training/runs/smoke
  python training/train_qlora.py --dataset data/prepped --max-steps 5000 \
      --out-dir training/runs/final --export-merged
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# unsloth MUST be imported before transformers/trl/peft (it patches them)
try:
    from unsloth import FastLanguageModel, UnslothTrainer
    from trl import SFTConfig
    from trl.trainer.sft_trainer import selective_log_softmax
except ImportError as e:  # pragma: no cover
    print(f"[train] missing dependency: {e}")
    print("[train] run from the unsloth venv: /home/penhfel/unsloth_uv/bin/python")
    sys.exit(1)

import torch
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.common import SPECIAL_TOKENS, TOKEN_WEIGHT

# paper: user_is_speaking=1, finish=10, interruption=5, backchannel=2,
# thinking=1, system backchannel=3 (SPEC 6.3). zero weight = fully masked.
SPECIAL_IDS: dict[str, int] = {}


class DuplexSFTTrainer(UnslothTrainer):
    """Unsloth trainer with the paper's weighted/masked CE (SPEC 6.3).

    The prepped dataset's labels are ALREADY shifted (labels[i] = target of
    position i, -100 on user positions); weights[i] holds the loss weight of
    the predicted token. We override compute_loss to apply them per-token
    and normalize by the supervised-token count (num_items_in_batch),
    mirroring trl's dft_loss convention so HF grad-accum scaling stays exact.
    """

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
    ):
        labels = inputs.pop("labels")
        weights = inputs.pop("weights")
        input_ids = inputs.pop("input_ids")
        attention_mask = inputs.pop("attention_mask")

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )

        logits = outputs.logits  # [B, T, V], aligned: logits[t] predicts labels[t]
        flat_labels = labels.reshape(-1).to(logits.device)
        valid = flat_labels != -100
        flat_logits = logits.reshape(-1, logits.size(-1))
        safe_labels = flat_labels.masked_fill(~valid, 0)  # weights guard the rest
        logprobs = selective_log_softmax(flat_logits, safe_labels)
        flat_weights = weights.reshape(-1).to(logprobs.dtype)

        per_token = -logprobs * flat_weights  # weights are 0.0 on masked positions
        if num_items_in_batch is None:
            num_items_in_batch = valid.sum()
        loss = per_token.sum() / num_items_in_batch

        # cheap token accuracy for logging
        with torch.no_grad():
            preds = flat_logits.argmax(dim=-1)
            acc = ((preds == flat_labels) & valid).sum().float() / valid.sum().clamp(min=1)

        self._last_accuracy = acc.item()
        self._last_supervised_tokens = valid.sum().item()
        if not return_outputs:
            return loss
        return loss, outputs

    def training_step(self, *args, **kwargs):
        with self.maybe_activation_offload_context:
            return super().training_step(*args, **kwargs)

    def log(self, logs, start_time=None):
        acc = getattr(self, "_last_accuracy", None)
        if acc is not None:
            logs["supervised_token_accuracy"] = acc
        super().log(logs, start_time)


def _pad_batch(features):
    from transformers import BatchEncoding

    keys = list(features[0].keys())
    max_len = max(len(f["input_ids"]) for f in features)
    out: dict[str, torch.Tensor | list] = {}
    for k in keys:
        vals = [
            torch.as_tensor(f[k]) if hasattr(f[k], "__len__") and not isinstance(f[k], str) else f[k]
            for f in features
        ]
        if isinstance(vals[0], torch.Tensor) and vals[0].dim() >= 1:
            pad_dims = max_len if vals[0].dim() == 1 else vals[0].size(1)
            pad = torch.full(
                (len(vals), pad_dims), -100 if k == "labels" else 0, dtype=vals[0].dtype
            )
            for i, v in enumerate(vals):
                pad[i, : v.size(0)] = v
        elif isinstance(vals[0], torch.Tensor):
            pad = torch.stack(vals)
        else:
            out[k] = [str(v) for v in vals]
            continue
        out[k] = pad
    return BatchEncoding(out)


def lazy_init_special_embeddings(model, tokenizer, sigma: float = 0.02) -> None:
    """Gaussian-init the rows of the 7 new special tokens (SPEC 6.4)."""
    emb = model.get_input_embeddings()
    n = len(SPECIAL_TOKENS)
    start = emb.weight.size(0) - n
    with torch.no_grad():
        emb.weight.data[start:] = torch.randn_like(emb.weight.data[start:]) * sigma
    if model.config.tie_word_embeddings:
        model.get_output_embeddings().weight.data[start:] = emb.weight.data[start:]
    print(f"[train] special embeddings (last {n} rows) gaussian-init sigma={sigma}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="data/prepped")
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--out-dir", default="/mnt/f/duplex_cascade_runs/duplex-pt")
    p.add_argument("--max-seq-length", type=int, default=4096)
    p.add_argument("--per-device-batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--embedding-lr", type=float, default=3e-4)
    p.add_argument("--r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--logging-steps", type=int, default=5)
    p.add_argument("--save-steps", type=int, default=0, help="0 = save only at the end")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--export-merged", action="store_true",
                   help="also export a 16-bit merged model + train_cfg.json (SPEC 6.5)")
    p.add_argument("--limit", type=int, default=None, help="cap train rows (smoke)")
    p.add_argument("--eval-limit", type=int, default=16)
    args = p.parse_args()

    t0 = time.time()

    ds = load_from_disk(args.dataset)
    train = ds["train"]
    if args.limit:
        train = train.select(range(min(args.limit, len(train))))
    eval_ds = ds["eval"].select(range(min(args.eval_limit, len(ds["eval"]))))

    print(f"[train] {len(train)} train / {len(eval_ds)} eval sequences")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=False)
    n_added = tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    print(f"[train] added {n_added} special tokens; vocab = {len(tokenizer)}")

    model, _tk = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
        trust_remote_code=False,
    )
    model.resize_token_embeddings(len(tokenizer))
    # Only gaussian-init the special-token rows when they are brand new (fresh
    # base, n_added > 0). When continuing from a merged model the specials
    # already exist and their embeddings were trained - re-randomizing them
    # would wipe the learned turn-taking behavior.
    if n_added > 0:
        lazy_init_special_embeddings(model, tokenizer)

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        modules_to_save=["embed_tokens", "lm_head"],
    )
    # embed_tokens & lm_head are weight-tied (tie_word_embeddings=True); the flag
    # makes the saved adapter mergeable by unsloth (SPEC 6.5 vocab/merge gotcha).
    for _pc in model.peft_config.values():
        _pc.ensure_weight_tying = True
    print("[train] PEFT applied (all linear, r=%d a=%d, dropout=0)" % (args.r, args.lora_alpha))

    for tid in SPECIAL_TOKENS:
        ids = tokenizer.encode(tid, add_special_tokens=False)
        assert len(ids) == 1, tid
        SPECIAL_IDS[tid] = ids[0]

    sft_cfg = SFTConfig(
        output_dir=str(Path(args.out_dir) / "adapter"),
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps if args.save_steps > 0 else None,
        save_strategy="no" if args.save_steps <= 0 else "steps",
        eval_strategy="steps",
        eval_steps=max(1, args.logging_steps),
        seed=args.seed,
        fp16=False,
        bf16=True,
        weight_decay=0.0,
        packing=False,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=0,
        dataloader_drop_last=False,
        dataset_kwargs={"skip_prepare_dataset": True},
    )
    # unsloth's optimizer splits embedding params into their own LR group
    sft_cfg.embedding_learning_rate = args.embedding_lr

    trainer = DuplexSFTTrainer(
        model=model,
        args=sft_cfg,
        processing_class=tokenizer,
        train_dataset=train,
        eval_dataset=eval_ds,
        data_collator=_pad_batch,
    )

    print(f"[train] starting {args.max_steps} steps (lr={args.lr}, embed lr={args.embedding_lr})")
    trainer.train()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out_dir / "adapter"))
    print(f"[train] adapter saved -> {out_dir / 'adapter'}")

    if args.export_merged:
        merged_dir = out_dir / "merged_16bit"
        FastLanguageModel.save_pretrained_merged(
            model, tokenizer, merged_dir, save_method="merged_16bit"
        )
        train_cfg = {
            "model": {"name": f"file://{merged_dir}", "trust_remote_code": False},
            "tokenizer": f"file://{merged_dir}",
            "special_tokens": SPECIAL_TOKENS,
            "token_weights": TOKEN_WEIGHT,
        }
        (merged_dir / "train_cfg.json").write_text(json.dumps(train_cfg, ensure_ascii=False, indent=2))
        print(f"[train] merged 16-bit export + train_cfg.json -> {merged_dir}")

    print(f"[train] done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()