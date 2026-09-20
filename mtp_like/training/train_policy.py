"""M2 — fine-tune the MTP-like policy companion with Unsloth (SPEC 6).

Trains Qwen3.5-0.8B (text-only) on the prepped policy dataset
(/mnt/f/duplex_cascade_runs/mtp_like/prepped_policy) to predict the duplex protocol tag at every item
boundary (SPEC 5): labels are masked (-100) everywhere except item
boundaries, where the target is one of the 6 duplex tags or the label-only
<|no tag|> token.

Stack:
  * unsloth (FastLanguageModel + UnslothTrainer), LoRA r=32 on all linears
    plus modules_to_save=[embed_tokens, lm_head] so the 8 new special-token
    embeddings are trained (embedding_learning_rate).
  * bf16, NO gradient checkpointing: measured on the 4080, unsloth GC +
    4-bit are ~10x slower on the Qwen3.5 GatedDeltaNet layers
    (~300 tok/s vs ~3100 tok/s for bf16 LoRA without GC).
  * Weighted CE (custom trainer): every boundary label carries its per-tag
    weight from data.common + the retuned POLICY_WEIGHTS (SPEC 5.3c),
    <user interruption>=5, ...) while <|no tag|> (the dominant EMPTY class)
    carries --no-tag-weight (default 0.3). This forces the model to learn
    the rare but critical controller actions (barge-in -> interruption).
  * Eval: restricted-argmax over the protocol vocabulary (no_tag + 6 tags)
    at boundary positions -> per-tag P/R/F1 (watch eval_user_interruption_f1).

The GatedDeltaNet fast path needs `fla` + `causal-conv1d` compiled against
torch 2.11/cu130 + CUDA 13.0 (pip-installed nvidia-cuda-nvcc, see
LEARNINGS/notes). If the libs are missing the model still runs (torch
fallback) but ~50x slower.

Usage (smoke -> full):
  python training/train_policy.py --dataset /mnt/f/duplex_cascade_runs/mtp_like/prepped_policy \\
      --max-steps 300 --limit 4000 --out-dir /mnt/f/duplex_cascade_runs/mtp_like/runs/smoke
  python training/train_policy.py --dataset /mnt/f/duplex_cascade_runs/mtp_like/prepped_policy \\
      --max-steps 4000 --out-dir /mnt/f/duplex_cascade_runs/mtp_like/runs/final
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import AutoTokenizer, EvalPrediction, TrainingArguments

# unsloth MUST be imported before transformers/trl/peft (it patches them)
from unsloth import FastLanguageModel, UnslothTrainer, unsloth_save_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.common import NO_TAG, SPECIAL_TOKENS, TOKEN

# Retuned class weights for the 7-way decision (SPEC 5.3c). The
# paper's weights (finish=10, interruption=5, ...) over-fire the rare heavy
# classes when the ENTIRE signal is the tag (measured: 879 false "finish"
# vs 72 true on the eval split). Weights here balance decision precision.
POLICY_WEIGHTS = {
    "user_is_speaking": 2,
    "user_finish_speaking": 3,
    "user_interruption": 6,
    "user_backchannel": 3,
    "user_is_thinking": 2,
    "system_backchannel": 3,
    "system_take_floor": 5,
    "system_handover": 3,
}
NO_TAG_WEIGHT = 0.5

# protocol vocabulary order used for restricted-argmax decoding / eval
TAG_VOCAB = ["no_tag"] + list(POLICY_WEIGHTS)


class PolicyCollator:
    """Pad tensors to the batch max (capped at max_seq_length, tail kept)."""

    def __init__(self, tokenizer, max_seq_length: int):
        self.pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        self.max_seq_length = max_seq_length

    def __call__(self, features):
        keys = [k for k in features[0] if k not in ("dialogue_id", "kind")]
        max_len = min(
            self.max_seq_length,
            max(len(f["input_ids"]) for f in features),
        )
        out = {}
        for k in keys:
            vals = [torch.as_tensor(f[k]) for f in features]
            if vals[0].dim() == 1:
                pad = torch.full(
                    (len(vals), max_len),
                    -100 if k == "labels" else (self.pad_id if k == "input_ids" else 0),
                    dtype=vals[0].dtype,
                )
                for i, v in enumerate(vals):
                    # keep the TAIL when truncating: the boundary label is
                    # the LAST position of the record
                    t = v[-max_len:]
                    pad[i, : len(t)] = t
                out[k] = pad
            else:
                out[k] = torch.stack(vals)
        return out


class PolicyTrainer(UnslothTrainer):
    """Unsloth trainer with per-token label weights (tags via POLICY_WEIGHTS,
    no_tag via NO_TAG_WEIGHT, -100 masked). CE over the full vocab at
    boundary positions only, normalized by the supervised-token count.

    SDFT/distill option (--kl-weight > 0, ported from the distill project):
    at non-boundary (content) positions the student's distribution is pinned
    to the FROZEN BASE via forward-KL over the teacher's top-k support —
    teacher = the same weights with the LoRA adapters disabled (no second
    model, no extra VRAM). This preserves the companion's language /
    instruction-following capability, which the promptable-controller rules
    depend on."""

    kl_weight: float = 0.0
    teacher_temperature: float = 1.0
    top_k: int = 32

    def __init__(self, *args, label_weights: torch.Tensor | None = None, **kwargs):
        # MUST be set before super().__init__: UnslothSFTTrainer.__init__
        # calls _prepare_dataset DURING construction (it silently replaces
        # the passed data_collator with DataCollatorForSeq2Seq whenever the
        # dataset has a "labels" column — dropping loss_weight and removing
        # the max-seq-length cap, which caused VRAM spikes and the WSL2 dxg
        # host-memory OOM). We re-force the real collator after the pass.
        self._real_collator = kwargs.get("data_collator")
        super().__init__(*args, **kwargs)
        self._label_weights = (
            label_weights.cuda() if label_weights is not None else None
        )

    def _prepare_dataset(self, dataset, processing_class, args, packing,
                         formatting_func, dataset_name):
        ds = super()._prepare_dataset(
            dataset, processing_class, args, packing,
            formatting_func, dataset_name,
        )
        if self._real_collator is not None:
            self.data_collator = self._real_collator
        return ds

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
    ):
        labels = inputs.pop("labels")
        input_ids = inputs.pop("input_ids")
        attention_mask = inputs.pop("attention_mask")
        loss_weight = inputs.pop("loss_weight", None)
        V = model.config.vocab_size

        # ---- teacher forward: frozen base (adapters disabled) ----
        t_val = t_idx = None
        if self.kl_weight > 0:
            with torch.no_grad(), model.disable_adapter():
                t_logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                ).logits
                t_val, t_idx = t_logits.topk(self.top_k, dim=-1)  # [B,T,k]
                t_val = t_val / self.teacher_temperature
                del t_logits

        # ---- student forward ----
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        logits = outputs.logits  # [B, T, V]; logits[t] predicts labels[t]
        flat_logits = logits.reshape(-1, V)
        flat_labels = labels.reshape(-1)
        valid = flat_labels != -100
        safe = flat_labels.masked_fill(~valid, 0)

        ce = F.cross_entropy(flat_logits, safe, reduction="none")  # -log p(target)
        weights = self._label_weights[safe] * valid
        if loss_weight is not None:
            weights = weights * loss_weight.unsqueeze(1).expand_as(labels).reshape(-1)
        per_token = ce * weights
        if num_items_in_batch is None:
            num_items_in_batch = valid.sum()
        loss = per_token.sum() / num_items_in_batch.clamp(min=1)

        # ---- content positions: forward-KL vs the frozen base ----
        if self.kl_weight > 0 and t_val is not None:
            content = (~valid) & (attention_mask.reshape(-1) == 1)
            s_val = flat_logits.gather(-1, t_idx.reshape(-1, self.top_k))
            s_lse = torch.logsumexp(flat_logits, dim=-1)
            p = torch.softmax(t_val.reshape(-1, self.top_k), dim=-1)
            logq = s_val - s_lse.unsqueeze(-1)
            kl = -(p * logq).sum(dim=-1)
            if loss_weight is not None:
                kl = kl * loss_weight.unsqueeze(1).expand_as(labels).reshape(-1)
            loss = loss + self.kl_weight * kl[content].sum() / num_items_in_batch.clamp(min=1)
            self._last_kl_mean = kl[content].mean().item() if content.any() else 0.0

        with torch.no_grad():
            tag_ids = self.model._tag_ids
            sub = flat_logits[..., tag_ids]  # [n, K] restricted logits
            pred = sub.argmax(dim=-1)  # class index into tag_ids
            cls_of = torch.full(
                (flat_logits.size(-1),), -1, dtype=torch.long, device=flat_logits.device
            )
            for i, tid in enumerate(tag_ids.tolist()):
                cls_of[tid] = i
            true_cls = cls_of[safe]
            hit = (pred == true_cls) & valid
            acc = hit.float().sum() / valid.sum().clamp(min=1)
            self._boundary_acc = acc.item()
        if not return_outputs:
            return loss
        return loss, outputs

    def log(self, logs, start_time=None):
        acc = getattr(self, "_boundary_acc", None)
        if acc is not None:
            logs["boundary_acc"] = acc
        kl = getattr(self, "_last_kl_mean", None)
        if kl is not None:
            logs["kl_mean"] = kl
        super().log(logs, start_time)


def make_preprocess_logits(tag_ids: list[int]):
    """Restrict eval logits to the protocol vocabulary on GPU, so the CPU
    gather stays tiny ([B, T, 7] instead of [B, T, 248k])."""

    def preprocess(logits, labels):
        return logits[..., tag_ids]

    return preprocess


def make_compute_metrics(tag_ids: list[int], tag_names: list[str]):
    """Restricted-argmax per-tag P/R/F1 over the protocol vocabulary."""

    def compute_metrics(eval_pred: EvalPrediction) -> dict:
        logits, labels = eval_pred.predictions, eval_pred.label_ids
        if isinstance(logits, tuple):
            logits = logits[0]
        logits = torch.as_tensor(logits)
        labels = torch.as_tensor(labels)
        # prediction_step already restricted logits to the tag columns
        if logits.shape[-1] == len(tag_ids):
            pred = logits.argmax(dim=-1)  # class index directly
        else:
            pred = logits[..., tag_ids].argmax(dim=-1)
        valid = labels != -100
        # map vocab ids -> class index (labels are vocab ids); size by the
        # highest protocol-token id (labels can only be tag ids or -100)
        cls_of = torch.full((max(tag_ids) + 1,), -1, dtype=torch.long)
        for i, tid in enumerate(tag_ids):
            cls_of[tid] = i
        true_cls = cls_of[labels.clamp(min=0)]
        mask = valid & (true_cls >= 0)
        n = mask.sum().item()
        if n == 0:
            return {"eval_acc": 0.0}
        match = (pred == true_cls) & mask
        acc = match.float().sum().item() / n
        metrics = {"eval_acc": acc, "eval_n": n}
        for i, name in enumerate(tag_names):
            tp = ((pred == i) & (true_cls == i) & mask).sum().item()
            fp = ((pred == i) & (true_cls != i) & mask).sum().item()
            fn = ((pred != i) & (true_cls == i) & mask).sum().item()
            p = tp / (tp + fp) if tp + fp else 0.0
            r = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * p * r / (p + r) if p + r else 0.0
            metrics[f"eval_{name}_p"] = p
            metrics[f"eval_{name}_r"] = r
            metrics[f"eval_{name}_f1"] = f1
        return metrics

    return compute_metrics


def lazy_init_special_embeddings(model, n_specials: int, sigma: float = 0.02) -> None:
    """Gaussian-init the rows of the newly added special tokens."""
    emb = model.get_input_embeddings()
    start = emb.weight.size(0) - n_specials
    with torch.no_grad():
        emb.weight.data[start:] = torch.randn_like(emb.weight.data[start:]) * sigma
    print(f"[train] special embeddings (last {n_specials} rows) gaussian-init sigma={sigma}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="/mnt/f/duplex_cascade_runs/mtp_like/prepped_policy")
    p.add_argument("--model", default="principled-intelligence/Qwen3.5-0.8B-text-only",
                   help="text-only Qwen3.5-0.8B (Qwen3_5ForCausalLM)")
    p.add_argument("--continue-from", default=None,
                   help="merged model dir to continue from (instead of the base)")
    p.add_argument("--out-dir", default="/mnt/f/duplex_cascade_runs/mtp_like/runs/policy")
    p.add_argument("--max-seq-length", type=int, default=1024)
    p.add_argument("--per-device-batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--embedding-lr", type=float, default=3e-4)
    p.add_argument("--r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--no-tag-weight", type=float, default=0.5)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--eval-steps", type=int, default=None,
                   help="eval frequency (default: same as --logging-steps)")
    p.add_argument("--save-steps", type=int, default=0, help="0 = save only at the end")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None, help="cap train rows (smoke)")
    p.add_argument("--eval-limit", type=int, default=128)
    # SDFT/distill options (ported from the distill project)
    p.add_argument("--kl-weight", type=float, default=0.3,
                   help="lambda: content forward-KL weight vs tag CE (0 disables distill)")
    p.add_argument("--teacher-temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=32,
                   help="teacher top-k next tokens used as the KL support")
    args = p.parse_args()

    t0 = time.time()

    ds = load_from_disk(args.dataset)
    train = ds["train"]
    if args.limit:
        train = train.select(range(min(args.limit, len(train))))
    eval_ds = ds["eval"].select(range(min(args.eval_limit, len(ds["eval"]))))
    print(f"[train] {len(train)} train / {len(eval_ds)} eval records")

    tokenizer = AutoTokenizer.from_pretrained(args.model, token=False)
    n_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": SPECIAL_TOKENS + [NO_TAG]}
    )
    print(f"[train] added {n_added} special tokens; vocab now {len(tokenizer)}")

    load_from = args.continue_from or args.model
    model, _tk = FastLanguageModel.from_pretrained(
        model_name=load_from,
        max_seq_length=args.max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=False,  # measured ~10x faster than 4-bit on GDN layers
        use_gradient_checkpointing=False,  # unsloth GC is slow on GDN layers
        token=False,
    )
    print(f"[train] loaded {type(model).__name__} from {load_from}")

    # model vocab 248320 (padded) >= tokenizer 248077+8 -> resize truncates
    # the unused tail rows (sft-proven) so the 8 new tokens own the last
    # rows, which are then gaussian-initialized
    model.resize_token_embeddings(len(tokenizer))
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        modules_to_save=["embed_tokens", "lm_head"],
        random_state=args.seed,
    )
    for _pc in model.peft_config.values():
        _pc.ensure_weight_tying = True
    if n_added > 0:
        lazy_init_special_embeddings(model, n_added)

    # label-id -> loss weight lookup over the full vocab
    label_weights = torch.zeros(len(tokenizer), dtype=torch.float32)
    for key, weight in POLICY_WEIGHTS.items():
        tid = tokenizer.encode(TOKEN[key], add_special_tokens=False)[0]
        label_weights[tid] = weight
    label_weights[tokenizer.encode(NO_TAG, add_special_tokens=False)[0]] = args.no_tag_weight
    tag_ids = [tokenizer.encode(NO_TAG, add_special_tokens=False)[0]] + [
        tokenizer.encode(TOKEN[key], add_special_tokens=False)[0]
        for key in list(POLICY_WEIGHTS)
    ]
    model._tag_ids = torch.tensor(tag_ids, dtype=torch.long, device="cuda")

    sft_cfg = TrainingArguments(
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
        eval_steps=args.eval_steps or max(1, args.logging_steps),
        seed=args.seed,
        fp16=False,
        bf16=True,
        weight_decay=0.0,
        max_grad_norm=1.0,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=0,
        dataloader_drop_last=False,
    )
    sft_cfg.embedding_learning_rate = args.embedding_lr

    trainer = PolicyTrainer(
        model=model,
        args=sft_cfg,
        processing_class=tokenizer,
        train_dataset=train,
        eval_dataset=eval_ds,
        data_collator=PolicyCollator(tokenizer, args.max_seq_length),
        compute_metrics=make_compute_metrics(tag_ids, TAG_VOCAB),
        preprocess_logits_for_metrics=make_preprocess_logits(tag_ids),
        label_weights=label_weights,
    )
    trainer.kl_weight = args.kl_weight
    trainer.teacher_temperature = args.teacher_temperature
    trainer.top_k = args.top_k

    print(f"[train] starting {args.max_steps} steps (lr={args.lr}, "
          f"embed lr={args.embedding_lr}, no_tag weight={args.no_tag_weight})")
    trainer.train()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = out_dir / "adapter"
    model.save_pretrained(str(adapter_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(adapter_dir))
    meta = {
        "tag_vocab": TAG_VOCAB,
        "tag_ids": {name: tid for name, tid in zip(TAG_VOCAB, tag_ids)},
        "special_tokens": SPECIAL_TOKENS + [NO_TAG],
        "no_tag_weight": args.no_tag_weight,
        "base_model": load_from,
    }
    (adapter_dir / "policy_cfg.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2)
    )
    print(f"[train] LoRA adapter + tokenizer + policy_cfg.json -> {adapter_dir}")
    print("[train] merge to a standalone bf16 model with "
          "training/export_policy.py --adapter ...")
    print(f"[train] done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()