"""M2 — DuplexCascade-Distill training: tags CE + content KL to the frozen base.

Fork of duplex_cascade/training/train_qlora.py implementing the SDFT-derived
loss of SPEC §4.3 (variant D2, on-the-fly teacher):

  L = Σ_{tags}   w_t · CE(student, y_t)                          (as today)
    + λ · Σ_{content}  D_KL(teacher(·|prefix) ∥ student(·|prefix))   (forward KL)

  (optional anchor rows, all positions are content -> pure KL on general
   prompts, pinning the LoRA delta to ~0 outside the duplex protocol)

The teacher is the SAME weights as the student minus the LoRA adapters, so a
second no_grad forward under `model.disable_adapters()` gives the frozen base
distribution with ZERO extra VRAM. The KL support is the teacher's top-k next
tokens (k = --top-k); student logits are gathered at those ids. Teacher logits
are released right after top-k/lse, keeping peak memory to one full [B,T,V].

Artifact masking (first content tokens after <|user finish speaking|>) is done
at prep time (--mask-first-tokens in prep_dataset.py), so this trainer needs no
extra masking.

Usage (smoke -> full):
  python training/train_distill.py --dataset data/prepped --max-steps 100 \\
      --per-device-batch-size 1 --grad-accum 8 --max-seq-length 2048 \\
      --out-dir /mnt/f/duplex_cascade_runs/distill/smoke \\
      --kl-weight 0.3 --teacher-temperature 1.0 --top-k 32
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
from datasets import load_from_disk
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.common import SPECIAL_TOKENS, TOKEN_WEIGHT

SPECIAL_IDS: list[int] = []


class DuplexDistillTrainer(UnslothTrainer):
    """Weighted/masked tag CE + forward-KL to the frozen base (SPEC §4.3)."""

    kl_weight: float = 0.3
    teacher_temperature: float = 1.0
    top_k: int = 32
    teacher_cache: tuple | None = None  # (topk_ids list, topk_logits list) or None

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
        # is_anchor is informational only: anchor rows have no special tokens,
        # so they are pure-content rows already handled by the content-KL term.
        inputs.pop("is_anchor", None)
        inputs.pop("cache_idx", None)

        B, T = input_ids.shape
        V = model.config.vocab_size
        special_t = torch.tensor(SPECIAL_IDS, dtype=torch.long, device=input_ids.device)

        if self.teacher_cache is None:
            # ---- teacher forward: frozen base (same weights, no adapters) ----
            with torch.no_grad(), model.disable_adapter():
                teacher_logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                ).logits
                t_val, t_idx = teacher_logits.topk(self.top_k, dim=-1)  # [B,T,k]
                del teacher_logits
        else:
            # ---- cached teacher top-k (precomputed, SPEC §5.3) ----
            t_idx = inputs.pop("topk_ids").to(input_ids.device)     # [B,T,k]
            t_val = inputs.pop("topk_logits").to(input_ids.device).to(torch.bfloat16)
        t_val, t_idx = t_val.reshape(-1, self.top_k), t_idx.reshape(-1, self.top_k)

        # ---- student forward ----
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        student_logits = outputs.logits  # [B,T,V]

        flat_labels = labels.reshape(-1)
        flat_weights = weights.reshape(-1).to(student_logits.dtype)
        valid = flat_labels != -100
        is_special = torch.isin(flat_labels, special_t)

        # tag positions: weighted CE to the target (identical to the original)
        safe_labels = flat_labels.masked_fill(~valid, 0)
        tag_logprobs = selective_log_softmax(
            student_logits.reshape(-1, V), safe_labels
        )
        tag_ce = (-(tag_logprobs * flat_weights))[valid & is_special]

        # content positions: forward-KL (CE form) vs teacher top-k support
        preds = student_logits.reshape(-1, V).argmax(dim=-1)  # for logging only
        s_val = student_logits.reshape(-1, V).gather(-1, t_idx)
        s_lse = torch.logsumexp(student_logits.reshape(-1, V), dim=-1)  # [B*T]
        del student_logits
        p_logits = t_val / self.teacher_temperature
        p = torch.softmax(p_logits, dim=-1)                       # teacher probs
        logq = s_val - s_lse.unsqueeze(-1)                        # student log-probs
        kl = -(p * logq).sum(dim=-1)                              # forward-KL CE
        del s_val, logq, p, p_logits, t_val, t_idx

        if num_items_in_batch is None:
            num_items_in_batch = valid.sum()
        loss = (tag_ce.sum() + self.kl_weight * kl[valid & ~is_special].sum()) / num_items_in_batch

        with torch.no_grad():
            tag_acc = ((preds == flat_labels) & is_special).sum().float() / \
                is_special.sum().clamp(min=1)
            self._last_tag_accuracy = tag_acc.item()
            self._last_kl_mean = kl[valid & ~is_special].mean().item() if (valid & ~is_special).any() else 0.0
            self._last_tag_ce_mean = tag_ce.mean().item() if len(tag_ce) else 0.0
            self._last_supervised_tokens = valid.sum().item()

        if not return_outputs:
            return loss
        return loss, outputs

    def training_step(self, *args, **kwargs):
        with self.maybe_activation_offload_context:
            return super().training_step(*args, **kwargs)

    def log(self, logs, start_time=None):
        for key in ("_last_tag_accuracy", "_last_kl_mean", "_last_tag_ce_mean"):
            val = getattr(self, key, None)
            if val is not None:
                logs[key.removeprefix("_last_")] = val
        super().log(logs, start_time)


class DistillCollator:
    """Pads 1-D fields to the batch max length; when a teacher cache is attached,
    also looks up and pads each row's [T, K] top-k arrays to [max_T, K]."""

    def __init__(self, teacher_cache: tuple | None = None):
        self.cache = teacher_cache  # (topk_ids list, topk_logits list) or None

    def __call__(self, features):
        from transformers import BatchEncoding

        keys = list(features[0].keys())
        max_len = max(len(f["input_ids"]) for f in features)
        out: dict[str, torch.Tensor | list] = {}
        for k in keys:
            if k == "cache_idx":
                continue  # used only for the cache lookup below
            vals = [
                torch.as_tensor(f[k]) if hasattr(f[k], "__len__") and not isinstance(f[k], str) else f[k]
                for f in features
            ]
            if isinstance(vals[0], torch.Tensor) and vals[0].dim() >= 1:
                pad_shape = (len(vals), max_len) + tuple(vals[0].size()[1:])
                pad_val = -100 if k == "labels" else 0
                pad = torch.full(pad_shape, pad_val, dtype=vals[0].dtype)
                for i, v in enumerate(vals):
                    pad[i, : v.size(0)] = v
                out[k] = pad
            elif all(isinstance(v, (int,)) for v in vals):
                out[k] = torch.tensor([int(v) for v in vals])
            elif isinstance(vals[0], torch.Tensor):
                out[k] = torch.stack(vals)
            else:
                out[k] = [str(v) for v in vals]
        if self.cache is not None:
            ids_l, logits_l = self.cache
            K = ids_l[0].shape[1]
            topk_ids = torch.zeros((len(features), max_len, K), dtype=torch.long)
            topk_logits = torch.zeros((len(features), max_len, K), dtype=torch.float16)
            for i, f in enumerate(features):
                ids = torch.as_tensor(ids_l[int(f["cache_idx"])])
                vals = torch.as_tensor(logits_l[int(f["cache_idx"])])
                Ti = ids.size(0)
                topk_ids[i, :Ti] = ids
                topk_logits[i, :Ti] = vals
            out["topk_ids"] = topk_ids
            out["topk_logits"] = topk_logits
        return BatchEncoding(out)


def lazy_init_special_embeddings(model, tokenizer, sigma: float = 0.02) -> None:
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
    p.add_argument("--out-dir", default="/mnt/f/duplex_cascade_runs/distill")
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
                   help="also export a 16-bit merged model + train_cfg.json")
    p.add_argument("--limit", type=int, default=None, help="cap train rows (smoke)")
    p.add_argument("--eval-limit", type=int, default=16)
    # SDFT/distill hyper-parameters (SPEC §4.3, §5.2)
    p.add_argument("--kl-weight", type=float, default=0.3,
                   help="lambda: weight of the content forward-KL term vs tag CE")
    p.add_argument("--teacher-temperature", type=float, default=1.0,
                   help="temperature applied to the teacher logits for the soft targets")
    p.add_argument("--top-k", type=int, default=32,
                   help="teacher top-k next tokens used as the KL support")
    p.add_argument("--teacher-cache", default=None,
                   help="dir with topk_ids.pkl/topk_logits.pkl + dataset/ (offline teacher; "
                        "skips the in-loop teacher forward, SPEC §5.3)")
    p.add_argument("--resume-from", default=None,
                   help="checkpoint dir to resume from (e.g. .../adapter/checkpoint-500)")
    p.add_argument("--no-eval", action="store_true",
                   help="disable in-loop eval (real evaluation happens separately in M3). "
                        "The periodic eval at the 16 GB ceiling triggers the "
                        "CUDA-allocator fragmentation spiral.")
    args = p.parse_args()

    t0 = time.time()

    ds = load_from_disk(args.dataset)
    train = ds["train"]
    if args.limit:
        train = train.select(range(min(args.limit, len(train))))
    eval_ds = ds["eval"].select(range(min(args.eval_limit, len(ds["eval"]))))

    print(f"[train] {len(train)} train / {len(eval_ds)} eval sequences")

    teacher_cache = None
    if args.teacher_cache:
        import pickle

        cache_dir = Path(args.teacher_cache)
        with (cache_dir / "topk_ids.pkl").open("rb") as f:
            ids_l = pickle.load(f)
        with (cache_dir / "topk_logits.pkl").open("rb") as f:
            logits_l = pickle.load(f)
        teacher_cache = (ids_l, logits_l)
        print(f"[train] teacher cache loaded: {len(ids_l)} sequences, K={ids_l[0].shape[1]}")

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
    for _pc in model.peft_config.values():
        _pc.ensure_weight_tying = True
    print("[train] PEFT applied (all linear, r=%d a=%d, dropout=0)" % (args.r, args.lora_alpha))

    for tid in SPECIAL_TOKENS:
        ids = tokenizer.encode(tid, add_special_tokens=False)
        assert len(ids) == 1, tid
        SPECIAL_IDS.append(ids[0])
    print(f"[train] special ids: {SPECIAL_IDS}")

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
        eval_strategy="no" if args.no_eval else "steps",
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
    sft_cfg.embedding_learning_rate = args.embedding_lr

    trainer = DuplexDistillTrainer(
        model=model,
        args=sft_cfg,
        processing_class=tokenizer,
        train_dataset=train,
        eval_dataset=None if args.no_eval else eval_ds,
        data_collator=DistillCollator(teacher_cache),
    )
    trainer.kl_weight = args.kl_weight
    trainer.teacher_temperature = args.teacher_temperature
    trainer.top_k = args.top_k
    trainer.teacher_cache = teacher_cache

    print(f"[train] starting {args.max_steps} steps (lr={args.lr}, embed lr={args.embedding_lr}, "
          f"kl_weight={args.kl_weight}, teacher_temp={args.teacher_temperature}, "
          f"top_k={args.top_k}, cache={'yes' if teacher_cache else 'no'})")
    trainer.train(resume_from_checkpoint=args.resume_from)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out_dir / "adapter"))
    print(f"[train] adapter saved -> {out_dir / 'adapter'}")

    if args.export_merged:
        from unsloth.save import unsloth_save_pretrained_merged

        merged_dir = out_dir / "merged_16bit"
        # modern unsloth API (>= 2026.8): module-level, model as first arg
        unsloth_save_pretrained_merged(
            model, str(merged_dir), tokenizer=tokenizer, save_method="merged_16bit"
        )
        train_cfg = {
            "model": {"name": f"file://{merged_dir}", "trust_remote_code": False},
            "tokenizer": f"file://{merged_dir}",
            "special_tokens": SPECIAL_TOKENS,
            "token_weights": TOKEN_WEIGHT,
            "distill": {
                "kl_weight": args.kl_weight,
                "teacher_temperature": args.teacher_temperature,
                "top_k": args.top_k,
                "teacher": "frozen-base-disable-adapters",
            },
        }
        (merged_dir / "train_cfg.json").write_text(json.dumps(train_cfg, ensure_ascii=False, indent=2))
        print(f"[train] merged 16-bit export + train_cfg.json -> {merged_dir}")

    print(f"[train] done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()