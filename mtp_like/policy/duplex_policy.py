"""M3 — the trained companion as a sidecar policy model (SPEC 4.3).

DuplexPolicy loads the merged bf16 companion, renders the duplex context
(items) into the ChatML stream used at training time, and returns the
protocol tag with calibrated probabilities. Thresholds decide whether a tag
fires; a tag that fires is emitted (added to the context), otherwise the
companion emits nothing (the "empty string not added to the context").

Guards (controller safety):
  * a tag only fires if P(tag) >= its threshold,
  * <|user interruption|> never fires when the current user item is
    <|no voice|> (the orchestrator knows it inserted the silence marker;
    silence must never hard-stop the assistant),
  * <|user finish speaking|> requires P >= its threshold (it is the
    answer-trigger; a false finish starts the assistant too early).

The duplex context is a list of items exactly like the training data:
  {"role": "user"|"assistant", "text": str}  (text may contain the tag
  tokens or <|no voice|>, and may be a PARTIAL item — no <|im_end|> —
  when the speaker is still producing it).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.common import (
    IM_END,
    IM_END_NL,
    IM_START_ASSISTANT,
    IM_START_SYSTEM,
    IM_START_USER,
    TOKEN,
)

# default per-tag decision thresholds (tuned on the eval split)
DEFAULT_THRESHOLDS = {
    "user_is_speaking": 0.35,
    "user_finish_speaking": 0.40,
    "user_interruption": 0.55,
    "user_backchannel": 0.60,
    "user_is_thinking": 0.35,
    "system_backchannel": 0.60,
    # the promptable-controller tags: the rule is only in the context when
    # it is configured, so false fires are contained; the barge-in must
    # fire reliably on the trigger (observed 0.86 with the normalized
    # digit stream; a dictation-like stream without letters rates ~0.40)
    "system_take_floor": 0.50,
    "system_handover": 0.30,
}


def render_items(items: list[dict]) -> str:
    """Render the duplex context to the training ChatML format.

    Matches the training distribution exactly (SPEC 5.4):
      * USER items are ALWAYS closed (<|im_end|>\n) — even when their
        content is a partial chunk (words still arriving); the SAS records
        train exactly this shape. An open user item never occurs in the
        training data.
      * ASSISTANT items are closed, EXCEPT an in-progress assistant item
        (the barge-in pattern) which is rendered OPEN (no <|im_end|>) —
        exactly like the barge-in augmentation records.
    """
    out: list[str] = []
    for it in items:
        role = it["role"]
        if role == "system":
            out.append(IM_START_SYSTEM + it["text"] + IM_END_NL)
        elif role == "user":
            out.append(IM_START_USER + it["text"] + IM_END_NL)
        else:
            tail = "" if it.get("partial") else IM_END
            out.append(IM_START_ASSISTANT + it["text"] + tail)
    return "".join(out)


class DuplexPolicy:
    def __init__(
        self,
        model_dir: str,
        thresholds: dict[str, float] | None = None,
        device: str = "cuda",
    ):
        self.model_dir = Path(model_dir)
        cfg = json.loads((self.model_dir / "policy_cfg.json").read_text())
        self.tok = AutoTokenizer.from_pretrained(model_dir, token=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, dtype=torch.bfloat16, device_map=device, token=False,
        ).eval()
        self.tag_vocab: list[str] = cfg["tag_vocab"]
        self.vocab_ids = [cfg["tag_ids"][name] for name in self.tag_vocab]
        self.thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
        self.no_voice = TOKEN["no_voice"]

    @torch.no_grad()
    def query(self, items: list[dict]) -> tuple[str, dict[str, float]]:
        """Predict the tag that should fire at the current point.

        Returns (tag, probs). `tag` is "no_tag" when nothing fires
        (the emit-nothing case); otherwise the protocol tag to inject.
        """
        text = render_items(items)
        ids = self.tok(text, return_tensors="pt").to(self.model.device)
        logits = self.model(**ids).logits[0, -1, self.vocab_ids]
        probs = torch.softmax(logits.float(), dim=-1)
        probs = {name: float(p) for name, p in zip(self.tag_vocab, probs)}
        argmax = self.tag_vocab[int(logits.argmax())]

        if argmax == "no_tag":
            return "no_tag", probs
        if probs[argmax] < self.thresholds.get(argmax, 0.5):
            return "no_tag", probs
        if argmax == "user_interruption" and self._last_is_no_voice(items):
            return "no_tag", probs  # silence never hard-stops the assistant
        return argmax, probs

    @staticmethod
    def _last_is_no_voice(items: list[dict]) -> bool:
        """True only when the query is PURE silence (no user words): the
        orchestrator-inserted marker must never hard-stop the assistant.
        If real user words precede the marker, an interruption IS the user
        talking over us and must be allowed."""
        if not items:
            return False
        last = items[-1]
        if last["role"] != "user" or last["text"].strip() != TOKEN["no_voice"]:
            return False
        prev = items[-2] if len(items) >= 2 else None
        if prev is not None and prev["role"] == "user":
            if prev["text"].strip() and prev["text"].strip() != TOKEN["no_voice"]:
                return False  # the user actually spoke before the silence
        return True


if __name__ == "__main__":
    import argparse
    import sys

    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/mnt/f/duplex_cascade_runs/mtp_like/runs/final_v2/merged")
    p.add_argument("--text", default=None, help="context to classify (raw string)")
    args = p.parse_args()
    pol = DuplexPolicy(args.model)
    if args.text:
        items = [{"role": "user", "text": args.text}]
        tag, probs = pol.query(items)
        top = sorted(probs.items(), key=lambda kv: -kv[1])[:4]
        print(f"tag: {tag}")
        for name, p in top:
            print(f"  {name:24s} {p:.3f}")
    else:
        print("usage: --text '<duplex context>'", file=sys.stderr)