"""Runtime companion for the fine-tuned Laya decision model.

The policy keeps the exact state contract used in training: the newest item is
the stream event, and the event carries ``kind`` (partial / final / silence),
``ends_turn`` and ``system_speaking``.  The FDB adapters add those fields; a
caller that only has role+text still works, but then a final chunk is inferred
from ``ends_turn`` and system speech from the preceding assistant item.

The answer is a single protocol tag, guarded by the same turn-end rules as the
data generator.  No text is generated here: the adapter hands a fired
``user_finish_speaking`` to its content model.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import unsloth  # noqa: F401  (must be imported before transformers/torch users)
import torch
from unsloth import FastDecisionModel

from laya_approach.data.common import (
    ACTION_KEYS,
    DEFAULT_THRESHOLDS,
    NO_TAG,
    NO_VOICE,
    action_question,
    resolve_action,
    state_from_items,
)


class LayaPolicy:
    def __init__(
        self,
        model_dir: str | Path,
        *,
        device: str = "cuda",
        thresholds: dict[str, float] | None = None,
        max_seq_length: int | None = None,
        guard: bool | None = None,
        stream: str = "text",
    ) -> None:
        self.model_dir = Path(model_dir)
        self.stream = stream
        cfg_path = self.model_dir / "policy_cfg.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
        self.tag_vocab: list[str] = cfg.get("tag_vocab") or list(ACTION_KEYS)
        self.no_voice: str = cfg.get("no_voice") or NO_VOICE
        if guard is None:
            guard = bool(cfg.get("guard", True))
            if os.environ.get("LAYA_GUARD", "1").lower() in ("0", "false", "no"):
                guard = False
        self.guard = bool(guard)
        self.thresholds = {
            **DEFAULT_THRESHOLDS,
            **(cfg.get("thresholds") or {}),
            **(thresholds or {}),
        }
        self.model, self.tokenizer = FastDecisionModel.from_pretrained(
            str(self.model_dir),
            max_seq_length=int(max_seq_length or cfg.get("max_len") or 640),
            dtype=torch.float16,
        )
        FastDecisionModel.for_inference(self.model)
        self.model.to(device)

    @torch.no_grad()
    def query(self, items: list[dict]) -> tuple[str, dict[str, float]]:
        """Return ``(tag, probabilities)`` for the newest stream event."""
        if not items:
            return NO_TAG, {tag: 0.0 for tag in self.tag_vocab}
        if items[-1].get("role") != "user":
            return NO_TAG, {tag: 0.0 for tag in self.tag_vocab}
        state, event = state_from_items(items, stream=self.stream)
        answer = FastDecisionModel.predict(
            self.model,
            self.tokenizer,
            state,
            {"action": action_question()},
        )["action"]
        probs = {tag: float(answer["probabilities"].get(tag, 0.0)) for tag in self.tag_vocab}
        if not self.guard:
            tag = max(probs, key=probs.__getitem__)
            return tag, probs
        tag, _ = resolve_action(probs, event, state, self.thresholds)
        return tag, probs

    def __call__(self, items: list[dict]) -> tuple[str, dict[str, float]]:
        return self.query(items)


def load(model_dir: str | Path, device: str = "cuda", **kwargs: Any) -> LayaPolicy:
    return LayaPolicy(model_dir, device=device, **kwargs)
