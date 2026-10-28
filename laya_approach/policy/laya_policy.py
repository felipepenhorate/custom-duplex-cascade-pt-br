"""Runtime companion for the fine-tuned Laya decision model.

The policy keeps the exact state contract used in training: the newest item is
the stream event, and the event carries ``kind`` (partial / final / silence),
``ends_turn`` and ``system_speaking``.  The FDB adapters add those fields; a
caller that only has role+text still works, but then a final chunk is inferred
from ``ends_turn`` and system speech from the preceding assistant item.

The answer is a single protocol tag, guarded by the same turn-end rules as the
data generator.  No text is generated here: the adapter hands a fired
``user_finish_speaking`` to its content model.

Quick start
-----------
Use the project's ``.venv-unsloth`` environment (unsloth must be imported before
transformers/torch) on a CUDA/ROCm GPU.  Download the published checkpoint,
which is the v7 model trained for both text and audio streams::

    huggingface-cli download lumierenoir/laya-duplex-cascade-pt-br \
        --repo-type model --local-dir models/laya-companion

Then call the policy on the conversation so far.  The **last** item must be the
user's newest stream event; the return value is ``(tag, probabilities)``::

    from laya_approach.policy import load

    policy = load("models/laya-companion", device="cuda", stream="text")

    items = [
        {"role": "assistant", "text": "Claro, posso ajudar."},
        {"role": "user", "text": "Eu queria saber",
         "kind": "partial", "ends_turn": False, "system_speaking": False},
    ]
    tag, probs = policy(items)
    assert tag == "user_is_speaking"          # punctuation is not a turn end

    items.append({"role": "user", "text": "o prazo de entrega",
                  "kind": "final", "ends_turn": True, "system_speaking": False})
    tag, probs = policy(items)
    # tag == "user_finish_speaking": the caller now runs its content model.

Each item is **one incremental stream piece**, in order.  Do not repeat the
whole utterance in every event: a final whose text is a punctuation-only edit of
the preceding partial is ambiguous, and the model correctly waits for more
speech.  For a cumulative ASR transcript, pass only the new suffix.  (The FDB
adapters already deliver incremental pieces.)

The item fields the state reads are:

=============  ==============================================================
``role``       ``"user"`` or ``"assistant"``; the newest item must be user.
``text``       ASR piece, or ``""`` for silence.
``kind``       ``"partial"`` (mid-turn), ``"final"`` (ASR/VAD final), or
               ``"silence"`` (an explicit no-speech tick).
``ends_turn``  whether this event ends the user's turn.  For ``stream="audio"``
               this is the VAD's utterance boundary, not necessarily a
               semantic turn end.
``system_speaking``  true while the assistant audio is still audible.
=============  ==============================================================

``stream`` selects the trained contract:

* ``"text"`` -- ``ends_turn`` is a semantic turn boundary; a final fires
  immediately.
* ``"audio"`` -- ``ends_turn`` is the VAD final.  Keep feeding silence ticks
  after it; the model was trained to finish on the final or the first two
  post-final ticks, and to wait through a pause/backchannel silence.

Silence is an explicit item, for example
``{"role": "user", "text": "", "kind": "silence",
"ends_turn": False, "system_speaking": False}``.  Do not reuse the previous
piece: the state distinguishes "user stopped talking" from "the same words
again".

The runtime guard is enabled by default.  It vetoes ``user_finish_speaking``
unless the event really ends the turn and the assistant is not speaking, and it
only allows interruption/backchannel while ``system_speaking`` is true.  Set
``guard=False`` or ``LAYA_GUARD=0`` to measure the raw model; production should
keep the guard.  ``probs`` always contains all seven tags, so callers can log
confidence or apply their own threshold on top.

Full training/eval code, data pipeline and benchmarks:
https://github.com/felipepenhorate/custom-duplex-cascade-pt-br/tree/main/laya_approach

Model card and ATA comparison:
https://huggingface.co/lumierenoir/laya-duplex-cascade-pt-br

Commercial/permissive retraining (MIT-only data):
``laya_approach/COMMERCIAL.md`` in the GitHub repository.
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
    """Load one decision checkpoint and answer protocol tags.

    Args:
        model_dir: Local directory with ``model.safetensors``, ``encoder/``,
            ``tokenizer/`` and ``policy_cfg.json``.  A Hub repo id works for
            the weights, but ``policy_cfg.json`` is only read from a local
            path, so prefer ``huggingface-cli download`` (see module docstring).
        device: Torch device, normally ``"cuda"``.
        thresholds: Per-tag overrides merged on top of ``policy_cfg.json`` and
            :data:`~laya_approach.data.common.DEFAULT_THRESHOLDS`.
        max_seq_length: Encoder limit; defaults to ``policy_cfg.json["max_len"]``
            (or 640).
        guard: Force the runtime guard on/off.  When ``None``, use
            ``policy_cfg.json["guard"]`` (default true) unless ``LAYA_GUARD``
            is ``0``/``false``/``no``.
        stream: ``"text"`` or ``"audio"``; selects the trained contract and
            must match the checkpoint's training mix.
    """

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
        """Return ``(tag, probabilities)`` for the newest stream event.

        ``items`` is the conversation so far with the user's latest event last
        (see the module docstring for the item schema).  The assistant history
        is optional; only the fields ``role``/``text`` are required for old
        items.  User events must be incremental pieces, not the cumulative
        transcript repeated.  If ``items`` is empty or the newest item is not a
        user event, the policy returns ``no_tag``.
        """
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
    """Convenience constructor; see :class:`LayaPolicy` for the arguments.

    Example::

        from laya_approach.policy import load
        policy = load("models/laya-companion", stream="audio")
        tag, probs = policy(items)
    """
    return LayaPolicy(model_dir, device=device, **kwargs)
