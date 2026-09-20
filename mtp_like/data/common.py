"""Shared datums for the mtp_like policy-companion data pipeline.

Duplex protocol vocabulary (exact tokens from sft/data/common.py), the
label-only ``<|no tag|>`` symbol (the companion's "emit nothing" output —
never part of the input stream, never added to the context), per-tag loss
weights (paper §4.1) and the ChatML framing used to render the duplex
context for the companion (same conventions as sft/training/prep_dataset.py).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Conversational special tokens (same as sft/data/common.py, paper §3.2)
# ---------------------------------------------------------------------------
TOKEN = {
    "no_voice": "<|no voice|>",
    "user_is_speaking": "<|user is speaking|>",
    "user_finish_speaking": "<|user finish speaking|>",
    "user_is_thinking": "<|user is thinking|>",
    "user_interruption": "<|user interruption|>",
    "user_backchannel": "<|user backchannel|>",
    "system_backchannel": "<|system backchannel|>",
    "system_take_floor": "<|system take floor|>",
    "system_handover": "<|system handover|>",
}

# Label-only symbol: the companion's "emit nothing" class. It is added to the
# tokenizer as a special token but NEVER rendered into the input stream; it
# exists purely as a classification target at item boundaries.
NO_TAG = "<|no tag|>"

# The tags the companion can EMIT (the protocol vocabulary). `no_voice` is a
# context marker only (orchestrator state) and is never a prediction target.
# The two `system_*` tags are the promptable-controller additions (v6):
#   system_take_floor  -> the SYSTEM interrupts the user and speaks now
#   system_handover    -> a prompt rule fired; hand control to the big model,
#                         which decides what to do (maybe nothing)
TAG_KEYS = [
    "user_is_speaking",
    "user_finish_speaking",
    "user_interruption",
    "user_backchannel",
    "user_is_thinking",
    "system_backchannel",
    "system_take_floor",
    "system_handover",
]

SPECIAL_TOKENS = list(TOKEN.values())

# Per-tag loss weights (paper §4.1, same as sft). `no_tag` gets its own small
# weight in the prep script (the EMPTY class dominates).
TOKEN_WEIGHT = {
    "user_is_speaking": 1,
    "user_finish_speaking": 10,
    "user_interruption": 5,
    "user_backchannel": 2,
    "user_is_thinking": 1,
    "system_backchannel": 3,
}

# Weights for the promptable-controller tags (v6) — rare and critical
SYSTEM_WEIGHTS = {
    "system_take_floor": 5,
    "system_handover": 3,
}

# ChatML framing (same as sft/training/prep_dataset.py):
#   user      -> <|im_start|>user\n ... <|im_end|>\n
#   assistant -> <|im_start|>assistant\n ... <|im_end|>
IM_START_USER = "<|im_start|>user\n"
IM_START_ASSISTANT = "<|im_start|>assistant\n"
IM_START_SYSTEM = "<|im_start|>system\n"
IM_END = "<|im_end|>"
IM_END_NL = "<|im_end|>\n"

ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_SYSTEM = "system"

# pt-BR conversational backchannels (same as sft/data/common.py). Used both
# as context content (user backchannel items in the sft data) and as sampled
# barge-in negatives (a backchannel word is NOT an interruption).
PT_BACKCHANNELS = [
    "aham",
    "uhum",
    "tá",
    "sim",
    "certo",
    "ok",
    "okay",
    "entendi",
    "é",
    "claro",
    "pois é",
    "né",
    "sei",
    "hmm",
    "ah tá",
]