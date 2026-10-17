"""Duplex protocol constants + prompt construction (single source of truth).

Everything here mirrors what the models were *trained* on and what the live
bridges *serve*:

* the special tokens come from each variant's own ``data/common.py`` (imported
  at runtime, never hardcoded, so the bench cannot drift from training),
* ``build_prompt`` is a faithful re-implementation of
  ``<variant>/DuplexCascade/server.py::BridgeServer.build_prompt`` — it is a
  pure function of ``history``/``prime`` with no ``self`` use, so
  ``tests/test_parity.py`` asserts byte equality against the live bridges,
* the per-variant system prompts are copied verbatim from the live bridges and
  asserted equal by the same parity test.
"""

from __future__ import annotations

import ast
import importlib.util
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_ROOT = Path("/mnt/f/duplex_cascade_runs")

VARIANT_DIRS = {
    "sft": REPO_ROOT / "sft",
    "distill": REPO_ROOT / "distill",
    "mtp_like": REPO_ROOT / "mtp_like",
}

# ---------------------------------------------------------------- tokens ----
# Canonical tag names (the keys of data/common.py::TOKEN). The mtp_like
# companion adds two more that never appear in an sft/distill vocabulary.
TAG_KEYS = (
    "no_voice",
    "user_is_speaking",
    "user_finish_speaking",
    "user_is_thinking",
    "user_interruption",
    "user_backchannel",
    "system_backchannel",
)
MTP_EXTRA_KEYS = ("system_take_floor", "system_handover")

# Cache of the imported TOKEN dicts, keyed by variant.
_TOKEN_CACHE: dict[str, dict[str, str]] = {}


# The upstream sbintuitions/DuplexCascade release uses DIFFERENT tag strings from
# this pt-BR fork (verified against its ``tokenizer/added_tokens.json``):
#
#   ours (pt-BR fork)              original (sbintuitions)
#   <|user is speaking|>           <|user is talking|>
#   <|user finish speaking|>       <|user finish talking|>
#   <|user is thinking|>           <|user is thinking|>
#   <|user interruption|>          <|user interruption|>
#   <|user backchannel|>           <|user backchannel|>
#   <|system backchannel|>         (does not exist)
#
# and it has no ``<|system backchannel|>`` token at all. So it cannot share
# ``data/common.py``; it carries its own vocabulary, with ``system_backchannel``
# mapped to the wait tag purely so the simulator's tag->action table does not
# have to special-case it (the token is never emitted and never trained).
ORIGINAL_TOKENS: dict[str, str] = {
    "no_voice": "<|no voice|>",
    "user_is_speaking": "<|user is talking|>",
    "user_finish_speaking": "<|user finish talking|>",
    "user_is_thinking": "<|user is thinking|>",
    "user_interruption": "<|user interruption|>",
    "user_backchannel": "<|user backchannel|>",
    # NOTE: there is deliberately NO "system_backchannel" entry. Upstream has no
    # such token, and aliasing it onto an existing tag made the adapter's
    # reverse map ambiguous — `<|user is talking|>` then resolved to
    # "system_backchannel" (last key wins in the comprehension), so every
    # "user is talking" was executed as a canned backchannel. The simulator only
    # looks for that key when a model emits it, which for this vocabulary is
    # impossible.
}


def load_tokens(variant: str = "sft") -> dict[str, str]:
    """Import ``<variant>/data/common.py`` and return its ``TOKEN`` dict."""
    if variant == "original_dc":
        return dict(ORIGINAL_TOKENS)
    if variant in _TOKEN_CACHE:
        return _TOKEN_CACHE[variant]
    common_py = VARIANT_DIRS[variant] / "data" / "common.py"
    spec = importlib.util.spec_from_file_location(f"fdb_{variant}_common", common_py)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"cannot load {common_py}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    tokens = dict(mod.TOKEN)
    _TOKEN_CACHE[variant] = tokens
    return tokens


# ------------------------------------------------------------ ChatML I/O ----
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

_SPECIAL_RE = re.compile(r"(<\|[^|>]+\|>)")


def build_prompt(history: list[dict[str, Any]], prime: str = "",
                 bos: bool = False) -> str:
    """Build the ChatML duplex micro-turn prompt.

    Byte-for-byte identical to ``BridgeServer.build_prompt`` in
    ``sft/DuplexCascade/server.py`` (the distill bridge inherits it unchanged):

        <|im_start|>system\\n{content}<|im_end|>\\n
        <|im_start|>user\\n{content}<|im_end|>\\n
        <|im_start|>assistant\\n{content}<|im_end|>          <- no trailing \\n
        <|im_start|>assistant\\n{prime}                       <- open turn

    Assistant items are closed with ``<|im_end|>`` and carry no trailing newline;
    user/system items are closed with ``<|im_end|>\\n``. ``prime`` is appended
    right after the final assistant header so the model continues that turn.
    """
    parts: list[str] = []
    for m in history:
        role = m.get("role", "user")
        content = str(m.get("content", ""))
        if role == "system":
            parts.append(f"{IM_START}system\n{content}{IM_END}\n")
        elif role == "assistant":
            parts.append(f"{IM_START}assistant\n{content}{IM_END}")
        else:
            parts.append(f"{IM_START}user\n{content}{IM_END}\n")
    parts.append(f"{IM_START}assistant\n" + prime)
    # The upstream sbintuitions release prepends BOS (`<|endoftext|>`) — its
    # server builds `history_ids = [bos_id, ...]` before the first micro-turn.
    # This fork's bridges do not. It is not cosmetic for the original: without
    # BOS its top token at a completed user turn is `<|user is talking|>` (p .90,
    # i.e. it stays silent), with BOS it is the first content token (p .958, it
    # answers). So the flag is per-model, matching each codebase.
    return ("<|endoftext|>" if bos else "") + "".join(parts)


def split_special(text: str) -> list[tuple[str, str]]:
    """Split ``text`` into ``(kind, payload)`` pairs; kind in {special, text}.

    Same regex and semantics as ``server.py::_split_special``.
    """
    parts: list[tuple[str, str]] = []
    for tok in _SPECIAL_RE.split(text):
        if not tok:
            continue
        if tok.startswith("<|") and tok.endswith("|>"):
            parts.append(("special", tok))
        else:
            parts.append(("text", tok))
    return parts


def strip_trailing_eos(text: str) -> str:
    """Drop everything from ``<|im_end|>`` on (the EOS that ends a micro-turn)."""
    return text.split(f"{IM_END}")[0]


def leading_special(text: str) -> str | None:
    """The first special token in ``text`` (the model's turn-taking decision)."""
    for kind, payload in split_special(text):
        if kind == "special":
            return payload
        if payload.strip():  # content before any tag => no decision token
            return None
    return None


def text_of(text: str) -> str:
    """All non-special content of ``text``, stripped."""
    return "".join(p for kind, p in split_special(text) if kind == "text").strip()


def has_any(text: str, tokens: tuple[str, ...]) -> bool:
    return any(t in text for t in tokens)


# --------------------------------------------------------- system prompts ---
# Copied verbatim from the live bridges (asserted equal by tests/test_parity.py).

SFT_SYSTEM = (
    "Você é um assistente de voz em português. "
    "Responda de forma curta e natural."
)

DISTILL_SYSTEM = (
    "Você é um assistente de voz full-duplex do sistema DuplexCascade. "
    "Numa conversa duplex, sistema e usuário falam em micro-turnos curtos e "
    "naturais, com sobreposição, como numa conversa de voz real. "
    "Você gerencia a tomada de turno com tags especiais: "
    "<|user is speaking|> (usuário está falando), "
    "<|user finish speaking|> (usuário terminou de falar), "
    "<|user is thinking|> (usuário está pensando), "
    "<|user interruption|> (usuário interrompeu), "
    "<|user backchannel|> (usuário faz um backchannel), "
    "<|system backchannel|> (backchannel do sistema) e "
    "<|no voice|> (silêncio). "
    "Responda de forma natural e completa, em português brasileiro coloquial, "
    "como uma fala falada em voz alta. Responda apenas com a fala — "
    "não imprima nenhuma tag especial. Não se repita: diga cada informação "
    "uma única vez. "
    "Quando terminar de responder, emita <|user is thinking|> para indicar "
    "que seu turno terminou."
)

# The upstream sbintuitions release ships no inference system prompt (its
# released train_cfg.json records only LoRA settings). It gets the same neutral
# prompt as the SFT control, in the language of the run.
ORIGINAL_SYSTEM = (
    "Você é um assistente de voz em português. "
    "Responda de forma curta e natural."
)

SYSTEM_PROMPTS = {"sft": SFT_SYSTEM, "distill": DISTILL_SYSTEM, "base": SFT_SYSTEM}

# A system prompt that *permits* the protocol tags. Used by the "tag" prompt
# mode to isolate a model's protocol ability from prompt-induced tag
# suppression.
TAG_PERMITTING_SYSTEM = (
    "Você é um assistente de voz em português. Responda de forma curta e natural. "
    "Antes de qualquer fala, emita a tag especial correspondente ao estado da "
    "conversa (por exemplo <|user finish speaking|>, <|user is speaking|> ou "
    "<|user is thinking|>) e só depois fale."
)

# The mtp_like companion's system item. mtp_like's training data *does* carry a
# system instruction on ~50% of records (policy/prep_policy_dataset.py:120), so
# this is the on-distribution generic instruction for the companion — the
# bank/CPF-specific rule from demo/server.py is a demo persona, not the protocol.
MTP_COMPANION_SYSTEM = "Gerencie a conversa normalmente, com turnos naturais."
MTP_COMPANION_SYSTEM_EN = "Manage the conversation normally, with natural turns."

# ------------------------------------------------- duplex-activating prompts ---
# The distill teacher prompts, imported from the training code so they cannot
# drift (distill/data/teacher_prompts.py). These are the prompts that ask for
# the full-duplex protocol explicitly:
#
#   ``TRIGGER_SYSTEM``              frames the model as a duplex assistant and
#                                   lists the tags, but says nothing about the
#                                   speech itself.
#   ``TRIGGER_SYSTEM_ALLOW_TAGS``   the above + "follow the duplex protocol and
#                                   use the special tags" — the prompt designed
#                                   to elicit tags *and* coherent content (the
#                                   "ICL assumption" ablation, SPEC §8).
#   ``TRIGGER_SYSTEM_CONTENT_ONLY`` the one the teacher and the live distill
#                                   bridge use; it explicitly FORBIDS tags
#                                   ("não imprima nenhuma tag especial").
_DISTILL_PROMPTS: dict[str, str] | None = None


def distill_teacher_prompts() -> dict[str, str]:
    """The three trigger prompts from the distill training code."""
    global _DISTILL_PROMPTS
    if _DISTILL_PROMPTS is None:
        src = VARIANT_DIRS["distill"] / "data" / "teacher_prompts.py"
        _DISTILL_PROMPTS = {
            name: extract_bridge_symbol(src, name)
            for name in ("TRIGGER_SYSTEM", "TRIGGER_SYSTEM_CONTENT_ONLY",
                         "TRIGGER_SYSTEM_ALLOW_TAGS")
        }
    return _DISTILL_PROMPTS


# The prompt mode used by default. "none" is the training distribution: the
# duplex sequences in data/duplex_train*.jsonl start directly with
# <|im_start|>user and carry no system item at all. Measuring the protocol under
# that distribution is the only apples-to-apples way to compare the variants,
# because each bridge then *adds* an inference-time prompt of its own — and
# the distill one explicitly forbids the tags ("não imprima nenhuma tag
# especial"), which alone decides every turn-taking outcome.
DEFAULT_SYSTEM_PROMPT_MODE = "none"

# Tags that mean "keep waiting, do not take the floor".
WAIT_TOKENS = ("user_is_speaking", "user_is_thinking")
# Tags that end the assistant's current turn.
TURN_END_TOKENS = ("user_is_thinking", "user_is_speaking", "user_interruption")


# --------------------------------------------------------------- registry ---
# GGUFs / merged dirs for every model the bench can drive. Paths follow the
# run_services.sh defaults of each variant.
MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "sft": {
        "kind": "llama_duplex",
        "variant": "sft",
        "gguf": RUNS_ROOT / "continue/export/gguf/DuplexCascade-PT-q4_k_m.gguf",
        "system": SFT_SYSTEM,
        "label": "SFT (original recipe, continue)",
    },
    "sft_m2": {
        "kind": "llama_duplex",
        "variant": "sft",
        "gguf": RUNS_ROOT / "full/export/gguf/DuplexCascade-PT-q4_k_m.gguf",
        "system": SFT_SYSTEM,
        "label": "SFT (original recipe, M2)",
    },
    "distill": {
        "kind": "llama_duplex",
        "variant": "distill",
        "gguf": RUNS_ROOT / "distill/full/export/gguf/DuplexCascade-Distill-q4_k_m.gguf",
        "system": DISTILL_SYSTEM,
        "label": "Distill (self-distillation)",
    },
    "mtp_like": {
        "kind": "policy_companion",
        "variant": "mtp_like",
        "companion": RUNS_ROOT / "mtp_like/runs/final_v6/merged",
        "big_gguf": None,  # the runner serves a stock Qwen3-4B GGUF
        "system": MTP_COMPANION_SYSTEM,
        "label": "MTP-like (0.8B policy companion + stock Qwen3-4B)",
    },
    "base": {
        "kind": "llama_duplex",
        "variant": "sft",  # same protocol vocabulary, but UNTRAINED control
        "gguf": None,  # set by the runner; stock Qwen3-4B-Instruct-2507 GGUF
        "system": SFT_SYSTEM,
        "label": "Base (untrained Qwen3-4B control)",
    },
    # Two ways to run the upstream checkpoint. The GGUF variant is preferred for
    # comparison because it goes through the *identical* serving stack as the
    # three pt-BR variants (same llama-server, same greedy decoding, same raw
    # /v1/completions prompts) — only the weights and the tag vocabulary differ.
    "original_dc": {
        "kind": "llama_duplex",
        "variant": "original_dc",
        "gguf": RUNS_ROOT / "original_dc/gguf/DuplexCascade-Original-q4_k_m.gguf",
        "system": ORIGINAL_SYSTEM,
        "label": "Original DuplexCascade (sbintuitions, q4_k_m)",
    },
    "original_dc_hf": {
        "kind": "original_dc",
        "variant": "original_dc",
        "state_dict": None,  # gated HF repo, see README "Adding the original"
        "system": ORIGINAL_SYSTEM,
        "label": "Original DuplexCascade (sbintuitions, HF bf16)",
    },
}


# ------------------------------------------------- live-bridge extraction ---
def extract_bridge_symbol(source: Path, name: str) -> Any:
    """Extract a top-level constant from another module, by name.

    Evaluates the module's top-level *simple assignments and annotations in
    order* into a namespace, so a constant defined as an expression over earlier
    constants resolves correctly (``TRIGGER_SYSTEM_CONTENT_ONLY = TRIGGER_SYSTEM
    + (...)`` is a ``BinOp``, which ``ast.literal_eval`` cannot read).

    Only assignments and function definitions are executed — no imports, no
    calls, nothing else — so this is safe to point at the training code and it
    lets the bench reuse those prompts verbatim instead of copying them.
    """
    tree = ast.parse(source.read_text(encoding="utf-8"))
    ns: dict[str, Any] = {}
    for node in tree.body:
        try:
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.FunctionDef,
                                 ast.ClassDef, ast.Import, ast.ImportFrom)):
                mod = ast.Module(body=[node], type_ignores=[])
                exec(compile(ast.fix_missing_locations(mod), str(source), "exec"), ns)
        except Exception:  # noqa: BLE001 - unrelated top-level code may need imports
            continue
        if name in ns:
            return ns[name]
    raise KeyError(f"{name} not found in {source}")


def bridge_method(source: Path, cls: str, method: str) -> Any:
    """ast-extract a ``BridgeServer`` method (e.g. ``build_prompt``)."""
    tree = ast.parse(source.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name == method:
                    fn = ast.FunctionDef(
                        name=sub.name,
                        args=sub.args,
                        body=sub.body,
                        decorator_list=[],
                        returns=None,
                        type_params=[],
                    )
                    ns: dict[str, Any] = {}
                    mod = ast.Module(body=[fn], type_ignores=[])
                    exec(compile(ast.fix_missing_locations(mod), str(source), "exec"), ns)
                    return ns[sub.name]
    raise KeyError(f"{cls}.{method} not found in {source}")
