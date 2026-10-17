"""Adapter for the *original* DuplexCascade model (sbintuitions, English).

The upstream model lives at https://huggingface.co/sbintuitions/DuplexCascade
and is **gated** (auto-approved, EU excluded). Its checkpoint is a raw
``model_state.safetensors`` state dict against ``Qwen/Qwen2-7B-Instruct``, not
a standard HF export — which is exactly what the vendored
``sft/DuplexCascade/model.py`` loads:

    model = Model(tokenizer, model_name="Qwen/Qwen2-7B-Instruct")
    model.load_state_dict(torch.load("model_state.safetensors"))

To run it in the bench:

1. accept the license at https://huggingface.co/sbintuitions/DuplexCascade
2. ``hf download sbintuitions/DuplexCascade model_state.safetensors \\
       tokenizer --local-dir /mnt/f/duplex_cascade_runs/original_dc``
3. pass ``--original-state-dict <path>`` to ``run_inference.py``.

Until then the published FDB numbers in ``REFERENCE.md`` are the comparison
baseline, and :func:`build_original_adapter` fails with instructions instead of
silently falling back.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

STATE_DICT_ENV = "FDB_ORIGINAL_STATE_DICT"
DEFAULT_DIR = Path("/mnt/f/duplex_cascade_runs/original_dc")

# The upstream model is English-only and its repository ships no inference system
# prompt (the released ``train_cfg.json`` records only the LoRA settings and the
# paper's repo documents none), so it runs with the same neutral prompt the SFT
# control uses. For the pt-BR pass the prompt is left in Portuguese: the model is
# out-of-distribution either way, and holding the prompt constant across models
# is what makes the cross-model comparison meaningful.
ORIGINAL_SYSTEM = (
    "Você é um assistente de voz em português. "
    "Responda de forma curta e natural."
)
ORIGINAL_SYSTEM_EN = (
    "You are a voice assistant in English. "
    "Reply briefly and naturally."
)


def resolve_state_dict(explicit: str | None = None) -> Path | None:
    """Find the downloaded checkpoint, if any."""
    import os

    for cand in (explicit, os.environ.get(STATE_DICT_ENV), str(DEFAULT_DIR)):
        if not cand:
            continue
        p = Path(cand)
        if p.is_dir():
            hit = p / "model_state.safetensors"
            if hit.is_file():
                return hit
        elif p.is_file():
            return p
    return None


def build_original_adapter(
    spec: dict[str, Any],
    *,
    api_base: str,
    lang: str = "pt",
    prime_mode: str = "decision",
    prompt_candidate: str = "deployed",
    state_dict: str | None = None,
    device: str = "cuda",
) -> Any:
    """Load the original checkpoint in-process and wrap it as an adapter.

    Raises :class:`FileNotFoundError` with setup instructions when the gated
    checkpoint has not been downloaded — never falls back to another model.
    """
    import torch  # imported lazily: only this adapter needs the GPU stack

    from .protocol import REPO_ROOT

    sd_path = resolve_state_dict(state_dict or spec.get("state_dict"))
    if sd_path is None:
        raise FileNotFoundError(
            "The original DuplexCascade checkpoint is not available.\n"
            "  1. accept the license: https://huggingface.co/sbintuitions/DuplexCascade\n"
            "  2. hf download sbintuitions/DuplexCascade model_state.safetensors tokenizer "
            f"--local-dir {DEFAULT_DIR}\n"
            "  3. re-run with --original-state-dict <path>\n"
            "The published FDB numbers in REFERENCE.md are used as the baseline until then."
        )

    # reuse the vendored loader rather than reimplementing it
    import importlib.util

    vendored = REPO_ROOT / "sft" / "DuplexCascade" / "model.py"
    mod_spec = importlib.util.spec_from_file_location("duplex_dc_model", vendored)
    if mod_spec is None or mod_spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"cannot import {vendored}")
    mod = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(mod)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(sd_path.parent / "tokenizer")
                                        if (sd_path.parent / "tokenizer").is_dir()
                                        else str(sd_path.parent))
    model = mod.Model(tok, model_name="Qwen/Qwen2-7B-Instruct", torch_dtype=torch.bfloat16)
    state = torch.load(str(sd_path), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model = model.to(device).eval()

    from .adapters import HFDuplexAdapter  # in-process twin of LlamaDuplexAdapter

    from .adapters import SFT_GEN, HFDuplexAdapter
    return HFDuplexAdapter(
        model, tok, variant="original_dc",
        name="original_dc",
        system_prompt=ORIGINAL_SYSTEM_EN if lang == "en" else ORIGINAL_SYSTEM,
        prime_mode=prime_mode,
        gen=dict(SFT_GEN),
    )
