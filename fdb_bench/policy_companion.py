"""Load the mtp_like policy companion for the bench.

Thin wrapper so :mod:`fdb_bench.adapters` does not have to touch torch /
transformers at import time (the tag-emitting adapters never need them).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from .protocol import REPO_ROOT

_MTP_ROOT = REPO_ROOT / "mtp_like"


def load_policy(model_dir: str | Path, device: str = "cuda") -> Any:
    """Instantiate ``mtp_like.policy.duplex_policy.DuplexPolicy``.

    The companion module does ``from data.common import ...`` and
    ``from policy.duplex_policy import ...``, so ``mtp_like`` must be on
    ``sys.path``. It is prepended (not replaced) and restored afterwards so a
    later import of our own ``data`` packages cannot be shadowed.
    """
    root = str(_MTP_ROOT)
    added = root not in sys.path
    if added:
        sys.path.insert(0, root)
    try:
        from policy.duplex_policy import DuplexPolicy  # type: ignore
        return DuplexPolicy(str(model_dir), device=device)
    finally:
        if added:
            try:
                sys.path.remove(root)
            except ValueError:
                pass


def available_checkpoints() -> list[str]:
    """Every ``runs/*/merged`` companion the repo has on disk."""
    runs = Path("/mnt/f/duplex_cascade_runs/mtp_like/runs")
    if not runs.is_dir():
        return []
    out = []
    for d in sorted(runs.iterdir()):
        if (d / "merged" / "policy_cfg.json").is_file():
            out.append(str(d / "merged"))
    return out
