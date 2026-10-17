"""Published Full-Duplex-Bench numbers, for the comparison column.

Sources
-------
* **DuplexCascade / DuplexCascade-β** — Table 1 of the DuplexCascade paper
  (arXiv:2603.09180), reproduced in the vendored
  ``sft/DuplexCascade/assets/full-duplex-bench.png``.
* **FDB v1.0 baselines** — the Full-Duplex-Bench README result table
  (arXiv:2503.04721).

Caveat, stated once and repeated in every table this module feeds: these are
**English, audio-domain** numbers produced by real speech models through the
full ASR–LLM–TTS stack. Our rows are **text-domain** numbers from a simulated
clock. Only the *ordering* of our own models is directly comparable; the
published row is a reference point, not an apples-to-apples baseline. The
``ata`` column in particular is computed by us (FDB does not define it in the
released code), so ours and theirs are not the same estimator.
"""

from __future__ import annotations

from typing import Any

# Table 1 of the DuplexCascade paper (arXiv:2603.09180).
PUBLISHED_DUPLEXCASCADE_ROWS: list[dict[str, Any]] = [
    {
        "label": "DuplexCascade (paper)",
        "lang": "en",
        "pause_synthetic_tor": 0.058,
        "pause_candor_tor": 0.222,
        "bc_tor": 0.218,
        "bc_freq": 0.009,
        "bc_jsd": 0.949,
        "smooth_tor": 0.832,
        "smooth_latency_s": 1.724,
        "intr_tor": 0.955,
        "intr_rating": 4.016,
        "intr_latency_s": 1.225,
        "ata": 0.858,
    },
    {
        "label": "DuplexCascade-β (paper)",
        "lang": "en",
        "pause_synthetic_tor": 0.343,
        "pause_candor_tor": 0.458,
        "bc_tor": 0.309,
        "bc_freq": 0.034,
        "bc_jsd": 0.811,
        "smooth_tor": 0.899,
        "smooth_latency_s": 0.567,
        "intr_tor": 0.950,
        "intr_rating": 4.011,
        "intr_latency_s": 0.850,
        "ata": 0.748,
    },
]

# FDB v1.0 baselines, for context on what the metric ranges look like.
FDB_V1_BASELINES: list[dict[str, Any]] = [
    {"label": "Moshi (FDB v1.0)", "pause_synthetic_tor": 0.985, "pause_candor_tor": 0.980,
     "bc_tor": 1.000, "bc_freq": 0.001, "bc_jsd": 0.957, "smooth_tor": 0.941,
     "smooth_latency_s": 0.265, "intr_tor": 1.000, "intr_rating": 0.765,
     "intr_latency_s": 0.257, "ata": 0.395},
    {"label": "dGSLM (FDB v1.0)", "pause_synthetic_tor": 0.934, "pause_candor_tor": 0.935,
     "bc_tor": 0.691, "bc_freq": 0.015, "bc_jsd": 0.934, "smooth_tor": 0.975,
     "smooth_latency_s": 0.352, "intr_tor": 0.917, "intr_rating": 0.201,
     "intr_latency_s": 2.531, "ata": 0.466},
    {"label": "GPT-Realtime (FDB v1.0)", "pause_synthetic_tor": 0.010,
     "pause_candor_tor": 0.120, "bc_tor": 0.000, "bc_freq": 0.007, "bc_jsd": 0.980,
     "smooth_tor": 1.000, "smooth_latency_s": 1.470, "intr_tor": 0.970,
     "intr_rating": 3.850, "intr_latency_s": 1.500, "ata": None},
]

def published_rows() -> list[dict[str, Any]]:
    """The DuplexCascade reference rows, ready for the table renderer."""
    return [dict(r) for r in PUBLISHED_DUPLEXCASCADE_ROWS]
