"""The pt-BR vs English comparison, after the rebuild.

Both rows are measured the same way: the same harness, the same silence gate,
the same rendering, and the same level normalisation before recognition. The two
rows differ in two things only, and both are deliberate:

* the **language** of the set (Portuguese re-voicing vs the English recording),
* the **recogniser** that reads the assistant's replies (whisper large-v3 --
  ``--language pt`` -- vs the benchmark's own parakeet, which is English-only).

The table prints the difference the rebuild was about: the same model, measured
on its in-distribution audio and on the out-of-distribution one.
"""
from __future__ import annotations

import json
from pathlib import Path

from fdb_bench.metrics import averaged_turn_taking_accuracy as ata

LABEL = {"original_dc": "Original DuplexCascade (upstream)",
         "sft": "SFT (pt-BR)",
         "distill": "Self-distillation (pt-BR)",
         "mtp_like": "Companion (MTP-style)"}
MODELS = ("original_dc", "sft", "distill", "mtp_like")


def load(p) -> dict | None:
    p = Path(p)
    if not p.is_file():
        return None
    return json.loads(p.read_text())["official_task_metrics"]


def row(tag: str, o: dict | None):
    if not o:
        return None
    g = lambda s, k: o.get(s, {}).get(k)  # noqa: E731
    p1, p2 = g("synthetic_pause_handling", "tor"), g("candor_pause_handling", "tor")
    bc = g("icc_backchannel", "tor")
    sm, it = g("candor_turn_taking", "tor"), g("synthetic_user_interruption", "tor")
    sml = g("candor_turn_taking", "latency_s")
    itl = g("synthetic_user_interruption", "latency_s")
    rel = g("synthetic_user_interruption", "relevance")
    a = ata({"pause_synthetic_tor": p1, "pause_candor_tor": p2,
             "smooth_tor": sm, "intr_tor": it})
    return {"tag": tag, "pause_syn": p1, "pause_cand": p2, "bc": bc, "smooth": sm,
            "smooth_lat": sml, "intr": it, "rel": rel, "intr_lat": itl, "ata": a}


def fmt(r) -> str:
    return (f"| {r['tag']} | {r['pause_syn']:.3f} | {r['pause_cand']:.3f} | "
            f"{r['bc']:.3f} | {r['smooth']:.3f} | {r['smooth_lat']:.2f} | "
            f"{r['intr']:.3f} | {r['rel']:.2f} | {r['intr_lat']:.2f} | "
            f"**{r['ata']:.3f}** |")


def main() -> None:
    L = [
        "# Full-Duplex-Bench on real audio: pt-BR (in-distribution) vs English (out-of-distribution)",
        "",
        "The three pt-BR variants are fine-tuned on pt-BR duplex data, so the "
        "Portuguese set is their **in-distribution** set and the English "
        "recording is **out-of-distribution**. Both rows are measured by the "
        "same harness: the same silence gate, the same rendering, the same "
        "level normalisation before recognition, and the benchmark's own "
        "`eval_*.py`. They differ in the language of the set and in the "
        "recogniser that reads the assistant's replies (whisper large-v3 "
        "`--language pt` vs the benchmark's English-only parakeet).",
        "",
        "The Portuguese set is a re-voicing: the benchmark's own audio is "
        "segmented, translated, spoken by pocket-tts and put back on the "
        "timeline with the gaps preserved and the annotations re-timed "
        "(`build_pt_audio.py`; `check_pt_structure.py` is the gate that "
        "compares the rebuilt audio with the recording it was made from).",
        "",
        "## Portuguese (in-distribution for the three trained variants)",
        "",
        "| model | pause_syn ↓ | pause_cand ↓ | BC TOR ↓ | smooth ↑ | smooth lat ↓ | intr ↑ | relevance ↑ | intr lat ↓ | **ATA** ↑ |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    pt_rows = {}
    for m in MODELS:
        r = row(LABEL[m], load(f"results_audio_pt/decision/{m}/metrics_whisper.json"))
        if r:
            pt_rows[m] = r
            L.append(fmt(r))
    dep = row("Self-distillation, production configuration (deployed prompt, decision mode)",
              load("results_audio_pt_deployed/decision/distill/metrics_whisper.json"))
    if dep:
        L.append(fmt(dep))

    L += ["", "## English (out-of-distribution for the three trained variants)", "",
          "| model | pause_syn ↓ | pause_cand ↓ | BC TOR ↓ | smooth ↑ | smooth lat ↓ | intr ↑ | relevance ↑ | intr lat ↓ | **ATA** ↑ |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    en_rows = {}
    for m in MODELS:
        r = row(LABEL[m], load(f"results_audio/decision/{m}/metrics.json"))
        if r:
            en_rows[m] = r
            L.append(fmt(r))

    if pt_rows and en_rows:
        L += ["", "## The difference the rebuild was about", "",
              "PT is the in-distribution set for the trained variants, so a "
              "negative number means the model did better on its own language. "
              "The upstream model is English-trained, so for it the sign is "
              "expected to point the other way.", "",
              "| model | ATA EN (OOD) | ATA PT (ID) | Δ (PT−EN) |", "|---|---|---|---|"]
        for m in MODELS:
            if m in pt_rows and m in en_rows:
                L.append(f"| {LABEL[m]} | {en_rows[m]['ata']:.3f} | "
                         f"{pt_rows[m]['ata']:.3f} | "
                         f"{pt_rows[m]['ata'] - en_rows[m]['ata']:+.3f} |")

    out = Path("results_audio_pt/PT_COMPARISON.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
