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
            "smooth_lat": sml, "intr": it, "rel": rel, "intr_lat": itl, "ata": a,
            "raw": (1.0 - p1 + 1.0 - p2 + sm + it) / 4.0}


def language(results_dir: str) -> dict | None:
    """The LLM-judged reply-language summary of a run, if it has one."""
    p = Path(results_dir) / "language_llm.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text())


_NON_LATIN = __import__("re").compile(
    r"[\uac00-\ud7af\u1100-\u11ff\u3040-\u30ff\u4e00-\u9fff\u0400-\u04ff]")


def drift(results_dir: str) -> dict:
    """Objective non-Latin drift: replies containing (or opening with) a stray
    non-Latin token.

    The language judge answers "which language is this text", and a reply that
    *opens* with a stray Korean interjection gets labelled Korean even when the
    rest of it is English. This counts the artefact itself, with no judge and
    no judgement call: the share of replies containing a non-Latin character,
    and the share that open with one.
    """
    n = with_any = opens_with = 0
    for td in Path(results_dir).glob("*/*/trace.json"):
        d = json.loads(td.read_text())
        for t in d.get("turns") or []:
            txt = (t.get("text") or "")
            if not txt.strip():
                continue
            n += 1
            if _NON_LATIN.search(txt):
                with_any += 1
                if _NON_LATIN.match(txt.strip()):
                    opens_with += 1
    return {"n_turns": n, "with_non_latin": with_any, "opening_non_latin": opens_with,
            "share_with": with_any / max(1, n), "share_opening": opens_with / max(1, n)}


def _words_per_turn(results_dir: str) -> float:
    """Mean length of an assistant turn, in words (the degeneracy column)."""
    n = 0.0
    c = 0
    for td in Path(results_dir).glob("*/*/trace.json"):
        d = json.loads(td.read_text())
        for t in d.get("turns") or []:
            if (t.get("text") or "").strip():
                n += t.get("n_words") or 0
                c += 1
    return n / max(1, c)


def fmt(r, results_dir: str = "") -> str:
    tail = ""
    lang = language(results_dir) if results_dir else None
    if lang:
        other = ", ".join(f"{k}:{v}" for k, v in
                          list(lang.get("other_breakdown", {}).items())[:3])
        tail = (f" | {lang.get('share_turns_pt', 0):.1%} / "
                f"{lang.get('share_scenarios_pt', 0):.1%}"
                f" | {other or '-'}")
    else:
        tail = ""
    d = drift(results_dir) if results_dir else None
    if d:
        tail += f" | {d['share_with']:.1%} / {d['share_opening']:.1%}"
    return (f"| {r['tag']} | {r['pause_syn']:.3f} | {r['pause_cand']:.3f} | "
            f"{r['bc']:.3f} | {r['smooth']:.3f} | {r['smooth_lat']:.2f} | "
            f"{r['intr']:.3f} | {r['rel']:.2f} | {r['intr_lat']:.2f} | "
            f"**{r['ata']:.3f}** | {r['raw']:.3f} |{tail}|")


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
        "Two ATA columns: the project's convention with its two guards, and the "
        "plain four-term mean (`raw`), which is what the published row is. The "
        "reply-language columns are the local Qwen3.5-9B judging each assistant "
        "turn `pt`, `en`, or another language: the share of turns in Portuguese "
        "and the share of scenarios whose answers are, then the languages that "
        "showed up in the rest.",
        "The Portuguese set is a re-voicing: the benchmark's own audio is "
        "segmented, translated, spoken by pocket-tts and put back on the "
        "timeline with the gaps preserved and the annotations re-timed "
        "(`build_pt_audio.py`; `check_pt_structure.py` is the gate that "
        "compares the rebuilt audio with the recording it was made from).",
        "",
        "## Portuguese (in-distribution for the three trained variants)",
        "",
        "| model | pause_syn ↓ | pause_cand ↓ | BC TOR ↓ | smooth ↑ | smooth lat ↓ | intr ↑ | relevance ↑ | intr lat ↓ | **ATA** ↑ | ATA raw | reply in pt | other langs | non-latin drift |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    pt_rows = {}
    for m in MODELS:
        r = row(LABEL[m], load(f"results_audio_pt/decision/{m}/metrics_whisper.json"))
        if r:
            pt_rows[m] = r
            L.append(fmt(r, f"results_audio_pt/decision/{m}"))
    dep = row("Self-distillation, production configuration (deployed prompt, decision mode)",
              load("results_audio_pt_deployed_tags/decision/distill/metrics_whisper.json"))
    if dep:
        L.append(fmt(dep, "results_audio_pt_deployed_tags/decision/distill"))

    L += ["", "## English (out-of-distribution for the three trained variants)", "",
          "| model | pause_syn ↓ | pause_cand ↓ | BC TOR ↓ | smooth ↑ | smooth lat ↓ | intr ↑ | relevance ↑ | intr lat ↓ | **ATA** ↑ | ATA raw | reply in pt | other langs |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    en_rows = {}
    for m in MODELS:
        r = row(LABEL[m], load(f"results_audio/decision/{m}/metrics.json"))
        if r:
            en_rows[m] = r
            L.append(fmt(r, f"results_audio/decision/{m}"))

    # ------------------------------------------------ the degeneration table
    L += ["", "## The distilled model's degeneration, and the prompt that fixes it",
          "",
          "The language judge labels a reply by the language it judges the turn "
          "to be in, and it reads a turn from its first word. The distilled "
          "model opens many replies with a stray non-Latin token "
          "(\"\uc544! You're saying you know *a lot* of people ...\"), so the "
          "judge calls those turns Korean even though the body of the reply is "
          "in the model's own language. The `main language` column is the same "
          "judgement on the same reply with that leading token removed: it is "
          "what the model was actually saying.",
          "",
          "| run | reply: pt / en / other | main language: pt / en / other | replies opening with a stray token | words per turn | replies | relevance | floor latency | ATA |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for label, res in (("PT, no prompt", "results_audio_pt/decision/distill"),
                       ("PT, production prompt", "results_audio_pt_deployed_tags/decision/distill"),
                       ("EN, no prompt", "results_audio/decision/distill")):
        lang = language(res)
        clean = json.loads((Path(res) / "language_llm_clean.json").read_text()) \
            if (Path(res) / "language_llm_clean.json").is_file() else None
        m = row(LABEL["distill"], load(Path(res) / "metrics_whisper.json")) or \
            row(LABEL["distill"], load(Path(res) / "metrics.json"))
        d = drift(res)
        if not (lang and m):
            continue
        L.append(
            f"| {label} | {lang.get('share_turns_pt',0):.1%} / "
            f"{lang.get('share_turns_en',0):.1%} / {lang.get('share_turns_other',0):.1%} | "
            + (f"{clean.get('share_turns_pt',0):.1%} / {clean.get('share_turns_en',0):.1%} / "
               f"{clean.get('share_turns_other',0):.1%} | "
               if clean else "| | ")
            + f"{d['share_opening']:.1%} | {_words_per_turn(res):.1f} | "
              f"{m['rel']:.2f} | {m['smooth_lat']:.2f} s | {m['ata']:.3f} (raw {m['raw']:.3f}) |")

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
