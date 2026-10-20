"""Merge parakeet and whisper-pt metrics into the pt-BR comparison table."""
from __future__ import annotations
import json
from pathlib import Path
from fdb_bench.metrics import averaged_turn_taking_accuracy as ata

LABEL = {"original_dc": "Original DuplexCascade (upstream, EN)",
         "sft": "SFT (pt-BR)", "distill": "Self-distillation (pt-BR)",
         "mtp_like": "Companion (MTP-style)",
         "distill_trigger": "Self-distillation + training prompt"}

def load(p):
    p = Path(p)
    return json.loads(p.read_text())["official_task_metrics"] if p.is_file() else None

def row(tag, o):
    if not o: return None
    g = lambda s, k: o.get(s, {}).get(k)
    p1, p2 = g("synthetic_pause_handling","tor"), g("candor_pause_handling","tor")
    bc, sm, it = g("icc_backchannel","tor"), g("candor_turn_taking","tor"), g("synthetic_user_interruption","tor")
    sml, itl = g("candor_turn_taking","latency_s"), g("synthetic_user_interruption","latency_s")
    rel = g("synthetic_user_interruption","relevance")
    a = ata({"pause_synthetic_tor":p1,"pause_candor_tor":p2,"smooth_tor":sm,"intr_tor":it})
    return (tag, p1, p2, bc, sm, sml, it, rel, itl, a)

def main():
    L=["# pt-BR Full-Duplex-Bench: real audio, two scorers\n",
       "`parakeet` is the benchmark's own recipe (`get_transcript/asr.py`) and is "
       "**English-only**; `whisper-pt` re-scores the identical audio and "
       "annotations with `faster-whisper large-v3 --language pt`. The models answer "
       "in Portuguese for 84-98% of turns on this set, so only the whisper rows are "
       "interpretable -- the parakeet rows are kept to show what the upstream "
       "recipe reports when it cannot read the replies.\n",
       "| model | scorer | pause_syn ↓ | pause_cand ↓ | BC TOR ↓ | smooth ↑ | smooth lat ↓ | intr ↑ | relevance ↑ | intr lat ↓ | **ATA** ↑ |",
       "|---|---|---|---|---|---|---|---|---|---|---|"]
    rows=[]
    for m in ("original_dc","sft","distill","mtp_like"):
        for sc, tag in ((f"results_audio_pt/decision/{m}/metrics.json","parakeet"),
                        (f"results_audio_pt/decision/{m}/metrics_whisper.json","whisper-pt")):
            r = row(LABEL.get(m,m), load(sc))
            if r: rows.append((m, tag, r))
    for m, tag, r in rows:
        _,p1,p2,bc,sm,sml,it,rel,itl,a = r
        L.append(f"| {r[0]} | {tag} | {p1:.3f} | {p2:.3f} | {bc:.3f} | {sm:.3f} | {sml:.2f} | {it:.3f} | {rel:.2f} | {itl:.2f} | **{a:.3f}** |")
    dp = row("Self-distillation + deployed prompt (prime)", load("results_audio_pt_deployed/prime/distill/metrics_whisper.json"))
    if dp:
        _,p1,p2,bc,sm,sml,it,rel,itl,a = dp
        L.append(f"| {dp[0]} | whisper-pt | {p1:.3f} | {p2:.3f} | {bc:.3f} | {sm:.3f} | {sml:.2f} | {it:.3f} | {rel:.2f} | {itl:.2f} | **{a:.3f}** |")
    tr = row(LABEL["distill_trigger"], load("results_audio_pt_trigger/decision/distill/metrics_whisper.json"))
    if tr:
        _,p1,p2,bc,sm,sml,it,rel,itl,a = tr
        L.append(f"| {tr[0]} | whisper-pt | {p1:.3f} | {p2:.3f} | {bc:.3f} | {sm:.3f} | {sml:.2f} | {it:.3f} | {rel:.2f} | {itl:.2f} | **{a:.3f}** |")
    L += ["", "## English reference (parakeet is the right scorer there)", "",
          "| model | pause_syn ↓ | pause_cand ↓ | BC TOR ↓ | smooth ↑ | smooth lat ↓ | intr ↑ | relevance ↑ | intr lat ↓ | **ATA** ↑ |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for m in ("original_dc","sft","distill","mtp_like"):
        r = row(LABEL.get(m,m), load(f"results_audio/decision/{m}/metrics.json"))
        if r:
            _,p1,p2,bc,sm,sml,it,rel,itl,a = r
            L.append(f"| {r[0]} | {p1:.3f} | {p2:.3f} | {bc:.3f} | {sm:.3f} | {sml:.2f} | {it:.3f} | {rel:.2f} | {itl:.2f} | **{a:.3f}** |")
    out = Path("results_audio_pt/PT_COMPARISON.md"); out.write_text("\n".join(L)+"\n", encoding="utf-8")
    print("\n".join(L)); print(f"\nwrote {out}")

if __name__ == "__main__":
    main()
