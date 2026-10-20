"""Does the re-voiced pt-BR set still have the *structure* of the benchmark?

The first pt-BR build was faithful in its annotations and unfaithful in its
audio: every translated span was time-fitted into the English span it replaced
and the remainder was padded with digital silence. That padding sat *inside* the
user's turn, so the harness's 0.6 s silence gate fired on it and the model was
invited to take the floor in the middle of a sentence. Measured on the traces:
invitations landing in silence the benchmark does not annotate as a pause went
from 58 (English) to 303 (Portuguese) over the same 353 scenarios, and at every
class of invitation every model took the floor 1.2-2.0x more often -- including
the English-trained upstream model, which cannot be a language-skill effect.

This is the gate that would have caught it, and the gate for the rebuild: the
Portuguese audio must have the same spans and the same gaps as the recording it
was made from, up to the fact that a Portuguese speaker says it in a different
number of seconds.

    python -m fdb_bench.audio.check_pt_structure --en /mnt/f/fdb_data/v1.0 \\
        --pt /mnt/f/fdb_data/pt_v1.0
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import numpy as np

SILENCE_GATE_S = 0.6     # the harness's turn-end gate (stt.py)
TOL_SPOKE_FRAC = 0.12    # speech fraction may differ by this much
TOL_GAPS = 0.35          # gaps >= 0.6 s per scenario may differ by this much
TOL_PHANTOM = 0.60       # phantom silence per scenario may differ by this much


def spans_of(wav: np.ndarray) -> list[tuple[float, float]]:
    from .stt import silero_spans

    return silero_spans(wav)


def read_wav(path: Path) -> np.ndarray:
    import soundfile as sf

    w, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if w.ndim > 1:
        w = w.mean(axis=1)
    if sr != 24000:
        from scipy.signal import resample_poly

        g = np.gcd(int(sr), 24000)
        w = resample_poly(w, 24000 // g, int(sr) // g).astype(np.float32)
    return np.ascontiguousarray(w, dtype=np.float32)


def pause_windows(sample_dir: Path) -> list[tuple[float, float]]:
    p = sample_dir / "pause.json"
    if not p.is_file():
        return []
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return []
    return [(float(i["timestamp"][0]), float(i["timestamp"][1]))
            for i in doc if i.get("timestamp")]


def measure(wav: np.ndarray, pauses: list[tuple[float, float]]) -> dict:
    spans = spans_of(wav)
    total = wav.size / 24000
    speech = sum(max(0.0, e - s) for s, e in spans)
    gaps = [spans[i + 1][0] - spans[i][1] for i in range(len(spans) - 1)]
    long_gaps = [g for g in gaps if g >= SILENCE_GATE_S]
    inside = 0.0
    for a, b in pauses:
        inside += sum(max(0.0, min(e, b) - max(s, a)) for s, e in spans if e > a and s < b)
    ann = sum(max(0.0, b - a) for a, b in pauses)
    return {
        "total_s": total,
        "spans": len(spans),
        "speech_s": speech,
        "speech_frac": speech / max(1e-9, total),
        "n_long_gaps": len(long_gaps),
        "phantom_s": max(0.0, (total - speech) - (ann - inside)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--en", type=Path, default=Path("/mnt/f/fdb_data/v1.0"))
    ap.add_argument("--pt", type=Path, default=Path("/mnt/f/fdb_data/pt_v1.0"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    rows = []
    subsets = sorted(p.name for p in args.pt.iterdir() if p.is_dir())
    for subset in subsets:
        pt_root = args.pt / subset
        en_root = args.en / subset / subset
        if not en_root.is_dir():
            en_root = args.en / subset
        for d in sorted(pt_root.iterdir()):
            if not d.is_dir() or not (d / "input.wav").is_file():
                continue
            en_wav = en_root / d.name / "input.wav"
            if not en_wav.is_file():
                continue
            en_m = measure(read_wav(en_wav), pause_windows(en_root / d.name))
            pt_m = measure(read_wav(d / "input.wav"), pause_windows(d))
            rows.append({"subset": subset, "sid": d.name, "en": en_m, "pt": pt_m})
            if args.limit and len(rows) >= args.limit:
                break

    def avg(key, which, sub=None):
        vals = [r[which][key] for r in rows if sub is None or r["subset"] == sub]
        return st.mean(vals) if vals else float("nan")

    print(f"paired scenarios: {len(rows)}")
    hdr = (f"{'subset':28s} {'n':>4s} | {'speech frac EN':>14s} {'PT':>7s} | "
           f"{'gaps>=0.6s EN':>13s} {'PT':>6s} | {'phantom s EN':>13s} {'PT':>7s}")
    print(hdr)
    print("-" * len(hdr))
    for subset in sorted({r["subset"] for r in rows}):
        n = sum(1 for r in rows if r["subset"] == subset)
        print(f"{subset:28s} {n:4d} | {avg('speech_frac','en',subset):14.3f} "
              f"{avg('speech_frac','pt',subset):7.3f} | {avg('n_long_gaps','en',subset):13.2f} "
              f"{avg('n_long_gaps','pt',subset):6.2f} | {avg('phantom_s','en',subset):13.2f} "
              f"{avg('phantom_s','pt',subset):7.2f}")
    print("-" * len(hdr))
    d_frac = avg('speech_frac', 'pt') - avg('speech_frac', 'en')
    d_gaps = avg('n_long_gaps', 'pt') - avg('n_long_gaps', 'en')
    d_phant = avg('phantom_s', 'pt') - avg('phantom_s', 'en')
    print(f"{'ALL':28s} {len(rows):4d} | {avg('speech_frac','en'):14.3f} "
          f"{avg('speech_frac','pt'):7.3f} | {avg('n_long_gaps','en'):13.2f} "
          f"{avg('n_long_gaps','pt'):6.2f} | {avg('phantom_s','en'):13.2f} "
          f"{avg('phantom_s','pt'):7.2f}")
    ok = (abs(d_frac) <= TOL_SPOKE_FRAC and abs(d_gaps) <= TOL_GAPS
          and abs(d_phant) <= TOL_PHANTOM)
    print(f"\ndeltas: speech frac {d_frac:+.3f} (<= {TOL_SPOKE_FRAC}), "
          f"gaps>=0.6s {d_gaps:+.2f} (<= {TOL_GAPS}), "
          f"phantom silence {d_phant:+.2f}s (<= {TOL_PHANTOM})")
    print(f"STRUCTURE GATE: {'PASS' if ok else 'FAIL'}")
    if args.out:
        args.out.write_text(json.dumps({"rows": rows, "gate": "PASS" if ok else "FAIL"},
                                       indent=1), encoding="utf-8")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
