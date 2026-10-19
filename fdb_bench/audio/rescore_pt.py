"""Re-score a pt-BR run with a Portuguese recogniser.

The benchmark's scoring recipe is ``nvidia/parakeet-tdt-0.6b-v2``, which is
**English-only**. That is fine for the English audio set and fatal for the pt-BR
one: the models answer in Portuguese, parakeet returns gibberish or nothing, and
the benchmark's own rule -- "if there is no transcription, the model does not take
the turn" -- turns a perfectly good Portuguese answer into TOR 0. Measured on the
upstream model's first pt-BR run: 469 Portuguese turns against 6 English ones, with
the scorer transcribing "Economizar dinheiro..." as "Economies of efficiency, the
mobility of value..." and two of three sampled interruptions as the empty string.

So this re-scores with faster-whisper ``large-v3`` at ``language="pt"``, emitting
the same ``{"text", "chunks":[{"text","timestamp"}]}`` schema that ``asr.py``
produces, and then runs the *same* upstream evaluators unchanged. Everything else
-- annotations, audio, the judge -- is identical, so the two scorings are directly
comparable and the difference between them is exactly what the English-only ASR
was costing.

Run (needs faster-whisper, i.e. the unsloth env, not the NeMo one):

    python -m fdb_bench.audio.rescore_pt --model original_dc
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np

from .dataset import TASKS
from .score import EVAL_DIR, LocalJudgeShim, run_task_eval

if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

SUBSET_OF = {"pause_handling": None}  # unused; kept for readability

_PATTERNS = {
    "pause_handling": {"tor": r"Average take turn:\s*([0-9.]+)"},
    "smooth_turn_taking": {"tor": r"Average take turn:\s*([0-9.]+)",
                           "latency_s": r"Average latency:\s*([0-9.]+)"},
    "backchannel": {"jsd_mean": r"JSD - Mean:\s*([0-9.]+)\s*±\s*([0-9.]+)",
                    "tor": r"TOR - Mean:\s*([0-9.]+)\s*±\s*([0-9.]+)",
                    "freq_mean": r"Frequency - Mean:\s*([0-9.]+)\s*±\s*([0-9.]+)"},
    "user_interruption": {"relevance": r"Average rating:\s*([0-9.]+)",
                          "tor": r"Average take turn:\s*([0-9.]+)",
                          "latency_s": r"Average latency:\s*([0-9.]+)"},
}


class WhisperScorer:
    """Word-timestamped transcription with a Portuguese-capable model."""

    def __init__(self, model: str = "large-v3", language: str = "pt") -> None:
        from faster_whisper import WhisperModel

        self.language = language
        self.model = WhisperModel(model, device="cuda", compute_type="float16")

    def words(self, wav_path: Path, *, crop_from: float | None = None) -> list[dict]:
        import soundfile as sf

        wav, sr = sf.read(str(wav_path))
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        offset = 0.0
        if crop_from is not None:
            offset = float(crop_from)
            wav = wav[int(offset * sr):]
        if wav.size < int(0.05 * sr):
            return []
        segs, _ = self.model.transcribe(
            wav.astype(np.float32), language=self.language, beam_size=5,
            temperature=0.0, word_timestamps=True,
            vad_filter=True, vad_parameters={"min_silence_duration_ms": 300},
        )
        out: list[dict] = []
        for s in segs:
            for w in (getattr(s, "words", None) or []):
                t = str(w.word).strip()
                if not t:
                    continue
                out.append({"text": t,
                            "timestamp": [float(w.start) + offset, float(w.end) + offset]})
        return out

    def write_output_json(self, sample_dir: Path, task: str) -> dict:
        crop = None
        if task == "user_interruption":
            meta = json.loads((sample_dir / "interrupt.json").read_text(encoding="utf-8"))
            crop = float(meta[0]["timestamp"][1])
        chunks = self.words(sample_dir / "output.wav", crop_from=crop)
        doc = {"text": " ".join(c["text"] for c in chunks).strip(), "chunks": chunks}
        (sample_dir / "output.json").write_text(json.dumps(doc, indent=4), encoding="utf-8")
        return doc


def parse(task: str, text: str) -> dict:
    out: dict = {}
    for key, pat in _PATTERNS[task].items():
        m = re.search(pat, text)
        if not m:
            continue
        out[key] = float(m.group(1))
        if m.lastindex and m.lastindex > 1:
            out[f"{key}_std"] = float(m.group(2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", default="decision")
    ap.add_argument("--results", type=Path, default=Path("results_audio_pt"))
    ap.add_argument("--audio-root", type=Path, default=Path("/mnt/f/fdb_audio_pt"))
    ap.add_argument("--data-root", type=Path, default=Path("/mnt/f/fdb_data/pt_v1.0"))
    ap.add_argument("--score-tmp", type=Path, default=Path("/tmp/opencode/score_pt_whisper"))
    ap.add_argument("--whisper", default="large-v3")
    ap.add_argument("--judge-api-base", default="http://127.0.0.1:8099/v1")
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--out-name", default="metrics_whisper.json")
    args = ap.parse_args()

    tasks = list(TASKS) if args.tasks == "all" else args.tasks.split(",")
    scorer = WhisperScorer(args.whisper, "pt")
    judge = LocalJudgeShim(args.judge_api_base)
    metrics: dict = {}
    for subset in tasks:
        audio_dir = args.audio_root / args.mode / args.model / subset
        if not audio_dir.is_dir():
            continue
        sdirs = sorted([p for p in audio_dir.glob("*") if p.is_dir()
                        and (p / "output.wav").is_file()],
                       key=lambda p: int(p.name) if p.name.isdigit() else 0)
        if not sdirs:
            continue
        root = args.score_tmp / f"pt_{args.mode}_{args.model}" / subset
        src_root = args.data_root / subset
        if not src_root.is_dir():
            src_root = args.data_root / subset / subset
        print(f"[rescore] {subset}: {len(sdirs)} samples", flush=True)
        for n, sd in enumerate(sdirs, 1):
            d = root / sd.name
            d.mkdir(parents=True, exist_ok=True)
            for name in ("pause.json", "turn_taking.json", "interrupt.json"):
                src = src_root / sd.name / name
                if src.is_file():
                    shutil.copyfile(src, d / name)
            link = d / "output.wav"
            if link.is_symlink() or link.exists():
                link.unlink()
            os.symlink((sd / "output.wav").resolve(), link)
            scorer.write_output_json(d, TASKS[subset])
            if n % 100 == 0:
                print(f"   {n}/{len(sdirs)} transcribed", flush=True)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            try:
                from eval_pause_handling import eval_pause_handling
                from eval_smooth_turn_taking import eval_smooth_turn_taking
                from eval_backchannel import eval_backchannel
                from eval_user_interruption import eval_user_interruption
            except Exception as e:  # noqa: BLE001
                print("import failed", e)
            if TASKS[subset] == "pause_handling":
                eval_pause_handling(str(root))
            elif TASKS[subset] == "smooth_turn_taking":
                eval_smooth_turn_taking(str(root))
            elif TASKS[subset] == "backchannel":
                cwd = os.getcwd()
                try:
                    os.chdir(EVAL_DIR)
                    eval_backchannel(str(root))
                finally:
                    os.chdir(cwd)
            elif TASKS[subset] == "user_interruption":
                eval_user_interruption(str(root), judge)
        res = parse(TASKS[subset], buf.getvalue())
        metrics[subset] = res
        print(f"[rescore] {subset}: {json.dumps({k:v for k,v in res.items()})}", flush=True)

    out = args.results / args.mode / args.model / args.out_name
    out.write_text(json.dumps({
        "model": args.model, "asr": f"faster-whisper {args.whisper}", "language": "pt",
        "note": "parakeet (the upstream recipe) is English-only; this re-scores the "
                "same audio and annotations with a Portuguese recogniser",
        "official_task_metrics": metrics,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[rescore] wrote {out}")


if __name__ == "__main__":
    main()
