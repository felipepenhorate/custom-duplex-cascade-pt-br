"""Publish the pt-BR Full-Duplex-Bench to the Hugging Face Hub.

Two things get published, under one repository, as two configurations:

``pt_audio``
    The Portuguese re-voicing of FDB v1.0: 727 scenarios, each the benchmark's
    own scenario re-voiced in pt-BR (the user side), with the audio, the
    benchmark's annotations re-timed onto it, the Portuguese transcript, and the
    provenance of the re-voicing (the English source text of every span and
    what it was translated to). This is the set the evaluation in
    ``fdb_bench/audio/`` runs on, so a reader can reproduce any number in the
    paper from it.

``pt_text`` / ``en_text``
    The synthetic Portuguese (and its parallel English) Full-Duplex-Bench: the
    same five subsets and the same scenario structure as the benchmark, in text,
    which is what the text-domain harness measures.

The audio is stored as the exact bytes the evaluation consumed, so the published
set is byte-identical to the one the numbers come from.

    python -m fdb_bench.publish_dataset --repo lumierenoir/synthetic-full-duplex-bench-pt-br
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

DEFAULT_REPO = "lumierenoir/synthetic-full-duplex-bench-pt-br"
DEFAULT_PT_ROOT = Path("/mnt/f/fdb_data/pt_v1.0")
DEFAULT_OUT = Path("/tmp/opencode/hf_pt_br")

TASK_OF = {
    "candor_pause_handling": "pause_handling",
    "synthetic_pause_handling": "pause_handling",
    "candor_turn_taking": "smooth_turn_taking",
    "icc_backchannel": "backchannel",
    "synthetic_user_interruption": "user_interruption",
}

CARD = """---
configs:
- config_name: pt_audio
  data_files:
  - split: train
    path: data/pt_audio/train-*.parquet
- config_name: pt_text
  data_files:
  - split: train
    path: data/pt_text/train-*.parquet
- config_name: en_text
  data_files:
  - split: train
    path: data/en_text/train-*.parquet
---
# Full-Duplex-Bench, pt-BR

The Full-Duplex-Bench v1.0 (arXiv:2603.09180) in Brazilian Portuguese: the
benchmark's own five subsets, 727 scenarios, re-voiced.

## `pt_audio` -- the benchmark re-voiced in pt-BR

Each row is one scenario of FDB v1.0 with the **user side** in Portuguese. The
benchmark's audio is segmented with silero (the benchmark's own detector), each
span is translated and spoken by pocket-tts, and the result is put back on the
original timeline with the gaps preserved and the benchmark's annotations
re-timed onto it.

| column | what it is |
|---|---|
| `audio` | the scenario's user-side audio, 24 kHz mono, exactly the bytes the evaluation consumed |
| `words` | the Portuguese word-level transcript with its timestamps |
| `pause`, `turn_taking`, `interrupt` | the benchmark's annotations, re-timed onto the Portuguese timeline |
| `segments` | the provenance of the re-voicing: every span's English source text, its translation, and where it landed |
| `voice`, `policy`, `source_duration_s`, `built_duration_s` | how the audio was made |

**Why the set is built the way it is.** The obvious construction -- time-fit
each translated span into the English span it replaces and pad the remainder
with silence -- is wrong, and measurably so. The pad is silence *inside* the
user's turn; the harness's 0.6 s silence gate fires on it, so the model is
invited to take the floor in the middle of a sentence. In that first build 30%
of the segments carried such a pad and the turn-end invitations that landed in
silence the benchmark does not annotate as a pause went from 58 on the English
recording to 303 on the Portuguese one, over the same 353 scenarios; every
model answered more often because of it, the English-trained upstream release
included, so the artefact could not be a language effect.

This set is therefore built with **nothing stretched and nothing padded**: the
Portuguese speaker keeps their own speed, the gap between two consecutive spans
keeps its length exactly, and the annotations are re-timed with the same map.
The `segments` column records where each span landed. A structural gate
(`fdb_bench/audio/check_pt_structure.py`) compares the result with the
recording it was made from -- speech fraction, spans, gaps of 0.6 s or more,
silence outside the benchmark's own pause windows -- and the published set
matches it: speech fraction 0.549 against 0.583, gaps of 0.6 s or more 0.94
against 0.98 per scenario, phantom silence 8.51 s against 8.44 s.

## `pt_text` and `en_text` -- the synthetic bench

The same five subsets and the same scenario structure in text, in Portuguese and
in English. Each row carries the user-side stream (the pieces, whether the turn
ends at each, the silence ticks) and, for the interruption and backchannel
subsets, the interrupting turn and the ground-truth backchannel times.

The two languages share the generator, the scenario sizes, the chunking and the
topic inventories (the Portuguese and English topics are translations of each
other), but they are **two draws from that distribution**, not the same
dialogues in two languages: the seed is offset per language
(`build_all(lang=..., seed=1234)` in `fdb_bench/scenarios.py`), so a comparison
across the two is controlled in distribution rather than dialogue by dialogue.
Both corpora are deterministic and regenerate exactly.

## Provenance

The user-side audio and the annotations derive from Full-Duplex-Bench v1.0
(sbintuitions/DuplexCascade, arXiv:2603.09180); this repository contains the
Portuguese re-voicing of it and the synthetic text corpus, not the original
recording. The translations and the synthesised voice are ours.
"""


def _read_json(p: Path, default=None):
    if not p.is_file():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


def audio_rows(pt_root: Path) -> list[dict]:
    rows = []
    for subset_dir in sorted(p for p in pt_root.iterdir() if p.is_dir()):
        subset = subset_dir.name
        for d in sorted(subset_dir.iterdir()):
            wav = d / "input.wav"
            if not wav.is_file():
                continue
            tr = _read_json(d / "translation.json", {}) or {}
            segments = [{
                "text_en": s.get("text_en", ""),
                "text_pt": s.get("text_pt", ""),
                "en_start": s.get("start"),
                "en_end": s.get("end"),
                "start": s.get("new_start"),
                "end": s.get("new_end"),
                "gain": s.get("gain"),
            } for s in tr.get("segments", [])]
            rows.append({
                "subset": subset,
                "sid": d.name,
                "task": TASK_OF.get(subset, subset),
                "audio": {"bytes": wav.read_bytes(), "path": "input.wav"},
                "duration_s": round(tr.get("built_duration_s", 0.0), 3),
                "source_duration_s": round(tr.get("source_duration_s", 0.0), 3),
                "words": _read_json(d / "transcription.json", []) or [],
                "pause": _read_json(d / "pause.json", []) or [],
                "turn_taking": _read_json(d / "turn_taking.json", []) or [],
                "interrupt": _read_json(d / "interrupt.json", []) or [],
                "segments": segments,
                "voice": tr.get("voice"),
                "policy": tr.get("policy"),
            })
    return rows


def text_rows(lang: str) -> list[dict]:
    from .scenarios import build_all

    out = []
    for sc in build_all(lang=lang, seed=1234):
        out.append({
            "task": sc.task,
            "sid": sc.sid,
            "lang": sc.lang,
            "events": [{
                "kind": e.kind,
                "text": e.text,
                "ends_turn": bool(e.ends_turn),
                "is_backchannel": bool(e.is_backchannel),
                "ends_sentence": bool(e.ends_sentence),
                "after_turn_end": bool(e.after_turn_end),
                "duration": e.duration,
            } for e in sc.events],
            "context": sc.context,
            "interrupt": sc.interrupt,
            "interrupt_after_model_words": sc.interrupt_after_model_words,
            "gt_backchannel_times": list(sc.gt_backchannel_times),
        })
    return out


def write_config(rows: list[dict], out_dir: Path, name: str, *, audio: bool) -> None:
    """Write one configuration as parquet.

    Straight pyarrow, not the ``datasets`` library: the Audio feature there
    needs torchcodec to encode, and the on-disk layout it writes is the same
    ``{bytes, path}`` struct this writes by hand.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    ts = pa.list_(pa.float64())
    annotated = pa.struct([pa.field("text", pa.string()),
                           pa.field("timestamp", ts)])
    if audio:
        schema = pa.schema([
            pa.field("subset", pa.string()),
            pa.field("sid", pa.string()),
            pa.field("task", pa.string()),
            pa.field("audio", pa.struct([pa.field("bytes", pa.binary()),
                                         pa.field("path", pa.string())])),
            pa.field("duration_s", pa.float32()),
            pa.field("source_duration_s", pa.float32()),
            pa.field("words", pa.list_(annotated)),
            pa.field("pause", pa.list_(annotated)),
            pa.field("turn_taking", pa.list_(annotated)),
            pa.field("interrupt", pa.list_(pa.struct([
                pa.field("text", pa.string()),
                pa.field("timestamp", ts),
                pa.field("context", pa.string()),
                pa.field("interrupt", pa.string())]))),
            pa.field("segments", pa.list_(pa.struct([
                pa.field("text_en", pa.string()),
                pa.field("text_pt", pa.string()),
                pa.field("en_start", pa.float64()),
                pa.field("en_end", pa.float64()),
                pa.field("start", pa.float64()),
                pa.field("end", pa.float64()),
                pa.field("gain", pa.float64())]))),
            pa.field("voice", pa.string()),
            pa.field("policy", pa.string()),
        ])
    else:
        schema = pa.schema([
            pa.field("task", pa.string()),
            pa.field("sid", pa.string()),
            pa.field("lang", pa.string()),
            pa.field("events", pa.list_(pa.struct([
                pa.field("kind", pa.string()),
                pa.field("text", pa.string()),
                pa.field("ends_turn", pa.bool_()),
                pa.field("is_backchannel", pa.bool_()),
                pa.field("ends_sentence", pa.bool_()),
                pa.field("after_turn_end", pa.bool_()),
                pa.field("duration", pa.float64())]))),
            pa.field("context", pa.string()),
            pa.field("interrupt", pa.string()),
            pa.field("interrupt_after_model_words", pa.int32()),
            pa.field("gt_backchannel_times", pa.list_(pa.float64())),
        ])

    target = out_dir / "data" / name
    target.mkdir(parents=True, exist_ok=True)
    path = target / "train-00000-of-00001.parquet"
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, str(path), compression="zstd")
    print(f"  {name}: {table.num_rows} rows, {path.stat().st_size / 1e6:.0f} MB "
          f"-> {path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--pt-root", type=Path, default=DEFAULT_PT_ROOT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()

    if args.out.exists():
        shutil.rmtree(args.out)
    print("building the configurations", flush=True)
    write_config(audio_rows(args.pt_root), args.out, "pt_audio", audio=True)
    write_config(text_rows("pt"), args.out, "pt_text", audio=False)
    write_config(text_rows("en"), args.out, "en_text", audio=False)
    (args.out / "README.md").write_text(CARD, encoding="utf-8")

    if args.no_upload:
        print(f"staged at {args.out} (not uploaded)")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", exist_ok=True)
    print(f"uploading to {args.repo}", flush=True)
    api.upload_folder(
        repo_id=args.repo, repo_type="dataset",
        folder_path=str(args.out), commit_message=(
            "Full-Duplex-Bench in pt-BR: the re-voiced benchmark (727 scenarios "
            "with audio, annotations and the re-voicing provenance) and the "
            "synthetic Portuguese corpus with its parallel English"),
    )
    print(f"uploaded: https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
