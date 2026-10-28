"""Phase 4: the audio Full-Duplex-Bench run.

For every scenario in the five v1.0 subsets: replay the benchmark's own user WAV
through the deployed STT semantics, let the model decide as the words settle, turn
its replies into real speech with pocket-tts, and render an ``output.wav`` that
shares the input's time origin. Then score it with the benchmark's own scripts.

Run one model per invocation (the LLM is served once and the whole sweep runs
against it):

    python -m fdb_bench.audio.run_audio --model sft --out results_audio/en/sft

An interruption judge server (llama.cpp, Qwen3.5-9B) is needed for the relevance
column; ``--skip-score`` runs inference only.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .. import protocol as P
from ..adapters import build_adapter
from .dataset import DEFAULT_ROOT, TASKS, iter_scenarios
from .render import render_output, write_wav
from .session import AudioDuplexSession
from .stt import AudioUserStream, TraceUserStream, load_wav_24k
from .tts import AssistantVoice

TASK_OF_SUBSET = dict(TASKS)


def _serialise(res, render_info: dict) -> dict:
    return {
        "model": res.model, "subset": res.subset, "task": res.task, "sid": res.sid,
        "mode": res.mode, "duration_s": round(res.duration_s, 3),
        "end_time": round(res.end_time, 3),
        "gt_user_speech_s": round(res.gt_user_speech_s, 3),
        "spoke": res.spoke(), "formal_turn_by_audio": res.formal_turn_by_audio(),
        "first_speech_start": round(min([t.start for t in res.turns], default=-1), 3),
        "audible_within_input": bool(render_info["placed_within_input_s"] > 0.0),
        "speech_beyond_input_s": render_info["speech_beyond_input_s"],
        "speech_intervals": [[round(a, 3), round(b, 3)] for a, b in res.speech_intervals],
        "n_turns": len(res.turns),
        "n_decisions": res.n_decisions,
        "compute_s": round(res.compute_s, 3),
        "max_decision_s": round(res.max_decision_s, 3),
        "tts_compute_s": round(res.tts_compute_s, 3),
        "stt_s": round(res.stt_s, 3), "stt_partial_s": round(res.stt_partial_s, 3),
        "stt_final_s": round(res.stt_final_s, 3), "stt_calls": res.stt_calls,
        "floor_compute_s": round(res.floor_compute_s, 4),
        "yield_compute_s": round(res.yield_compute_s, 4),
        "asr_settle_latency_s": res.asr_settle_latency_s,
        "barge_ins": res.barge_ins,
        "utt_finalized": res.utt_finalized,
        "utt_skipped": res.utt_skipped,
        "audio_beyond_input_s": round(res.audio_beyond_input_s, 3),
        "tts_capped": res.tts_capped, "tts_capped_words": res.tts_capped_words,
        "paused_floor_take": res.paused_floor_take,
        "interrupt_span": list(res.interrupt_span) if res.interrupt_span else None,
        "post_interrupt_text": res.post_interrupt_text[:2000],
        "took_floor_after_interrupt": res.took_floor_after_interrupt,
        "tags": res.tags,
        "turns": [
            {"start": round(t.start, 3), "end": round(t.end, 3),
             "natural_end": round(t.natural_end, 3), "dur": round(t.duration_s, 3),
             "text": t.text[:400], "tag": t.tag, "kind": t.kind,
             "n_words": t.n_words, "stopped": t.stop_at is not None}
            for t in res.turns
        ],
        "user_pieces": res.user_pieces,
        "user_speech_spans": [[a, b] for a, b in res.user_speech_spans],
        "decisions": res.decisions,
        "render": render_info,
    }


def _aggregate(results_dir: Path, data_root: Path) -> dict:
    """Task-level roll-up from the per-scenario traces (audio ground truth)."""
    agg: dict = {}
    for subset in TASK_OF_SUBSET:
        d = results_dir / subset
        if not d.is_dir():
            continue
        rows = [json.loads(p.read_text(encoding="utf-8"))
                for p in sorted(d.glob("*/trace.json"))]
        if not rows:
            continue
        late_only = [r for r in rows
                     if r["spoke"] and not r.get("audible_within_input", True)]
        block = {
            "n": len(rows),
            # the model's decision: did it emit a turn with any audio at all
            "spoke_rate": round(sum(r["spoke"] for r in rows) / len(rows), 4),
            # what FDB's ASR would see: speech that actually landed in the window
            "audible_within_input_rate": round(
                sum(1 for r in rows if r.get("audible_within_input")) / len(rows), 4),
            # the two differ when a response began after the input ended
            "spoke_but_late_only_rate": round(len(late_only) / len(rows), 4),
            "mean_first_speech_start_s": round(
                sum(r.get("first_speech_start", -1) for r in rows if r["spoke"])
                / max(1, sum(r["spoke"] for r in rows)), 3),
            "formal_turn_by_audio": round(
                sum(r["formal_turn_by_audio"] for r in rows) / len(rows), 4),
            "mean_compute_s": round(sum(r["compute_s"] for r in rows) / len(rows), 3),
            "max_decision_s": round(max(r["max_decision_s"] for r in rows), 3),
            "mean_tts_s": round(sum(r.get("tts_compute_s", 0.0) for r in rows) / len(rows), 3),
            "mean_stt_s": round(sum(r.get("stt_s", 0.0) for r in rows) / len(rows), 3),
            "mean_stt_final_s": round(sum(r.get("stt_final_s", 0.0) for r in rows) / len(rows), 3),
            "tts_capped_turns": sum(r.get("tts_capped", 0) for r in rows),
            "n_finalized": sum(r["utt_finalized"] for r in rows),
            "n_skipped_utt": sum(sum(r["utt_skipped"].values()) for r in rows),
            "clipped_speech_s": round(sum(r["render"]["clipped_speech_s"] for r in rows), 3),
        }
        # latency against the benchmark's own reference point, from real audio:
        # (first assistant speech) - (annotated turn end / interruption end)
        lat = []
        for r in rows:
            ref = None
            if subset == "candor_turn_taking":
                p = data_root / subset / subset / r["sid"] / "turn_taking.json"
                if p.is_file():
                    ref = float(json.loads(p.read_text())[0]["timestamp"][0])
            elif subset == "synthetic_user_interruption" and r["interrupt_span"]:
                ref = float(r["interrupt_span"][1])
            if ref is None:
                continue
            after = [t["start"] for t in r["turns"] if t["start"] >= ref - 0.25]
            if after and r["spoke"]:
                lat.append(min(after) - ref)
        if lat:
            block["latency_vs_reference_s"] = round(sum(lat) / len(lat), 3)
            block["latency_n"] = len(lat)
        agg[subset] = block
    return agg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(P.MODEL_REGISTRY))
    ap.add_argument("--out", required=True, type=Path,
                    help="repo results dir: traces, transcripts, metrics")
    ap.add_argument("--audio-root", type=Path,
                    default=Path("/mnt/f/fdb_audio"),
                    help="where the rendered output.wav files go. They are the "
                         "bulk of a run (hundreds of MB) and FDB's evaluators "
                         "only need them at scoring time, so they live outside "
                         "the repo and are symlinked into a scratch tree to score.")
    ap.add_argument("--score-tmp", type=Path, default=None,
                    help="scratch tree for scoring (default /tmp/opencode/audio_score)")
    ap.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--tasks", default="all",
                    help="comma-separated subset names, or 'all'")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--lang", default="en", choices=["pt", "en"],
                    help="prompt/language pack; FDB v1.0 audio is English")
    ap.add_argument("--mode", default="decision", choices=["decision", "prime"])
    ap.add_argument("--api-base", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--big-api-base", default=None)
    ap.add_argument("--judge-api-base", default="http://127.0.0.1:8099/v1")
    ap.add_argument("--prompt-candidate", default=None)
    ap.add_argument("--floor-by-text", action="store_true",
                    help="treat any assistant text as taking the floor. The deployed "
                         "distill bridge works this way: its prompt says not to print "
                         "tags, the text goes straight to TTS, and the turn ends on "
                         "<|user is thinking|>.")
    ap.add_argument("--system-override", default=None,
                    help="replace the model's system prompt entirely (used to "
                         "measure whether pinning the reply language fixes a "
                         "model that code-switches)")
    ap.add_argument("--device", default="cuda")
    # STT / VAD
    ap.add_argument("--whisper-model", default="medium")
    ap.add_argument("--whisper-compute", default="float16")
    ap.add_argument("--vad", default="silero", choices=["silero", "energy"])
    ap.add_argument("--silence-s", type=float, default=0.6)
    ap.add_argument("--partial-interval-s", type=float, default=0.6)
    ap.add_argument("--partial-commit-s", type=float, default=0.5)
    ap.add_argument("--stt-trace-root", type=Path, default=None,
                    help="replay the recorded ASR pieces in "
                         "<root>/<subset>/<sid>/trace.json instead of running "
                         "faster-whisper. Used for a paired audio run against a "
                         "stored model without paying for streaming ASR.")
    # TTS
    ap.add_argument("--tts-language", default="english")
    ap.add_argument("--tts-temp", type=float, default=0.7)
    ap.add_argument("--tail-s", type=float, default=4.0,
                    help="render this many seconds past the end of the input so a "
                         "late answer still lands (FDB's own outputs are exactly "
                         "input-length; see render.py)")
    ap.add_argument("--max-speech-s", type=float, default=4.0,
                    help="synthesise at most this much past the end of the input")
    ap.add_argument("--no-tts", action="store_true",
                    help="skip TTS (debug: no audio, no scoring)")
    # scoring
    ap.add_argument("--skip-score", action="store_true")
    ap.add_argument("--score-only", action="store_true",
                    help="reuse existing output.wav files and only rescore")
    ap.add_argument("--judge-max-tokens", type=int, default=320)
    args = ap.parse_args()

    tasks = list(TASK_OF_SUBSET) if args.tasks == "all" else args.tasks.split(",")
    for t in tasks:
        if t not in TASK_OF_SUBSET:
            raise SystemExit(f"unknown subset {t!r}; known: {list(TASK_OF_SUBSET)}")

    scenarios = iter_scenarios(
        tasks, root=args.data_root,
        limits={t: args.limit for t in tasks} if args.limit else None,
    )
    print(f"[audio] {len(scenarios)} scenarios across {len(tasks)} subsets", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)

    adapter = voice = whisper = None
    if not args.score_only:
        adapter = build_adapter(
            args.model, api_base=args.api_base, big_api_base=args.big_api_base,
            device=args.device, prime_mode=args.mode, lang=args.lang,
            prompt_candidate=args.prompt_candidate,
            system_override=args.system_override,
            floor_by_text=(True if args.floor_by_text else None),
            stream="audio",
        )
        from faster_whisper import WhisperModel

        whisper = None if args.stt_trace_root else WhisperModel(
            args.whisper_model, device=args.device,
            compute_type=args.whisper_compute)
        if not args.no_tts:
            voice = AssistantVoice(args.tts_language, device=args.device,
                                   temp=args.tts_temp)
            print(f"[audio] tts voice={voice.voice} language={voice.language}", flush=True)

    # ---------------------------------------------------------------- pass 1
    n = 0
    t0 = time.time()
    for sc in scenarios:
        sample_dir = args.out / sc.subset / sc.sid
        trace_path = sample_dir / "trace.json"
        wav_path = args.audio_root / args.mode / args.model / sc.subset / sc.sid / "output.wav"
        if args.score_only:
            continue  # scoring is driven by the folders that exist on disk
        if wav_path.is_file() and trace_path.is_file():
            n += 1
            continue
        sample_dir.mkdir(parents=True, exist_ok=True)
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        pcm = load_wav_24k(sc.input_wav)
        if args.stt_trace_root:
            trace_file = args.stt_trace_root / sc.subset / sc.sid / "trace.json"
            if not trace_file.is_file():
                raise SystemExit(f"missing ASR trace {trace_file}")
            stream = TraceUserStream(
                json.loads(trace_file.read_text(encoding="utf-8")))
        else:
            stream = AudioUserStream(
                whisper, language=args.lang, vad=args.vad, silence_s=args.silence_s,
                partial_interval_s=args.partial_interval_s,
                partial_commit_s=args.partial_commit_s,
            )
        session = AudioDuplexSession(adapter, stream, voice, mode=args.mode,
                                     max_speech_s=args.max_speech_s)
        res = session.run(sc, pcm)
        samples, info = render_output(res, sc.duration_s, tail_s=args.tail_s)
        write_wav(wav_path, samples)
        trace_path.write_text(
            json.dumps(_serialise(res, info), ensure_ascii=False), encoding="utf-8")
        n += 1
        if n % 25 == 0 or n <= 3:
            rate = (time.time() - t0) / max(n, 1)
            print(f"[audio/{args.model}] {n}/{len(scenarios)} queue "
                  f"({rate:.1f}s/scenario, {res.duration_s:.1f}s audio) "
                  f"spoke={res.spoke()} pieces={res.n_decisions}", flush=True)

    # ---------------------------------------------------------------- pass 2
    metrics: dict = {}
    if not args.skip_score and not args.no_tts:
        from .score import LocalJudgeShim, ParakeetScorer, run_task_eval

        import os

        scorer = ParakeetScorer()
        judge = LocalJudgeShim(args.judge_api_base, max_tokens=args.judge_max_tokens)
        # The upstream evaluators insist on one folder per sample holding the
        # audio, the transcript *and* the task annotation (`turn_taking.json`,
        # `pause.json`, `interrupt.json`). The audio stays on /mnt/f, so scoring
        # builds that folder as a symlink farm: nothing is copied except the
        # annotations (tiny) and the parakeet transcript (kept in the repo as
        # evidence). The evaluators themselves see an ordinary directory.
        annotations = ("pause.json", "turn_taking.json", "interrupt.json")
        # Keyed by dataset as well as mode+model: without the data root, a second
        # run over a *different* audio set reused the first run's score tree, and
        # since `output.json` is only regenerated when it is missing, the
        # evaluators scored the previous set's transcripts against the new
        # annotations. Measured on the pt-BR set: every output.json was a leftover
        # from the English run, and only the one column the evaluator derives from
        # the audio itself (backchannel) came out different. Costs a re-ASR pass,
        # which is the point.
        score_root = (args.score_tmp or Path("/tmp/opencode/audio_score")) \
            / f"{Path(args.data_root).name}_{args.mode}_{args.model}"
        for subset in tasks:
            audio_dir = args.audio_root / args.mode / args.model / subset
            # published layout is <root>/<subset>/<subset>/; the pt-BR
            # re-voicing writes <root>/<subset>/ directly
            src_root = args.data_root / subset / subset
            if not src_root.is_dir():
                src_root = args.data_root / subset
            sdirs = sorted([p for p in audio_dir.glob("*") if p.is_dir()
                            and (p / "output.wav").is_file()],
                           key=lambda p: int(p.name) if p.name.isdigit() else 0)
            if not sdirs:
                continue
            root = score_root / subset
            for sd in sdirs:
                d = root / sd.name
                d.mkdir(parents=True, exist_ok=True)
                if src_root.is_dir():
                    for name in annotations:
                        src = src_root / sd.name / name
                        if src.is_file():
                            (d / name).write_text(src.read_text(encoding="utf-8"),
                                                  encoding="utf-8")
                link = d / "output.wav"
                if link.is_symlink() or link.exists():
                    link.unlink()
                os.symlink((sd / "output.wav").resolve(), link)
            print(f"[score] {subset}: {len(sdirs)} samples -> parakeet", flush=True)
            for sd in sdirs:
                d = root / sd.name
                if not (d / "output.json").is_file() or not args.score_only:
                    scorer.write_output_json(d, TASK_OF_SUBSET[subset])
                # keep the transcript in the repo next to the trace
                (args.out / subset / sd.name).mkdir(parents=True, exist_ok=True)
                (args.out / subset / sd.name / "output.json").write_text(
                    (d / "output.json").read_text(encoding="utf-8"), encoding="utf-8")
            res = run_task_eval(root, TASK_OF_SUBSET[subset], judge=judge)
            metrics[subset] = res
            print(f"[score] {subset}: {json.dumps(res, ensure_ascii=False)[:300]}", flush=True)

    audio_agg = _aggregate(args.out, args.data_root)
    (args.out / "metrics.json").write_text(json.dumps({
        "model": args.model, "mode": args.mode, "lang": args.lang,
        "vad": args.vad, "silence_s": args.silence_s,
        "partial_interval_s": args.partial_interval_s,
        "partial_commit_s": args.partial_commit_s,
        "whisper": args.whisper_model, "tts_language": args.tts_language,
        "tts_temp": args.tts_temp, "asr_scorer": "nvidia/parakeet-tdt-0.6b-v2",
        "stt": (f"trace:{args.stt_trace_root}" if args.stt_trace_root
                else f"faster-whisper:{args.whisper_model}"),
        "judge": args.judge_api_base,
        "official_task_metrics": metrics,
        "audio_rollup": audio_agg,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.out / "meta.json").write_text(json.dumps({
        "model": args.model, "mode": args.mode, "n_scenarios": len(scenarios),
        "limit": args.limit, "tasks": tasks,
    }, indent=2), encoding="utf-8")
    print(f"[audio] done -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
