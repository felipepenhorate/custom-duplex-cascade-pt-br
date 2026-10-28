"""Phase 1: run every scenario for one model and dump raw timelines.

One model per invocation (a model is served once, then the whole sweep runs
against it) so a 4 GB GGUF is not reloaded 727 times:

    python -m fdb_bench.run_inference --model distill --lang pt --out results/pt/distill

Writes ``raw.jsonl`` (one :class:`SimResult` per line) plus ``meta.json``.
Interruption scenarios are left un-rated here; ``run_judge.py`` scores them
afterwards, so the expensive judge model never blocks the inference sweep.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from . import protocol as P
from . import scenarios as S


def resolved_candidate(adapter, args) -> str:
    """The prompt candidate actually used, for provenance in meta.json.

    Recording the literal string "default" would make two runs of the same model
    under different defaults indistinguishable in the results table.
    """
    from .adapters import DEFAULT_PROMPT_CANDIDATE
    return (getattr(adapter, "prompt_candidate", None)
            or args.prompt_candidate
            or DEFAULT_PROMPT_CANDIDATE.get(args.model, "none"))


def build_chunker(prefer_training: bool = True, kind: str = "training"):
    """The user-side chunker. The single most important fidelity knob in the bench.

    ``kind`` selects the arm of the input-stream ablation:
      * ``training`` (default) — the training algorithm, 1--7-token pieces;
      * ``sentence`` — whole sentences, the naive arm that flatters the model;
      * ``word`` — tokenizer-free approximation with matched piece lengths.

    See ``fdb_bench/chunking.py``; falls back to ``word`` if no tokenizer can be
    loaded for ``training``.
    """
    if kind != "training":
        from .chunking import CHUNKERS
        print(f"chunker: {kind} (ablation arm — see FINDINGS.md)")
        return CHUNKERS[kind]
    from .chunking import TrainingChunker, word_chunker
    if not prefer_training:
        return word_chunker
    try:
        from transformers import AutoTokenizer
        src = P.MODEL_REGISTRY.get("sft", {}).get("gguf")
        # any of the merged checkpoints carries the same tokenizer
        merged = P.RUNS_ROOT / "continue/export/merged_bf16"
        try:
            tok = AutoTokenizer.from_pretrained(str(merged), token=False)
        except Exception:
            # The /mnt/f run store is not present on every machine; the same
            # Qwen3-4B tokenizer is in the HF cache, and the chunker must not
            # silently degrade to the word approximation when it is available.
            tok = AutoTokenizer.from_pretrained(
                "Qwen/Qwen3-4B-Instruct-2507", token=False, local_files_only=True
            )
            print(f"chunker: TrainingChunker (tokenizer from HF cache; {merged} absent)")
        return TrainingChunker(tok)
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: falling back to the word chunker ({type(e).__name__}: {e})")
        print("WARNING: the ASR stream will NOT match the training distribution")
        return word_chunker


def _git_rev() -> str:
    """Short git revision of this checkout, for provenance."""
    import subprocess
    try:
        return subprocess.run(
            ["git", "-C", str(P.REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"
from .clock import SpeechClock
from .simulate import DuplexSimulator, SimResult


def _serialise(res: SimResult) -> dict:
    return {
        "sid": res.sid, "task": res.task, "lang": res.lang, "model": res.model,
        "context": res.context, "interrupt": res.interrupt,
        "user_events": [
            {"t_start": e.t_start, "t_end": e.t_end, "kind": e.kind, "text": e.text,
             "ends_turn": e.ends_turn, "is_backchannel": e.is_backchannel,
             "index": e.index}
            for e in res.user_events
        ],
        "model_turns": [
            {"start": m.start, "end": m.end, "text": m.text, "tag": m.tag,
             "kind": m.kind}
            for m in res.model_turns
        ],
        "tags": res.tags,
        "gt_backchannel_times": res.gt_backchannel_times,
        "interrupt_time": res.interrupt_time,
        "interrupt_end": res.interrupt_end,
        "post_interrupt_text": res.post_interrupt_text,
        "took_floor_after_interrupt": res.took_floor_after_interrupt,
        "yield_time": res.yield_time,
        "response_compute_s": res.response_compute_s,
        "floor_compute_s": res.floor_compute_s,
        "model_words_at_interrupt": res.model_words_at_interrupt,
        "model_speaking_at_interrupt": res.model_speaking_at_interrupt,
        "yield_compute_s": res.yield_compute_s,
        "end_time": res.end_time,
        "error": res.error,
        "n_decisions": res.n_decisions,
        "compute_s": res.compute_s,
        "trace": res.trace[-40:],
        "decisions": res.decisions,
        "empty_floor_takings": res.empty_floor_takings,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=sorted(P.MODEL_REGISTRY))
    ap.add_argument("--lang", default="pt", choices=S.LANGS)
    ap.add_argument("--tasks", default="all",
                    help="comma-separated subset names, or 'all'")
    ap.add_argument("--limit", type=int, default=None,
                    help="scenarios per subset (default: the FDB v1.0 size)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--api-base", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--big-api-base", default=None,
                    help="llama-server for the mtp_like stock content model")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--chunker", default="training",
                    choices=["training", "sentence", "word"],
                    help="user-stream chunker; 'sentence' is the ablation arm "
                         "that does NOT match training (see FINDINGS.md)")
    ap.add_argument("--turn-style", default="training",
                    choices=["training", "long"],
                    help="scenario turn length; 'long' is the ablation arm that "
                         "does NOT match training (see FINDINGS.md)")
    ap.add_argument("--prompt-candidate", default=None,
                    help="named system-prompt configuration (see "
                         "adapters.PROMPT_CANDIDATES). Defaults to each "
                         "variant's calibrated choice "
                         "(adapters.DEFAULT_PROMPT_CANDIDATE)")
    ap.add_argument("--prime-mode", default="decision",
                    choices=["decision", "prime"],
                    help="'decision' = let the model choose the turn (VAD-free, "
                         "what FDB measures); 'prime' = hand it "
                         "<|user finish speaking|> as the deployed bridges do")
    ap.add_argument("--words-per-second", type=float, default=3.25)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--original-state-dict", default=None)
    args = ap.parse_args()

    tasks = S.TASKS if args.tasks == "all" else tuple(
        t.strip() for t in args.tasks.split(",") if t.strip()
    )
    for t in tasks:
        if t not in S.TASKS:
            raise SystemExit(f"unknown task {t!r}; expected {S.TASKS}")

    spec = P.MODEL_REGISTRY[args.model]
    from .adapters import build_adapter

    if spec["kind"] == "original_dc" and args.original_state_dict:
        # the resolver reads the env var the adapter factory consults
        import os

        from . import original_dc
        os.environ[original_dc.STATE_DICT_ENV] = args.original_state_dict

    adapter = build_adapter(
        args.model, api_base=args.api_base, big_api_base=args.big_api_base,
        device=args.device, prime_mode=args.prime_mode, lang=args.lang,
        prompt_candidate=args.prompt_candidate,
    )

    clock = SpeechClock(words_per_second=args.words_per_second)
    sim = DuplexSimulator(adapter, clock)
    chunker = build_chunker(prefer_training=True, kind=args.chunker)

    args.out.mkdir(parents=True, exist_ok=True)
    raw_path = args.out / "raw.jsonl"
    t_start = time.time()
    n = n_err = 0

    with raw_path.open("w", encoding="utf-8") as fh:
        for task in tasks:
            for sc in S.build_scenarios(task, limit=args.limit, seed=args.seed,
                                        lang=args.lang, chunker=chunker,
                                        turn_style=args.turn_style):
                res = sim.run(sc)
                res.compute_s = getattr(adapter, "last_call_s", 0.0)
                fh.write(json.dumps(_serialise(res), ensure_ascii=False) + "\n")
                fh.flush()
                n += 1
                if res.error:
                    n_err += 1
                if n % 10 == 0:
                    rate = (time.time() - t_start) / n
                    print(f"[{args.model}/{args.lang}] {n} done "
                          f"({rate:.2f}s/scenario, {n_err} errors)", flush=True)

    meta = {
        "model": args.model,
        "label": spec.get("label", args.model),
        "kind": spec["kind"],
        "lang": args.lang,
        "tasks": list(tasks),
        "limit": args.limit,
        "seed": args.seed,
        "prime_mode": args.prime_mode,
        "prompt_candidate": resolved_candidate(adapter, args),
        "system_prompt": adapter.system_prompt or "<none>",
        "words_per_second": args.words_per_second,
        # Record the *arm* (`training` / `word` / `sentence`), not the Python
        # callable's type: every arm is a function, so ``type(chunker)`` made
        # distinct chunkers indistinguishable in the published tables.
        "chunker": args.chunker,
        "chunker_fn": getattr(chunker, "__name__", type(chunker).__name__),
        "turn_style": args.turn_style,
        "n_server_rejected": getattr(getattr(adapter, "client", None), "n_rejected", 0),
        "n_scenarios": n,
        "n_errors": n_err,
        "wall_s": round(time.time() - t_start, 1),
        # provenance: the bench constants live in git, so record the revision
        # and the decoding settings that produced this run
        "git_rev": _git_rev(),
        "generation": {k: v for k, v in getattr(adapter, "gen", {}).items()},
        "adapter": type(adapter).__name__,
        "judge": "Qwen3.5-9B-UD-Q4_K_XL.gguf (see fdb_bench/judge.py)",
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    if n_err:
        print(f"WARNING: {n_err} scenarios errored — see raw.jsonl 'error' fields")


if __name__ == "__main__":
    main()
