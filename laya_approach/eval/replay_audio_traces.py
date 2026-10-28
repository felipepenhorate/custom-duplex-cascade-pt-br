"""Replay the recorded audio ASR streams through a decision companion.

The audio benchmark itself is out of reach on this machine (no pocket-tts, no
whisper), but the archived runs keep every piece the ASR handed the policy:
``<traces>/<subset>/<sid>/trace.json`` -> ``user_pieces`` with the exact
``kind``, ``text``, ``ends_turn`` and arrival time.  That is the input side of
the failure this project fixes, so it can be re-decided offline.

Two modes:

``open`` (default)
    No assistant history, ``system_speaking=False``: the pre-answer condition
    where the over-eager finishes happened.  A finish here is a false floor
    take by construction, because the turn has not ended.

``reference``
    Replay the reference run's assistant turns as history and set
    ``system_speaking`` while they are audible.  This is closer to the live
    context, but the assistant turns belong to the reference policy, so it is
    a secondary view.

The reference run's own tags are read from the same trace, giving a paired
before/after on identical input: for each stream piece, "did the old policy
finish mid-turn?" and "does this policy finish mid-turn?".

    python -m laya_approach.eval.replay_audio_traces \
        --model laya_approach/runs/laya_companion_v1/merged \
        --traces-root custom-duplex-cascade-pt-br/results_audio_pt/decision/mtp_like
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from laya_approach.data.common import resolve_action, state_from_items

FINISH = "user_finish_speaking"
TASKS = (
    "synthetic_pause_handling",
    "candor_pause_handling",
    "icc_backchannel",
    "candor_turn_taking",
    "synthetic_user_interruption",
)


def load_traces(root: Path) -> list[dict]:
    traces = []
    for subset in TASKS:
        folder = root / subset
        if not folder.is_dir():
            continue
        for sid_dir in sorted(folder.iterdir(), key=lambda p: p.name):
            path = sid_dir / "trace.json"
            if not path.is_file():
                continue
            trace = json.loads(path.read_text(encoding="utf-8"))
            trace["_path"] = str(path)
            traces.append(trace)
    return traces


def reference_tags(trace: dict) -> list[str]:
    """One tag per piece, stripped of the adapter's ``:empty`` suffix.

    The audio adapter appends ``:inaudible``/``:empty`` to the tag for logging;
    the tag itself is the part before the colon.  There can be one extra
    terminal decision (after the last piece), so the list is truncated.
    """
    pieces = trace.get("user_pieces") or []
    tags = [str(t).split(":", 1)[0] for t in (trace.get("tags") or [])]
    return tags[: len(pieces)]


def _utt_outcomes(pieces: list[dict], tags: list[str], finish: str = FINISH) -> Counter:
    """First floor-take per user turn, not per stream piece.

    A policy that fires on every silence after the same finalized turn would
    otherwise look like it took the floor many times.  The question that
    matters is per turn: did the *first* finish arrive before the turn's last
    piece (premature), on its last piece while the stream still says the turn
    continues (open), or on a closed final (legitimate)?
    """
    groups: dict[int, list[int]] = {}
    for i, piece in enumerate(pieces):
        groups.setdefault(piece.get("utt"), []).append(i)
    out = Counter()
    for indices in groups.values():
        out["utts"] += 1
        # Trailing silences share the utterance index but are not its end;
        # the turn ends on the last *speech* piece.
        speech = [i for i in indices if pieces[i].get("kind") != "silence"]
        end = speech[-1] if speech else indices[-1]
        fired = [i for i in indices if tags[i] == finish]
        if not fired:
            out["no_finish"] += 1
            continue
        first = fired[0]
        out["with_finish"] += 1
        if first < end:
            out["premature"] += 1
        elif pieces[end].get("ends_turn"):
            out["closed_end"] += 1
        else:
            out["open_end"] += 1
    return out


def summarise(
    traces: list[dict],
    rows_by_trace: list[list[dict]],
    reference: list[list[str]],
) -> dict:
    by_subset: dict[str, Counter] = {}
    total = Counter()
    for trace, rows, ref_tags in zip(traces, rows_by_trace, reference):
        subset = str(trace.get("subset") or trace.get("task") or "?")
        c = by_subset.setdefault(subset, Counter())
        for row, ref_tag in zip(rows, ref_tags):
            c["pieces"] += 1
            total["pieces"] += 1
            silence = row["kind"] == "silence"
            mid = not row["ends_turn"] and not silence
            if silence:
                c["silence_pieces"] += 1
                total["silence_pieces"] += 1
            elif mid:
                c["mid_turn_pieces"] += 1
                total["mid_turn_pieces"] += 1
            else:
                c["turn_end_pieces"] += 1
                total["turn_end_pieces"] += 1
            if row["raw_tag"] == FINISH:
                c["raw_finish"] += 1
                total["raw_finish"] += 1
                if silence:
                    c["raw_finish_on_silence"] += 1
                    total["raw_finish_on_silence"] += 1
                elif mid:
                    c["raw_finish_mid_turn"] += 1
                    total["raw_finish_mid_turn"] += 1
                else:
                    c["raw_finish_at_end"] += 1
                    total["raw_finish_at_end"] += 1
            if row["tag"] == FINISH:
                c["finish"] += 1
                total["finish"] += 1
                if silence:
                    c["finish_on_silence"] += 1
                    total["finish_on_silence"] += 1
                elif mid:
                    c["finish_mid_turn"] += 1
                    total["finish_mid_turn"] += 1
                else:
                    c["finish_at_end"] += 1
                    total["finish_at_end"] += 1
            if ref_tag == FINISH:
                c["reference_finish"] += 1
                total["reference_finish"] += 1
                if silence:
                    c["reference_finish_on_silence"] += 1
                    total["reference_finish_on_silence"] += 1
                elif mid:
                    c["reference_finish_mid_turn"] += 1
                    total["reference_finish_mid_turn"] += 1
                else:
                    c["reference_finish_at_end"] += 1
                    total["reference_finish_at_end"] += 1
            if row["tag"] == "user_backchannel":
                c["backchannel"] += 1
                total["backchannel"] += 1
            if row["tag"] == "user_interruption":
                c["interruption"] += 1
                total["interruption"] += 1
            c[f"tag:{row['tag']}"] += 1
            total[f"tag:{row['tag']}"] += 1
        pieces = trace.get("user_pieces") or []
        limit = min(len(pieces), len(rows))
        for prefix, tags in (
            ("utt", [row["tag"] for row in rows[:limit]]),
            ("raw_utt", [row["raw_tag"] for row in rows[:limit]]),
            ("ref_utt", ref_tags[:limit]),
        ):
            for key, value in _utt_outcomes(pieces[:limit], tags).items():
                c[f"{prefix}_{key}"] += value
                total[f"{prefix}_{key}"] += value
    return {
        "subsets": {k: dict(v) for k, v in sorted(by_subset.items())},
        "total": dict(total),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--traces-root",
        default="custom-duplex-cascade-pt-br/results_audio_pt/decision/mtp_like",
    )
    ap.add_argument("--mode", default="open", choices=("open", "reference"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-traces", type=int, default=0)
    args = ap.parse_args()

    traces = load_traces(Path(args.traces_root))
    if args.max_traces:
        traces = traces[: args.max_traces]
    if not traces:
        raise SystemExit(f"no traces under {args.traces_root}")

    import unsloth  # noqa: F401  (must precede transformers/torch users)
    import torch
    from unsloth import FastDecisionModel

    thresholds: dict[str, float] = {}
    cfg_path = Path(args.model) / "policy_cfg.json"
    if cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        thresholds = {
            str(tag): float(value)
            for tag, value in (cfg.get("thresholds") or {}).items()
            if isinstance(value, (int, float))
        }

    model, tokenizer = FastDecisionModel.from_pretrained(
        args.model, max_seq_length=640, dtype=torch.float16
    )
    FastDecisionModel.for_inference(model)

    # One flat batch is far faster than one forward per piece; the rows are
    # built first, then scored with the same FastDecisionModel.predict path
    # the runtime uses.
    from laya_approach.data.common import action_question  # local import: light

    flat_states: list[dict] = []
    refs: list[list[str]] = []
    for ti, trace in enumerate(traces):
        refs.append(reference_tags(trace))
        items: list[dict] = []
        turns = trace.get("turns") or []
        injected = 0
        for pi, piece in enumerate(trace.get("user_pieces") or []):
            t = float(piece.get("t_avail") or 0.0)
            speaking = False
            if args.mode == "reference":
                while injected < len(turns) and float(turns[injected]["start"]) <= t:
                    items.append(
                        {"role": "assistant", "text": turns[injected].get("text", "")}
                    )
                    injected += 1
                speaking = any(
                    float(turn["start"]) <= t < float(turn["end"]) for turn in turns
                )
            items.append(
                {
                    "role": "user",
                    "text": piece.get("text", ""),
                    "kind": piece.get("kind", "partial"),
                    "ends_turn": bool(piece.get("ends_turn", False)),
                    "system_speaking": speaking,
                }
            )
            state, _event = state_from_items(items, stream="audio")
            flat_states.append(
                {
                    "state": state,
                    "questions": {"action": action_question()},
                    "gold": {"action": FINISH},
                    "_meta": {"ti": ti, "pi": pi},
                }
            )

    items, report = FastDecisionModel.build_dataset(flat_states, tokenizer, model)
    if report["skipped"]:
        raise SystemExit(report["reason"])
    from unsloth.models.decision import _decision_logits
    from unsloth.models._decision_common import (
        _internal,
        _option_keys,
        _served_temperatures,
    )

    logits, items = _decision_logits(model, tokenizer, items, args.batch_size)
    temps = _served_temperatures(model.decision_config, logits, items)

    rows_by_trace: list[list[dict]] = [[] for _ in traces]
    preds: list[str] = []
    for z, item, temp in zip(logits, items, temps):
        row = flat_states[item["row"]]
        keys = _option_keys(_internal(row["questions"]["action"]))
        probs = dict(zip(keys, torch.softmax(z / temp, -1).tolist()))
        state = row["state"]
        raw_tag = max(probs, key=probs.__getitem__)
        tag, _ = resolve_action(probs, state["event"], state, thresholds)
        ti = row["_meta"]["ti"]
        piece = traces[ti]["user_pieces"][row["_meta"]["pi"]]
        rows_by_trace[ti].append(
            {
                "sid": traces[ti].get("sid"),
                "subset": traces[ti].get("subset"),
                "task": traces[ti].get("task"),
                "kind": piece.get("kind", "partial"),
                "text": piece.get("text", ""),
                "ends_turn": bool(state["event"]["ends_turn"]),
                "system_speaking": bool(state["event"]["system_speaking"]),
                "raw_tag": raw_tag,
                "tag": tag,
                "p_finish": round(float(probs.get(FINISH, 0.0)), 4),
            }
        )
        preds.append(tag)

    summary = summarise(traces, rows_by_trace, refs)
    summary.update(
        {
            "model": args.model,
            "mode": args.mode,
            "traces": len(traces),
            "pieces": len(preds),
            "tag_hist": dict(Counter(preds)),
            "raw_tag_hist": dict(
                Counter(row["raw_tag"] for rows in rows_by_trace for row in rows)
            ),
        }
    )
    out = Path(args.out) if args.out else Path(args.model) / f"audio_replay_{args.mode}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({**summary, "rows": rows_by_trace}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[replay] model={args.model} mode={args.mode} "
          f"traces={len(traces)} pieces={len(preds)}")
    print(f"{'subset':26s} {'pieces':>6s} {'utts':>5s} {'prem':>5s} "
          f"{'raw':>4s} {'ref':>4s} {'midfin':>6s} {'rawmid':>6s} "
          f"{'grdmid':>6s} {'fin_end':>7s}")
    for subset, counts in summary["subsets"].items():
        print(f"{subset:26s} {counts.get('pieces', 0):6d} "
              f"{counts.get('utt_utts', 0):5d} "
              f"{counts.get('utt_premature', 0):5d} "
              f"{counts.get('raw_utt_premature', 0):4d} "
              f"{counts.get('ref_utt_premature', 0):4d} "
              f"{counts.get('finish_mid_turn', 0):6d} "
              f"{counts.get('raw_finish_mid_turn', 0):6d} "
              f"{counts.get('finish_mid_turn', 0):6d} "
              f"{counts.get('finish_at_end', 0):7d}")
    total = summary["total"]
    print(f"{'TOTAL':26s} {total.get('pieces', 0):6d} "
          f"{total.get('utt_utts', 0):5d} "
          f"{total.get('utt_premature', 0):5d} "
          f"{total.get('raw_utt_premature', 0):4d} "
          f"{total.get('ref_utt_premature', 0):4d} "
          f"{total.get('finish_mid_turn', 0):6d} "
          f"{total.get('raw_finish_mid_turn', 0):6d} "
          f"{total.get('finish_mid_turn', 0):6d} "
          f"{total.get('finish_at_end', 0):7d}")
    print(f"[replay] wrote {out}")


if __name__ == "__main__":
    main()
