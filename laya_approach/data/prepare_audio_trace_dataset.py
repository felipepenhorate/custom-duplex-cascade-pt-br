"""Build decision rows from the recorded FDB audio traces + annotations.

The translated dialogues always have a true end of turn.  The audio pause clips
do not: the recording stops mid-turn, so their last ASR final must be a *wait*.
This script labels each recorded user piece from the benchmark's own annotations
instead of from a generated dialogue:

* pause_handling: every piece is wait/backchannel; no finish anywhere;
* smooth_turn_taking: finish the final nearest the annotated turn-taking time;
* user_interruption: finish the context turn, wait/interrupt during the
  annotation span, finish the post-interruption turn;
* backchannel: wait/backchannel for the whole monologue, no formal finish.

The assistant history is the reference turn only before the decision point; for
pause/backchannel the erroneous reference answers are omitted entirely.

    python -m laya_approach.data.prepare_audio_trace_dataset \
        --out-dir laya_approach/data/generated_audio_fix
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from laya_approach.data.common import action_question, build_state

ACKS = {
    "uhum", "uhum!", "sim", "sim.", "aham", "aham.", "certo", "certo.", "ok",
    "okay", "yeah", "yep", "right", "hum", "hmm", "é", "é.", "isso", "isso.",
}

TRACE_SETS = (
    # label, traces root, data root, language
    ("pt", "custom-duplex-cascade-pt-br/results_audio_pt/decision/mtp_like",
     "laya_approach/data/fdb_pt_audio/v1.0"),
    ("en", "custom-duplex-cascade-pt-br/results_audio/decision/mtp_like",
     "laya_approach/data/fdb_en_audio/v1.0"),
)

SUBSETS = (
    "candor_pause_handling", "synthetic_pause_handling",
    "candor_turn_taking", "icc_backchannel", "synthetic_user_interruption",
)


def _ann(root: Path, subset: str, sid: str, name: str) -> list[dict]:
    path = root / subset / sid / name
    if not path.is_file():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8")) or []
    except json.JSONDecodeError:
        return []


def _is_ack(text: str) -> bool:
    words = re.findall(r"[^\W\d_]+", text.lower(), re.UNICODE)
    return 0 < len(words) <= 2 and text.strip().lower().rstrip(".!?…") in {
        a.rstrip(".!?…") for a in ACKS
    }


def _target_final(pieces: list[dict], when: float, *, after: bool) -> int | None:
    finals = [i for i, p in enumerate(pieces)
              if p.get("kind") == "final" and p.get("ends_turn")
              and p.get("text", "").strip()]
    if not finals:
        finals = [i for i, p in enumerate(pieces)
                  if p.get("kind") == "final" and p.get("ends_turn")]
    if not finals:
        return None
    if after:
        cand = [i for i in finals if float(pieces[i]["t_avail"]) >= when - 0.5]
        return min(cand, key=lambda i: abs(float(pieces[i]["t_avail"]) - when)) if cand else finals[-1]
    cand = [i for i in finals if float(pieces[i]["t_avail"]) <= when + 0.4]
    return max(cand, key=lambda i: float(pieces[i]["t_avail"])) if cand else None


def rows_for_trace(trace: dict, data_root: Path, lang: str) -> list[dict]:
    subset = trace["subset"]
    sid = str(trace["sid"])
    pieces = trace.get("user_pieces") or []
    if not pieces:
        return []
    task = trace.get("task") or subset
    pause = _ann(data_root, subset, sid, "pause.json")
    turn_taking = _ann(data_root, subset, sid, "turn_taking.json")
    interrupt = _ann(data_root, subset, sid, "interrupt.json")

    is_pause = subset in ("candor_pause_handling", "synthetic_pause_handling")
    is_backchannel = subset == "icc_backchannel"
    is_smooth = subset == "candor_turn_taking"
    is_interrupt = subset == "synthetic_user_interruption"

    target: int | None = None
    context_target: int | None = None
    post_target: int | None = None
    int_start = int_end = None
    if is_smooth and turn_taking:
        when = float(turn_taking[0]["timestamp"][0])
        target = _target_final(pieces, when, after=True)
    if is_interrupt and interrupt:
        int_start = float(interrupt[0]["timestamp"][0])
        int_end = float(interrupt[0]["timestamp"][1])
        context_target = _target_final(pieces, int_start, after=False)
        post_target = _target_final(pieces, int_end, after=True)
        if context_target == post_target:
            context_target = None

    # Assistant history: only the reference turns that start before the current
    # decision point.  Pause/backchannel reference answers are errors, so omit
    # them; smooth has no prior answer; interruption keeps the context answer.
    include_assistant = is_interrupt
    turns = sorted(trace.get("turns") or [], key=lambda t: float(t["start"]))

    history: list[dict] = []
    ai = 0
    finished_at: float | None = None
    post_finish_ticks = 0
    rows: list[dict] = []

    def append_assistants(until: float) -> None:
        nonlocal ai
        if not include_assistant:
            return
        while ai < len(turns) and float(turns[ai]["start"]) <= until:
            turn = turns[ai]
            text = str(turn.get("text") or "").strip()
            if text:
                history.append({"role": "assistant", "text": text})
            ai += 1

    for i, piece in enumerate(pieces):
        t_avail = float(piece.get("t_avail", piece.get("t_start", 0.0)))
        append_assistants(t_avail)
        speaking = include_assistant and any(
            float(t["start"]) <= t_avail < float(t["end"]) for t in turns
        )
        kind = str(piece.get("kind") or "partial")
        text = str(piece.get("text") or "")
        if kind == "silence":
            text = ""
        elif finished_at is not None:
            # A new word after a completed turn starts a new user turn, not a
            # continuation of the post-answer silence.
            finished_at = None

        if is_pause or is_backchannel:
            if kind == "silence":
                label = "no_tag" if speaking else "user_is_speaking"
            elif speaking and _is_ack(text):
                label = "user_backchannel"
            else:
                label = "user_is_speaking"
        elif is_smooth:
            if i == target and not speaking:
                label = "user_finish_speaking"
                finished_at = t_avail
                post_finish_ticks = 0
            elif finished_at is not None:
                if kind == "silence" and not speaking and post_finish_ticks < 2:
                    label = "user_finish_speaking"
                    post_finish_ticks += 1
                else:
                    label = "no_tag" if speaking else "user_is_thinking"
            elif kind == "silence":
                label = "no_tag" if speaking else "user_is_speaking"
            elif speaking:
                label = "user_backchannel" if _is_ack(text) else "user_interruption"
            else:
                label = "user_is_speaking"
        elif is_interrupt:
            if i == context_target or i == post_target:
                if speaking:
                    label = "user_interruption" if not _is_ack(text) else "user_backchannel"
                else:
                    label = "user_finish_speaking"
                    finished_at = t_avail
                    post_finish_ticks = 0
            elif finished_at is not None:
                if kind == "silence" and not speaking and post_finish_ticks < 2:
                    label = "user_finish_speaking"
                    post_finish_ticks += 1
                else:
                    label = "no_tag" if speaking else "user_is_thinking"
            elif kind == "silence":
                label = "no_tag" if speaking else "user_is_speaking"
            elif speaking:
                label = "user_backchannel" if _is_ack(text) else "user_interruption"
            else:
                label = "user_is_speaking"
        else:
            label = "user_is_speaking"

        event = {
            "kind": kind,
            "ends_turn": bool(piece.get("ends_turn", False)),
            "text": text,
            "system_speaking": speaking,
            "stream": "audio",
        }
        state = build_state(event, history)
        rows.append({
            "state": state,
            "questions": {"action": action_question()},
            "gold": {"action": label},
            "_meta": {
                "source": "audio_trace",
                "lang": lang,
                "task": task,
                "subset": subset,
                "sid": sid,
                "bucket": f"audio_{subset}",
                "trace_i": i,
            },
        })

        history.append({
            "role": "user",
            "text": text,
            "kind": kind,
            "ends_turn": bool(piece.get("ends_turn", False)),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="laya_approach/data/generated_audio_fix")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    seen: Counter = Counter()
    for lang, traces_root, data_root in TRACE_SETS:
        root = Path(traces_root)
        dr = Path(data_root)
        for subset in SUBSETS:
            folder = root / subset
            if not folder.is_dir():
                continue
            for trace_path in sorted(folder.glob("*/trace.json"),
                                     key=lambda p: int(p.parent.name) if p.parent.name.isdigit() else 0):
                trace = json.loads(trace_path.read_text(encoding="utf-8"))
                sid = str(trace.get("sid"))
                bucket = int(hashlib.sha1(f"{lang}:{subset}:{sid}".encode()).hexdigest(), 16) % 10
                split = "test" if bucket == 0 else ("val" if bucket == 1 else "train")
                rows = rows_for_trace(trace, dr, lang)
                if not rows:
                    continue
                splits[split].extend(rows)
                seen[(split, subset)] += 1

    for split, rows in splits.items():
        groups: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            groups[json.dumps(row["state"], ensure_ascii=False, sort_keys=True)].append(row)
        kept: list[dict] = []
        dropped = 0
        for group in groups.values():
            if len({r["gold"]["action"] for r in group}) > 1:
                dropped += len(group)
            else:
                kept.extend(group)
        splits[split] = kept
        if dropped:
            print(f"[audio-data] {split}: dropped {dropped} ambiguous-state rows")
        path = out / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for row in kept:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        labels = Counter(r["gold"]["action"] for r in kept)
        print(f"[audio-data] {split}: {len(kept):,} rows, labels={dict(labels)}")
    print("[audio-data] traces per split/subset:", dict(seen))

    # Audit exact-state conflicts.
    for split, rows in splits.items():
        by: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            by[json.dumps(row["state"], ensure_ascii=False, sort_keys=True)].add(row["gold"]["action"])
        conflicts = sum(1 for row in rows
                        if len(by[json.dumps(row["state"], ensure_ascii=False, sort_keys=True)]) > 1)
        print(f"[audio-data] {split}: conflicts={conflicts}")

    (out / "audit.json").write_text(json.dumps({
        "splits": {k: len(v) for k, v in splits.items()},
        "labels": {k: dict(Counter(r["gold"]["action"] for r in v))
                   for k, v in splits.items()},
        "traces": {f"{k[0]}/{k[1]}": v for k, v in sorted(seen.items())},
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
