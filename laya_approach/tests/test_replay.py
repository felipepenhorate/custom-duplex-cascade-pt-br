"""Replay-evaluator bookkeeping: tag alignment and the mid-turn counters."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from laya_approach.eval.replay_audio_traces import (  # noqa: E402
    _utt_outcomes,
    reference_tags,
    summarise,
)


class UtteranceOutcomeTests(unittest.TestCase):
    def test_first_finish_per_turn_is_not_double_counted(self) -> None:
        pieces = [
            {"utt": 0, "kind": "partial", "ends_turn": False},
            {"utt": 0, "kind": "final", "ends_turn": True},
            # repeated silence after the turn shares the utterance index
            {"utt": 0, "kind": "silence", "ends_turn": False},
            {"utt": 1, "kind": "partial", "ends_turn": False},
            {"utt": 1, "kind": "partial", "ends_turn": False},  # turn continues
        ]
        tags = ["no_tag", "user_finish_speaking", "user_finish_speaking",
                "user_finish_speaking", "user_is_speaking"]
        out = _utt_outcomes(pieces, tags)
        self.assertEqual(out["utts"], 2)
        self.assertEqual(out["closed_end"], 1)
        self.assertEqual(out["premature"], 1)
        self.assertEqual(out["with_finish"], 2)

    def test_no_finish_turn(self) -> None:
        out = _utt_outcomes(
            [{"utt": 7, "ends_turn": False}], ["user_is_speaking"]
        )
        self.assertEqual(out["utts"], 1)
        self.assertEqual(out["no_finish"], 1)
        self.assertEqual(out["with_finish"], 0)


def row(kind: str, ends_turn: bool, raw: str, guarded: str) -> dict:
    return {
        "kind": kind,
        "ends_turn": ends_turn,
        "raw_tag": raw,
        "tag": guarded,
    }


class ReplaySummaryTests(unittest.TestCase):
    def test_reference_tags_strip_suffix_and_truncate(self) -> None:
        trace = {
            "user_pieces": [{"text": "a"}, {"text": "b"}],
            "tags": [
                "user_is_speaking:empty",
                "user_finish_speaking",
                "user_finish_speaking:inaudible",
            ],
        }
        self.assertEqual(
            reference_tags(trace), ["user_is_speaking", "user_finish_speaking"]
        )

    def test_mid_turn_end_and_silence_are_counted_separately(self) -> None:
        traces = [{"subset": "pause"}]
        rows = [[
            row("partial", False, "user_finish_speaking", "user_is_speaking"),
            row("final", False, "user_finish_speaking", "user_is_speaking"),
            row("final", True, "user_finish_speaking", "user_finish_speaking"),
            row("silence", False, "user_finish_speaking", "user_finish_speaking"),
        ]]
        reference = [["user_finish_speaking"] * 4]
        out = summarise(traces, rows, reference)
        total = out["total"]
        self.assertEqual(total["pieces"], 4)
        self.assertEqual(total["mid_turn_pieces"], 2)
        self.assertEqual(total["turn_end_pieces"], 1)
        self.assertEqual(total["silence_pieces"], 1)
        # Raw model: 4 finishes, 2 of them mid-turn.  Guarded: only the legal
        # turn-end and post-turn silence finishes survive.
        self.assertEqual(total["raw_finish_mid_turn"], 2)
        self.assertEqual(total.get("finish_mid_turn", 0), 0)
        self.assertEqual(total["finish_at_end"], 1)
        self.assertEqual(total["finish_on_silence"], 1)
        self.assertEqual(total["reference_finish_mid_turn"], 2)


if __name__ == "__main__":
    unittest.main()
