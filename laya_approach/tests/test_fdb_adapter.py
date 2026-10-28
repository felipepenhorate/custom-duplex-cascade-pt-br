"""Adapter-level tests: stream metadata must reach Laya and only Laya."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2] / "custom-duplex-cascade-pt-br"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from fdb_bench.adapters import LayaPolicyAdapter, PolicyAdapter  # noqa: E402


class RecordingPolicy:
    tag_vocab = ["no_tag", "user_is_speaking", "user_finish_speaking"]
    no_voice = "<|no voice|>"

    def __init__(self, tag: str = "user_finish_speaking") -> None:
        self.tag = tag
        self.seen: list[dict] = []

    def query(self, items):
        self.seen = [dict(item) for item in items]
        return self.tag, {name: 1.0 for name in self.tag_vocab}


class FakeClient:
    def chat(self, messages, **gen):
        return "Claro, posso ajudar."


class AdapterMetadataTests(unittest.TestCase):
    def test_laya_receives_empty_final_and_system_speech(self) -> None:
        policy = RecordingPolicy()
        adapter = LayaPolicyAdapter(policy, FakeClient(), name="laya_companion")
        items = [
            {"role": "user", "text": "qual o prazo", "kind": "partial",
             "ends_turn": False, "system_speaking": True},
            {"role": "user", "text": "", "kind": "final",
             "ends_turn": True, "system_speaking": False},
        ]
        tag, text = adapter.act(items)
        self.assertEqual(tag, "user_finish_speaking")
        self.assertTrue(text)
        self.assertEqual(policy.seen[-1]["kind"], "final")
        self.assertTrue(policy.seen[-1]["ends_turn"])
        self.assertFalse(policy.seen[-1]["system_speaking"])
        self.assertEqual(items[-1]["role"], "assistant")

    def test_mtp_like_adapter_strips_metadata(self) -> None:
        policy = RecordingPolicy("no_tag")
        adapter = PolicyAdapter(policy, FakeClient(), name="mtp_like")
        adapter.act([
            {"role": "user", "text": "oi", "kind": "partial",
             "ends_turn": False, "system_speaking": False},
        ])
        self.assertEqual(set(policy.seen[-1]), {"role", "text"})


class EndOfTurnPolicy(RecordingPolicy):
    """Finish on a final event, wait otherwise — enough to run the simulator."""

    def query(self, items):
        super().query(items)
        event = items[-1]
        if event.get("kind") == "final" and event.get("ends_turn"):
            return "user_finish_speaking", {}
        return "no_tag", {}


class SimulatorIntegrationTests(unittest.TestCase):
    def test_every_task_feeds_laya_the_stream_metadata(self) -> None:
        from fdb_bench import scenarios as S
        from fdb_bench.chunking import word_chunker
        from fdb_bench.clock import SpeechClock
        from fdb_bench.simulate import DuplexSimulator

        policy = EndOfTurnPolicy()
        adapter = LayaPolicyAdapter(policy, FakeClient(), name="laya_companion")
        sim = DuplexSimulator(adapter, SpeechClock(words_per_second=3.25))
        for task in S.TASKS:
            for scenario in S.build_scenarios(
                task, limit=1, lang="pt", chunker=word_chunker
            ):
                res = sim.run(scenario)
                self.assertIsNone(res.error, f"{task}/{scenario.sid}: {res.error}")
                self.assertTrue(policy.seen, task)
                # Assistant turns stay role-only; every user item is a stream
                # event, and the newest one (the decision point) always carries
                # its kind / turn-end / system-speech metadata.
                user_kinds = {
                    item.get("kind")
                    for item in policy.seen
                    if item["role"] == "user"
                }
                self.assertLessEqual(user_kinds, {"partial", "final", "silence"})
                self.assertIn(
                    policy.seen[-1].get("kind"), {"partial", "final", "silence"}, task
                )
                self.assertTrue(
                    all("system_speaking" in item for item in policy.seen
                        if item["role"] == "user"),
                    task,
                )


if __name__ == "__main__":
    unittest.main()
