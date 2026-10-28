"""Schema and guard tests for the Laya companion (no model load)."""

from __future__ import annotations

import unittest

from laya_approach.data.common import (
    NO_TAG,
    event_bucket,
    finish_is_allowed,
    resolve_action,
    state_from_items,
)


def probs(top: str, value: float = 0.8) -> dict[str, float]:
    out = {tag: (1.0 - value) / 6 for tag in (
        NO_TAG,
        "user_is_speaking",
        "user_finish_speaking",
        "user_interruption",
        "user_backchannel",
        "user_is_thinking",
        "system_backchannel",
    )}
    out[top] = value
    return out


class StateTests(unittest.TestCase):
    def test_partial_with_punctuation_is_mid_turn(self) -> None:
        state, event = state_from_items([
            {"role": "user", "text": "Qual é o prazo?", "kind": "partial",
             "ends_turn": False, "system_speaking": False},
        ])
        self.assertEqual(state["event"]["kind"], "partial")
        self.assertTrue(state["event"]["terminal_punctuation"])
        self.assertEqual(state["event"]["user_turn_state"], "in_turn")
        self.assertFalse(finish_is_allowed(event, state))
        self.assertEqual(resolve_action(probs("user_finish_speaking"), event, state)[0], NO_TAG)

    def test_vetoed_finish_falls_back_to_keep_listening(self) -> None:
        state, event = state_from_items([
            {"role": "user", "text": "Qual é o prazo?", "kind": "partial",
             "ends_turn": False, "system_speaking": False},
        ])
        dist = probs("user_finish_speaking", 0.52)
        dist["user_is_speaking"] = 0.40
        tag, _ = resolve_action(dist, event, state)
        self.assertEqual(tag, "user_is_speaking")

    def test_final_with_punctuation_is_complete(self) -> None:
        state, event = state_from_items([
            {"role": "user", "text": "Qual é o prazo?", "kind": "final",
             "ends_turn": True, "system_speaking": False},
        ])
        self.assertTrue(finish_is_allowed(event, state))
        self.assertEqual(
            resolve_action(probs("user_finish_speaking"), event, state)[0],
            "user_finish_speaking",
        )

    def test_empty_final_is_complete(self) -> None:
        state, event = state_from_items([
            {"role": "user", "text": "Qual é o prazo?", "kind": "partial",
             "ends_turn": False, "system_speaking": False},
            {"role": "user", "text": "", "kind": "final",
             "ends_turn": True, "system_speaking": False},
        ])
        self.assertEqual(state["event"]["kind"], "final")
        self.assertFalse(state["event"]["has_words"])
        self.assertTrue(finish_is_allowed(event, state))
        self.assertEqual(event_bucket(event, state), "empty_final")

    def test_finish_never_fires_over_system_speech(self) -> None:
        state, event = state_from_items([
            {"role": "assistant", "text": "Deixa eu te contar..."},
            {"role": "user", "text": "pera", "kind": "final",
             "ends_turn": True, "system_speaking": True},
        ])
        self.assertFalse(finish_is_allowed(event, state))
        self.assertEqual(resolve_action(probs("user_finish_speaking"), event, state)[0], NO_TAG)

    def test_interruption_requires_system_speech(self) -> None:
        state, event = state_from_items([
            {"role": "user", "text": "uma pergunta nova", "kind": "final",
             "ends_turn": True, "system_speaking": False},
        ])
        dist = probs("user_interruption", 0.50)
        dist["user_finish_speaking"] = 0.46
        tag, _ = resolve_action(dist, event, state)
        # Interruption is vetoed because the system is silent; the constrained
        # argmax falls back to finish, which this completed turn does allow.
        self.assertNotEqual(tag, "user_interruption")
        self.assertEqual(tag, "user_finish_speaking")


if __name__ == "__main__":
    unittest.main()
