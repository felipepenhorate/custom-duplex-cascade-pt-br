"""Evaluator bookkeeping: bucket rates must use the right denominator."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from laya_approach.eval.evaluate_laya import (  # noqa: E402
    _bucket_rate,
    _load_thresholds,
    critical_rates,
)


def row(bucket: str, gold: str) -> dict:
    return {"_meta": {"bucket": bucket}, "gold": {"action": gold}}


class BucketRateTests(unittest.TestCase):
    def test_interruption_recall_is_over_gold_not_bucket_size(self) -> None:
        rows = [
            row("final_during_system", "user_interruption"),
            row("final_during_system", "user_interruption"),
            row("final_during_system", "user_backchannel"),
            row("final_during_system", "user_backchannel"),
        ]
        pred = ["user_interruption", "user_backchannel",
                "user_backchannel", "user_backchannel"]
        # 1 of 2 true interruptions, not 1 of 4 bucket rows.
        self.assertEqual(
            _bucket_rate(rows, pred, "final_during_system", "user_interruption", "gold"),
            0.5,
        )
        self.assertEqual(
            _bucket_rate(rows, pred, "final_during_system", "user_interruption"),
            0.25,
        )

    def test_false_finish_rate_stays_over_the_whole_bucket(self) -> None:
        rows = [row("partial_terminal_punct", "user_is_speaking")] * 4
        pred = ["user_finish_speaking", "user_is_speaking",
                "user_is_speaking", "user_is_speaking"]
        self.assertEqual(
            _bucket_rate(rows, pred, "partial_terminal_punct",
                         "user_finish_speaking", "bucket"),
            0.25,
        )

    def test_critical_rates_skip_absent_buckets(self) -> None:
        rows = [row("final_mid_turn", "user_is_speaking")]
        out = critical_rates(rows, ["user_finish_speaking"],
                             ["user_is_speaking"])
        self.assertEqual(out["final_mid_turn_false_finish_raw"], 1.0)
        self.assertEqual(out["final_mid_turn_false_finish_guarded"], 0.0)
        self.assertNotIn("partial_punct_false_finish_raw", out)


class ThresholdLoadingTests(unittest.TestCase):
    def test_flat_report_and_checkpoint_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            flat = root / "flat.json"
            flat.write_text(json.dumps({"user_is_speaking": 0.4}))
            self.assertEqual(_load_thresholds(str(flat)),
                             {"user_is_speaking": 0.4})

            report = root / "report.json"
            report.write_text(json.dumps({"recommended": {"no_tag": 0.0}}))
            self.assertEqual(_load_thresholds(str(report)), {"no_tag": 0.0})

            ckpt = root / "policy_cfg.json"
            ckpt.write_text(json.dumps({"thresholds": {"no_tag": 0.0},
                                        "fine_tuned": True}))
            self.assertEqual(_load_thresholds(str(ckpt)), {"no_tag": 0.0})


if __name__ == "__main__":
    unittest.main()
