from __future__ import annotations

import unittest

from experiments.promotion_gate import evaluate_metrics


RULES = {
    "metrics": {
        "recall": {"direction": "higher", "min_delta": 0.05},
        "dice": {"direction": "higher", "max_drop": 0.002},
        "false_repair_rate": {"direction": "lower", "max_value": 0.01},
    }
}


class PromotionGateTests(unittest.TestCase):
    def test_promote(self):
        result = evaluate_metrics(
            {"recall": 0.80, "dice": 0.90, "false_repair_rate": 0.02},
            {"recall": 0.86, "dice": 0.899, "false_repair_rate": 0.005},
            RULES,
        )
        self.assertEqual(result.decision, "promote")

    def test_reject_regression(self):
        result = evaluate_metrics(
            {"recall": 0.80, "dice": 0.90, "false_repair_rate": 0.005},
            {"recall": 0.86, "dice": 0.88, "false_repair_rate": 0.005},
            RULES,
        )
        self.assertEqual(result.decision, "reject")
        self.assertIn("dice", result.failed_checks)


if __name__ == "__main__":
    unittest.main()
