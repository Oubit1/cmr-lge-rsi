"""Offline unit tests for the read-only pending CMR topology shadow scanner.

No external data, images, masks or network access required.
"""

from __future__ import annotations

import unittest

from experiments.shadow_cmr_pending_topology_scan import (
    EXPECTED_PENDING,
    TOPOLOGY_POPULATED_RATE_MIN,
    _select_non_holdout,
    _topology_populated_rate,
    run_evaluation,
)


def _record(case_id, area=500.0, cavity=0.4, **extra):
    rec = {
        "case_id": case_id,
        "reference_area": 1000.0,
        "before_area": area,
        "before_cavity": cavity,
        "after_area": area,
        "after_cavity": cavity,
        "quality_before": 80.0,
        "quality_after": 80.0,
        "quality_delta": 0.0,
        "decision": "REVIEW",
        "quality_status": "SCORED",
        "quality_flags_before": [],
        "quality_flags_after": [],
        "quality_subscores_before": {
            "integrity": 100.0,
            "topology": 80.0,
            "temporal": 60.0,
            "image_support": 100.0,
        },
        "quality_subscores_after": {
            "integrity": 100.0,
            "topology": 80.0,
            "temporal": 60.0,
            "image_support": 100.0,
        },
    }
    rec.update(extra)
    return rec


class SelectNonHoldoutTest(unittest.TestCase):
    def test_excludes_holdout_and_dedups(self):
        records = [
            _record("a"),
            _record("b"),
            _record("a"),
            _record("h"),
        ]
        out = _select_non_holdout(records, ["h"])
        self.assertEqual([r["case_id"] for r in out["selected"]], ["a", "b"])
        self.assertEqual(out["holdout_selected"], 0)
        self.assertEqual(out["holdout_excluded_count"], 1)
        self.assertEqual(out["duplicate_case_ids"], ["a"])


class TopologyPopulatedRateTest(unittest.TestCase):
    def test_rate_unavailable_when_empty(self):
        out = _topology_populated_rate({}, 0)
        self.assertIsNone(out["rate"])
        self.assertEqual(out["status"], "unavailable_empty_denominator")

    def test_rate_counts_populated_cells(self):
        cases = {
            "a": {"topology": {"components": {"value": 1}}},
            "b": {"topology": {"components": {"value": None}}},
        }
        out = _topology_populated_rate(cases, 2)
        self.assertIsNotNone(out["rate"])
        self.assertGreaterEqual(out["rate"], 0.0)
        self.assertLessEqual(out["rate"], 1.0)


class RunEvaluationTest(unittest.TestCase):
    def test_zero_write_and_holdout_excluded(self):
        records = [_record("a"), _record("b"), _record("h")]
        report = run_evaluation(
            {"records": records, "holdout_case_ids": ["h"]}
        )
        self.assertFalse(report["label_write"])
        self.assertEqual(report["holdout_selected"], 0)
        self.assertEqual(report["volumes_scanned"], 2)
        self.assertEqual(report["writes"]["labels"], 0)
        self.assertNotEqual(
            report["target_path"], "app/check_auto_apply.py"
        )
        self.assertIn("module_hash", report)
        self.assertIn("rollback_command", report)

    def test_no_transition_always_has_reason(self):
        records = [_record("a"), _record("b"), _record("c")]
        report = run_evaluation({"records": records})
        self.assertEqual(report["no_transition_without_reason_count"], 0)
        self.assertEqual(report["no_transition_without_reason"], [])

    def test_topology_rate_reported(self):
        records = [_record("a"), _record("b")]
        report = run_evaluation({"records": records})
        rate = report["topology_fields_populated_rate"]
        self.assertIsNotNone(rate)
        self.assertGreaterEqual(rate, 0.0)
        self.assertLessEqual(rate, 1.0)
        self.assertEqual(
            report["topology_fields_populated_rate_threshold"],
            TOPOLOGY_POPULATED_RATE_MIN,
        )

    def test_expected_pending_constant(self):
        self.assertEqual(EXPECTED_PENDING, 61)

    def test_unavailable_metrics_not_imputed(self):
        report = run_evaluation({"records": [_record("a")]})
        self.assertIn("unavailable", report)
        self.assertIn("concordance_vs_independent_labels", report["unavailable"])
        self.assertIn("precision", report["unavailable"])
        self.assertIn("ci", report["unavailable"])


if __name__ == "__main__":
    unittest.main()
