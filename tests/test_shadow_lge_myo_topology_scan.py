"""Offline unit tests for the read-only LGE myo topology shadow scanner.

No external data, no network, no production writes.
"""

from __future__ import annotations

import unittest

from experiments.shadow_lge_myo_topology_scan import (
    EXPECTED_NON_HOLDOUT,
    FILL_RATE_MIN,
    _reconcile_continuity,
    run_evaluation,
)


def _rec(case_id, **overrides):
    rec = {
        "case_id": case_id,
        "reference_area": 1000.0,
        "before_area": 500.0,
        "before_cavity": 0.3,
        "after_area": 500.0,
        "after_cavity": 0.3,
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
        "component_count": 1,
        "hole_count": 0,
    }
    rec.update(overrides)
    return rec


class ReconcileContinuityTest(unittest.TestCase):
    def test_missing_mask_measurement_is_unavailable(self):
        proxy = {"continuous": True, "reason": "no_layer_jump_flag"}
        mask = {"available": False, "layer_jump": None, "reason": "missing_mask_layer_jump"}
        out = _reconcile_continuity(proxy, mask)
        self.assertEqual(out["status"], "unavailable_no_mask_layer_jump")
        self.assertIsNone(out["agrees"])
        self.assertIsNone(out["difference"])

    def test_agreement_when_both_continuous(self):
        proxy = {"continuous": True}
        mask = {"available": True, "layer_jump": False}
        out = _reconcile_continuity(proxy, mask)
        self.assertEqual(out["status"], "reconciled")
        self.assertTrue(out["agrees"])
        self.assertEqual(out["difference"], 0)

    def test_disagreement_records_difference(self):
        proxy = {"continuous": True}
        mask = {"available": True, "layer_jump": True}
        out = _reconcile_continuity(proxy, mask)
        self.assertFalse(out["agrees"])
        self.assertEqual(out["difference"], 1)
        self.assertEqual(out["reason"], "proxy_mask_disagreement")


class RunEvaluationTest(unittest.TestCase):
    def test_zero_label_write_and_holdout_excluded(self):
        records = [_rec("case_a"), _rec("case_b")]
        holdout = ["case_b"]
        report = run_evaluation({"records": records, "holdout_case_ids": holdout})
        self.assertFalse(report["label_write"])
        self.assertEqual(report["holdout_selected"], 0)
        self.assertEqual(report["holdout_excluded_count"], 1)
        self.assertEqual(report["non_holdout_volumes_scanned"], 1)
        self.assertEqual(report["writes"], {"labels": 0, "code": 0, "datasets": 0})

    def test_topology_fields_populated_and_no_reason_count_zero(self):
        records = [_rec(f"case_{i}") for i in range(EXPECTED_NON_HOLDOUT)]
        report = run_evaluation({"records": records})
        self.assertGreater(report["component_count_fill_rate"], FILL_RATE_MIN)
        self.assertGreater(report["hole_count_fill_rate"], FILL_RATE_MIN)
        self.assertGreater(report["cross_slice_continuity_fill_rate"], FILL_RATE_MIN)
        self.assertEqual(report["no_reason_count"], 0)
        self.assertTrue(report["no_reason_count_is_zero"])

    def test_reconciliation_against_mask_layer_jump(self):
        records = [
            _rec("case_agree", mask_layer_jump=False),
            _rec("case_disagree", mask_layer_jump=True),
            _rec("case_missing"),
        ]
        report = run_evaluation({"records": records})
        self.assertEqual(report["mask_layer_jump_filled"], 2)
        self.assertEqual(report["continuity_reconciliation_filled"], 2)
        self.assertEqual(report["continuity_reconciliation_agreements"], 1)
        self.assertEqual(report["continuity_reconciliation_disagreements"], 1)
        diffs = report["continuity_reconciliation_differences"]
        self.assertEqual(len(diffs), 1)
        self.assertEqual(diffs[0]["case_id"], "case_disagree")
        self.assertEqual(diffs[0]["difference"], 1)
        self.assertEqual(
            report["cases"]["case_missing"]["continuity_reconciliation"]["status"],
            "unavailable_no_mask_layer_jump",
        )

    def test_missing_measurement_reason_populated(self):
        records = [_rec("case_missing", before_cavity=None)]
        report = run_evaluation({"records": records})
        self.assertEqual(report["no_reason_count"], 0)
        self.assertEqual(
            report["cases"]["case_missing"]["suppression_reason"],
            "missing_measurement",
        )


if __name__ == "__main__":
    unittest.main()
