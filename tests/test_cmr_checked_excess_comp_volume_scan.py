"""Offline unit tests for the S825 checked excess-component volume scan.

No external data, no network, no production writes.
"""

from __future__ import annotations

import unittest

from experiments.cmr_checked_excess_comp_volume_scan import (
    EXPECTED_CHECKED,
    _secondary_component_fields,
    report_to_json,
    run_evaluation,
)


def _record(cid, area, cavity, **extra):
    rec = {
        "case_id": cid,
        "reference_area": 1000.0,
        "before_area": area,
        "before_cavity": cavity,
        "after_area": area,
        "after_cavity": cavity,
        "decision": "REVIEW",
        "quality_status": "SCORED",
    }
    rec.update(extra)
    return rec


class SecondaryComponentFieldsTest(unittest.TestCase):
    def test_ratio_derived_from_supplied_areas(self):
        out = _secondary_component_fields(
            {
                "secondary_component_count": 2,
                "primary_component_area": 200.0,
                "secondary_component_area": 10.0,
                "secondary_adjacent_to_primary": True,
                "hole_count": 1,
                "touches_border": False,
                "centroid_drift": 0.5,
            }
        )
        self.assertAlmostEqual(out["secondary_primary_area_ratio"], 0.05)
        self.assertTrue(out["secondary_fields_available"])

    def test_missing_fields_are_unavailable_not_imputed(self):
        out = _secondary_component_fields({"case_id": "c"})
        self.assertIsNone(out["secondary_component_count"])
        self.assertIsNone(out["secondary_primary_area_ratio"])
        self.assertFalse(out["secondary_fields_available"])

    def test_zero_primary_area_does_not_divide(self):
        out = _secondary_component_fields(
            {"primary_component_area": 0.0, "secondary_component_area": 5.0}
        )
        self.assertIsNone(out["secondary_primary_area_ratio"])


class RunEvaluationTest(unittest.TestCase):
    def test_empty_input_is_safe(self):
        report = run_evaluation({})
        self.assertEqual(report["volumes_scanned"], 0)
        self.assertEqual(report["holdout_selected"], 0)
        self.assertFalse(report["label_write"])
        self.assertEqual(report["writes"]["labels"], 0)
        self.assertNotEqual(report["target_path"], "app/check_auto_apply.py")

    def test_per_volume_fields_and_suppression_reasons(self):
        records = [
            _record("case_a", 500.0, 0.3),
            _record("case_b", 200.0, 0.2),
            _record("case_c", 900.0, 0.9),
        ]
        report = run_evaluation({"records": records})
        self.assertEqual(report["volumes_scanned"], 3)
        for cid in ("case_a", "case_b", "case_c"):
            case = report["cases"][cid]
            self.assertIn("secondary_components", case)
            self.assertIn("secondary_primary_area_ratio", case["secondary_components"])
            self.assertIn("secondary_adjacent_to_primary", case["secondary_components"])
            self.assertIn("suppression_reason", case)
        self.assertEqual(report["no_transition_without_reason_count"], 0)
        self.assertEqual(
            sum(report["suppression_reasons"].values()), report["volumes_scanned"]
        )

    def test_denominator_mismatch_marks_reconciliation_unavailable(self):
        report = run_evaluation({"records": [_record("case_a", 500.0, 0.3)]})
        self.assertFalse(report["checked_denominator_matches_expected"])
        self.assertEqual(
            report["reconciliation_status"],
            "unavailable_checked_denominator_mismatch",
        )
        self.assertIsNone(report["no_transition_reconciliation_delta_vs_s783"])

    def test_expected_checked_constant(self):
        self.assertEqual(EXPECTED_CHECKED, 184)

    def test_report_is_json_serializable(self):
        report = run_evaluation({"records": [_record("case_a", 500.0, 0.3)]})
        text = report_to_json(report)
        self.assertIn("volumes_scanned", text)


if __name__ == "__main__":
    unittest.main()
