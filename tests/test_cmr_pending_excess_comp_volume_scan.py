"""Offline unit tests for the S793 pending excess-component volume scan.

No external data, images or masks are required. Tests cover the 86-denominator
reconciliation, unavailable-metric reporting, zero-write contract and the
per-volume secondary-component field extraction.
"""

from __future__ import annotations

import unittest

from experiments.cmr_pending_excess_comp_volume_scan import (
    EXPECTED_PENDING,
    PENDING_BASELINE_S792,
    _reconciliation_table,
    _secondary_component_fields,
    run_evaluation,
)
from experiments.cmr_low_cavity_discriminator_scan import (
    NO_TRANSITION_BASELINE_S783,
)


def _record(case_id: str, area: float, cavity: float, **extra) -> dict:
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


class ReconciliationTableTest(unittest.TestCase):
    def test_denominator_match_reports_deltas(self) -> None:
        table = _reconciliation_table(EXPECTED_PENDING, 40, 30, 16)
        self.assertTrue(table["pending_denominator_matches_expected"])
        self.assertEqual(table["status"], "ok")
        self.assertEqual(
            table["no_transition_delta_vs_s792_pending_baseline"],
            16 - PENDING_BASELINE_S792,
        )
        self.assertEqual(
            table["no_transition_delta_vs_s783"],
            16 - NO_TRANSITION_BASELINE_S783,
        )

    def test_denominator_mismatch_marks_unavailable(self) -> None:
        table = _reconciliation_table(EXPECTED_PENDING - 1, 40, 30, 15)
        self.assertFalse(table["pending_denominator_matches_expected"])
        self.assertEqual(table["status"], "unavailable_pending_denominator_mismatch")
        self.assertIsNone(table["no_transition_delta_vs_s792_pending_baseline"])
        self.assertIsNone(table["no_transition_delta_vs_s783"])


class SecondaryComponentFieldsTest(unittest.TestCase):
    def test_ratio_derived_when_areas_supplied(self) -> None:
        fields = _secondary_component_fields(
            {
                "secondary_component_count": 2,
                "primary_component_area": 200.0,
                "secondary_component_area": 20.0,
                "secondary_adjacent_to_primary": True,
                "hole_count": 1,
                "touches_border": False,
                "centroid_drift": 0.5,
            }
        )
        self.assertAlmostEqual(fields["secondary_primary_area_ratio"], 0.1)
        self.assertTrue(fields["secondary_fields_available"])

    def test_missing_fields_are_unavailable_not_imputed(self) -> None:
        fields = _secondary_component_fields({"case_id": "case_x"})
        self.assertIsNone(fields["secondary_component_count"])
        self.assertIsNone(fields["secondary_primary_area_ratio"])
        self.assertFalse(fields["secondary_fields_available"])


class RunEvaluationTest(unittest.TestCase):
    def _records(self, n: int) -> list:
        return [
            _record(f"case_{i:04d}", 300.0 + i, 0.3, secondary_component_count=1)
            for i in range(n)
        ]

    def test_86_denominator_reconciles_and_sums(self) -> None:
        report = run_evaluation({"records": self._records(EXPECTED_PENDING)})
        self.assertEqual(report["n_records"], EXPECTED_PENDING)
        self.assertTrue(report["pending_denominator_matches_expected"])
        self.assertEqual(report["simulated_total"], EXPECTED_PENDING)
        self.assertTrue(report["simulated_total_matches_denominator"])
        self.assertEqual(report["reconciliation_status"], "ok")
        self.assertEqual(
            report["simulated_checked_total"]
            + report["simulated_trash_total"]
            + report["simulated_no_transition_total"],
            EXPECTED_PENDING,
        )

    def test_denominator_mismatch_marks_reconciliation_unavailable(self) -> None:
        report = run_evaluation({"records": self._records(EXPECTED_PENDING - 1)})
        self.assertFalse(report["pending_denominator_matches_expected"])
        self.assertEqual(
            report["reconciliation_status"],
            "unavailable_pending_denominator_mismatch",
        )
        self.assertIsNone(
            report["no_transition_reconciliation_delta_vs_s792_pending_baseline"]
        )

    def test_unavailable_metrics_are_not_imputed(self) -> None:
        report = run_evaluation({"records": self._records(EXPECTED_PENDING)})
        self.assertEqual(report["topology_deltas"]["status"], "unavailable_no_masks")
        self.assertIsNone(report["topology_deltas"]["components_delta"])
        self.assertIn("concordance_vs_independent_labels", report["unavailable"])
        self.assertIn("precision", report["unavailable"])
        self.assertIn("ci", report["unavailable"])

    def test_zero_write_contract(self) -> None:
        report = run_evaluation({"records": self._records(EXPECTED_PENDING)})
        self.assertEqual(report["writes"], {"labels": 0, "code": 0, "datasets": 0})
        # audit_diff returns a structured zero-write record, not an empty dict.
        audit = report["audit_diff"]
        self.assertIsInstance(audit, dict)
        self.assertEqual(audit.get("changed_cases"), [])
        self.assertEqual(audit.get("writes", {}).get("labels", 0), 0)
        self.assertEqual(audit.get("writes", {}).get("code", 0), 0)
        self.assertEqual(audit.get("writes", {}).get("datasets", 0), 0)
        self.assertIn("rollback_command", report)
        self.assertIn("module_hash", report)

    def test_per_volume_secondary_fields_present(self) -> None:
        report = run_evaluation({"records": self._records(EXPECTED_PENDING)})
        case = report["cases"]["case_0000"]
        self.assertIn("secondary_components", case)
        self.assertEqual(case["secondary_components"]["secondary_component_count"], 1)


if __name__ == "__main__":
    unittest.main()
