"""Offline tests for the read-only pending-volume cluster check (S779)."""

from __future__ import annotations

import unittest

from experiments.cmr_pending_volume_cluster_check import (
    CLUSTERS,
    EXPECTED_TOTAL,
    run_evaluation,
)


def _rec(cid, ref, area, cavity, **extra):
    rec = {
        "case_id": cid,
        "reference_area": ref,
        "before_area": area,
        "before_cavity": cavity,
    }
    rec.update(extra)
    return rec


class PendingVolumeClusterCheckTest(unittest.TestCase):
    def test_three_cluster_counts_sum_to_expected(self):
        records = []
        # 112 adequate_area_low_cavity: ratio >= 0.15, cavity <= 0.5
        for i in range(112):
            records.append(_rec(f"a{i}", 1000.0, 500.0, 0.2))
        # 8 low_area_adequate_cavity: ratio < 0.15, cavity > 0.5
        for i in range(8):
            records.append(_rec(f"b{i}", 1000.0, 100.0, 0.8))
        # 36 low_area_low_cavity: ratio < 0.15, cavity <= 0.5
        for i in range(36):
            records.append(_rec(f"c{i}", 1000.0, 100.0, 0.2))
        report = run_evaluation({"records": records})
        self.assertEqual(report["n_records"], EXPECTED_TOTAL)
        self.assertEqual(report["classified_total"], EXPECTED_TOTAL)
        self.assertTrue(report["classified_total_matches_expected"])
        self.assertEqual(report["cluster_counts"][CLUSTERS[0]]["count"], 112)
        self.assertEqual(report["cluster_counts"][CLUSTERS[1]]["count"], 8)
        self.assertEqual(report["cluster_counts"][CLUSTERS[2]]["count"], 36)

    def test_missing_fields_reported_unavailable_not_imputed(self):
        records = [
            _rec("ok", 1000.0, 500.0, 0.2),
            {"case_id": "missing_area"},
        ]
        report = run_evaluation({"records": records})
        self.assertIn("missing_area", report["missing_measurement_cases"])
        self.assertIn("missing_area", report["unclassified_cases"])
        self.assertIsNone(report["cases"]["missing_area"]["cluster"])
        self.assertIsNone(report["cases"]["missing_area"]["area_ratio"])
        for cond, info in report["suppression_reason_distribution"].items():
            self.assertIn(info["status"], ("ok", "unavailable_no_masks_or_geometry"))

    def test_zero_write_audit_diff_empty(self):
        report = run_evaluation({"records": [_rec("x", 1000.0, 500.0, 0.2)]})
        self.assertEqual(report["writes"], {"labels": 0, "code": 0, "datasets": 0})
        self.assertEqual(report["audit_diff"]["production_writes"], 0)
        self.assertEqual(report["audit_diff"]["changed_cases"], [])
        self.assertEqual(report["audit_diff"]["removed_cases"], [])

    def test_simulated_transition_totals_and_unavailable_metrics(self):
        records = [
            _rec("a", 1000.0, 500.0, 0.2),
            _rec("b", 1000.0, 100.0, 0.8),
            _rec("c", 1000.0, 100.0, 0.2),
        ]
        report = run_evaluation({"records": records})
        self.assertIsInstance(report["simulated_checked_total"], int)
        self.assertIsInstance(report["simulated_trash_total"], int)
        self.assertEqual(
            report["simulated_checked_total"] + report["simulated_trash_total"],
            sum(
                report["simulated_transitions"][c]["checked"]
                + report["simulated_transitions"][c]["trash"]
                for c in CLUSTERS
            ),
        )
        self.assertEqual(
            report["unavailable"]["concordance_vs_independent_labels"],
            "no independent reference labels supplied",
        )
        self.assertEqual(
            report["unavailable"]["precision"],
            "no independent reference labels supplied",
        )
        self.assertEqual(
            report["unavailable"]["ci"],
            "no independent reference labels supplied",
        )
        self.assertTrue(report["module_hash"])
        self.assertIn("git checkout", report["rollback_command"])


if __name__ == "__main__":
    unittest.main()
