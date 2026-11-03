"""Offline unit tests for the read-only S829 pending small-area cluster scan."""

from __future__ import annotations

import unittest

from experiments.cmr_pending_small_area_cluster_scan_exec_13 import (
    EXPECTED_PENDING,
    run_evaluation,
)


def _record(cid, before_area, reference_area, cavity):
    return {
        "case_id": cid,
        "reference_area": reference_area,
        "before_area": before_area,
        "before_cavity": cavity,
        "after_area": before_area,
        "after_cavity": cavity,
        "decision": "REVIEW",
        "quality_status": "SCORED",
    }


def _spatial(cid, decision, reason, rank, core=None):
    rec = {
        "case_id": cid,
        "state": "待筛",
        "decision": decision,
        "reason": reason,
        "rank": rank,
    }
    if core is not None:
        rec["core"] = core
    return rec


class PendingSmallAreaClusterScanExec13Test(unittest.TestCase):
    def _payload(self):
        records = [
            _record("case_a", 100.0, 1000.0, 0.2),
            _record("case_b", 200.0, 1000.0, 0.3),
            _record("case_c", 300.0, 1000.0, 0.4),
            _record("case_d", 400.0, 1000.0, 0.5),
            _record("case_e", 500.0, 1000.0, 0.6),
            _record("case_f", 600.0, 1000.0, 0.7),
            _record("case_g", 700.0, 1000.0, 0.8),
            _record("case_h", 800.0, 1000.0, 0.9),
            _record("case_i", 900.0, 1000.0, 1.0),
            _record("case_j", 150.0, 1000.0, 0.25),
            _record("case_k", 250.0, 1000.0, 0.35),
            _record("case_l", 350.0, 1000.0, 0.45),
            _record("case_m", 450.0, 1000.0, 0.55),
        ]
        spatial = {
            "status": "AVAILABLE",
            "records": [
                _spatial("case_a", "spatial_boundary_candidate", "outside_unique_sax_core", 1, [3, 11]),
                _spatial("case_b", "unresolved", "inside_sax_core", 11, [3, 11]),
                _spatial("case_c", "unresolved", "inside_sax_core", 0, [0, 10]),
                _spatial("case_d", "unresolved", "inside_sax_core", 1, [1, 10]),
                _spatial("case_e", "unresolved", "inside_sax_core", 11, [0, 11]),
                _spatial("case_f", "unresolved", "inside_sax_core", 0, [0, 11]),
                _spatial("case_g", "unresolved", "inside_sax_core", 6, [0, 6]),
                _spatial("case_h", "unresolved", "inside_sax_core", 0, [0, 6]),
                _spatial("case_i", "unresolved", "inside_sax_core", 9, [1, 10]),
                _spatial("case_j", "unresolved", "inside_sax_core", 10, [1, 10]),
                _spatial("case_k", "unresolved", "inside_sax_core", 11, [0, 11]),
                _spatial("case_l", "unresolved", "inside_sax_core", 0, [0, 11]),
                _spatial("case_m", "unresolved", "inside_sax_core", 1, [0, 11]),
            ],
        }
        return {"records": records, "spatial_control": spatial}

    def test_scan_counts_and_zero_write(self):
        report = run_evaluation(self._payload())
        self.assertEqual(report["non_holdout_volumes_scanned"], EXPECTED_PENDING)
        self.assertEqual(report["holdout_selected"], 0)
        self.assertFalse(report["label_write"])
        self.assertNotEqual(report["target_path"], "app/check_auto_apply.py")
        self.assertEqual(report["writes"], {"labels": 0, "code": 0, "datasets": 0})
        self.assertTrue(report["simulated_total_matches_denominator"])

    def test_per_cluster_transitions_non_negative(self):
        report = run_evaluation(self._payload())
        for cluster, count in report["simulated_transitions_per_cluster"].items():
            self.assertGreaterEqual(count, 0, cluster)

    def test_every_no_transition_has_reason(self):
        report = run_evaluation(self._payload())
        self.assertEqual(report["no_transition_without_reason_count"], 0)
        for cid, case in report["cases"].items():
            if case["simulated_transition"] == "no_transition":
                self.assertTrue(case["suppression_reason"], cid)

    def test_unavailable_metrics_not_imputed(self):
        report = run_evaluation(self._payload())
        self.assertEqual(
            report["unavailable"]["concordance_vs_independent_labels"],
            "no independent reference labels supplied",
        )
        self.assertEqual(report["topology_deltas"]["status"], "unavailable_no_masks")

    def test_empty_input_is_safe(self):
        report = run_evaluation({"records": []})
        self.assertEqual(report["non_holdout_volumes_scanned"], 0)
        self.assertFalse(report["label_write"])
        self.assertEqual(report["no_transition_without_reason_count"], 0)


if __name__ == "__main__":
    unittest.main()
