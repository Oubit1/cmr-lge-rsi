"""Offline unit tests for the S761 frozen-holdout area-ratio check."""

from __future__ import annotations

import unittest

from experiments.cmr_area_ratio_frozen_holdout_check import (
    ALTERNATE_AREA_RATIO_MIN,
    AREA_RATIO_MIN,
    CLUSTERS,
    run_evaluation,
)


def _rec(case_id, ref, area, cavity, **extra):
    rec = {
        "case_id": case_id,
        "reference_area": ref,
        "before_area": area,
        "before_cavity": cavity,
    }
    rec.update(extra)
    return rec


class FrozenHoldoutCheckTest(unittest.TestCase):
    def test_per_cluster_counts_and_denominators(self):
        records = [
            _rec("c1", 1000.0, 100.0, 0.2),
            _rec("c2", 1000.0, 100.0, 0.9),
            _rec("c3", 1000.0, 900.0, 0.2),
        ]
        report = run_evaluation({"records": records})
        self.assertEqual(report["n_records"], 3)
        counts = report["primary"]["cluster_counts"]
        for c in CLUSTERS:
            self.assertEqual(counts[c]["denominator"], 3)
        self.assertEqual(sum(counts[c]["count"] for c in CLUSTERS), 3)

    def test_missing_measurement_not_imputed(self):
        records = [_rec("c1", 1000.0, None, 0.2)]
        report = run_evaluation({"records": records})
        self.assertIn("c1", report["primary"]["missing_measurement_cases"])
        self.assertIsNone(report["primary"]["cases"]["c1"]["area_ratio"])

    def test_threshold_reconciliation_reports_both(self):
        records = [
            _rec("c1", 1000.0, 200.0, 0.2),
            _rec("c2", 1000.0, 400.0, 0.2),
        ]
        report = run_evaluation({"records": records})
        recon = report["threshold_reconciliation"]
        self.assertEqual(recon["primary_threshold"], AREA_RATIO_MIN)
        self.assertEqual(recon["alternate_threshold"], ALTERNATE_AREA_RATIO_MIN)
        self.assertGreaterEqual(recon["primary_flagged_total"], recon["alternate_flagged_total"])

    def test_ledger_deltas_unavailable_without_ledger(self):
        report = run_evaluation({"records": [_rec("c1", 1000.0, 100.0, 0.2)]})
        self.assertEqual(report["ledger_deltas"]["status"], "unavailable_no_ledger_supplied")

    def test_ledger_deltas_computed_with_ledger(self):
        ledger = {c: {"checked": 0, "trash": 0} for c in CLUSTERS}
        report = run_evaluation({
            "records": [_rec("c1", 1000.0, 100.0, 0.2)],
            "ledger": ledger,
        })
        self.assertEqual(report["ledger_deltas"]["status"], "ok")
        for c in CLUSTERS:
            self.assertIsNotNone(report["ledger_deltas"][c]["checked_delta"])

    def test_zero_write_audit(self):
        report = run_evaluation({"records": []})
        self.assertEqual(report["writes"], {"labels": 0, "code": 0, "datasets": 0})
        self.assertEqual(report["audit_diff"]["production_writes"], 0)
        self.assertTrue(report["module_hash"])

    def test_concordance_reported_unavailable(self):
        report = run_evaluation({"records": [_rec("c1", 1000.0, 100.0, 0.2)]})
        self.assertIn("concordance_vs_independent_labels", report["unavailable"])
        self.assertIn("precision", report["unavailable"])
        self.assertIn("ci", report["unavailable"])


if __name__ == "__main__":
    unittest.main()
