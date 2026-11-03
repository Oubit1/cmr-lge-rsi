"""Offline tests for the read-only volume-level area-ratio check (S759)."""

import json
import unittest

from experiments.cmr_area_ratio_volume_check import (
    CLUSTERS,
    LEDGER_BUCKETS,
    ROLLBACK_COMMAND,
    audit_diff,
    module_hash,
    report_to_json,
    run_evaluation,
)


def _rec(cid, ref, area, cavity, after_area=None, after_cavity=None):
    return {
        "case_id": cid,
        "reference_area": ref,
        "before_area": area,
        "before_cavity": cavity,
        "after_area": area if after_area is None else after_area,
        "after_cavity": cavity if after_cavity is None else after_cavity,
    }


class AreaRatioVolumeCheckTest(unittest.TestCase):
    def test_cluster_counts_and_denominators(self):
        data = {
            "records": [
                _rec("c1", 1000.0, 600.0, 0.2),   # adequate area, low cavity
                _rec("c2", 1000.0, 100.0, 0.5),   # low area, adequate cavity
                _rec("c3", 1000.0, 100.0, 0.2),   # low area, low cavity
            ]
        }
        report = run_evaluation(data)
        self.assertEqual(report["n_records"], 3)
        for c in CLUSTERS:
            self.assertEqual(report["cluster_counts"][c]["denominator"], 3)
        total = sum(report["cluster_counts"][c]["count"] for c in CLUSTERS)
        self.assertEqual(total, 3)

    def test_missing_measurement_not_imputed(self):
        data = {"records": [{"case_id": "c1", "reference_area": 1000.0}]}
        report = run_evaluation(data)
        self.assertIn("c1", report["missing_measurement_cases"])
        self.assertIsNone(report["cases"]["c1"]["area_ratio"])
        self.assertIsNone(report["cases"]["c1"]["myo_cavity_area_ratio"])

    def test_ledger_deltas_unavailable_without_ledger(self):
        report = run_evaluation({"records": [_rec("c1", 1000.0, 600.0, 0.2)]})
        self.assertEqual(
            report["ledger_deltas"]["status"], "unavailable_no_ledger_supplied"
        )
        self.assertEqual(report["ledger_deltas"]["buckets"], list(LEDGER_BUCKETS))

    def test_ledger_deltas_computed_when_supplied(self):
        data = {
            "records": [_rec("c1", 1000.0, 600.0, 0.2)],
            "ledger": {c: {"checked": 0, "trash": 0} for c in CLUSTERS},
        }
        report = run_evaluation(data)
        deltas = report["ledger_deltas"]
        self.assertEqual(deltas["status"], "ok")
        for c in CLUSTERS:
            self.assertIn("checked_delta", deltas[c])
            self.assertIn("trash_delta", deltas[c])

    def test_zero_write_audit_and_rollback(self):
        report = run_evaluation({"records": [_rec("c1", 1000.0, 600.0, 0.2)]})
        self.assertEqual(report["writes"], {"labels": 0, "code": 0, "datasets": 0})
        self.assertEqual(report["audit_diff"]["production_writes"], 0)
        self.assertEqual(report["rollback_command"], ROLLBACK_COMMAND)
        self.assertEqual(len(report["module_hash"]), 64)
        self.assertEqual(report["module_hash"], module_hash())

    def test_unavailable_metrics_reported(self):
        report = run_evaluation({"records": []})
        for key in ("concordance", "precision", "ci"):
            self.assertIn(key, report["unavailable"])
        for cond in report["suppression_coverage"]:
            self.assertIsNone(report["suppression_coverage"][cond]["count"])

    def test_report_is_json_serializable(self):
        report = run_evaluation({"records": [_rec("c1", 1000.0, 600.0, 0.2)]})
        json.loads(report_to_json(report))

    def test_audit_diff_baseline_none(self):
        report = run_evaluation({"records": [_rec("c1", 1000.0, 600.0, 0.2)]})
        diff = audit_diff(None, report)
        self.assertEqual(diff["baseline"], "none")
        self.assertEqual(diff["production_writes"], 0)


if __name__ == "__main__":
    unittest.main()
