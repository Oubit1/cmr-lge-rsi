"""Offline tests for the S759 read-only area-ratio skill draft."""

import json
import unittest

from experiments.cmr_area_ratio_skill_draft import (
    CLUSTERS,
    ROUTING_MODE,
    TOPOLOGY_INVARIANTS,
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


class TestAreaRatioSkillDraft(unittest.TestCase):
    def test_three_cluster_routing_and_denominators(self):
        data = {
            "records": [
                _rec("c1", 1000.0, 900.0, 0.2),   # adequate area, low cavity
                _rec("c2", 1000.0, 100.0, 0.9),   # low area, adequate cavity
                _rec("c3", 1000.0, 100.0, 0.2),   # low area, low cavity
            ]
        }
        report = run_evaluation(data)
        self.assertEqual(report["routing_mode"], ROUTING_MODE)
        self.assertEqual(report["n_records"], 3)
        for c in CLUSTERS:
            self.assertEqual(report["cluster_counts"][c]["denominator"], 3)
        total = sum(report["cluster_counts"][c]["count"] for c in CLUSTERS)
        self.assertEqual(total, 3)

    def test_missing_measurements_not_imputed(self):
        data = {"records": [{"case_id": "c1", "reference_area": 0.0, "before_area": None}]}
        report = run_evaluation(data)
        self.assertIn("c1", report["missing_measurement_cases"])
        self.assertIsNone(report["cases"]["c1"]["area_ratio"])
        self.assertIsNone(report["cases"]["c1"]["myo_cavity_area_ratio"])

    def test_ledger_deltas_unavailable_without_ledger(self):
        report = run_evaluation({"records": [_rec("c1", 1000.0, 900.0, 0.2)]})
        self.assertEqual(
            report["ledger_deltas"]["status"], "unavailable_no_ledger_supplied"
        )

    def test_ledger_deltas_computed_when_supplied(self):
        ledger = {c: {"checked": 0, "trash": 0} for c in CLUSTERS}
        report = run_evaluation(
            {"records": [_rec("c1", 1000.0, 900.0, 0.2)], "ledger": ledger}
        )
        self.assertEqual(report["ledger_deltas"]["status"], "ok")
        for c in CLUSTERS:
            self.assertIsNotNone(report["ledger_deltas"][c]["checked_delta"])

    def test_topology_invariants_declared_but_unmeasured(self):
        report = run_evaluation({"records": []})
        clauses = report["skill_clauses"]
        self.assertEqual(clauses["topology_invariants"], list(TOPOLOGY_INVARIANTS))
        self.assertEqual(
            clauses["topology_invariants_status"],
            "declared_unmeasured_no_masks_or_geometry",
        )
        self.assertIn("topology_invariants", report["unavailable"])

    def test_concordance_precision_ci_unavailable(self):
        report = run_evaluation({"records": []})
        for key in ("concordance", "precision", "ci"):
            self.assertIn(key, report["unavailable"])

    def test_zero_write_accounting_and_rollback(self):
        report = run_evaluation({"records": []})
        self.assertEqual(report["writes"], {"labels": 0, "code": 0, "datasets": 0})
        self.assertEqual(report["audit_diff"]["production_writes"], 0)
        self.assertIn("git checkout", report["rollback_command"])
        self.assertTrue(report["module_hash"])

    def test_audit_diff_detects_changes(self):
        prev = {"module_hash": "x", "cases": {"c1": {"cluster": "a"}}}
        cur = {"module_hash": "y", "cases": {"c1": {"cluster": "b"}, "c2": {}}}
        diff = audit_diff(prev, cur)
        self.assertEqual(diff["changed_cases"], ["c1"])
        self.assertEqual(diff["added_cases"], ["c2"])
        self.assertEqual(diff["production_writes"], 0)

    def test_report_is_json_serializable(self):
        report = run_evaluation({"records": [_rec("c1", 1000.0, 900.0, 0.2)]})
        parsed = json.loads(report_to_json(report))
        self.assertEqual(parsed["module_version"], report["module_version"])

    def test_module_hash_stable(self):
        self.assertEqual(module_hash(), module_hash())


if __name__ == "__main__":
    unittest.main()
