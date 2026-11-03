"""Offline tests for the read-only S780 unrouted-cavity scan."""

from __future__ import annotations

import unittest

from experiments.cmr_unrouted_cavity_scan import (
    EXPECTED_UNROUTED,
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


def _spatial(cid, state, decision, reason, rank, core=None):
    rec = {
        "case_id": cid,
        "state": state,
        "decision": decision,
        "reason": reason,
        "rank": rank,
    }
    if core is not None:
        rec["core"] = core
    return rec


class UnroutedCavityScanTest(unittest.TestCase):
    def _data(self):
        records = []
        spatial = []
        # 22 unrouted cases: cavity > 0.5 -> cluster None.
        for i in range(EXPECTED_UNROUTED):
            cid = f"unrouted_{i:02d}"
            records.append(_rec(cid, 1000.0, 500.0, 0.6 + 0.01 * i))
            spatial.append(
                _spatial(cid, "待筛", "spatial_boundary_candidate",
                         "outside_unique_sax_core", i, [1, 8])
            )
        # 2 routed cases: cavity <= 0.5 -> cluster not None.
        records.append(_rec("routed_0", 1000.0, 500.0, 0.2))
        records.append(_rec("routed_1", 1000.0, 500.0, 0.4))
        spatial.append(_spatial("routed_0", "checked", "unresolved",
                                "inside_sax_core", 0, [0, 8]))
        spatial.append(_spatial("routed_1", "checked", "unresolved",
                                "inside_sax_core", 1, [0, 8]))
        return {"records": records, "spatial_control": {"records": spatial}}

    def test_unrouted_count_matches_expected(self):
        report = run_evaluation(self._data())
        self.assertEqual(report["unrouted_count"], EXPECTED_UNROUTED)
        self.assertTrue(report["unrouted_count_matches_expected"])
        self.assertEqual(report["unrouted_denominator"], 24)

    def test_spatial_rule_overlap_reported(self):
        report = run_evaluation(self._data())
        self.assertEqual(report["spatial_rule_overlap_count"], EXPECTED_UNROUTED)
        self.assertEqual(report["area_rule_overlap_count"], EXPECTED_UNROUTED)
        self.assertEqual(report["spatial_and_area_overlap_count"], EXPECTED_UNROUTED)

    def test_simulated_transitions_sum_to_unrouted(self):
        report = run_evaluation(self._data())
        t = report["simulated_transitions"]
        self.assertEqual(t["total"], EXPECTED_UNROUTED)
        self.assertEqual(t["denominator"], EXPECTED_UNROUTED)
        self.assertEqual(t["checked"] + t["trash"] + t["no_transition"],
                         EXPECTED_UNROUTED)

    def test_missing_topology_fields_unavailable(self):
        report = run_evaluation(self._data())
        for field, info in report["topology_unavailable_counts"].items():
            self.assertEqual(info["count"], EXPECTED_UNROUTED, field)
            self.assertEqual(info["denominator"], EXPECTED_UNROUTED, field)
        for cid, case in report["cases"].items():
            for field, info in case["topology"].items():
                self.assertIsNone(info["value"], (cid, field))
                self.assertEqual(info["status"],
                                 "unavailable_no_masks_or_geometry")

    def test_zero_writes_and_audit_diff(self):
        report = run_evaluation(self._data())
        self.assertEqual(report["writes"],
                         {"labels": 0, "code": 0, "datasets": 0})
        self.assertEqual(report["audit_diff"]["production_writes"], 0)
        self.assertEqual(report["audit_diff"]["changed_cases"], [])

    def test_missing_spatial_record_unavailable(self):
        data = self._data()
        data["spatial_control"]["records"] = [
            r for r in data["spatial_control"]["records"]
            if r["case_id"] != "unrouted_00"
        ]
        report = run_evaluation(data)
        case = report["cases"]["unrouted_00"]
        self.assertEqual(case["spatial_rule"]["status"], "unavailable")
        self.assertFalse(case["spatial_rule_overlap"])


if __name__ == "__main__":
    unittest.main()
