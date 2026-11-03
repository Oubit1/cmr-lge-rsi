"""Offline unit tests for the CMR topology-delta shadow scan."""

from __future__ import annotations

import unittest

from experiments.cmr_topology_delta_scan_exec_13 import (
    EXPECTED_NON_HOLDOUT,
    run_evaluation,
)


def _rec(cid, **kw):
    base = {
        "case_id": cid,
        "reference_area": 1000.0,
        "before_area": 400.0,
        "after_area": 400.0,
        "before_cavity": 0.3,
        "after_cavity": 0.3,
        "quality_before": 70.0,
        "quality_after": 70.0,
        "quality_delta": 0.0,
        "quality_flags_before": [],
        "quality_flags_after": [],
        "quality_subscores_before": {"topology": 80.0},
        "quality_subscores_after": {"topology": 80.0},
    }
    base.update(kw)
    return base


def _spatial(cid, decision, reason="outside_unique_sax_core"):
    return {"case_id": cid, "state": "待筛", "decision": decision, "reason": reason}


class TopologyDeltaScanTests(unittest.TestCase):
    def test_holdout_excluded_and_zero_writes(self):
        records = [_rec(f"case_{i:02d}") for i in range(15)]
        holdout = ["case_00", "case_01"]
        data = {"records": records, "holdout_case_ids": holdout}
        report = run_evaluation(data)
        self.assertEqual(report["holdout_selected"], 0)
        self.assertEqual(report["holdout_excluded_count"], 2)
        self.assertEqual(report["n_records"], 13)
        self.assertFalse(report["label_write"])
        self.assertEqual(report["writes"], {"labels": 0, "code": 0, "datasets": 0})
        self.assertEqual(report["expected_non_holdout"], EXPECTED_NON_HOLDOUT)

    def test_deltas_filled_when_geometry_supplied(self):
        rec = _rec(
            "case_a",
            before_components=1,
            after_components=3,
            before_holes=0,
            after_holes=2,
            before_touches_border=False,
            after_touches_border=True,
            before_centroid=[10.0, 20.0],
            after_centroid=[12.0, 24.0],
        )
        data = {"records": [rec], "holdout_case_ids": []}
        report = run_evaluation(data)
        deltas = report["cases"]["case_a"]["topology_deltas"]
        self.assertEqual(deltas["component_delta"], 2)
        self.assertEqual(deltas["hole_delta"], 2)
        self.assertEqual(deltas["border_delta"], 1)
        self.assertEqual(deltas["centroid_delta"]["dx"], 2.0)
        self.assertEqual(deltas["centroid_delta"]["dy"], 4.0)
        self.assertTrue(deltas["component_delta_available"])
        self.assertTrue(deltas["centroid_delta_available"])
        self.assertEqual(report["topology_delta_filled"]["component_delta"], 1)

    def test_missing_geometry_reported_unavailable_not_imputed(self):
        rec = _rec("case_b")
        data = {"records": [rec], "holdout_case_ids": []}
        report = run_evaluation(data)
        deltas = report["cases"]["case_b"]["topology_deltas"]
        self.assertIsNone(deltas["component_delta"])
        self.assertIsNone(deltas["hole_delta"])
        self.assertIsNone(deltas["border_delta"])
        self.assertIsNone(deltas["centroid_delta"])
        self.assertFalse(deltas["component_delta_available"])
        self.assertIn("component_delta", report["cases"]["case_b"]["unavailable_fields"])
        self.assertEqual(report["topology_delta_unavailable"]["component_delta"], 1)

    def test_guard_applicable_and_no_transition_reason(self):
        records = [
            _rec("case_c", before_area=200.0, after_area=200.0, reference_area=1000.0),
            _rec("case_d", before_area=900.0, after_area=900.0, reference_area=1000.0),
        ]
        spatial = {
            "records": [
                _spatial("case_c", "spatial_boundary_candidate"),
                _spatial("case_d", "unresolved", "inside_sax_core"),
            ]
        }
        data = {
            "records": records,
            "holdout_case_ids": [],
            "spatial_control": spatial,
        }
        report = run_evaluation(data)
        self.assertEqual(report["topology_guard_applicable"], 1)
        self.assertEqual(report["topology_guard_applicable_rate"], 0.5)
        self.assertEqual(report["no_transition_reason_populated"], 2)
        self.assertEqual(report["no_transition_reason_populated_rate"], 1.0)
        self.assertEqual(
            report["cases"]["case_c"]["guard"]["guard_route"],
            "expand_undersegmented_region",
        )
        self.assertEqual(
            report["cases"]["case_d"]["guard"]["no_transition_reason"],
            "inside_sax_core",
        )

    def test_fragmentation_blocks_expansion(self):
        rec = _rec(
            "case_e",
            before_area=200.0,
            after_area=200.0,
            reference_area=1000.0,
            quality_flags_before=["fragmentation_candidate"],
        )
        spatial = {"records": [_spatial("case_e", "spatial_boundary_candidate")]}
        data = {"records": [rec], "holdout_case_ids": [], "spatial_control": spatial}
        report = run_evaluation(data)
        guard = report["cases"]["case_e"]["guard"]
        self.assertFalse(guard["guard_applicable"])
        self.assertEqual(guard["guard_route"], "hold_for_review")
        self.assertEqual(guard["no_transition_reason"], "fragmentation_blocks_expansion")

    def test_duplicate_case_ids_deduplicated(self):
        records = [_rec("case_f"), _rec("case_f")]
        data = {"records": records, "holdout_case_ids": []}
        report = run_evaluation(data)
        self.assertEqual(report["n_records"], 1)
        self.assertEqual(report["duplicate_case_ids"], ["case_f"])


if __name__ == "__main__":
    unittest.main()
