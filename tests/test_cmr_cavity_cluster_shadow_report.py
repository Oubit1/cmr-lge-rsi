"""Offline unit tests for the read-only cavity-cluster shadow report."""

import unittest

from experiments.cmr_cavity_cluster_shadow_report import (
    CLUSTERS,
    SUPPRESSION_REASONS,
    FrameFlag,
    VolumeInput,
    build_report,
    report_with_provenance,
    triage_volume,
)


def _vol(case_id, n, flags, reasons=None, refs=None):
    reasons = reasons or [None] * n
    frames = [FrameFlag(i, flags[i], reasons[i]) for i in range(n)]
    return VolumeInput(case_id=case_id, n_frames=n, frames=frames, reference_flags=refs)


class TestTriage(unittest.TestCase):
    def test_cluster_assignment_and_counts(self):
        v = _vol("c1", 9, [True] * 9)
        tri = triage_volume(v)
        self.assertEqual(tri["cluster_counts"]["cavity_base"]["total"], 3)
        self.assertEqual(tri["cluster_counts"]["cavity_mid"]["total"], 3)
        self.assertEqual(tri["cluster_counts"]["cavity_apex"]["total"], 3)
        for c in CLUSTERS:
            self.assertEqual(tri["cluster_counts"][c]["flagged"], 3)

    def test_suppression_reasons_and_denominator(self):
        v = _vol(
            "c2",
            4,
            [False, False, False, False],
            reasons=["empty", "border_touch", "temporal_hole", "centroid_drift"],
        )
        tri = triage_volume(v)
        for r in SUPPRESSION_REASONS:
            self.assertEqual(tri["suppression_counts"][r], 1)
        self.assertEqual(tri["suppression_denominator"], 4)

    def test_unknown_reason_rejected(self):
        v = _vol("c3", 1, [False], reasons=["bogus"])
        with self.assertRaises(ValueError):
            triage_volume(v)

    def test_frame_index_out_of_range(self):
        v = VolumeInput(case_id="c4", n_frames=2, frames=[FrameFlag(5, True)])
        with self.assertRaises(ValueError):
            triage_volume(v)


class TestReport(unittest.TestCase):
    def test_concordance_unavailable_without_reference(self):
        vols = [_vol("a", 3, [True, False, True]), _vol("b", 3, [False, False, True])]
        rep = build_report(vols)
        self.assertFalse(rep["concordance"]["available"])
        self.assertEqual(rep["writes"], {"labels": 0, "code": 0})
        self.assertEqual(rep["n_volumes"], 2)

    def test_concordance_with_reference(self):
        vols = [
            _vol("a", 3, [True, False, True], refs=[True, True, True]),
            _vol("b", 3, [False, False, True], refs=[False, False, True]),
        ]
        rep = build_report(vols)
        self.assertTrue(rep["concordance"]["available"])
        for c in CLUSTERS:
            entry = rep["concordance"]["per_cluster"][c]
            self.assertEqual(entry["total"], 2)
            self.assertIsNotNone(entry["ci95"])
            lo, hi = entry["ci95"]
            self.assertLessEqual(lo, hi)

    def test_aggregate_suppression_denominator(self):
        vols = [
            _vol("a", 2, [False, False], reasons=["empty", None]),
            _vol("b", 2, [False, False], reasons=["border_touch", "empty"]),
        ]
        rep = build_report(vols)
        self.assertEqual(rep["suppression"]["counts"]["empty"], 2)
        self.assertEqual(rep["suppression"]["counts"]["border_touch"], 1)
        self.assertEqual(rep["suppression"]["denominator"], 3)

    def test_provenance_fields_present(self):
        rep = report_with_provenance([_vol("a", 1, [True])])
        prov = rep["provenance"]
        self.assertEqual(len(prov["module_hash"]), 64)
        self.assertIn("read-only", prov["audit_diff"])
        self.assertIn("git checkout", prov["rollback_command"])


if __name__ == "__main__":
    unittest.main()
