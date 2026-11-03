"""Offline unit tests for the read-only LGE MI component shadow scanner.

Covers the intensity-proxy continuity / border-contact estimators, the
non-holdout selection, the populated-reason contract (no_reason_count==0),
the cross_slice_continuity_agreement metric and the zero-write contract.
No external data, masks or labels are required.
"""

from __future__ import annotations

import unittest

from experiments.shadow_lge_mi_component_scan import (
    CONTINUITY_AGREEMENT_MIN,
    EXPECTED_NON_HOLDOUT,
    INTENSITY_PROXY_TOPOLOGY_FLOOR,
    ROUTED_REASON,
    _intensity_proxy_border_contact,
    _intensity_proxy_continuity,
    _select_non_holdout,
    run_evaluation,
)


def _rec(cid, area=500.0, cavity=0.3, topology=80.0, flags=None, **extra):
    rec = {
        "case_id": cid,
        "before_area": area,
        "before_cavity": cavity,
        "quality_flags_before": list(flags or []),
        "quality_subscores_before": {
            "integrity": 100.0,
            "topology": topology,
            "temporal": 60.0,
            "image_support": 100.0,
        },
    }
    rec.update(extra)
    return rec


class IntensityProxyContinuityTest(unittest.TestCase):
    def test_above_floor_is_continuous(self):
        out = _intensity_proxy_continuity(_rec("c1", topology=80.0))
        self.assertTrue(out["continuous"])
        self.assertEqual(out["floor"], INTENSITY_PROXY_TOPOLOGY_FLOOR)

    def test_below_floor_is_discontinuous(self):
        out = _intensity_proxy_continuity(_rec("c1", topology=33.5))
        self.assertFalse(out["continuous"])
        self.assertEqual(out["reason"], "topology_subscore_below_floor")

    def test_missing_subscore_is_unavailable(self):
        rec = _rec("c1")
        rec["quality_subscores_before"] = {}
        out = _intensity_proxy_continuity(rec)
        self.assertIsNone(out["continuous"])
        self.assertEqual(out["reason"], "missing_measurement")

    def test_border_contact_proxy(self):
        out = _intensity_proxy_border_contact(
            _rec("c1", flags=["border_contact_review"])
        )
        self.assertTrue(out["touches_border"])
        out2 = _intensity_proxy_border_contact(_rec("c2"))
        self.assertFalse(out2["touches_border"])


class SelectionTest(unittest.TestCase):
    def test_holdout_excluded_and_dedup(self):
        records = [
            _rec("h1"),
            _rec("p1"),
            _rec("p1"),
            _rec("p2"),
        ]
        sel = _select_non_holdout(records, ["h1"])
        self.assertEqual(sel["holdout_selected"], 0)
        self.assertEqual(sel["holdout_excluded_count"], 1)
        self.assertEqual([r["case_id"] for r in sel["selected"]], ["p1", "p2"])
        self.assertEqual(sel["duplicate_case_ids"], ["p1"])


class RunEvaluationTest(unittest.TestCase):
    def _data(self):
        records = [
            _rec("h1"),
            _rec("p1", topology=80.0),
            _rec("p2", topology=33.5, flags=["fragmentation_candidate"]),
            _rec("p3", topology=80.0, flags=["border_contact_review"]),
        ]
        return {
            "records": records,
            "holdout_case_ids": ["h1"],
            "spatial_control": {"records": []},
        }

    def test_zero_label_write_and_no_reason(self):
        report = run_evaluation(self._data())
        self.assertFalse(report["label_write"])
        self.assertEqual(report["writes"]["labels"], 0)
        self.assertEqual(report["no_reason_count"], 0)
        self.assertTrue(report["no_reason_count_is_zero"])
        self.assertEqual(report["holdout_selected"], 0)
        self.assertEqual(report["non_holdout_volumes_scanned"], 3)

    def test_continuity_agreement_metric(self):
        report = run_evaluation(self._data())
        self.assertIsNotNone(report["cross_slice_continuity_agreement"])
        self.assertEqual(report["cross_slice_continuity_agreement_denominator"], 3)
        self.assertGreaterEqual(
            report["cross_slice_continuity_agreement"], 0.0
        )
        self.assertLessEqual(
            report["cross_slice_continuity_agreement"], 1.0
        )
        self.assertEqual(
            report["cross_slice_continuity_agreement_threshold"],
            CONTINUITY_AGREEMENT_MIN,
        )

    def test_every_candidate_has_reason(self):
        report = run_evaluation(self._data())
        for cand in report["suppression_candidates"]:
            self.assertTrue(cand["reason"])
            self.assertNotEqual(cand["reason"], "")

    def test_expected_non_holdout_constant(self):
        self.assertEqual(EXPECTED_NON_HOLDOUT, 13)

    def test_routed_reason_present(self):
        report = run_evaluation(self._data())
        self.assertIn(ROUTED_REASON, report["suppression_reasons"])


if __name__ == "__main__":
    unittest.main()
