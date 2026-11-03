"""Offline tests for the S777 checked-control discriminator in
scripts/shadow_spatial_boundary.py.

These tests exercise the pure discriminator helpers only; they do not load
nibabel/numpy/scipy or touch any dataset. No external data is required.
"""
import importlib.util
import sys
import unittest
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "shadow_spatial_boundary.py"
    spec = importlib.util.spec_from_file_location("shadow_spatial_boundary", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["shadow_spatial_boundary"] = mod
    spec.loader.exec_module(mod)
    return mod


class DiscriminatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_module()

    def test_checked_misroute_is_suppressed(self):
        rows = [
            {"state": "checked", "decision": "spatial_boundary_candidate",
             "reason": "outside_unique_sax_core", "rank": 11, "core": [2, 9]},
        ]
        observed = self.mod._checked_control_set(rows)
        out = self.mod._discriminate(rows[0], observed)
        self.assertEqual(out["decision"], "unresolved")
        self.assertTrue(out["suppressed"])

    def test_pending_unobserved_rank_is_preserved(self):
        rows = [
            {"state": "checked", "decision": "spatial_boundary_candidate",
             "reason": "outside_unique_sax_core", "rank": 11, "core": [2, 9]},
            {"state": "待筛", "decision": "spatial_boundary_candidate",
             "reason": "outside_unique_sax_core", "rank": 1, "core": [3, 11]},
        ]
        observed = self.mod._checked_control_set(rows)
        out = self.mod._discriminate(rows[1], observed)
        self.assertEqual(out["decision"], "spatial_boundary_candidate")
        self.assertFalse(out["suppressed"])

    def test_pending_observed_rank_is_suppressed(self):
        rows = [
            {"state": "checked", "decision": "spatial_boundary_candidate",
             "reason": "outside_unique_sax_core", "rank": 11, "core": [2, 9]},
            {"state": "待筛", "decision": "spatial_boundary_candidate",
             "reason": "outside_unique_sax_core", "rank": 11, "core": [2, 9]},
        ]
        observed = self.mod._checked_control_set(rows)
        out = self.mod._discriminate(rows[1], observed)
        self.assertEqual(out["decision"], "unresolved")

    def test_missing_fields_reported_unavailable(self):
        rows = [{"state": "checked", "decision": "spatial_boundary_candidate",
                 "reason": "outside_unique_sax_core", "rank": None, "core": [2, 9]}]
        observed = self.mod._checked_control_set(rows)
        out = self.mod._discriminate(rows[0], observed)
        self.assertEqual(out["status"], "unavailable_missing_fields")
        self.assertIn("rank", out["missing_fields"])

    def test_checked_control_set_excludes_missing(self):
        rows = [
            {"state": "checked", "decision": "spatial_boundary_candidate",
             "reason": "outside_unique_sax_core", "rank": 11, "core": [2, 9]},
            {"state": "checked", "decision": "spatial_boundary_candidate",
             "reason": "outside_unique_sax_core", "rank": None, "core": [2, 9]},
            {"state": "checked", "decision": "unresolved",
             "reason": "inside_sax_core", "rank": 1, "core": [2, 9]},
        ]
        observed = self.mod._checked_control_set(rows)
        self.assertEqual(observed, {((2, 9), 11)})


if __name__ == "__main__":
    unittest.main()
