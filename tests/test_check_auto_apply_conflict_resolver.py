"""Offline unit tests for the route-conflict discriminator in app/check_auto_apply.py.

These tests exercise only the pure ``resolve_route_conflict`` hook and the
preview JSON contract.  They do not read datasets, masks, runtime databases or
the frozen holdout, and they never write labels.
"""
from __future__ import annotations

import unittest

from app.check_auto_apply import (
    CONFLICT_MAX_RANK,
    CONFLICT_MIN_QUALITY,
    CONFLICT_MIN_RANK,
    resolve_route_conflict,
)


def _evaluation(**overrides):
    base = {
        "quality_status": "SCORED",
        "quality_before": 82.0,
        "quality_after": 82.0,
        "quality_delta": 0.0,
        "quality_flags_after": [],
    }
    base.update(overrides)
    return base


def _spatial(**overrides):
    base = {
        "decision": "spatial_boundary_candidate",
        "reason": "outside_unique_sax_core",
        "rank": 1,
        "core": [2, 7],
    }
    base.update(overrides)
    return base


class ConflictResolverTests(unittest.TestCase):
    def test_checked_route_has_reason(self):
        result = resolve_route_conflict(
            evaluation=_evaluation(),
            candidate_records=[{"passed": True}],
            spatial_record=_spatial(),
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["route"], "checked")
        self.assertTrue(result["reason"].strip())
        self.assertIn("conflict_quality_C_pass", result["reason"])

    def test_trash_route_has_reason(self):
        result = resolve_route_conflict(
            evaluation=_evaluation(quality_before=40.0, quality_after=45.0,
                                   quality_delta=-1.0),
            candidate_records=[{"passed": True}],
            spatial_record=_spatial(decision="spatial_boundary_candidate"),
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["route"], "trash")
        self.assertTrue(result["reason"].strip())
        self.assertIn("conflict_quality_below_C", result["reason"])

    def test_unresolved_spatial_decision_can_resolve(self):
        # Preflight root cause: pending queue is dominated by 'unresolved'.
        result = resolve_route_conflict(
            evaluation=_evaluation(),
            candidate_records=[{"passed": True}],
            spatial_record=_spatial(decision="unresolved",
                                    reason="inside_sax_core"),
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["route"], "checked")
        self.assertTrue(result["reason"].strip())

    def test_rank_zero_is_in_range(self):
        result = resolve_route_conflict(
            evaluation=_evaluation(),
            candidate_records=[{"passed": True}],
            spatial_record=_spatial(rank=0),
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["route"], "checked")

    def test_rank_out_of_range_abstains(self):
        result = resolve_route_conflict(
            evaluation=_evaluation(),
            candidate_records=[{"passed": True}],
            spatial_record=_spatial(rank=CONFLICT_MAX_RANK + 1),
        )
        self.assertIsNone(result)

    def test_missing_core_abstains(self):
        record = _spatial()
        record.pop("core")
        result = resolve_route_conflict(
            evaluation=_evaluation(),
            candidate_records=[{"passed": True}],
            spatial_record=record,
        )
        self.assertIsNone(result)

    def test_unknown_decision_abstains(self):
        result = resolve_route_conflict(
            evaluation=_evaluation(),
            candidate_records=[{"passed": True}],
            spatial_record=_spatial(decision="something_else"),
        )
        self.assertIsNone(result)

    def test_unscored_abstains(self):
        result = resolve_route_conflict(
            evaluation=_evaluation(quality_status="UNSCORED"),
            candidate_records=[{"passed": True}],
            spatial_record=_spatial(),
        )
        self.assertIsNone(result)

    def test_defect_flag_blocks_checked_route(self):
        result = resolve_route_conflict(
            evaluation=_evaluation(quality_flags_after=["fragmentation_candidate"]),
            candidate_records=[{"passed": True}],
            spatial_record=_spatial(),
        )
        self.assertIsNone(result)

    def test_quality_threshold_constant_is_c_grade(self):
        self.assertEqual(CONFLICT_MIN_QUALITY, 70.0)
        self.assertEqual(CONFLICT_MIN_RANK, 0)


if __name__ == "__main__":
    unittest.main()
