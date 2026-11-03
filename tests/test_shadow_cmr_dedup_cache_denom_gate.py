"""Offline unittest for the read-only dedup cache + denominator gate."""

from __future__ import annotations

import unittest

from experiments.shadow_cmr_dedup_cache_denom_gate import (
    DENOM_FULL_PENDING,
    DENOM_NON_HOLDOUT,
    DENOM_SPATIAL_PENDING,
    ISSUE_TYPES,
    run_evaluation,
)


def _record(case_id, area=500.0, cavity=0.3):
    return {
        "case_id": case_id,
        "reference_area": 1000.0,
        "before_area": area,
        "before_cavity": cavity,
        "after_area": area,
        "after_cavity": cavity,
        "quality_before": 80.0,
        "quality_after": 80.0,
        "quality_delta": 0.0,
        "decision": "REVIEW",
        "quality_status": "SCORED",
        "quality_flags_before": [],
        "quality_flags_after": [],
        "quality_subscores_before": {
            "integrity": 100.0,
            "topology": 80.0,
            "temporal": 60.0,
            "image_support": 100.0,
        },
        "quality_subscores_after": {
            "integrity": 100.0,
            "topology": 80.0,
            "temporal": 60.0,
            "image_support": 100.0,
        },
    }


def _base_input(records, holdout_ids=None):
    return {
        "records": records,
        "holdout_case_ids": holdout_ids or [],
        "full_pending_queue": DENOM_FULL_PENDING,
        "frozen_split": {
            "status": "AVAILABLE",
            "non_holdout_in_queue": DENOM_NON_HOLDOUT,
        },
        "spatial_control": {
            "status": "AVAILABLE",
            "pending_candidates": DENOM_SPATIAL_PENDING,
            "records": [],
        },
    }


class DedupCacheDenomGateTest(unittest.TestCase):
    def test_cache_hit_rate_and_token_delta(self):
        records = [_record("case_a"), _record("case_b")]
        report = run_evaluation(_base_input(records))
        cache = report["cache"]
        # 2 cases x 5 issue types = 10 lookups, all misses on first pass.
        self.assertEqual(cache["total_lookups"], 2 * len(ISSUE_TYPES))
        self.assertEqual(cache["hits"], 0)
        self.assertEqual(cache["token_delta"], 0)
        self.assertFalse(report["denominator_gate"]["denominator_mismatch_blocked"])

    def test_duplicate_case_ids_are_deduplicated(self):
        records = [_record("case_a"), _record("case_a"), _record("case_b")]
        report = run_evaluation(_base_input(records))
        self.assertEqual(report["n_records"], 2)
        self.assertEqual(report["duplicate_case_ids"], ["case_a"])

    def test_denominator_mismatch_blocks_execution(self):
        records = [_record("case_a")]
        data = _base_input(records)
        data["frozen_split"]["non_holdout_in_queue"] = 99
        report = run_evaluation(data)
        gate = report["denominator_gate"]
        self.assertTrue(gate["denominator_mismatch_blocked"])
        self.assertFalse(gate["executed"])
        self.assertIn("non_holdout_in_queue", gate["mismatches"])
        self.assertEqual(report["cache"]["scans"], 0)
        self.assertEqual(report["cache"]["token_delta"], 0)

    def test_holdout_excluded_by_case_id(self):
        records = [_record("case_a"), _record("case_h")]
        report = run_evaluation(_base_input(records, holdout_ids=["case_h"]))
        self.assertEqual(report["n_records"], 1)
        self.assertEqual(report["holdout_excluded_count"], 1)

    def test_zero_write_contract(self):
        report = run_evaluation(_base_input([_record("case_a")]))
        self.assertFalse(report["label_write"])
        self.assertEqual(report["writes"], {"labels": 0, "code": 0, "datasets": 0})


if __name__ == "__main__":
    unittest.main()
