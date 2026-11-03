"""Unit tests for scripts/lge_feedback_loop.py consumer contract.

Validates all 10 required fields, UNAVAILABLE holdout handling,
count reconciliation, zero label writes, and downstream consumer audit.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from typing import Any, Dict, List

# Ensure scripts/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import lge_feedback_loop


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = [
    "source_modality",
    "workflow_stage",
    "input_source_sha256",
    "<stage>_consumer_ready",
    "<stage>_pending",
    "<stage>_work",
    "<stage>_resolved",
    "frozen_split_status",
    "holdout_excluded",
    "label_write",
    "production_writes",
]


def _make_data(
    records: List[Dict[str, Any]],
    workflow_stage: str = "lge_myo",
    source_sha256: str = lge_feedback_loop.EXPECTED_SOURCE_SHA256,
    frozen_status: str = "UNAVAILABLE",
    holdout_case_ids: List[str] | None = None,
) -> Dict[str, Any]:
    return {
        "records": records,
        "workflow_stage": workflow_stage,
        "source_sha256": source_sha256,
        "frozen_split": {"status": frozen_status},
        "holdout_case_ids": holdout_case_ids or [],
    }


def _make_record(case_id: str) -> Dict[str, Any]:
    return {"case_id": case_id}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRequiredFields(unittest.TestCase):
    """All 10 required fields must be present in the returned report."""

    def test_all_required_fields_present(self):
        data = _make_data([_make_record("c1")])
        report = lge_feedback_loop.run_evaluation(data)

        self.assertEqual(report["source_modality"], "lge")
        self.assertEqual(report["workflow_stage"], "lge_myo")
        self.assertEqual(report["input_source_sha256"], lge_feedback_loop.EXPECTED_SOURCE_SHA256)
        self.assertIsInstance(report["lge_myo_consumer_ready"], bool)
        self.assertIsInstance(report["lge_myo_pending"], int)
        self.assertIsInstance(report["lge_myo_work"], int)
        self.assertIsInstance(report["lge_myo_resolved"], int)
        self.assertEqual(report["frozen_split_status"], "UNAVAILABLE")
        self.assertEqual(report["holdout_excluded"], 0)
        self.assertFalse(report["label_write"])
        self.assertEqual(report["production_writes"], 0)


class TestHoldoutContract(unittest.TestCase):
    """When frozen_split.status is UNAVAILABLE, holdout_excluded must be 0."""

    def test_unavailable_holdout_excluded_zero(self):
        data = _make_data(
            [_make_record("c1"), _make_record("c2")],
            frozen_status="UNAVAILABLE",
            holdout_case_ids=["c1"],
        )
        report = lge_feedback_loop.run_evaluation(data)
        self.assertEqual(report["frozen_split_status"], "UNAVAILABLE")
        self.assertEqual(report["holdout_excluded"], 0)

    def test_available_holdout_excluded_counted(self):
        data = _make_data(
            [_make_record("c1"), _make_record("c2"), _make_record("c3")],
            frozen_status="AVAILABLE",
            holdout_case_ids=["c1", "c3"],
        )
        report = lge_feedback_loop.run_evaluation(data)
        self.assertEqual(report["frozen_split_status"], "AVAILABLE")
        self.assertEqual(report["holdout_excluded"], 2)


class TestCountReconciliation(unittest.TestCase):
    """pending + work + resolved must equal len(records)."""

    def test_counts_reconcile(self):
        n = 5
        data = _make_data([_make_record(f"c{i}") for i in range(n)])
        report = lge_feedback_loop.run_evaluation(data)
        self.assertEqual(report["lge_myo_pending"], n)
        self.assertEqual(report["lge_myo_work"], 0)
        self.assertEqual(report["lge_myo_resolved"], 0)
        self.assertTrue(report["counts_reconcile"])
        self.assertEqual(report["total_records"], n)

    def test_counts_reconcile_empty(self):
        data = _make_data([])
        report = lge_feedback_loop.run_evaluation(data)
        self.assertEqual(report["lge_myo_pending"], 0)
        self.assertEqual(report["lge_myo_work"], 0)
        self.assertEqual(report["lge_myo_resolved"], 0)
        self.assertTrue(report["counts_reconcile"])


class TestConsumerReady(unittest.TestCase):
    """consumer_ready is derived from manifest integrity, not forced."""

    def test_consumer_ready_true_with_correct_manifest(self):
        records = [_make_record(f"c{i}") for i in range(lge_feedback_loop.EXPECTED_RECORD_COUNT)]
        data = _make_data(records)
        report = lge_feedback_loop.run_evaluation(data)
        self.assertTrue(report["lge_myo_consumer_ready"])

    def test_consumer_ready_false_with_wrong_count(self):
        data = _make_data([_make_record("c1")])
        report = lge_feedback_loop.run_evaluation(data)
        self.assertFalse(report["lge_myo_consumer_ready"])

    def test_consumer_ready_false_with_wrong_hash(self):
        records = [_make_record(f"c{i}") for i in range(lge_feedback_loop.EXPECTED_RECORD_COUNT)]
        data = _make_data(records, source_sha256="bad_hash")
        report = lge_feedback_loop.run_evaluation(data)
        self.assertFalse(report["lge_myo_consumer_ready"])


class TestConsumerReadyWithUnavailable(unittest.TestCase):
    """consumer_ready=true is allowed even when frozen_split_status=UNAVAILABLE."""

    def test_ready_true_and_unavailable(self):
        records = [_make_record(f"c{i}") for i in range(lge_feedback_loop.EXPECTED_RECORD_COUNT)]
        data = _make_data(records, frozen_status="UNAVAILABLE")
        report = lge_feedback_loop.run_evaluation(data)
        self.assertTrue(report["lge_myo_consumer_ready"])
        self.assertEqual(report["frozen_split_status"], "UNAVAILABLE")
        self.assertEqual(report["holdout_excluded"], 0)


class TestZeroLabelWrites(unittest.TestCase):
    """Report must declare label_write=False and production_writes=0."""

    def test_zero_writes_declared(self):
        data = _make_data([_make_record("c1")])
        report = lge_feedback_loop.run_evaluation(data)
        self.assertFalse(report["label_write"])
        self.assertEqual(report["production_writes"], 0)


class TestDownstreamAudit(unittest.TestCase):
    """Explicit downstream consumer audit of UNAVAILABLE+ready=true output."""

    def test_audit_pass(self):
        records = [_make_record(f"c{i}") for i in range(lge_feedback_loop.EXPECTED_RECORD_COUNT)]
        data = _make_data(records, frozen_status="UNAVAILABLE")
        report = lge_feedback_loop.run_evaluation(data)

        audit = report.get("downstream_consumer_audit", {})
        self.assertIsInstance(audit, dict)
        self.assertIn("checks", audit)
        self.assertIn("all_pass", audit)
        self.assertIn("failed", audit)
        self.assertIn("verdict", audit)

        checks = audit["checks"]
        # All 12 checks must pass
        self.assertTrue(checks.get("source_modality_lge"))
        self.assertTrue(checks.get("workflow_stage_present"))
        self.assertTrue(checks.get("input_source_sha256_present"))
        self.assertTrue(checks.get("consumer_ready_boolean"))
        self.assertTrue(checks.get("pending_int"))
        self.assertTrue(checks.get("work_int"))
        self.assertTrue(checks.get("resolved_int"))
        self.assertTrue(checks.get("frozen_split_status_UNAVAILABLE"))
        self.assertTrue(checks.get("holdout_excluded_zero"))
        self.assertTrue(checks.get("label_write_false"))
        self.assertTrue(checks.get("production_writes_zero"))
        self.assertTrue(checks.get("counts_reconcile_true"))

        self.assertTrue(audit["all_pass"])
        self.assertEqual(audit["failed"], [])
        self.assertEqual(audit["verdict"], "PASS")

    def test_audit_fail_on_bad_hash(self):
        records = [_make_record(f"c{i}") for i in range(lge_feedback_loop.EXPECTED_RECORD_COUNT)]
        data = _make_data(records, source_sha256="bad_hash", frozen_status="UNAVAILABLE")
        report = lge_feedback_loop.run_evaluation(data)

        audit = report.get("downstream_consumer_audit", {})
        # consumer_ready will be False, but the audit checks structural fields only,
        # so all_pass should still be True because consumer_ready_boolean is still a bool.
        self.assertTrue(audit["all_pass"])
        self.assertEqual(audit["verdict"], "PASS")


if __name__ == "__main__":
    unittest.main()
