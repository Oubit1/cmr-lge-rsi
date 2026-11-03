import unittest
from experiments.evidence_rollup_guard import rollup_evidence, validate_extractor_guard, process_qc_payload


class TestEvidenceRollupGuard(unittest.TestCase):
    def setUp(self):
        self.raw_records = [
            {"case_id": "C001", "modality": "CMR", "issue_type": "LGE_MISMATCH", "slice_index": 5, "severity": "HIGH"},
            {"case_id": "C001", "modality": "CMR", "issue_type": "LGE_MISMATCH", "slice_index": 6, "severity": "HIGH"},
            {"case_id": "C001", "modality": "CMR", "issue_type": "BORDER_BLUR", "slice_index": 5, "severity": "MEDIUM"},
            {"case_id": "C002", "modality": "LGE", "issue_type": "FRAGMENTATION", "slice_index": 2, "severity": "LOW"},
        ]
        self.valid_digests = {"run-4", "run-5", "run-9", "run-13", "run-14", "run-15", "run-16"}

    def test_deduplication_and_one_record_per_case_modality(self):
        result = rollup_evidence(self.raw_records)
        self.assertEqual(len(result), 2)
        c001_cmr = next(r for r in result if r["case_id"] == "C001" and r["modality"] == "CMR")
        self.assertIn("LGE_MISMATCH", c001_cmr["merged_issue_types"])
        self.assertIn("BORDER_BLUR", c001_cmr["merged_issue_types"])
        self.assertEqual(c001_cmr["slice_indices_by_issue"]["LGE_MISMATCH"], [5, 6])
        self.assertEqual(c001_cmr["slice_indices_by_issue"]["BORDER_BLUR"], [5])
        self.assertEqual(c001_cmr["rollup_metrics"]["count"], 3)
        self.assertEqual(c001_cmr["rollup_metrics"]["worst_slice_index"], 6)
        self.assertEqual(c001_cmr["rollup_metrics"]["severity_tier"], "HIGH")

    def test_guard_blocks_valid_digest(self):
        guard = validate_extractor_guard("run-9", self.valid_digests, "run-10")
        self.assertEqual(guard["guard_status"], "BLOCKED")
        self.assertIn("Cache-reuse guard triggered", guard["mismatch_report"])
        self.assertFalse(guard["reproposal_flagged"])

    def test_guard_flags_reproposal(self):
        guard = validate_extractor_guard("new-digest", self.valid_digests, "run-5")
        self.assertEqual(guard["guard_status"], "ALLOWED")
        self.assertTrue(guard["reproposal_flagged"])
        self.assertIn("Re-proposal guard triggered", guard["mismatch_report"])

    def test_guard_both_triggers(self):
        guard = validate_extractor_guard("run-4", self.valid_digests, "run-6")
        self.assertEqual(guard["guard_status"], "BLOCKED")
        self.assertTrue(guard["reproposal_flagged"])
        self.assertIn("Additionally, re-proposal", guard["mismatch_report"])

    def test_process_qc_payload_integration(self):
        payload = process_qc_payload(self.raw_records, self.valid_digests, "run-13", "run-8")
        self.assertEqual(len(payload["rolled_up_records"]), 2)
        self.assertEqual(payload["guard_check"]["guard_status"], "BLOCKED")
        self.assertFalse(payload["guard_check"]["reproposal_flagged"])

    def test_zero_side_effects(self):
        before = len(self.raw_records)
        rollup_evidence(self.raw_records)
        after = len(self.raw_records)
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
