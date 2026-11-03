import unittest
import sys
import os

# Ensure experiments package is importable when running tests directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from experiments.e1_evidence_dedup_rollup import (
    deduplicate_evidence,
    roll_up_high_volume_flags,
    CacheReuseGuard,
    HIGH_VOLUME_MODALITIES,
    VALID_DIGEST_RUNS,
    GUARDED_RUN_CLASSES,
)


class TestEvidenceDedupRollup(unittest.TestCase):
    def setUp(self):
        self.sample_records = [
            {"case_id": "C001", "modality": "CMR", "issue_type": "artifact", "slice_index": 5},
            {"case_id": "C001", "modality": "CMR", "issue_type": "artifact", "slice_index": 10},
            {"case_id": "C001", "modality": "CMR", "issue_type": "motion", "slice_index": 12},
            {"case_id": "C002", "modality": "LGE", "issue_type": "enhancement", "slice_index": 3},
            {"case_id": "C002", "modality": "T1w", "issue_type": "noise", "slice_index": 1},
        ]

    def test_deduplicate_one_record_per_case_modality(self):
        rolled = deduplicate_evidence(self.sample_records)
        self.assertEqual(len(rolled), 3)  # C001-CMR, C002-LGE, C002-T1w

        c001_cmr = next(
            r for r in rolled if r["case_id"] == "C001" and r["modality"] == "CMR"
        )
        self.assertEqual(set(c001_cmr["issue_types"]), {"artifact", "motion"})
        self.assertEqual(
            sorted(c001_cmr["slice_indices_by_issue"]["artifact"]), [5, 10]
        )
        self.assertEqual(c001_cmr["slice_indices_by_issue"]["motion"], [12])

    def test_roll_up_high_volume_flags(self):
        rolled = deduplicate_evidence(self.sample_records)
        rolled_with_flags = roll_up_high_volume_flags(rolled)

        c001_cmr = next(
            r for r in rolled_with_flags if r["case_id"] == "C001" and r["modality"] == "CMR"
        )
        self.assertEqual(c001_cmr["flag_count"], 3)
        self.assertEqual(c001_cmr["worst_slice_index"], 12)
        self.assertEqual(c001_cmr["severity_tier"], "low")  # 3 < 7
        self.assertIn(5, c001_cmr["addressable_slices"])
        self.assertIn(10, c001_cmr["addressable_slices"])
        self.assertIn(12, c001_cmr["addressable_slices"])

        # T1w should not be modified
        t1w_rec = next(r for r in rolled_with_flags if r["modality"] == "T1w")
        self.assertNotIn("flag_count", t1w_rec)

    def test_cache_reuse_guard_blocks_on_valid_digest(self):
        guard = CacheReuseGuard()
        guard.register_valid_digest("run-5", "abc123")

        # Should block
        self.assertTrue(guard.check_and_block_extractor("run-5", "abc123"))
        # Should allow different digest
        self.assertFalse(guard.check_and_block_extractor("run-5", "xyz789"))
        # Should allow non-guarded run
        self.assertFalse(guard.check_and_block_extractor("run-1", "abc123"))

    def test_proposal_flagging_for_guarded_runs(self):
        guard = CacheReuseGuard()
        res = guard.evaluate_proposal("run-4")
        self.assertTrue(res["flagged"])
        self.assertIn("run-4", guard._flagged_proposals)

        res_normal = guard.evaluate_proposal("run-10")
        self.assertFalse(res_normal["flagged"])

    def test_zero_recall_preservation(self):
        # Verify that merging does not drop slice indices
        rolled = deduplicate_evidence(self.sample_records)
        c001_cmr = next(
            r for r in rolled if r["case_id"] == "C001" and r["modality"] == "CMR"
        )
        all_slices = []
        for slices in c001_cmr["slice_indices_by_issue"].values():
            all_slices.extend(slices)
        self.assertEqual(sorted(all_slices), [5, 10, 12])


if __name__ == "__main__":
    unittest.main()
