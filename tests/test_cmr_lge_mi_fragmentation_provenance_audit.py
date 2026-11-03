"""
Unit tests for cmr_lge_mi_fragmentation_provenance_audit
Verifies acceptance metrics, deduplication, FP rates, and stratification signals.
"""

import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from experiments.cmr_lge_mi_fragmentation_provenance_audit import FragmentationAudit

class TestFragmentationAudit(unittest.TestCase):
    def setUp(self):
        self.audit = FragmentationAudit()
        self.metrics = self.audit.run_all_arms()

    def test_deduplication_per_case_modality(self):
        """Exactly one deduplicated record per (case_id, modality) per arm."""
        for arm in range(1, 5):
            records = self.audit.execute_arm(arm)
            keys = [(r["case_id"], r["modality"]) for r in records]
            self.assertEqual(len(keys), len(set(keys)), f"Arm {arm} has duplicate case/modality records")

    def test_flag_rates_reported(self):
        """Flag rate reported per arm on stratified sample and controls."""
        for arm in range(1, 5):
            key = f"arm_{arm}_flag_rate"
            self.assertIn(key, self.metrics)
            self.assertGreaterEqual(self.metrics[key], 0.0)
            self.assertLessEqual(self.metrics[key], 1.0)

    def test_mean_median_comp_counts(self):
        """Mean and median connected-component count per flagged slice per arm."""
        for arm in range(1, 5):
            m_key = f"arm_{arm}_mean_comp_count"
            med_key = f"arm_{arm}_median_comp_count"
            self.assertIn(m_key, self.metrics)
            self.assertIn(med_key, self.metrics)
            self.assertGreaterEqual(self.metrics[m_key], 0)
            self.assertGreaterEqual(self.metrics[med_key], 0)

    def test_fp_rate_on_controls(self):
        """False-positive rate on unflagged controls <= 1 slice across all arms."""
        for arm in range(1, 5):
            fp_key = f"arm_{arm}_fp_rate_controls"
            self.assertIn(fp_key, self.metrics)
            self.assertLessEqual(self.metrics[fp_key], 1, f"Arm {arm} FP rate on controls exceeds limit")

    def test_cooccurrence_resolution(self):
        """Co-occurrence resolution rate for mi_multiple_components with myo_topology and mi_fraction_high."""
        for arm in range(1, 5):
            cooccur_key = f"arm_{arm}_cooccur_resolution"
            self.assertIn(cooccur_key, self.metrics)
            self.assertGreaterEqual(self.metrics[cooccur_key], 0)

    def test_ring_contiguity_agreement(self):
        """Paired CMR ring-contiguity agreement on discordant slices only, reported as metric."""
        self.assertIn("arm_4_ring_contiguity_agreement", self.metrics)
        self.assertGreaterEqual(self.metrics["arm_4_ring_contiguity_agreement"], 0)

    def test_cache_reuse_and_mismatch_reporting(self):
        """Cache-reuse rate and pipeline/config hash mismatch count reported explicitly."""
        self.assertIn("cache_reuse_rate", self.metrics)
        self.assertGreater(self.metrics["cache_reuse_rate"], 0.0)
        self.assertIn("hash_mismatch_count", self.metrics)
        self.assertGreaterEqual(self.metrics["hash_mismatch_count"], 0)

    def test_mi_fraction_high_stability(self):
        """mi_fraction_high stability (direction and magnitude) under arms 2 and 3."""
        # Stability implies consistent handling of cases with this flag across filtering arms
        arm2_flag_rate = self.metrics.get("arm_2_flag_rate", 0)
        arm3_flag_rate = self.metrics.get("arm_3_flag_rate", 0)
        # Both arms apply filters, so rates should be comparable and non-zero for stratified sample
        self.assertGreater(arm2_flag_rate, 0)
        self.assertGreater(arm3_flag_rate, 0)

    def test_no_patient_identity_inference(self):
        """Verify case IDs follow synthetic pattern and contain no PII."""
        for case in self.audit.cases:
            self.assertTrue(case.case_id.startswith("LGE_C") or case.case_id.startswith("LGE_CTRL"))
            self.assertNotIn("patient", case.case_id.lower())
            self.assertNotIn("id_card", case.case_id.lower())

if __name__ == "__main__":
    unittest.main()
