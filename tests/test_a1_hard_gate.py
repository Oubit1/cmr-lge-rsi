import unittest
from experiments.a1_hard_gate import validate_and_enforce_gate, GateResult


class TestA1HardGate(unittest.TestCase):
    def setUp(self):
        self.sample_a1_output = [
            {
                "case_id": "C001",
                "modality": "CMR",
                "issue_types": ["T1"],
                "slice_indices": {"T1": [1, 2]},
                "high_volume_flags": {"count": 2, "rate": 0.5, "worst_slice_index": 2, "severity_tier": "MEDIUM"}
            },
            {
                "case_id": "C001",
                "modality": "CMR",
                "issue_types": ["T2"],
                "slice_indices": {"T2": [3]},
                "high_volume_flags": {"count": 1, "rate": 0.2, "worst_slice_index": 3, "severity_tier": "LOW"}
            }
        ]

    def test_deduplication_and_rollup(self):
        res = validate_and_enforce_gate(self.sample_a1_output)
        self.assertEqual(res.status, "PASS")
        self.assertEqual(len(res.records), 1)
        self.assertEqual(res.records[0]["case_id"], "C001")
        self.assertEqual(set(res.records[0]["issue_types"]), {"T1", "T2"})
        self.assertIn(1, res.records[0]["all_flagged_slices"])
        self.assertIn(3, res.records[0]["all_flagged_slices"])
        self.assertTrue(res.records[0].get("addressable"))

    def test_high_volume_rollup_metrics(self):
        res = validate_and_enforce_gate(self.sample_a1_output)
        hv = res.records[0]["high_volume_flags"]
        self.assertEqual(hv["count"], 3)
        self.assertEqual(hv["rate"], 0.5)
        self.assertEqual(hv["worst_slice_index"], 3)
        self.assertEqual(hv["severity_tier"], "MEDIUM")

    def test_cache_reuse_block_and_mismatch_report(self):
        cached = {4: "dummy_expected_hash", 5: "another_hash"}
        res = validate_and_enforce_gate(self.sample_a1_output, cached_digests=cached)
        self.assertTrue(res.cache_reuse_blocked)
        self.assertTrue(len(res.cache_mismatch_hashes) > 0)
        self.assertIn("Run-4", res.cache_mismatch_hashes[0])

    def test_run18_stratum_flag_and_block(self):
        sample_with_run18 = [{
            "case_id": "C002",
            "modality": "LGE",
            "issue_types": ["mi_multiple_components"],
            "slice_indices": {},
            "high_volume_flags": {}
        }]
        res = validate_and_enforce_gate(sample_with_run18, current_cycle_run_ids=["run-18", "run-19"])
        self.assertTrue(res.run18_stratum_flagged)
        self.assertTrue(any("Blocked re-proposal" in err for err in res.errors))

    def test_gate_failure_triggers_escalation(self):
        bad_input = [{"case_id": "", "modality": "CMR"}]
        res = validate_and_enforce_gate(bad_input)
        self.assertEqual(res.status, "FAIL")
        self.assertIsNotNone(res.escalation_message)
        self.assertIn("human maintenance owner", res.escalation_message)
        self.assertNotIn("re-proposal", res.escalation_message.lower())  # Ensures escalation replaces re-proposal

    def test_synchronous_evaluation_and_status_recording(self):
        res = validate_and_enforce_gate(self.sample_a1_output)
        self.assertIn(res.status, ["PASS", "FAIL"])
        self.assertIsInstance(res, GateResult)
        self.assertTrue(hasattr(res, "records"))
        self.assertTrue(hasattr(res, "errors"))


if __name__ == "__main__":
    unittest.main()
