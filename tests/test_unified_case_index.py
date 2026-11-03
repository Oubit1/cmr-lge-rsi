import unittest
from experiments.unified_case_index import UnifiedCaseIndex


class TestUnifiedCaseIndex(unittest.TestCase):
    def setUp(self):
        self.index = UnifiedCaseIndex()
        self.data = [
            {"case_id": "C1", "slice_index": 1, "stratum": "S1", "flags": ["F1"]},
            {"case_id": "C1", "slice_index": 1, "stratum": "S2", "flags": ["F2"]},
            {"case_id": "C1", "slice_index": 2, "stratum": "S1", "flags": ["F1"]},
            {"case_id": "C2", "slice_index": 1, "stratum": "S1", "flags": ["F1"]},
        ]
        self.index.build_index(self.data)

    def test_builds_index_correctly(self):
        self.assertIn("C1", self.index.index)
        self.assertIn(1, self.index.index["C1"]["slices"])
        self.assertEqual(len(self.index.index["C1"]["slices"][1]["strata"]), 2)

    def test_prevents_duplicate_sampling(self):
        plan = self.index.get_sampling_plan("C1")
        # Slice 1 should appear in only one stratum's plan
        slice_1_count = 0
        for p in plan["plan"]:
            slice_1_count += 1 if 1 in p["slices"] else 0
        self.assertEqual(slice_1_count, 1)

    def test_skips_resampled_slices(self):
        plan = self.index.get_sampling_plan("C1")
        skipped_slices = [s["slice"] for s in plan["skipped"]]
        self.assertIn(1, skipped_slices)

    def test_handles_missing_case(self):
        plan = self.index.get_sampling_plan("NONEXISTENT")
        self.assertEqual(plan, {"plan": [], "skipped": []})


if __name__ == "__main__":
    unittest.main()
