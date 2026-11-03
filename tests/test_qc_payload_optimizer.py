import sys
import os
import json
import unittest
from typing import Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from experiments.qc_payload_optimizer import QCPayloadOptimizer


class TestQCPayloadOptimizer(unittest.TestCase):
    def setUp(self) -> None:
        self.opt = QCPayloadOptimizer()
        self.issue_types = [f"ISSUE_{i}" for i in range(13)]
        self.raw_records: List[Dict] = []

        # Generate 83 raw records across 20 unique (case_id, modality) pairs.
        # QC logs for the same case/modality/issue are typically identical or near-identical.
        # We generate 4 identical records per case to simulate repeated pipeline passes,
        # ensuring >70% serialization reduction after deduplication.
        for i in range(20):
            cid = f"CASE_{i:02d}"
            mod = "CMR" if i % 2 == 0 else "LGE"
            iss = self.issue_types[i % 13]
            base_slices = {iss: [i * 10 + k for k in range(3)]}

            for j in range(4):
                self.raw_records.append(
                    {
                        "case_id": cid,
                        "modality": mod,
                        "issue_types": [iss],
                        "slice_indices": base_slices,
                    }
                )

        # Pad to exactly 83 records with additional duplicates of existing cases
        while len(self.raw_records) < 83:
            idx = len(self.raw_records) % 20
            cid = f"CASE_{idx:02d}"
            mod = "CMR" if idx % 2 == 0 else "LGE"
            iss = self.issue_types[idx % 13]
            base_slices = {iss: [idx * 10 + k for k in range(3)]}
            self.raw_records.append(
                {
                    "case_id": cid,
                    "modality": mod,
                    "issue_types": [iss],
                    "slice_indices": base_slices,
                }
            )

    def test_dedup_coverage_and_size_reduction(self) -> None:
        optimized = self.opt.deduplicate_evidence(self.raw_records)
        self.assertEqual(len(optimized), 20, "Expected 20 unique (case_id, modality) pairs")

        raw_json = json.dumps(self.raw_records, separators=(",", ":"))
        opt_json = json.dumps(optimized, separators=(",", ":"))
        reduction = 1 - len(opt_json) / len(raw_json)
        self.assertGreaterEqual(
            reduction, 0.70, f"Serialization reduction {reduction:.2%} < 70%"
        )

    def test_flagged_slices_preserved(self) -> None:
        optimized = self.opt.deduplicate_evidence(self.raw_records)
        all_original = set()
        for r in self.raw_records:
            for _, idxs in r.get("slice_indices", {}).items():
                all_original.update(idxs)

        found = set()
        for o in optimized:
            for _, idxs in o.get("slice_indices", {}).items():
                found.update(idxs)
        self.assertEqual(all_original, found, "All original slice indices must remain addressable")

    def test_roll_up_logic(self) -> None:
        meta = {
            "flagged_slices": [1, 2, 3, 5, 8],
            "total_slices": 10,
            "worst_severity": "high",
        }
        rolled = self.opt.roll_up_cmr_flags(meta)
        self.assertEqual(rolled["flag_count"], 5)
        self.assertAlmostEqual(rolled["flag_rate"], 0.5)
        self.assertEqual(rolled["worst_slice_index"], 8)
        self.assertEqual(rolled["severity_tier"], "high")
        self.assertEqual(rolled["addressable_slices"], [1, 2, 3, 5, 8])

    def test_digest_cache_reference(self) -> None:
        sd = {
            "mi_components": [
                {"volume": 10, "distance": 2},
                {"volume": 20, "distance": 1},
            ],
            "total_volume": 100,
            "myo_topology": True,
            "threshold": 0.8,
        }
        d1 = self.opt.get_qc_digest("C1", sd)
        d2 = self.opt.get_qc_digest("C1", sd)
        self.assertIs(d1, d2, "Digest must be cached by reference")
        self.assertEqual(d1["mi_component_count"], 2)
        self.assertEqual(d1["mi_fraction"], 0.3)
        self.assertEqual(d1["min_inter_component_distance"], 1)
        self.assertTrue(d1["myo_topology_flag"])
        self.assertEqual(d1["threshold_value"], 0.8)

    def test_specialist_pass_gate(self) -> None:
        pairs_pass = [("CMR_A", "LGE_A"), ("CMR_B", "LGE_B"), ("CMR_C", "LGE_C")]
        res_pass = self.opt.gate_specialist_pass(pairs_pass)
        self.assertEqual(res_pass["status"], "pass")
        self.assertIn("finding", res_pass)

        pairs_fail = [("CMR_X", "LGE_X")]
        res_fail = self.opt.gate_specialist_pass(pairs_fail)
        self.assertEqual(res_fail["status"], "not_applicable")
        self.assertIn("evidence_gap", res_fail["contraindications"])
        self.assertIn("note", res_fail)


if __name__ == "__main__":
    unittest.main()
