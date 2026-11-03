"""Offline unit tests for the A1 output-side hard gate.

All fixtures are inline dictionaries; no datasets, caches or network access.
"""

import unittest

from experiments.a1_output_dedup_rollup import (
    dedup_rows,
    guard_extractor_creation,
    is_valid_cache_summary,
    rollup_by_case,
    run_a1_output_gate,
    severity_tier,
)


class DedupTests(unittest.TestCase):
    def test_duplicate_rows_rejected_and_not_delivered(self):
        row = {
            "case_id": "C1",
            "modality": "LGE",
            "issue_type": "mi_fragmentation",
            "slices": [7, 9],
            "frames": [12],
            "severity": "high",
        }
        reordered = dict(row, slices=[9, 7])
        result = dedup_rows([dict(row), reordered])

        self.assertEqual(len(result["records"]), 1)
        self.assertEqual(result["input_rows"], 2)
        self.assertEqual(result["accepted_rows"], 1)
        self.assertEqual(result["duplicate_rows"], 1)
        self.assertEqual(result["rejected"][0]["reason"], "duplicate_row")

        record = result["records"][0]
        self.assertEqual(record["source_rows"], 1)
        self.assertEqual(record["issue_counts"]["mi_fragmentation"], 1)
        self.assertEqual(record["issue_slices"]["mi_fragmentation"], [7, 9])

    def test_merge_issue_slices_frames_and_counts(self):
        rows = [
            {
                "case_id": "C1",
                "modality": "LGE",
                "issue_type": "mi_fragmentation",
                "slices": [9, 7],
                "frames": [12],
                "severity": "low",
            },
            {
                "case_id": "C1",
                "modality": "LGE",
                "issue_type": "mi_fragmentation",
                "slices": [5, 7],
                "frames": [13, 12],
                "severity": "high",
            },
            {
                "case_id": "C1",
                "modality": "LGE",
                "issue_type": "myo_border",
                "slices": [3],
                "frames": [],
                "severity": "moderate",
            },
        ]
        result = dedup_rows(rows)

        self.assertEqual(len(result["records"]), 1)
        self.assertEqual(result["duplicate_rows"], 0)
        record = result["records"][0]
        self.assertEqual(record["issue_types"], ["mi_fragmentation", "myo_border"])
        self.assertEqual(record["issue_counts"], {"mi_fragmentation": 2, "myo_border": 1})
        self.assertEqual(record["source_rows"], 3)
        self.assertEqual(record["issue_slices"]["mi_fragmentation"], [5, 7, 9])
        self.assertEqual(record["issue_frames"]["mi_fragmentation"], [12, 13])
        self.assertEqual(record["issue_severities"]["mi_fragmentation"], "high")
        self.assertEqual(record["index_map"]["myo_border"], {"slices": [3], "frames": []})
        self.assertEqual(record["all_slices"], [3, 5, 7, 9])

    def test_one_record_per_case_modality(self):
        rows = [
            {"case_id": "C1", "modality": "LGE", "issue_type": "a", "slices": [1]},
            {"case_id": "C1", "modality": "SAX", "issue_type": "b", "slices": [2]},
        ]
        result = dedup_rows(rows)
        keys = {(record["case_id"], record["modality"]) for record in result["records"]}
        self.assertEqual(keys, {("C1", "LGE"), ("C1", "SAX")})
        self.assertEqual(len(keys), len(result["records"]))

    def test_invalid_rows_reported_not_delivered(self):
        rows = [
            {"case_id": "C1", "modality": "", "issue_type": "a", "slices": [1]},
            {"case_id": "C1", "modality": "LGE", "issue_type": "a", "slices": [1]},
        ]
        result = dedup_rows(rows)
        self.assertEqual(len(result["records"]), 1)
        self.assertEqual(len(result["invalid_rows"]), 1)
        self.assertIn("modality", result["invalid_rows"][0])


class RollupTests(unittest.TestCase):
    def _rows(self):
        return [
            {"case_id": "C1", "modality": "LGE", "issue_type": "mi_fragmentation", "slices": [7, 9]},
            {"case_id": "C1", "modality": "LGE", "issue_type": "mi_fragmentation", "slices": [5]},
            {"case_id": "C1", "modality": "SAX", "issue_type": "myo_border", "slices": [3]},
        ]

    def test_case_counts_ratios_worst_slice_and_tier(self):
        dedup = dedup_rows(self._rows())
        cases = rollup_by_case(dedup["records"])
        case = cases["C1"]

        self.assertEqual(case["record_count"], 2)
        self.assertEqual(case["modalities"], ["LGE", "SAX"])
        self.assertEqual(case["source_rows"], 3)
        self.assertEqual(case["issue_counts"], {"mi_fragmentation": 2, "myo_border": 1})
        self.assertEqual(case["issue_worst_slices"], {"mi_fragmentation": 9, "myo_border": 3})
        self.assertEqual(case["worst_slice"], 9)
        self.assertEqual(case["worst_slice_issue"], "mi_fragmentation")
        self.assertEqual(case["severity_tier"], "high")
        self.assertAlmostEqual(case["issue_ratios"]["mi_fragmentation"], 2.0 / 3.0, places=5)

        # reconciliation: case rollup counts equal the raw flagged slices fed in
        self.assertEqual(sum(case["issue_counts"].values()), len(self._rows()))
        self.assertEqual(
            sum(sum(c["issue_counts"].values()) for c in cases.values()),
            sum(record["source_rows"] for record in dedup["records"]),
        )

    def test_high_frequency_rollup(self):
        dedup = dedup_rows(self._rows())
        cases = rollup_by_case(dedup["records"])
        case = cases["C1"]
        flagged = [entry["issue_type"] for entry in case["high_frequency_issues"]]
        self.assertEqual(flagged, ["mi_fragmentation"])
        self.assertEqual(case["high_frequency_issue_count"], 1)
        entry = case["high_frequency_issues"][0]
        self.assertEqual(entry["count"], 2)
        self.assertEqual(entry["worst_slice"], 9)
        self.assertEqual(entry["severity_tier"], "high")

    def test_severity_tier_boundaries(self):
        self.assertEqual(severity_tier(0, 0.0), "none")
        self.assertEqual(severity_tier(1, 0.1), "low")
        self.assertEqual(severity_tier(1, 0.3), "moderate")
        self.assertEqual(severity_tier(2, 0.6), "high")
        self.assertEqual(severity_tier(6, 0.8), "critical")


class CacheGuardTests(unittest.TestCase):
    def _cache(self):
        return {"hash": "sha256:abc", "entries": {"C1": {"LGE": 3}}}

    def _proposals(self):
        return [
            {"name": "extractor_a", "source_hash": "sha256:abc"},
            {"name": "extractor_b", "source_hash": "sha256:def"},
            {"name": "extractor_c"},
        ]

    def test_is_valid_cache_summary(self):
        self.assertTrue(is_valid_cache_summary(self._cache()))
        self.assertFalse(is_valid_cache_summary(None))
        self.assertFalse(is_valid_cache_summary({"entries": [1]}))
        self.assertFalse(is_valid_cache_summary({"hash": "sha256:abc", "entries": []}))
        self.assertTrue(is_valid_cache_summary({"hash": "sha256:abc", "count": 2}))

    def test_valid_cache_blocks_every_new_extractor(self):
        allowed, report = guard_extractor_creation(self._cache(), self._proposals())
        self.assertEqual(allowed, [])
        self.assertTrue(report["cache_valid"])
        self.assertEqual(report["created"], 0)
        self.assertEqual(report["hash_mismatch_count"], 1)
        self.assertEqual(report["missing_hash_count"], 1)
        self.assertTrue(report["alerts"])

        reasons = {item["extractor"]: item["reason"] for item in report["blocked"]}
        self.assertEqual(reasons["extractor_a"], "cached_extractor_reused")
        self.assertEqual(reasons["extractor_b"], "hash_mismatch")
        self.assertEqual(reasons["extractor_c"], "missing_source_hash")
        self.assertEqual(len(report["hash_mismatches"]), 2)
        self.assertEqual(report["cache_hash"], "sha256:abc")

    def test_missing_cache_allows_extractors(self):
        allowed, report = guard_extractor_creation(None, self._proposals())
        self.assertEqual(len(allowed), 3)
        self.assertFalse(report["cache_valid"])
        self.assertEqual(report["created"], 3)
        self.assertEqual(report["hash_mismatch_count"], 0)


class GateTests(unittest.TestCase):
    def test_gate_is_reconciled_and_blocks_extractors_with_valid_cache(self):
        rows = [
            {"case_id": "C1", "modality": "LGE", "issue_type": "mi_fragmentation", "slices": [7], "frames": [1]},
            {"case_id": "C1", "modality": "SAX", "issue_type": "myo_border", "slices": [3], "frames": [2]},
        ]
        cache = {"hash": "sha256:abc", "entries": {"C1": 2}}
        result = run_a1_output_gate(
            rows,
            cache_summary=cache,
            proposed_extractors=[{"name": "e1", "source_hash": "sha256:zzz"}],
        )
        report = result["report"]

        self.assertTrue(result["ok"])
        self.assertEqual(report["delivered_records"], 2)
        self.assertEqual(report["case_count"], 1)
        self.assertEqual(report["rejected_duplicate_rows"], 0)
        self.assertEqual(report["duplicate_rows_delivered"], 0)
        self.assertTrue(report["unique_row_compliance"])
        self.assertTrue(report["slice_addressability_ok"])
        self.assertTrue(report["reconciled"])
        self.assertEqual(report["extractors_created"], 0)
        self.assertEqual(report["extractor_report"]["hash_mismatch_count"], 1)
        self.assertEqual(result["extractors"], [])

        # every delivered (case_id, modality) is unique
        keys = [(record["case_id"], record["modality"]) for record in result["records"]]
        self.assertEqual(len(keys), len(set(keys)))

    def test_gate_rejects_duplicates_and_invalid_rows(self):
        row = {"case_id": "C2", "modality": "LGE", "issue_type": "mi_fragmentation", "slices": [4], "frames": [5]}
        rows = [
            dict(row),
            dict(row),
            {"case_id": "C2", "issue_type": "x"},
            {"case_id": "C2", "modality": "SAX", "issue_type": "myo_border", "slices": [6], "frames": []},
        ]
        result = run_a1_output_gate(
            rows,
            cache_summary=None,
            proposed_extractors=[{"name": "e1", "source_hash": "sha256:abc"}],
        )
        report = result["report"]

        self.assertTrue(result["ok"])
        self.assertEqual(report["rejected_duplicate_rows"], 1)
        self.assertEqual(report["duplicate_rows_delivered"], 0)
        self.assertEqual(report["invalid_rows"], 1)
        self.assertEqual(report["delivered_records"], 2)
        self.assertEqual(report["extractors_created"], 1)
        self.assertEqual(len(result["extractors"]), 1)

        slices = [
            slice_index
            for record in result["records"]
            for slice_index in record["all_slices"]
        ]
        self.assertIn(4, slices)
        self.assertIn(6, slices)


if __name__ == "__main__":
    unittest.main()
