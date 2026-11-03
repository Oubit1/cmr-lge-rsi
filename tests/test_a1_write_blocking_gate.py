"""Offline tests for the A1 write-side blocking hard gate.

No dataset, image, network call or runtime database is required: every fixture
below is inline evidence shaped like the run-31 A1 output bundle.
"""

import json
import unittest

from experiments.a1_write_blocking_gate import (
    A1GateInputError,
    A1WriteGate,
    ExtractorCacheGuard,
    build_canonical_row,
    index_addressability,
    record_key,
    replay_evidence,
    severity_tier,
)

RUN31_RECORDS = [
    {
        "case_id": "run31-c001",
        "modality": "LGE",
        "total_slices": 12,
        "issues": [
            {"issue_type": "mi_transmural", "slices": [3, 4], "frames": [1], "severity": "critical"},
        ],
    },
    {
        "case_id": "run31-c001",
        "modality": "lge",
        "total_slices": 12,
        "issues": [
            {"issue_type": "mi_fragmentation", "slices": [7], "frames": [2, 3], "severity": "review"},
        ],
    },
    {
        "case_id": "run31-c001",
        "modality": "LGE",
        "total_slices": 12,
        "issues": [
            {
                "issue_type": "mi_transmural",
                "slice_indices": [9],
                "frame_indices": [],
                "severity": "warning",
                "occurrences": 2,
            },
        ],
    },
    {
        "case_id": "run31-c001",
        "modality": "T2",
        "total_slices": 8,
        "issues": ["myo_small_area"],
    },
    {
        "case_id": "run31-c002",
        "modality": "LGE",
        "total_slices": 10,
        "issues": [],
    },
    {
        "case_id": "run31-c002",
        "modality": "LGE",
        "total_slices": 10,
        "issues": [
            {
                "type": "lge_mi_high_fraction",
                "slice_indices": [5],
                "frame_indices": [0],
                "severity": "review",
            },
        ],
    },
]


class Run31ReplayTest(unittest.TestCase):
    """Acceptance metric 1 + 2: replay run-31 evidence through the gate."""

    def setUp(self):
        self.report = replay_evidence(RUN31_RECORDS)
        self.gate = self.report["gate"]

    def row(self, case_id, modality):
        for row in self.gate["rows"]:
            if row["case_id"] == case_id and row["modality"] == modality:
                return row
        self.fail("missing merged row for {}::{}".format(case_id, modality))

    def test_duplicate_row_ratio_is_zero(self):
        self.assertEqual(self.gate["write_attempts"], 6)
        self.assertEqual(self.gate["blocked_writes"], 3)
        self.assertEqual(self.gate["row_count"], 3)
        self.assertEqual(self.gate["duplicate_row_count"], 0)
        self.assertEqual(self.gate["duplicate_row_ratio"], 0.0)

    def test_case_modality_uniqueness_is_total(self):
        self.assertEqual(self.gate["uniqueness_ratio"], 1.0)
        self.assertEqual(self.gate["unique_key_count"], self.gate["row_count"])
        keys = [record_key(row["case_id"], row["modality"]) for row in self.gate["rows"]]
        self.assertEqual(sorted(keys), sorted(set(keys)))

    def test_modality_case_is_normalised_before_keying(self):
        self.assertEqual(record_key("run31-c001", "lge"), "run31-c001::LGE")
        self.assertEqual(self.row("run31-c001", "LGE")["record_count"], 3)

    def test_merge_unions_issue_types_and_keeps_indices_addressable(self):
        row = self.row("run31-c001", "LGE")
        self.assertEqual(sorted(row["issue_types"]), ["mi_fragmentation", "mi_transmural"])
        self.assertEqual(row["flagged_slices"], [3, 4, 7, 9])
        self.assertEqual(row["flagged_frames"], [1, 2, 3])
        self.assertTrue(self.gate["all_indices_addressable"])
        self.assertTrue(all(merge["indices_preserved"] for merge in self.gate["merges"]))

    def test_merged_row_carries_counts_ratios_worst_slice_and_tier(self):
        row = self.row("run31-c001", "LGE")
        self.assertEqual(row["issue_count"], 2)
        self.assertEqual(row["issue_occurrences"], 4)
        self.assertEqual(row["flagged_slice_count"], 4)
        self.assertEqual(row["flagged_frame_count"], 3)
        self.assertEqual(row["worst_slice"], 3)
        self.assertEqual(row["severity_tier"], "critical")
        self.assertEqual(row["total_slices"], 12)
        self.assertAlmostEqual(row["flagged_slice_ratio"], 4.0 / 12.0, places=6)
        self.assertAlmostEqual(row["issue_occurrence_ratio"], 4.0 / 12.0, places=6)

    def test_second_case_merge_is_also_unique(self):
        row = self.row("run31-c002", "LGE")
        self.assertEqual(row["issue_types"], ["lge_mi_high_fraction"])
        self.assertEqual(row["flagged_slices"], [5])
        self.assertEqual(row["flagged_frames"], [0])
        self.assertEqual(row["worst_slice"], 5)
        self.assertEqual(row["severity_tier"], "review")

    def test_clean_row_keeps_a_neutral_tier(self):
        row = self.row("run31-c001", "T2")
        self.assertEqual(row["issue_types"], ["myo_small_area"])
        self.assertEqual(row["severity_tier"], "info")
        self.assertIsNone(row["worst_slice"])

    def test_evidence_report_is_json_serialisable(self):
        json.dumps(self.report)


class WriteGateUnitTest(unittest.TestCase):
    def test_first_write_is_accepted_and_second_is_blocked(self):
        gate = A1WriteGate()
        first = gate.write({"case_id": "c", "modality": "LGE", "issues": ["a"]})
        second = gate.write({"case_id": "c", "modality": "LGE", "issues": ["b"]})
        self.assertTrue(first["accepted"])
        self.assertFalse(second["accepted"])
        self.assertTrue(second["blocked"])
        self.assertEqual(len(gate.rows()), 1)
        self.assertEqual(sorted(gate.rows()[0]["issue_types"]), ["a", "b"])

    def test_distinct_keys_are_not_blocked(self):
        gate = A1WriteGate()
        gate.write({"case_id": "c", "modality": "LGE", "issues": ["a"]})
        result = gate.write({"case_id": "c", "modality": "T1", "issues": ["a"]})
        self.assertTrue(result["accepted"])
        self.assertEqual(gate.snapshot()["blocked_writes"], 0)

    def test_record_without_key_is_rejected(self):
        gate = A1WriteGate()
        with self.assertRaises(A1GateInputError):
            gate.write({"case_id": "c", "issues": ["a"]})
        with self.assertRaises(A1GateInputError):
            gate.write({"modality": "LGE", "issues": ["a"]})

    def test_non_integer_index_is_rejected(self):
        with self.assertRaises(A1GateInputError):
            build_canonical_row(
                [{"case_id": "c", "modality": "LGE", "issues": [{"issue_type": "a", "slices": ["x"]}]}]
            )

    def test_addressability_detects_a_lost_index(self):
        source = {
            "case_id": "c",
            "modality": "LGE",
            "total_slices": 4,
            "issues": [{"issue_type": "a", "slices": [1, 2], "frames": [0]}],
        }
        row = build_canonical_row([source])
        self.assertTrue(index_addressability(row, [source])["complete"])
        trimmed = dict(row, flagged_slices=[1], flagged_frames=[0])
        check = index_addressability(trimmed, [source])
        self.assertFalse(check["complete"])
        self.assertEqual(check["missing_slices"], [2])

    def test_severity_tier_picks_the_worst_label(self):
        self.assertEqual(severity_tier(["info", "review"]), "review")
        self.assertEqual(severity_tier([]), "info")
        self.assertEqual(severity_tier(["nonsense", "critical"]), "critical")


class ExtractorCacheGuardTest(unittest.TestCase):
    """Acceptance metric 3: no extractor when a valid cache digest exists."""

    def test_valid_cache_digest_match_reuses_without_creation(self):
        guard = ExtractorCacheGuard({"run31-c001::LGE": {"digest": "abc", "valid": True}})
        plan = guard.plan("run31-c001::LGE", "abc")
        self.assertEqual(plan["action"], "reuse")
        self.assertEqual(plan["creations"], 0)
        self.assertFalse(plan["report_required"])
        snapshot = guard.snapshot()
        self.assertEqual(snapshot["extractor_creations"], 0)
        self.assertEqual(snapshot["reports"], [])

    def test_digest_mismatch_builds_at_most_one_extractor_and_reports(self):
        guard = ExtractorCacheGuard({"k": {"digest": "abc", "valid": True}})
        first = guard.plan("k", "def")
        second = guard.plan("k", "ghi")
        self.assertEqual(first["action"], "recreate")
        self.assertEqual(first["creations"], 1)
        self.assertTrue(first["report_required"])
        self.assertEqual(second["action"], "blocked_recreate")
        self.assertEqual(second["creations"], 0)
        self.assertTrue(second["report_required"])
        snapshot = guard.snapshot()
        self.assertEqual(snapshot["extractor_creations"], 1)
        self.assertEqual(snapshot["max_creations_per_key"], 1)
        self.assertTrue(snapshot["report_required"])
        self.assertEqual(len(snapshot["reports"]), 2)
        self.assertTrue(all(report["reason"] == "hash_mismatch" for report in snapshot["reports"]))

    def test_missing_cache_builds_once_then_reuses(self):
        guard = ExtractorCacheGuard()
        first = guard.plan("k", "d1")
        second = guard.plan("k", "d1")
        self.assertEqual(first["action"], "create")
        self.assertEqual(first["creations"], 1)
        self.assertEqual(second["action"], "reuse")
        self.assertEqual(second["creations"], 0)
        self.assertEqual(guard.snapshot()["extractor_creations"], 1)
        self.assertEqual(guard.snapshot()["reports"], [])

    def test_invalid_cache_entry_matching_digest_still_builds_once(self):
        guard = ExtractorCacheGuard({"k": {"digest": "abc", "valid": False}})
        plan = guard.plan("k", "abc")
        self.assertEqual(plan["action"], "create")
        self.assertEqual(plan["creations"], 1)
        self.assertEqual(guard.snapshot()["extractor_creations"], 1)
        self.assertEqual(guard.snapshot()["reports"], [])

    def test_replay_runs_gate_and_guard_together(self):
        report = replay_evidence(
            RUN31_RECORDS,
            cache_state={"run31-c001::LGE": {"digest": "abc", "valid": True}},
            extractor_requests=[
                {"key": "run31-c001::LGE", "digest": "abc"},
                {"case_id": "run31-c001", "modality": "T2", "digest": "zz"},
                {"case_id": "run31-c001", "modality": "T2", "digest": "zz"},
            ],
        )
        self.assertEqual(report["gate"]["duplicate_row_ratio"], 0.0)
        self.assertEqual(report["extractor"]["extractor_creations"], 1)
        self.assertEqual(report["extractor_plans"][0]["action"], "reuse")
        self.assertEqual(report["extractor_plans"][1]["action"], "create")
        self.assertEqual(report["extractor_plans"][2]["action"], "reuse")
        json.dumps(report)


if __name__ == "__main__":
    unittest.main()
