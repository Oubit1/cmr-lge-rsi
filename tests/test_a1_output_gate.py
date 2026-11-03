"""Offline unit tests for the A1 output-side hard gate."""

import unittest

from experiments.a1_output_gate import (
    B1_IMPORT_EXTRACTOR,
    ExtractorGateError,
    enforce_extractor_gate,
    evaluate_extractor_gate,
    merge_a1_records,
    summarize_high_risk,
)


def base_rows():
    return [
        {
            "case_id": "c1",
            "modality": "LGE",
            "issue_type": "mi_fragmentation",
            "slice_indices": [3, 7],
            "severity": "high",
            "flags": {"scar"},
            "labels": ["mi"],
            "rule": "r1",
        },
        {
            "case_id": "c1",
            "modality": "LGE",
            "issue_type": "myo_excess",
            "slice_indices": [5],
            "severity": "medium",
            "flags": {"myo"},
            "labels": ["myo"],
            "rule": "r2",
        },
        {
            "case_id": "c1",
            "modality": "LGE",
            "issue_type": "mi_fragmentation",
            "slice_indices": [7, 9],
            "severity": "low",
            "flags": {"scar"},
            "labels": ["mi"],
            "rule": "r1",
        },
        {
            "case_id": "c2",
            "modality": "T1",
            "issue_type": "small_area",
            "slice_indices": [1],
            "severity": "critical",
            "flags": {"myo"},
            "labels": ["myo"],
            "rule": "r3",
        },
    ]


def record_for(records, case_id, modality):
    for record in records:
        if record["case_id"] == case_id and record["modality"] == modality:
            return record
    raise AssertionError("missing record for {0}/{1}".format(case_id, modality))


class MergeGateTest(unittest.TestCase):
    def test_duplicate_rows_rejected_and_unique_key_rate_is_one(self):
        rows = base_rows()
        rows.append(dict(rows[0]))
        rows.append(dict(rows[3]))
        report = merge_a1_records(rows)
        self.assertEqual(report.input_row_count, 6)
        self.assertEqual(report.duplicate_rows_rejected, 2)
        self.assertEqual(len(report.records), 2)
        self.assertEqual(report.unique_key_rate, 1.0)
        keys = sorted((record["case_id"], record["modality"]) for record in report.records)
        self.assertEqual(keys, [("c1", "LGE"), ("c2", "T1")])

    def test_issue_types_merged_and_slice_recall_identical(self):
        rows = base_rows()
        before = {
            (row["case_id"], row["modality"], row["issue_type"], int(slice_index))
            for row in rows
            for slice_index in row["slice_indices"]
        }
        report = merge_a1_records(rows)
        self.assertEqual(set(report.positive_slice_index), before)
        self.assertEqual(len(report.records), report.merged_group_count)
        merged = record_for(report.records, "c1", "LGE")
        self.assertEqual(merged["issue_types"], ("mi_fragmentation", "myo_excess"))
        self.assertEqual(merged["slice_indices_by_issue"]["mi_fragmentation"], (3, 7, 9))
        self.assertEqual(merged["slice_indices_by_issue"]["myo_excess"], (5,))
        self.assertEqual(merged["slice_indices"], (3, 5, 7, 9))

    def test_flags_labels_rules_and_severity_unchanged(self):
        report = merge_a1_records(base_rows())
        merged = record_for(report.records, "c1", "LGE")
        self.assertEqual(merged["flags"], ("myo", "scar"))
        self.assertEqual(merged["labels"], ("mi", "myo"))
        self.assertEqual(merged["rule_ids"], ("r1", "r2"))
        self.assertEqual(merged["severity"], "high")

    def test_false_repair_cap_is_never_raised(self):
        rows = [
            {
                "case_id": "c1",
                "modality": "LGE",
                "issue_type": "mi_fragmentation",
                "slice_indices": [1],
                "false_repair_cap": 4,
            },
            {
                "case_id": "c1",
                "modality": "LGE",
                "issue_type": "myo_excess",
                "slice_indices": [2],
                "false_repair_cap": 2,
            },
        ]
        merged = merge_a1_records(rows).records[0]
        self.assertEqual(merged["false_repair_cap"], 2)
        self.assertEqual(merged["slice_indices"], (1, 2))

    def test_row_and_token_reduction_without_content_loss(self):
        rows = base_rows()
        for row in list(rows):
            for _ in range(3):
                rows.append(dict(row))
        report = merge_a1_records(rows)
        self.assertEqual(report.duplicate_rows_rejected, 12)
        self.assertEqual(report.row_count, 2)
        self.assertLess(report.row_count, report.input_row_count)
        self.assertLess(report.payload_chars_after, report.payload_chars_before)
        self.assertGreater(report.token_reduction_ratio, 0.0)

    def test_non_mapping_row_is_rejected(self):
        with self.assertRaises(TypeError):
            merge_a1_records([base_rows()[0], "not-a-row"])


class HighRiskSummaryTest(unittest.TestCase):
    def test_summary_counts_rate_worst_slice_and_addressability(self):
        report = merge_a1_records(base_rows())
        summary = summarize_high_risk(report.records)
        self.assertEqual(summary.total_records, 2)
        self.assertEqual(summary.positive_records, 2)
        self.assertEqual(summary.positive_rate, 1.0)
        self.assertEqual(summary.issue_counts["mi_fragmentation"], 1)
        self.assertEqual(summary.issue_slice_counts["mi_fragmentation"], 3)
        self.assertEqual(summary.worst_slice_index, 9)
        self.assertEqual(summary.worst_slice_index_by_severity["high"], 9)
        self.assertEqual(summary.severity_counts["high"], 1)
        self.assertEqual(summary.severity_counts["critical"], 1)
        self.assertEqual(summary.flag_counts["scar"], 1)
        self.assertEqual(set(summary.positive_slice_index), set(report.positive_slice_index))
        self.assertEqual(len(summary.positive_slice_index), 5)

    def test_empty_input_is_summarised_safely(self):
        summary = summarize_high_risk(())
        self.assertEqual(summary.total_records, 0)
        self.assertEqual(summary.positive_rate, 0.0)
        self.assertIsNone(summary.worst_slice_index)
        self.assertEqual(summary.positive_slice_index, ())


class ExtractorGateTest(unittest.TestCase):
    def test_valid_cache_hash_forbids_new_extractor(self):
        decision = evaluate_extractor_gate(
            cache_hash="abc", expected_hash="abc", requested_extractor=B1_IMPORT_EXTRACTOR
        )
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "cache_hash_valid")
        self.assertIsNone(decision.extractor_kind)

    def test_hash_mismatch_allows_only_b1_import(self):
        allowed = evaluate_extractor_gate(
            cache_hash="stale", expected_hash="fresh", requested_extractor=B1_IMPORT_EXTRACTOR
        )
        self.assertTrue(allowed.allowed)
        self.assertEqual(allowed.extractor_kind, B1_IMPORT_EXTRACTOR)
        for kind in ("a1_shadow", "legacy_import", "b2_import", None, ""):
            decision = evaluate_extractor_gate(
                cache_hash="stale", expected_hash="fresh", requested_extractor=kind
            )
            self.assertFalse(decision.allowed, msg=str(kind))

    def test_second_b1_import_extractor_is_refused(self):
        decision = evaluate_extractor_gate(
            cache_hash="stale",
            expected_hash="fresh",
            requested_extractor=B1_IMPORT_EXTRACTOR,
            cached_extractors=(B1_IMPORT_EXTRACTOR,),
        )
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "b1_import_extractor_already_cached")

    def test_missing_expected_hash_never_allows(self):
        for cache_hash in ("abc", None, ""):
            decision = evaluate_extractor_gate(
                cache_hash=cache_hash, expected_hash="", requested_extractor=B1_IMPORT_EXTRACTOR
            )
            self.assertFalse(decision.allowed)
            self.assertEqual(decision.reason, "missing_expected_hash")

    def test_enforce_raises_on_valid_cache(self):
        with self.assertRaises(ExtractorGateError):
            enforce_extractor_gate(
                cache_hash="abc", expected_hash="abc", requested_extractor=B1_IMPORT_EXTRACTOR
            )
        decision = enforce_extractor_gate(
            cache_hash="stale", expected_hash="fresh", requested_extractor=B1_IMPORT_EXTRACTOR
        )
        self.assertTrue(decision.allowed)


if __name__ == "__main__":
    unittest.main()
