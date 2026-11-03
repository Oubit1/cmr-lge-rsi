"""Offline tests for the isolated-side coverage ledger closure."""

import unittest

from experiments.coverage_ledger_closure import (
    DONE,
    OPEN,
    CoverageLedger,
    CoverageLedgerError,
    canonical_rows,
    canonical_work_item,
    closure_metrics,
    content_hash,
    dedup_stats,
    duplicate_issue_row_count,
    duplicate_row_count,
    issue_content_hash,
    max_issue_multiplicity,
    merge_by_case_modality,
    reject_duplicate_rows,
    work_item_from_entry,
    work_items_for_open_issues,
)


def _row(
    case_id,
    modality,
    issue_type,
    slice_index=None,
    rule_version="rv1",
    status=OPEN,
):
    return {
        "case_id": case_id,
        "modality": modality,
        "issue_type": issue_type,
        "slice_index": slice_index,
        "rule_version": rule_version,
        "status": status,
    }


def _duplicate_heavy_rows():
    rows = []
    for _ in range(6):
        rows.append(_row("c1", "LAX", "myo_empty", slice_index=4))
    for _ in range(4):
        rows.append(_row("c1", "LAX", "lge_frag", slice_index=7))
    rows.append(_row("c2", "SAX", "myo_small", slice_index=2))
    return rows


class MergeTests(unittest.TestCase):
    def test_merge_unions_issue_types_and_preserves_slice_indices(self):
        rows = [
            _row("c1", "lax", "myo_empty", slice_index=3),
            _row("c1", "lax", "myo_empty", slice_index=9),
            _row("c1", "lax", "lge_frag", slice_index=9),
            _row("c2", "sax", "myo_empty", slice_index=1),
        ]
        merged = merge_by_case_modality(rows)
        self.assertEqual(len(merged), 2)
        by_key = {(e["case_id"], e["modality"]): e for e in merged}
        c1 = by_key[("c1", "LAX")]
        self.assertEqual(c1["issue_types"], ("lge_frag", "myo_empty"))
        self.assertEqual(c1["slice_indices"], (3, 9))
        self.assertEqual(c1["issue_slices"]["myo_empty"], (3, 9))
        self.assertEqual(c1["issue_slices"]["lge_frag"], (9,))
        self.assertEqual(c1["status"], OPEN)
        self.assertEqual(
            c1["entry_hash"],
            content_hash("c1", "LAX", ("lge_frag", "myo_empty"), (3, 9)),
        )
        self.assertEqual(
            issue_content_hash(c1, "myo_empty"),
            content_hash("c1", "LAX", ("myo_empty",), (3, 9)),
        )

    def test_duplicate_rows_collapse_to_one_entry(self):
        rows = [_row("c1", "LAX", "myo_empty", slice_index=3)] * 5
        merged = merge_by_case_modality(rows)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["issue_types"], ("myo_empty",))
        self.assertEqual(merged[0]["slice_indices"], (3,))
        self.assertEqual(duplicate_row_count(rows), 4)
        kept, dropped = reject_duplicate_rows(rows)
        self.assertEqual((len(kept), dropped), (1, 4))

    def test_case_modality_issue_is_unique_after_merge(self):
        rows = [
            _row("c1", "LAX", "myo_empty", slice_index=1),
            _row("c1", "LAX", "myo_empty", slice_index=2),
            _row("c1", "LAX", "lge_frag", slice_index=5),
            _row("c2", "SAX", "lge_frag", slice_index=5),
        ]
        merged = merge_by_case_modality(rows)
        self.assertEqual(max_issue_multiplicity(merged), 1)
        self.assertEqual(duplicate_issue_row_count(rows), 1)
        self.assertTrue(closure_metrics(rows)["unique_per_case_modality_issue"])

    def test_duplicate_reduction_meets_threshold(self):
        stats = dedup_stats(_duplicate_heavy_rows())
        self.assertGreaterEqual(stats["duplicate_reduction"], 0.9)
        self.assertEqual(stats["duplicate_rows_after"], 0)
        self.assertEqual(stats["duplicate_issue_rows_after"], 0)
        self.assertEqual(stats["input_rows"], 11)
        self.assertEqual(stats["unique_rows"], 3)

    def test_none_slice_index_is_not_invented(self):
        merged = merge_by_case_modality([_row("c1", "LAX", "myo_empty")])
        self.assertEqual(merged[0]["slice_indices"], ())
        self.assertEqual(merged[0]["issue_slices"]["myo_empty"], ())

    def test_merge_keeps_newest_rule_version_and_is_order_independent(self):
        forward = merge_by_case_modality(
            [
                _row("c1", "LAX", "myo_empty", 1, rule_version="rv1"),
                _row("c1", "LAX", "myo_empty", 1, rule_version="rv2"),
            ]
        )
        backward = merge_by_case_modality(
            [
                _row("c1", "LAX", "myo_empty", 1, rule_version="rv2"),
                _row("c1", "LAX", "myo_empty", 1, rule_version="rv1"),
            ]
        )
        self.assertEqual(forward[0]["rule_version"], "rv2")
        self.assertEqual(forward[0]["rule_versions"], ("rv1", "rv2"))
        self.assertEqual(forward[0]["entry_hash"], backward[0]["entry_hash"])

    def test_closure_metrics_summarise_batch(self):
        metrics = closure_metrics(_duplicate_heavy_rows())
        self.assertEqual(metrics["merged_entries"], 2)
        self.assertEqual(metrics["issue_types_total"], 3)
        self.assertLessEqual(metrics["max_issue_multiplicity"], 1)
        self.assertEqual(metrics["duplicate_rows_after"], 0)


class HashTests(unittest.TestCase):
    def test_hash_ignores_order_duplicates_and_modality_case(self):
        left = content_hash("c1", "LAX", ["x", "y"], [3, 1, 1])
        right = content_hash("c1", "lax", ["y", "x", "x"], [1, 3])
        self.assertEqual(left, right)

    def test_hash_changes_when_content_changes(self):
        self.assertNotEqual(
            content_hash("c1", "LAX", ["x"], [1]),
            content_hash("c1", "LAX", ["x"], [2]),
        )
        self.assertNotEqual(
            content_hash("c1", "LAX", ["x"], [1]),
            content_hash("c1", "LAX", ["y"], [1]),
        )


class ValidationTests(unittest.TestCase):
    def test_missing_required_fields_raise(self):
        with self.assertRaises(CoverageLedgerError):
            canonical_rows([{"case_id": "c1"}])
        with self.assertRaises(CoverageLedgerError):
            canonical_rows([{"case_id": "c1", "modality": "LAX"}])
        with self.assertRaises(CoverageLedgerError):
            canonical_rows(
                [{"case_id": "  ", "modality": "LAX", "issue_type": "x"}]
            )

    def test_unknown_status_raises(self):
        with self.assertRaises(CoverageLedgerError):
            canonical_rows([_row("c1", "LAX", "x", status="MAYBE")])

    def test_bad_slice_index_raises(self):
        with self.assertRaises(CoverageLedgerError):
            canonical_rows([_row("c1", "LAX", "x", slice_index="abc")])
        with self.assertRaises(CoverageLedgerError):
            canonical_rows([_row("c1", "LAX", "x", slice_index=True)])

    def test_non_mapping_row_raises(self):
        with self.assertRaises(CoverageLedgerError):
            canonical_rows(["not-a-row"])

    def test_work_item_accepts_single_slice_index_and_coerces_strings(self):
        item = canonical_work_item(
            {
                "case_id": "c1",
                "modality": "lax",
                "issue_type": "myo_empty",
                "slice_index": "4",
                "rule_version": "rv1",
            }
        )
        self.assertEqual(item["modality"], "LAX")
        self.assertEqual(item["slice_indices"], (4,))
        self.assertEqual(
            item["content_hash"],
            content_hash("c1", "LAX", ("myo_empty",), (4,)),
        )

    def test_work_item_rejects_string_slice_indices_field(self):
        with self.assertRaises(CoverageLedgerError):
            canonical_work_item(
                {
                    "case_id": "c1",
                    "modality": "LAX",
                    "issue_type": "x",
                    "slice_indices": "12",
                }
            )


def _fresh_ledger(rows):
    ledger = CoverageLedger()
    ledger.load(rows)
    return ledger


class DoneSentinelTests(unittest.TestCase):
    def test_done_entry_is_skipped_when_hash_and_rule_version_match(self):
        ledger = _fresh_ledger(
            [_row("c1", "LAX", "myo_empty", slice_index=4, status=DONE)]
        )
        self.assertTrue(ledger.is_done("c1", "lax", "myo_empty"))
        entry = ledger.entry_for("c1", "lax")
        item = work_item_from_entry(entry, "myo_empty")
        self.assertEqual(ledger.propose([item]), [])
        self.assertEqual(
            [event["event"] for event in ledger.audit_events], ["skip_done"]
        )

    def test_changed_slices_reopen_done_entry(self):
        ledger = _fresh_ledger(
            [_row("c1", "LAX", "myo_empty", slice_index=4, status=DONE)]
        )
        entry = ledger.entry_for("c1", "LAX")
        item = work_item_from_entry(entry, "myo_empty")
        widened = dict(item)
        widened["slice_indices"] = tuple(item["slice_indices"]) + (11,)
        proposals = ledger.propose([widened])
        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0]["slice_indices"], (4, 11))
        self.assertNotEqual(proposals[0]["content_hash"], item["content_hash"])

    def test_rule_version_bump_reopens_done_entry(self):
        ledger = _fresh_ledger(
            [_row("c1", "LAX", "myo_empty", slice_index=4, status=DONE)]
        )
        entry = ledger.entry_for("c1", "LAX")
        bumped = work_item_from_entry(entry, "myo_empty", rule_version="rv2")
        proposals = ledger.propose([bumped])
        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0]["rule_version"], "rv2")

    def test_partially_done_pair_stays_open(self):
        ledger = _fresh_ledger(
            [
                _row("c1", "LAX", "myo_empty", 4, status=DONE),
                _row("c1", "LAX", "lge_frag", 7, status=OPEN),
            ]
        )
        entry = ledger.entry_for("c1", "LAX")
        self.assertEqual(entry["status"], OPEN)
        self.assertFalse(ledger.is_done("c1", "LAX", "myo_empty"))
        issues = {item["issue_type"] for item in work_items_for_open_issues(ledger.entries)}
        self.assertEqual(issues, {"myo_empty", "lge_frag"})

    def test_mark_done_requires_validation_then_freezes(self):
        ledger = _fresh_ledger([])
        digest = content_hash("c1", "LAX", ("myo_empty",), (4,))
        self.assertFalse(
            ledger.mark_done(
                "c1", "LAX", "myo_empty", digest, "rv1", validation_passed=False
            )
        )
        self.assertFalse(ledger.is_done("c1", "LAX", "myo_empty"))
        self.assertTrue(
            ledger.mark_done(
                "c1", "LAX", "myo_empty", digest, "rv1", validation_passed=True
            )
        )
        self.assertTrue(ledger.is_done("c1", "LAX", "myo_empty"))
        candidate = {
            "case_id": "c1",
            "modality": "LAX",
            "issue_type": "myo_empty",
            "slice_indices": [4],
            "rule_version": "rv1",
        }
        self.assertEqual(ledger.propose([candidate]), [])
        events = [event["event"] for event in ledger.audit_events]
        self.assertEqual(
            events, ["mark_done_rejected", "mark_done", "skip_done"]
        )

    def test_mark_done_rejects_empty_hash(self):
        ledger = _fresh_ledger([])
        with self.assertRaises(CoverageLedgerError):
            ledger.mark_done(
                "c1", "LAX", "myo_empty", "", "rv1", validation_passed=True
            )


if __name__ == "__main__":
    unittest.main()
