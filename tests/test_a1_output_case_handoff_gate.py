"""Offline unit tests for the A1 output-side write-time hard gate.

No network, no datasets, no production store: everything runs on in-memory
fixtures.
"""

from __future__ import annotations

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.a1_output_case_handoff_gate import (  # noqa: E402
    A1Row,
    CaseModalityRecord,
    DuplicateRowError,
    ExtractorCreationBlocked,
    ExtractorReuseGuard,
    GateResult,
    InvalidRowError,
    RecallRegression,
    apply_hard_gate,
    apply_write_gate,
    build_case_handoff,
    duplicate_record_rate,
    effort_ledger,
    ensure_recall_not_decreased,
    legacy_facts,
    recall_coverage,
    run_a1_output_gate,
    validate_handoff,
)


MIXED_ROWS = [
    {
        "case_id": "c1",
        "modality": "LGE",
        "issue_types": ["fragmentation"],
        "slice_indices": [3, 4],
        "severity": "warn",
    },
    {
        "case_id": "c1",
        "modality": "LGE",
        "issue_types": ["transmural"],
        "slice_indices": [9],
        "severity": "high",
    },
    {
        "case_id": "c1",
        "modality": "LGE",
        "issue_types": ["fragmentation"],
        "slice_indices": [3, 4],
        "severity": "warn",
    },  # duplicate row
    {
        "case_id": "c1",
        "modality": "T2",
        "issue_types": ["temporal_hole"],
        "slice_indices": [1],
    },
    {
        "case_id": "c1",
        "modality": "T2",
        "issue_types": ["temporal_hole"],
        "slice_indices": [1],
    },  # duplicate row
    {"case_id": "c2", "modality": "LGE", "issue_types": ["fragmentation"], "slice_indices": [7]},
]


def _record(case_id, modality, issues=(), slices=(), severity="info", row_count=1):
    return CaseModalityRecord(
        case_id=case_id,
        modality=modality,
        issue_types=tuple(issues),
        slice_indices=tuple(slices),
        sources=(),
        severity=severity,
        row_count=row_count,
    )


class TestMergeAndReject(unittest.TestCase):
    def test_merges_issue_types_and_preserves_slice_indices(self):
        result = apply_hard_gate(MIXED_ROWS)
        lge = [r for r in result.records if r.identity == ("c1", "LGE")][0]
        t2 = [r for r in result.records if r.identity == ("c1", "T2")][0]

        self.assertEqual(lge.issue_types, ("fragmentation", "transmural"))
        self.assertEqual(lge.slice_indices, (3, 4, 9))
        self.assertEqual(lge.row_count, 2)
        self.assertEqual(lge.severity, "high")
        self.assertEqual(t2.issue_types, ("temporal_hole",))
        self.assertEqual(t2.slice_indices, (1,))

    def test_one_record_per_case_modality(self):
        result = apply_hard_gate(MIXED_ROWS)
        identities = [r.identity for r in result.records]
        self.assertEqual(len(identities), len(set(identities)))
        self.assertEqual(result.output_rows, 3)
        self.assertEqual(result.input_rows, 6)

    def test_duplicate_rows_are_rejected_and_rate_is_zero(self):
        result = apply_hard_gate(MIXED_ROWS)
        self.assertEqual(result.duplicate_rows, 2)
        self.assertTrue(all(item.reason == "duplicate_row" for item in result.rejected))
        self.assertEqual(result.duplicate_record_rate, 0.0)
        self.assertEqual(duplicate_record_rate(result.records), 0.0)

    def test_subset_row_is_treated_as_duplicate(self):
        rows = [
            {"case_id": "c3", "modality": "LGE", "issue_types": ["a", "b"], "slice_indices": [1, 2]},
            {"case_id": "c3", "modality": "LGE", "issue_types": ["a"], "slice_indices": [1]},
        ]
        result = apply_hard_gate(rows)
        self.assertEqual(result.output_rows, 1)
        self.assertEqual(result.duplicate_rows, 1)
        self.assertEqual(result.records[0].issue_types, ("a", "b"))

    def test_accepts_a1row_objects(self):
        rows = [A1Row("c4", "LGE", ("x",), (5,))]
        result = apply_hard_gate(rows)
        self.assertEqual(result.output_rows, 1)
        self.assertEqual(result.records[0].slice_indices, (5,))

    def test_normalisation_is_sorted_and_deduplicated(self):
        row = A1Row.from_mapping(
            {
                "case_id": " c5 ",
                "modality": "LGE",
                "issue_types": ["b", "a", "a"],
                "slice_indices": "4,3,3",
            }
        )
        self.assertEqual(row.case_id, "c5")
        self.assertEqual(row.issue_types, ("a", "b"))
        self.assertEqual(row.slice_indices, (3, 4))

    def test_field_aliases_are_supported(self):
        row = A1Row.from_mapping(
            {"caseId": "c6", "mod": "LGE", "issue_type": "x", "slice_index": 2}
        )
        self.assertEqual(row.identity, ("c6", "LGE"))
        self.assertEqual(row.issue_types, ("x",))
        self.assertEqual(row.slice_indices, (2,))


class TestInvalidRows(unittest.TestCase):
    def test_missing_case_id_raises(self):
        with self.assertRaises(InvalidRowError):
            apply_hard_gate([{"modality": "LGE"}])

    def test_empty_modality_raises(self):
        with self.assertRaises(InvalidRowError):
            apply_hard_gate([{"case_id": "c1", "modality": "   "}])

    def test_non_integer_slice_index_raises(self):
        with self.assertRaises(InvalidRowError):
            apply_hard_gate([{"case_id": "c1", "modality": "LGE", "slice_indices": ["abc"]}])

    def test_negative_slice_index_raises(self):
        with self.assertRaises(InvalidRowError):
            apply_hard_gate([{"case_id": "c1", "modality": "LGE", "slice_indices": [-1]}])


class TestRecallPreservation(unittest.TestCase):
    def test_legacy_recall_is_fully_covered(self):
        result = apply_hard_gate(MIXED_ROWS)
        facts = legacy_facts(MIXED_ROWS)
        self.assertTrue(facts)
        self.assertEqual(recall_coverage(facts, result.records), 1.0)
        self.assertEqual(ensure_recall_not_decreased(facts, result.records), 1.0)

    def test_empty_legacy_facts_count_as_full_recall(self):
        self.assertEqual(recall_coverage(set(), ()), 1.0)

    def test_missing_slice_raises_recall_regression(self):
        facts = {("c1", "LGE", "fragmentation", 4)}
        records = [_record("c1", "LGE", issues=("fragmentation",), slices=(3,))]
        self.assertLess(recall_coverage(facts, records), 1.0)
        with self.assertRaises(RecallRegression):
            ensure_recall_not_decreased(facts, records)

    def test_missing_issue_raises_recall_regression(self):
        facts = {("c1", "LGE", "transmural", None)}
        records = [_record("c1", "LGE", issues=("fragmentation",), slices=(3,))]
        with self.assertRaises(RecallRegression):
            ensure_recall_not_decreased(facts, records)


class TestHandoffArtifact(unittest.TestCase):
    def test_handoff_is_single_case_level_artifact(self):
        result = apply_hard_gate(MIXED_ROWS)
        handoff = build_case_handoff(result)

        self.assertEqual(handoff["case_count"], 2)
        self.assertEqual(handoff["record_count"], 3)
        self.assertEqual(handoff["duplicate_record_rate"], 0.0)
        self.assertEqual(len(handoff["cases"]), 2)
        case_one = [c for c in handoff["cases"] if c["case_id"] == "c1"][0]
        self.assertEqual(case_one["record_count"], 2)
        self.assertEqual([m["modality"] for m in case_one["modalities"]], ["LGE", "T2"])
        self.assertEqual(
            case_one["issue_types"], ["fragmentation", "temporal_hole", "transmural"]
        )
        self.assertEqual(case_one["slice_indices"], [1, 3, 4, 9])
        self.assertEqual(validate_handoff(handoff), [])
        json.dumps(handoff)  # must stay JSON serialisable

    def test_validate_handoff_flags_duplicate_records(self):
        broken = {
            "schema_version": "a1-case-handoff-v1",
            "case_count": 1,
            "record_count": 2,
            "duplicate_record_rate": 0.5,
            "cases": [
                {
                    "case_id": "c1",
                    "record_count": 2,
                    "modalities": [
                        {"modality": "LGE", "issue_types": [], "slice_indices": []},
                        {"modality": "LGE", "issue_types": [], "slice_indices": []},
                    ],
                }
            ],
        }
        problems = validate_handoff(broken)
        self.assertTrue(any("duplicate record" in problem for problem in problems))
        self.assertTrue(any("duplicate_record_rate" in problem for problem in problems))

    def test_duplicate_record_rate_helper(self):
        records = (
            _record("c1", "LGE", issues=("a",), slices=(1,)),
            _record("c1", "LGE", issues=("b",), slices=(2,)),
            _record("c2", "LGE", issues=("c",), slices=(3,)),
        )
        self.assertAlmostEqual(duplicate_record_rate(records), 1 / 3)
        self.assertEqual(duplicate_record_rate(()), 0.0)


class TestExtractorReuseGuard(unittest.TestCase):
    def test_valid_cache_hash_forbids_new_extractor(self):
        guard = ExtractorReuseGuard({("c1", "LGE"): "hash-c1-lge"})
        self.assertTrue(guard.has_valid_cache(("c1", "LGE")))
        with self.assertRaises(ExtractorCreationBlocked):
            guard.create_extractor(("c1", "LGE"))
        self.assertEqual(guard.extractor_calls, 0)
        self.assertEqual(guard.blocked_creations, 1)

        handle = guard.resolve_extractor(("c1", "LGE"))
        self.assertTrue(handle.reused)
        self.assertEqual(handle.extractor_id, "cached::c1::LGE")
        self.assertEqual(guard.cache_reuses, 1)
        self.assertEqual(guard.extractor_calls, 0)

    def test_creation_allowed_without_cache_then_reused(self):
        guard = ExtractorReuseGuard()
        self.assertFalse(guard.has_valid_cache(("c2", "LGE")))
        first = guard.resolve_extractor(("c2", "LGE"), cache_hash="h2")
        self.assertFalse(first.reused)
        self.assertEqual(guard.extractor_calls, 1)
        self.assertTrue(guard.has_valid_cache(("c2", "LGE")))

        second = guard.resolve_extractor(("c2", "LGE"), cache_hash="h2")
        self.assertTrue(second.reused)
        self.assertEqual(guard.cache_reuses, 1)
        self.assertEqual(guard.extractor_calls, 1)

    def test_hash_conflict_is_blocked(self):
        guard = ExtractorReuseGuard({("c3", "LGE"): "h3"})
        with self.assertRaises(ExtractorCreationBlocked):
            guard.resolve_extractor(("c3", "LGE"), cache_hash="different")
        self.assertEqual(guard.extractor_calls, 0)
        self.assertEqual(len(guard.conflicts), 1)

    def test_invalidation_reopens_creation(self):
        guard = ExtractorReuseGuard({("c4", "LGE"): "h4"})
        with self.assertRaises(ExtractorCreationBlocked):
            guard.create_extractor(("c4", "LGE"))
        guard.invalidate(("c4", "LGE"), reason="shadow rerun")
        handle = guard.create_extractor(("c4", "LGE"), cache_hash="h4b")
        self.assertFalse(handle.reused)
        self.assertEqual(guard.extractor_calls, 1)
        self.assertEqual(len(guard.invalidations), 1)

    def test_invalid_cache_entry_does_not_block(self):
        guard = ExtractorReuseGuard({("c5", "LGE"): {"cache_hash": "h5", "valid": False}})
        self.assertFalse(guard.has_valid_cache(("c5", "LGE")))
        handle = guard.create_extractor(("c5", "LGE"))
        self.assertFalse(handle.reused)


class TestEffortLedger(unittest.TestCase):
    def test_step_reduction_on_collapsed_batch(self):
        ledger = effort_ledger(6, record_count=3, cache_reuses=3, case_count=2)
        self.assertEqual(ledger.legacy_steps, 18)
        self.assertEqual(ledger.gated_steps, 5)
        self.assertGreaterEqual(ledger.reduction, 0.30)
        self.assertTrue(ledger.meets())

    def test_no_rows_means_no_reduction(self):
        ledger = effort_ledger(0, record_count=0)
        self.assertEqual(ledger.legacy_steps, 0)
        self.assertEqual(ledger.reduction, 0.0)

    def test_reuses_are_clamped_to_record_count(self):
        ledger = effort_ledger(3, record_count=1, cache_reuses=99, case_count=1)
        self.assertEqual(ledger.gated_steps, 2)


class TestWriteGateAndEntryPoint(unittest.TestCase):
    def test_write_gate_blocks_duplicates(self):
        duplicated = (_record("c1", "LGE"), _record("c1", "LGE"))
        with self.assertRaises(DuplicateRowError):
            apply_write_gate(duplicated)

    def test_write_gate_accepts_gate_result(self):
        result = GateResult(records=(_record("c1", "LGE"), _record("c1", "T2")))
        store = apply_write_gate(result)
        self.assertEqual(sorted(store), [("c1", "LGE"), ("c1", "T2")])

    def test_run_emits_one_handoff_and_no_extra_writes(self):
        writes = []
        run = run_a1_output_gate(
            MIXED_ROWS,
            cache={("c1", "LGE"): "hc1lge"},
            writer=writes.append,
        )
        self.assertEqual(run.writes, 3)
        self.assertEqual(len(writes), 3)
        self.assertEqual(len(run.store), 3)
        self.assertEqual(validate_handoff(run.handoff), [])
        self.assertEqual(run.recall_coverage, 1.0)
        self.assertTrue(run.guard.has_valid_cache(("c1", "LGE")))
        self.assertEqual(run.result.duplicate_record_rate, 0.0)
        self.assertGreaterEqual(run.ledger.reduction, 0.30)

    def test_run_without_cache_creates_extractors_once(self):
        run = run_a1_output_gate(MIXED_ROWS)
        self.assertEqual(run.guard.extractor_calls, 3)
        self.assertEqual(run.guard.cache_reuses, 0)
        self.assertEqual(run.guard.blocked_creations, 0)
        self.assertEqual(validate_handoff(run.handoff), [])

    def test_identical_inputs_are_deterministic(self):
        first = run_a1_output_gate(MIXED_ROWS)
        second = run_a1_output_gate(MIXED_ROWS)
        self.assertEqual(first.handoff, second.handoff)


if __name__ == "__main__":
    unittest.main()
