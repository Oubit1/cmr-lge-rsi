"""Offline tests for the write-side (case_id, modality) dedup hard gate.

These tests load ``experiments/cmr_write_side_dedup_enforcer.py`` directly by
file path, so they run without importing the ``experiments`` package, without
network access and without any dataset or credential.

The module must be registered in ``sys.modules`` *before* ``exec_module`` runs:
it uses ``from __future__ import annotations`` and CPython's ``dataclasses``
resolves the annotation strings through ``sys.modules[cls.__module__]`` while
the class bodies execute.  Loading it without that registration fails with
``AttributeError: 'NoneType' object has no attribute '__dict__'``.
"""

from __future__ import annotations

import copy
import importlib.util
import pathlib
import sys
import unittest

_MODULE_NAME = "cmr_write_side_dedup_enforcer"
_MODULE_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "experiments"
    / "cmr_write_side_dedup_enforcer.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _MODULE_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"cannot load module from {_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    # dataclasses reads sys.modules[cls.__module__].__dict__ to resolve the
    # string annotations, so the module has to be visible before it executes.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    return module


gate_mod = _load_module()


def _row(case_id, modality, issue_types=(), slice_indices=(), **extra):
    row = {
        "case_id": case_id,
        "modality": modality,
        "issue_types": list(issue_types),
        "slice_indices": list(slice_indices),
    }
    row.update(extra)
    return row


class NormalizationTests(unittest.TestCase):
    def test_module_is_registered_for_dataclass_resolution(self):
        self.assertIs(sys.modules.get(_MODULE_NAME), gate_mod)

    def test_normalize_row_merges_singular_and_plural_keys(self):
        record = gate_mod.normalize_row(
            {
                "case_id": " c1 ",
                "modality": "MYO",
                "issue_type": "excess_components",
                "issue_types": ["touches_border", "excess_components"],
                "slice_index": 4,
                "slice_indices": [2, 2, 9],
            }
        )
        self.assertEqual((record.case_id, record.modality), ("c1", "MYO"))
        self.assertEqual(
            record.issue_types, ("excess_components", "touches_border")
        )
        self.assertEqual(record.slice_indices, (2, 4, 9))

    def test_normalize_row_rejects_missing_key_fields(self):
        bad_rows = [
            {"modality": "MYO"},
            {"case_id": "c1"},
            {"case_id": "   ", "modality": "MYO"},
            {"case_id": "c1", "modality": ""},
            "not-a-row",
        ]
        for bad in bad_rows:
            with self.assertRaises(gate_mod.WriteGateError):
                gate_mod.normalize_row(bad)

    def test_cache_hash_is_order_insensitive_and_stable(self):
        self.assertEqual(
            gate_mod.compute_cache_hash({"a": 1, "b": [2, 3]}),
            gate_mod.compute_cache_hash({"b": [2, 3], "a": 1}),
        )
        self.assertNotEqual(
            gate_mod.compute_cache_hash({"a": 1}),
            gate_mod.compute_cache_hash({"a": 2}),
        )


class WriteSideDedupGateTests(unittest.TestCase):
    def setUp(self):
        self.gate = gate_mod.CaseModalityWriteDedupGate()

    def test_duplicate_rows_are_rejected_and_evidence_is_merged(self):
        rows = [
            {
                "case_id": "c1",
                "modality": "MYO",
                "issue_type": "excess_components",
                "slice_index": 3,
            },
            {
                "case_id": "c1",
                "modality": "MYO",
                "issue_type": "touches_border",
                "slice_index": 5,
            },
            {
                "case_id": "c1",
                "modality": "MYO",
                "issue_type": "excess_components",
                "slice_indices": [7, 3],
            },
        ]
        report = self.gate.ingest(rows)
        self.assertEqual(report.accepted_rows, 1)
        self.assertEqual(report.duplicate_rows, 2)
        self.assertEqual(report.rejected_duplicate_rows, 2)
        self.assertGreaterEqual(report.duplicate_rejection_rate, 0.95)

        records = self.gate.records()
        self.assertEqual(len(records), 1)
        self.assertEqual(
            records[0].issue_types, ("excess_components", "touches_border")
        )
        self.assertEqual(records[0].slice_indices, (3, 5, 7))

    def test_duplicate_rejection_rate_is_one_for_many_duplicates(self):
        rows = [_row("c1", "MYO", ["a"], [1])]
        rows += [_row("c1", "MYO", ["b"], [i]) for i in range(2, 42)]

        report = self.gate.ingest(rows)
        self.assertEqual(report.duplicate_rows, 40)
        self.assertEqual(report.rejected_duplicate_rows, 40)
        self.assertEqual(report.duplicate_rejection_rate, 1.0)
        self.assertGreaterEqual(report.duplicate_rejection_rate, 0.95)

        records = self.gate.records()
        self.assertEqual(len(records), 1)
        self.assertEqual(len(records[0].slice_indices), 41)

    def test_mixed_batch_keeps_rejection_rate_above_threshold(self):
        rows = []
        for i in range(10):
            rows.append(_row(f"c{i}", "MYO", [f"i{i}"], [i]))
            rows.append(_row(f"c{i}", "MYO", [f"j{i}"], [i + 100]))
            rows.append(_row(f"c{i}", "MYO", [f"k{i}"], [i + 200]))

        report = self.gate.ingest(rows)
        self.assertEqual(report.accepted_rows, 10)
        self.assertEqual(report.duplicate_rows, 20)
        self.assertEqual(report.rejected_duplicate_rows, 20)
        self.assertGreaterEqual(report.duplicate_rejection_rate, 0.95)

        records = self.gate.records()
        self.assertEqual(len(records), 10)
        for record in records:
            self.assertEqual(len(record.issue_types), 3)
            self.assertEqual(len(record.slice_indices), 3)

        validation = self.gate.validate()
        self.assertTrue(validation.passed, validation.problems)
        self.assertEqual(self.gate.propose(), ())

    def test_one_record_per_case_modality_with_complete_slice_indices(self):
        rows = []
        for case_id in ("c1", "c2"):
            for modality in ("MYO", "LGE"):
                rows.append(_row(case_id, modality, ["x"], [1]))
                rows.append(_row(case_id, modality, ["y"], [2, 3]))

        report = self.gate.ingest(rows)
        self.assertEqual(report.accepted_rows, 4)
        self.assertEqual(report.duplicate_rows, 4)

        records = self.gate.records()
        self.assertEqual(len(records), 4)
        keys = [record.key for record in records]
        self.assertEqual(len(keys), len(set(keys)))
        for record in records:
            self.assertEqual(record.issue_types, ("x", "y"))
            self.assertEqual(record.slice_indices, (1, 2, 3))

    def test_repeated_run_proposes_nothing_new(self):
        rows = [
            {"case_id": "c1", "modality": "MYO", "issue_type": "a", "slice_index": 1},
            {"case_id": "c2", "modality": "LGE", "issue_type": "b", "slice_index": 2},
        ]

        first = self.gate.ingest(rows)
        self.assertEqual(first.new_proposals, 2)
        records_after_first = self.gate.records()
        proposals_after_first = self.gate.propose()

        second = self.gate.ingest(rows)
        self.assertEqual(second.accepted_rows, 0)
        self.assertEqual(second.merged_rows, 0)
        self.assertEqual(second.new_proposals, 0)
        self.assertTrue(second.no_op)
        self.assertEqual(self.gate.new_proposals, first.new_proposals)
        self.assertEqual(self.gate.records(), records_after_first)
        self.assertEqual(self.gate.propose(), proposals_after_first)

        probe = self.gate.dry_run(rows)
        self.assertEqual(probe.new_proposals, 0)
        self.assertEqual(self.gate.records(), records_after_first)
        self.assertEqual(self.gate.propose(), proposals_after_first)

    def test_mismatched_cache_hash_falls_back_without_committing(self):
        self.gate.set_cache_hash({"build": "v1"})
        rows = [_row("c1", "MYO", ["a"], [1])]

        report = self.gate.ingest(rows, cache_hash="not-the-hash")
        self.assertTrue(report.cache_fallback)
        self.assertTrue(report.no_op)
        self.assertEqual(report.accepted_rows, 0)
        self.assertEqual(self.gate.records(), ())
        self.assertEqual(self.gate.total_rows, 0)

        good = self.gate.ingest(rows, cache_hash=self.gate.cache_hash)
        self.assertFalse(good.cache_fallback)
        self.assertEqual(good.accepted_rows, 1)
        self.assertEqual(len(self.gate.records()), 1)

    def test_cache_fallback_leaves_counters_and_proposals_untouched(self):
        self.gate.set_cache_hash({"build": "v3"})
        self.gate.ingest(
            [_row("c1", "MYO", ["a"], [1])], cache_hash=self.gate.cache_hash
        )
        before_records = self.gate.records()
        before_proposals = self.gate.propose()
        before_counters = dict(self.gate.state_snapshot()["counters"])

        report = self.gate.ingest(
            [_row("c2", "LGE", ["b"], [2]), _row("c1", "MYO", ["c"], [3])],
            cache_hash="stale-hash",
        )
        self.assertTrue(report.cache_fallback)
        self.assertTrue(report.no_op)
        self.assertEqual(report.new_proposals, 0)
        self.assertEqual(self.gate.records(), before_records)
        self.assertEqual(self.gate.propose(), before_proposals)
        self.assertEqual(
            dict(self.gate.state_snapshot()["counters"]), before_counters
        )

    def test_extractor_is_blocked_while_cache_hash_is_valid(self):
        self.gate.set_cache_hash({"build": "v1"})

        blocked = self.gate.request_extractor(cache_hash=self.gate.cache_hash)
        self.assertFalse(blocked.allowed)
        self.assertEqual(blocked.reason, "cache_hash_valid")

        fallback = self.gate.request_extractor(cache_hash="stale-hash")
        self.assertTrue(fallback.allowed)
        self.assertEqual(fallback.reason, "cache_hash_missing_or_stale")

        pristine = gate_mod.CaseModalityWriteDedupGate()
        self.assertTrue(pristine.request_extractor(cache_hash="whatever").allowed)

    def test_extractor_requests_are_denied_once_done(self):
        self.gate.set_cache_hash({"build": "v2"})
        self.gate.ingest(
            [_row("c1", "MYO", ["a"], [1])], cache_hash=self.gate.cache_hash
        )
        self.assertTrue(self.gate.validate().passed)

        decision = self.gate.request_extractor(cache_hash="stale-hash")
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "status_done")

    def test_validate_marks_done_and_stops_proposals(self):
        rows = [_row("c1", "MYO", ["a"], [1]), _row("c2", "LGE", ["b"], [2])]
        self.gate.ingest(rows)
        self.assertEqual(len(self.gate.propose()), 2)

        report = self.gate.validate()
        self.assertTrue(report.passed, report.problems)
        self.assertEqual(report.status, gate_mod.STATUS_DONE)
        self.assertTrue(self.gate.is_done)
        self.assertEqual(self.gate.propose(), ())

        after = self.gate.ingest([_row("c3", "MYO", ["c"], [3])])
        self.assertTrue(after.no_op)
        self.assertEqual(after.accepted_rows, 0)
        self.assertEqual(len(self.gate.records()), 2)
        self.assertEqual(self.gate.propose(), ())

    def test_validate_fails_and_stays_pending_without_slice_indices(self):
        self.gate.ingest([{"case_id": "c1", "modality": "MYO", "issue_type": "a"}])

        report = self.gate.validate()
        self.assertFalse(report.passed)
        self.assertEqual(report.status, gate_mod.STATUS_PENDING)
        self.assertFalse(self.gate.is_done)
        self.assertTrue(
            any("slice_indices" in problem for problem in report.problems)
        )
        self.assertEqual(len(self.gate.propose()), 1)

    def test_label_and_rule_writes_stay_at_zero(self):
        self.gate.ingest([_row("c1", "MYO", ["a"], [1])])
        self.gate.validate()

        self.assertEqual(self.gate.label_writes, 0)
        self.assertEqual(self.gate.rule_writes, 0)
        counters = self.gate.state_snapshot()["counters"]
        self.assertEqual(counters["label_writes"], 0)
        self.assertEqual(counters["rule_writes"], 0)

    def test_input_rows_are_not_mutated(self):
        rows = [
            {"case_id": "c1", "modality": "MYO", "issue_type": "a", "slice_index": 1},
            {"case_id": "c1", "modality": "MYO", "issue_type": "b", "slice_indices": [2]},
        ]
        snapshot = copy.deepcopy(rows)
        self.gate.ingest(rows)
        self.assertEqual(rows, snapshot)

    def test_constructor_rejects_out_of_range_threshold(self):
        with self.assertRaises(gate_mod.WriteGateError):
            gate_mod.CaseModalityWriteDedupGate(min_duplicate_rejection_rate=1.5)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
