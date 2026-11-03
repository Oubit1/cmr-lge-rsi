"""Offline unit tests for the cmr_centroid_drift co-occurrence shadow audit.

No network access, no datasets, no patient data: every fixture is a plain
in-memory ``ShadowCase`` record.
"""

import dataclasses
import importlib
import unittest

from experiments.cmr_centroid_drift_cooccurrence_stratum_audit import (
    COOCCURRENCE_FLAGS,
    EXPECTED_CONTROL_CASES,
    EXPECTED_POSITIVE_CASES,
    FALLBACK_EXTRACTOR_RUN,
    FALLBACK_EXTRACTOR_TAG,
    MAX_CONTROL_FALSE_POSITIVE_RATE,
    MIN_CACHE_HIT_RATE,
    MIN_POSITIVE_CONCORDANCE_LIFT,
    REUSABLE_RUNS,
    STRATUM_NONE,
    AuditInputError,
    ShadowCase,
    audit_shadow_cohort,
    cooccurrence_stratum,
    resolve_cache,
)

MODULE_NAME = "experiments.cmr_centroid_drift_cooccurrence_stratum_audit"

_POSITIVE_STRATA = (
    ({}, 10),
    ({"small_area": True}, 6),
    ({"excess_components": True}, 5),
    ({"border": True}, 4),
    ({"review_centroid_outlier": True}, 3),
    ({"small_area": True, "border": True}, 2),
)

_CONTROL_STRATA = (
    ({}, 4),
    ({"small_area": True}, 2),
    ({"excess_components": True}, 1),
    ({"border": True}, 1),
    ({"review_centroid_outlier": True}, 1),
    ({"small_area": True, "border": True}, 1),
)

_MISMATCH_TAIL = 4


def _make_case(**overrides):
    base = dict(
        case_id="X000",
        cohort="positive",
        flags={},
        cache_run=4,
        cache_hash="h0",
        expected_hash="h0",
        slices=5,
        baseline_hits=3,
        shadow_hits=4,
    )
    base.update(overrides)
    return ShadowCase(**base)


def _build_cohort():
    """30 positives + 10 controls, 4 of them with a stale cache hash."""

    cases = []
    runs = list(REUSABLE_RUNS)

    for flags, count in _POSITIVE_STRATA:
        for _ in range(count):
            idx = len(cases)
            cases.append(
                _make_case(
                    case_id="P{0:03d}".format(idx),
                    cohort="positive",
                    flags=dict(flags),
                    cache_run=runs[idx % len(runs)],
                    cache_hash="h{0}".format(idx),
                    expected_hash="h{0}".format(idx),
                    slices=5 if idx < 20 else 10,
                    baseline_hits=3 if idx < 20 else 6,
                    shadow_hits=4 if idx < 20 else 7,
                )
            )

    for flags, count in _CONTROL_STRATA:
        for _ in range(count):
            idx = len(cases)
            cases.append(
                _make_case(
                    case_id="C{0:03d}".format(idx),
                    cohort="control",
                    flags=dict(flags),
                    cache_run=runs[idx % len(runs)],
                    cache_hash="h{0}".format(idx),
                    expected_hash="h{0}".format(idx),
                    slices=10,
                    baseline_hits=6,
                    shadow_hits=1 if idx % 3 == 0 else 0,
                )
            )

    head, tail = cases[:-_MISMATCH_TAIL], cases[-_MISMATCH_TAIL:]
    tail = [
        dataclasses.replace(case, expected_hash=case.expected_hash + "-expected")
        for case in tail
    ]
    return head + tail


class CooccurrenceStratumTests(unittest.TestCase):
    def test_empty_pattern_maps_to_none(self):
        self.assertEqual(cooccurrence_stratum({}), STRATUM_NONE)
        self.assertEqual(
            cooccurrence_stratum({flag: False for flag in COOCCURRENCE_FLAGS}),
            STRATUM_NONE,
        )

    def test_keys_are_canonical_and_order_independent(self):
        a = cooccurrence_stratum({"border": True, "small_area": True})
        b = cooccurrence_stratum({"small_area": True, "border": True})
        self.assertEqual(a, b)
        self.assertEqual(a, "small_area|border")

    def test_single_flag_uses_declared_order(self):
        self.assertEqual(
            cooccurrence_stratum({"review_centroid_outlier": True}),
            "review_centroid_outlier",
        )
        self.assertLess(
            COOCCURRENCE_FLAGS.index("small_area"),
            COOCCURRENCE_FLAGS.index("border"),
        )

    def test_unknown_flag_is_rejected(self):
        with self.assertRaises(AuditInputError):
            cooccurrence_stratum({"not_a_flag": True})
        with self.assertRaises(AuditInputError):
            cooccurrence_stratum(None)


class ResolveCacheTests(unittest.TestCase):
    def test_matching_hash_on_reusable_run_is_a_hit(self):
        case = _make_case(cache_run=13, cache_hash="abc", expected_hash="abc")
        decision = resolve_cache(case)
        self.assertTrue(decision.cache_hit)
        self.assertFalse(decision.hash_mismatch)
        self.assertFalse(decision.registered)
        self.assertEqual(decision.used_run, 13)
        self.assertEqual(decision.extractor, "run-13:cache")

    def test_hash_mismatch_falls_back_to_run9_b1_and_is_registered(self):
        case = _make_case(cache_run=5, cache_hash="abc", expected_hash="xyz")
        decision = resolve_cache(case)
        self.assertFalse(decision.cache_hit)
        self.assertTrue(decision.hash_mismatch)
        self.assertTrue(decision.registered)
        self.assertEqual(decision.used_run, FALLBACK_EXTRACTOR_RUN)
        self.assertEqual(decision.extractor, FALLBACK_EXTRACTOR_TAG)
        self.assertEqual(decision.reason, "hash-mismatch")

    def test_run_outside_reusable_set_is_registered_but_not_a_hash_mismatch(self):
        self.assertNotIn(7, REUSABLE_RUNS)
        case = _make_case(cache_run=7, cache_hash="abc", expected_hash="abc")
        decision = resolve_cache(case)
        self.assertFalse(decision.cache_hit)
        self.assertFalse(decision.hash_mismatch)
        self.assertTrue(decision.registered)
        self.assertEqual(decision.extractor, FALLBACK_EXTRACTOR_TAG)


class ShadowCaseValidationTests(unittest.TestCase):
    def test_valid_record_is_accepted(self):
        case = _make_case()
        self.assertEqual(case.case_id, "X000")

    def test_bad_records_are_rejected(self):
        bad_overrides = (
            {"case_id": ""},
            {"cohort": "maybe"},
            {"flags": {"not_a_flag": True}},
            {"cache_run": "4"},
            {"cache_run": 0},
            {"cache_hash": ""},
            {"slices": 0},
            {"baseline_hits": 99},
            {"shadow_hits": -1},
        )
        for overrides in bad_overrides:
            with self.subTest(overrides=overrides):
                with self.assertRaises(AuditInputError):
                    _make_case(**overrides)


class AuditShadowCohortTests(unittest.TestCase):
    def setUp(self):
        self.cases = _build_cohort()
        self.report = audit_shadow_cohort(self.cases)

    def test_cohort_shape_matches_the_approved_design(self):
        self.assertEqual(len(self.cases), 40)
        self.assertEqual(self.report.positive_cases, EXPECTED_POSITIVE_CASES)
        self.assertEqual(self.report.control_cases, EXPECTED_CONTROL_CASES)
        self.assertTrue(self.report.acceptance["cohort_composition"])

    def test_acceptance_metrics_pass(self):
        self.assertIsNotNone(self.report.positive_lift)
        self.assertGreaterEqual(
            self.report.positive_lift, MIN_POSITIVE_CONCORDANCE_LIFT
        )
        self.assertLessEqual(
            self.report.control_false_positive_rate,
            MAX_CONTROL_FALSE_POSITIVE_RATE,
        )
        self.assertGreaterEqual(self.report.cache_hit_rate, MIN_CACHE_HIT_RATE)
        self.assertTrue(self.report.passed)
        self.assertTrue(all(self.report.acceptance.values()))

    def test_rates_are_slice_level(self):
        # positives: 20 cases x 5 slices + 10 cases x 10 slices = 200 slices
        self.assertAlmostEqual(self.report.baseline_positive_concordance, 0.60)
        self.assertAlmostEqual(self.report.shadow_positive_concordance, 0.75)
        self.assertAlmostEqual(self.report.positive_lift, 0.25)
        # controls: 10 cases x 10 slices = 100 slices, 4 spurious slices
        self.assertAlmostEqual(self.report.control_false_positive_rate, 0.04)

    def test_strata_partition_the_cohort(self):
        self.assertTrue(self.report.strata)
        self.assertEqual(
            sum(s.positive_cases for s in self.report.strata),
            EXPECTED_POSITIVE_CASES,
        )
        self.assertEqual(
            sum(s.control_cases for s in self.report.strata),
            EXPECTED_CONTROL_CASES,
        )
        self.assertEqual(
            sum(s.positive_slices for s in self.report.strata),
            200,
        )
        keys = [s.stratum for s in self.report.strata]
        self.assertEqual(keys, sorted(keys))
        self.assertEqual(len(keys), len(set(keys)))

    def test_strata_carry_the_expected_membership(self):
        by_key = {s.stratum: s for s in self.report.strata}
        self.assertEqual(by_key[STRATUM_NONE].positive_cases, 10)
        self.assertEqual(by_key[STRATUM_NONE].control_cases, 4)
        self.assertEqual(by_key["small_area"].positive_cases, 6)
        self.assertEqual(by_key["small_area|border"].control_cases, 1)
        for summary in self.report.strata:
            self.assertLessEqual(summary.control_false_positive_rate, 1.0)
            self.assertLessEqual(summary.cache_hit_rate, 1.0)

    def test_every_hash_mismatch_is_registered_with_run9_b1(self):
        mismatched = [d for d in self.report.decisions if d.hash_mismatch]
        self.assertEqual(len(mismatched), _MISMATCH_TAIL)
        self.assertEqual(len(self.report.ledger), _MISMATCH_TAIL)
        ledger_ids = {entry.case_id for entry in self.report.ledger}
        self.assertEqual(ledger_ids, {d.case_id for d in mismatched})
        for entry in self.report.ledger:
            self.assertEqual(entry.extractor, FALLBACK_EXTRACTOR_TAG)
            self.assertEqual(entry.fallback_run, FALLBACK_EXTRACTOR_RUN)
            self.assertTrue(entry.hash_mismatch)
            self.assertEqual(entry.reason, "hash-mismatch")
        self.assertTrue(self.report.acceptance["mismatch_ledger"])

    def test_cache_hit_rate_counts_only_reusable_matching_entries(self):
        hits = [d for d in self.report.decisions if d.cache_hit]
        self.assertEqual(len(hits), 40 - _MISMATCH_TAIL)
        self.assertAlmostEqual(self.report.cache_hit_rate, 0.90)
        for decision in hits:
            self.assertIn(decision.used_run, REUSABLE_RUNS)

    def test_audit_does_not_mutate_its_input(self):
        fresh = _build_cohort()
        before = [dataclasses.asdict(case) for case in fresh]
        audit_shadow_cohort(fresh)
        after = [dataclasses.asdict(case) for case in fresh]
        self.assertEqual(before, after)
        self.assertEqual(len(fresh), 40)

    def test_report_is_read_only(self):
        self.assertTrue(self.report.readonly)
        self.assertFalse(self.report.writes_performed)
        self.assertFalse(self.report.labels_modified)
        self.assertFalse(self.report.rules_modified)
        self.assertFalse(self.report.thresholds_modified)


class AuditFailureModeTests(unittest.TestCase):
    def test_empty_input_is_rejected(self):
        with self.assertRaises(AuditInputError):
            audit_shadow_cohort([])

    def test_duplicate_case_ids_are_rejected(self):
        cases = _build_cohort()
        cases[-1] = dataclasses.replace(cases[-1], case_id=cases[0].case_id)
        with self.assertRaises(AuditInputError):
            audit_shadow_cohort(cases)

    def test_insufficient_lift_fails_acceptance_without_raising(self):
        cases = [
            dataclasses.replace(case, baseline_hits=case.shadow_hits)
            if case.cohort == "positive"
            else case
            for case in _build_cohort()
        ]
        report = audit_shadow_cohort(cases)
        self.assertEqual(report.positive_lift, 0.0)
        self.assertFalse(report.acceptance["positive_concordance_lift"])
        self.assertFalse(report.passed)

    def test_control_false_positive_spike_fails_acceptance(self):
        cases = [
            dataclasses.replace(case, shadow_hits=case.slices)
            if case.cohort == "control"
            else case
            for case in _build_cohort()
        ]
        report = audit_shadow_cohort(cases)
        self.assertGreater(
            report.control_false_positive_rate, MAX_CONTROL_FALSE_POSITIVE_RATE
        )
        self.assertFalse(report.acceptance["control_false_positive_rate"])
        self.assertFalse(report.passed)

    def test_low_cache_hit_rate_fails_acceptance(self):
        cases = [
            dataclasses.replace(case, expected_hash=case.expected_hash + "-drift")
            for case in _build_cohort()
        ]
        report = audit_shadow_cohort(cases)
        self.assertEqual(report.cache_hit_rate, 0.0)
        self.assertFalse(report.acceptance["cache_hit_rate"])
        self.assertEqual(len(report.ledger), 40)
        for entry in report.ledger:
            self.assertEqual(entry.extractor, FALLBACK_EXTRACTOR_TAG)
        self.assertFalse(report.passed)

    def test_wrong_cohort_shape_fails_acceptance(self):
        cases = _build_cohort()[:-1]
        report = audit_shadow_cohort(cases)
        self.assertFalse(report.acceptance["cohort_composition"])
        self.assertFalse(report.passed)


class ReadOnlySourceTests(unittest.TestCase):
    def test_module_source_has_no_write_or_process_escape_hatches(self):
        module = importlib.import_module(MODULE_NAME)
        source = open(module.__file__, "r", encoding="utf-8").read()
        for forbidden in ("subprocess", "os.system", "shutil", "socket"):
            self.assertNotIn(forbidden, source)
        with open(module.__file__, "r", encoding="utf-8") as handle:
            self.assertTrue(handle.read())


if __name__ == "__main__":
    unittest.main()
