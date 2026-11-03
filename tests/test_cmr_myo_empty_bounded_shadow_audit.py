"""Offline unit tests for the bounded cmr_myo_empty shadow audit.

Every fixture is synthesised in memory: no dataset, no cache directory and no
network access is required.
"""

import copy
import hashlib
import unittest
from collections import Counter

from experiments.cmr_myo_empty_bounded_shadow_audit import (
    B1_EXTRACTOR_RUN,
    CACHE_RUNS,
    CONTROL_ARM,
    CONTROL_SAMPLE_SIZE,
    MAX_CONTROL_FALSE_POSITIVE_RATE,
    MIN_CACHE_REUSE_RATE,
    POSITIVE_ARM,
    POSITIVE_SAMPLE_SIZE,
    band_for_rate,
    format_report,
    run_bounded_shadow_audit,
    select_bounded_cohort,
)


def _digest(*parts):
    return hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def make_case(
    case_id,
    arm,
    *,
    run=5,
    slice_vacuum=True,
    content_hash=None,
    b1_slice_vacuum=None,
    with_cache=True,
):
    expected = _digest(case_id, arm, "expected")
    case = {"case_id": case_id, "arm": arm, "expected_hash": expected}
    if with_cache:
        case["cached_summary"] = {
            "run": run,
            "content_hash": expected if content_hash is None else content_hash,
            "slice_vacuum": slice_vacuum,
        }
    if b1_slice_vacuum is not None:
        case["b1_slice_vacuum"] = b1_slice_vacuum
    return case


def build_pool(n_pos=POSITIVE_SAMPLE_SIZE, n_ctl=CONTROL_SAMPLE_SIZE, *, positive_vacuum=27):
    """30 cached positives (27 valid) + 10 cached controls (0 valid) by default."""
    cases = []
    for index in range(n_pos):
        cases.append(
            make_case(
                f"P{index:04d}",
                POSITIVE_ARM,
                run=CACHE_RUNS[index % len(CACHE_RUNS)],
                slice_vacuum=index < positive_vacuum,
            )
        )
    for index in range(n_ctl):
        cases.append(
            make_case(
                f"C{index:04d}",
                CONTROL_ARM,
                run=CACHE_RUNS[index % len(CACHE_RUNS)],
                slice_vacuum=False,
            )
        )
    return cases


def case_by_id(cases, case_id):
    return next(case for case in cases if case["case_id"] == case_id)


class TestBoundedSelection(unittest.TestCase):
    def test_selects_exact_bounded_cohort(self):
        pool = build_pool(45, 20)
        cohort = select_bounded_cohort(pool)
        counts = Counter(case["arm"] for case in cohort)
        self.assertEqual(counts[POSITIVE_ARM], POSITIVE_SAMPLE_SIZE)
        self.assertEqual(counts[CONTROL_ARM], CONTROL_SAMPLE_SIZE)
        ids = [case["case_id"] for case in cohort]
        self.assertEqual(len(set(ids)), len(ids))
        self.assertTrue(set(ids).issubset({case["case_id"] for case in pool}))

    def test_selection_is_deterministic_for_the_same_pool(self):
        pool = build_pool(45, 20)
        first = [case["case_id"] for case in select_bounded_cohort(pool)]
        second = [case["case_id"] for case in select_bounded_cohort(pool)]
        self.assertEqual(first, second)

    def test_raises_when_a_stratum_is_too_small(self):
        with self.assertRaises(ValueError):
            select_bounded_cohort(build_pool(10, 10))
        with self.assertRaises(ValueError):
            select_bounded_cohort(build_pool(30, 3))


class TestCacheReplay(unittest.TestCase):
    def test_full_cache_hit_reports_reuse_and_band(self):
        report = run_bounded_shadow_audit(build_pool())
        self.assertEqual(report.positives_selected, 30)
        self.assertEqual(report.controls_selected, 10)
        self.assertEqual(report.cache_hits, 40)
        self.assertEqual(report.cache_misses, 0)
        self.assertAlmostEqual(report.cache_reuse_rate, 1.0)
        self.assertTrue(report.cache_reuse_ok)
        self.assertEqual(report.hash_mismatches, ())
        self.assertEqual(report.positive_resolved, 30)
        self.assertEqual(report.positive_valid_slice_vacuum, 27)
        self.assertAlmostEqual(report.positive_valid_ratio, 0.9)
        self.assertEqual(report.positive_band, "75-100%")
        self.assertEqual(report.control_false_positives, 0)
        self.assertTrue(report.control_false_positive_ok)
        self.assertTrue(report.read_only)

    def test_cache_miss_is_not_reused_and_is_reported(self):
        pool = build_pool()
        target = case_by_id(pool, "P0000")
        del target["cached_summary"]
        report = run_bounded_shadow_audit(pool)
        self.assertEqual(report.cache_hits, 39)
        self.assertEqual(report.cache_misses, 1)
        self.assertEqual(report.hash_mismatches, ())
        self.assertTrue(any("P0000" in note for note in report.notes))
        self.assertTrue(report.cache_reuse_ok)

    def test_reuse_below_floor_fails_acceptance(self):
        pool = build_pool()
        for case in pool[:9]:
            del case["cached_summary"]
        report = run_bounded_shadow_audit(pool)
        self.assertEqual(report.cache_hits, 31)
        self.assertLess(report.cache_reuse_rate, MIN_CACHE_REUSE_RATE)
        self.assertFalse(report.cache_reuse_ok)

    def test_run_outside_reuse_set_is_a_miss_not_a_mismatch(self):
        pool = build_pool()
        case_by_id(pool, "P0000")["cached_summary"]["run"] = 7
        report = run_bounded_shadow_audit(pool)
        self.assertEqual(report.cache_misses, 1)
        self.assertEqual(report.cache_hits, 39)
        self.assertEqual(report.hash_mismatches, ())
        self.assertIn("run 7 outside reuse set", report.notes[0])

    def test_injected_cache_loader_overrides_inline_summary(self):
        pool = build_pool()
        external = {case["case_id"]: case["cached_summary"] for case in pool}
        for case in pool:
            del case["cached_summary"]
        report = run_bounded_shadow_audit(
            pool, cache_loader=lambda case: external.get(case["case_id"])
        )
        self.assertEqual(report.cache_hits, 40)
        self.assertEqual(report.hash_mismatches, ())


class TestHashMismatchHandling(unittest.TestCase):
    def test_mismatch_uses_run9_b1_only_and_is_reported_per_case(self):
        pool = build_pool()
        target = case_by_id(pool, "P0028")
        target["cached_summary"]["content_hash"] = _digest("stale")
        target["cached_summary"]["slice_vacuum"] = False
        target["b1_slice_vacuum"] = True
        report = run_bounded_shadow_audit(pool)
        self.assertEqual(len(report.hash_mismatches), 1)
        line = report.hash_mismatches[0]
        self.assertIn("P0028", line)
        self.assertIn(f"run-{B1_EXTRACTOR_RUN}", line)
        self.assertEqual(report.b1_recovered, 1)
        self.assertEqual(report.cache_hits, 39)
        self.assertEqual(report.positive_valid_slice_vacuum, 28)
        self.assertEqual(
            report.cache_hits + report.cache_misses + len(report.hash_mismatches), 40
        )

    def test_mismatch_without_b1_determination_is_unresolved(self):
        pool = build_pool()
        target = case_by_id(pool, "P0028")
        target["cached_summary"]["content_hash"] = _digest("stale")
        report = run_bounded_shadow_audit(pool)
        self.assertEqual(len(report.hash_mismatches), 1)
        self.assertEqual(report.b1_recovered, 0)
        self.assertEqual(report.positive_resolved, 29)
        self.assertEqual(report.positive_valid_slice_vacuum, 27)
        self.assertTrue(any("no determination" in note for note in report.notes))

    def test_injected_b1_extractor_called_only_for_mismatches(self):
        pool = build_pool()
        target = case_by_id(pool, "P0028")
        target["cached_summary"]["content_hash"] = _digest("stale")
        calls = []

        def b1_extractor(case):
            calls.append(case["case_id"])
            return True

        report = run_bounded_shadow_audit(pool, b1_extractor=b1_extractor)
        self.assertEqual(calls, ["P0028"])
        self.assertEqual(report.b1_recovered, 1)
        self.assertEqual(report.cache_hits, 39)

    def test_mismatches_are_never_silently_downgraded(self):
        pool = build_pool()
        for case_id in ("P0001", "P0002", "C0003"):
            case_by_id(pool, case_id)["cached_summary"]["content_hash"] = _digest(
                "stale", case_id
            )
        report = run_bounded_shadow_audit(pool)
        self.assertEqual(len(report.hash_mismatches), 3)
        for case_id in ("P0001", "P0002", "C0003"):
            self.assertTrue(any(case_id in line for line in report.hash_mismatches))


class TestControlFalsePositiveGate(unittest.TestCase):
    def test_ten_percent_control_fpr_passes(self):
        pool = build_pool()
        case_by_id(pool, "C0000")["cached_summary"]["slice_vacuum"] = True
        report = run_bounded_shadow_audit(pool)
        self.assertEqual(report.control_resolved, 10)
        self.assertEqual(report.control_false_positives, 1)
        self.assertAlmostEqual(report.control_false_positive_rate, 0.1)
        self.assertLessEqual(report.control_false_positive_rate, MAX_CONTROL_FALSE_POSITIVE_RATE)
        self.assertTrue(report.control_false_positive_ok)

    def test_above_ten_percent_control_fpr_fails(self):
        pool = build_pool()
        for case_id in ("C0000", "C0001"):
            case_by_id(pool, case_id)["cached_summary"]["slice_vacuum"] = True
        report = run_bounded_shadow_audit(pool)
        self.assertEqual(report.control_false_positives, 2)
        self.assertAlmostEqual(report.control_false_positive_rate, 0.2)
        self.assertFalse(report.control_false_positive_ok)

    def test_unresolved_controls_do_not_pass_vacuously(self):
        pool = build_pool()
        for case in pool:
            if case["arm"] == CONTROL_ARM:
                del case["cached_summary"]
        report = run_bounded_shadow_audit(pool)
        self.assertEqual(report.control_resolved, 0)
        self.assertFalse(report.control_false_positive_ok)
        self.assertTrue(any("not evaluable" in note for note in report.notes))


class TestReadOnlyEnvelope(unittest.TestCase):
    def test_inputs_are_never_mutated(self):
        pool = build_pool()
        snapshot = copy.deepcopy(pool)
        report = run_bounded_shadow_audit(pool)
        self.assertEqual(pool, snapshot)
        self.assertTrue(report.read_only)

    def test_inputs_are_never_mutated_even_with_mismatches(self):
        pool = build_pool()
        case_by_id(pool, "P0028")["cached_summary"]["content_hash"] = _digest("stale")
        snapshot = copy.deepcopy(pool)
        report = run_bounded_shadow_audit(pool)
        self.assertEqual(pool, snapshot)
        self.assertTrue(report.read_only)

    def test_report_exposes_bands_without_new_thresholds(self):
        payload = run_bounded_shadow_audit(build_pool()).as_dict()
        self.assertIn(payload["positive_band"], {"0-25%", "25-50%", "50-75%", "75-100%"})
        self.assertFalse(any("threshold" in key for key in payload))


class TestRateBands(unittest.TestCase):
    def test_bands_are_half_open_intervals(self):
        self.assertEqual(band_for_rate(None), "unresolved")
        self.assertEqual(band_for_rate(0.0), "0-25%")
        self.assertEqual(band_for_rate(0.249), "0-25%")
        self.assertEqual(band_for_rate(0.25), "25-50%")
        self.assertEqual(band_for_rate(0.5), "50-75%")
        self.assertEqual(band_for_rate(0.75), "75-100%")
        self.assertEqual(band_for_rate(1.0), "75-100%")

    def test_out_of_range_rate_is_rejected(self):
        with self.assertRaises(ValueError):
            band_for_rate(1.5)


class TestReportText(unittest.TestCase):
    def test_each_mismatch_surfaces_as_one_line(self):
        pool = build_pool()
        for case_id in ("P0001", "P0002"):
            case = case_by_id(pool, case_id)
            case["cached_summary"]["content_hash"] = _digest("stale", case_id)
            case["b1_slice_vacuum"] = True
        report = run_bounded_shadow_audit(pool)
        text = format_report(report)
        self.assertEqual(len(report.hash_mismatches), 2)
        self.assertEqual(text.count("HASH MISMATCH"), 2)
        self.assertIn("read-only: True", text)
        self.assertIn("band=", text)


if __name__ == "__main__":
    unittest.main()
