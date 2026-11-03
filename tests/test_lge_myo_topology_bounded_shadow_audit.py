"""Offline unittest coverage for the bounded LGE myo_topology shadow audit.

All cases are synthetic and injected through the module's public callables,
so no cache files, patient data or network access are required.
"""
from __future__ import annotations

import hashlib
import unittest

from experiments.lge_myo_topology_bounded_shadow_audit import (
    AUDIT_CONTROL_N,
    AUDIT_POSITIVE_N,
    MAX_FALLBACK_EXTRACTIONS,
    MIN_CACHE_HIT_RATE,
    MIN_PPV,
    CacheEntry,
    bounded_select,
    run_bounded_shadow_audit,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class _FakeCache:
    def __init__(self, entries_by_case):
        self.entries_by_case = entries_by_case

    def __call__(self, case_id):
        return self.entries_by_case.get(case_id, ())


def _entry(case_id, run_id, prediction, *, digest=None, raw=None):
    return CacheEntry(
        case_id=case_id,
        run_id=run_id,
        artifact_sha256=digest if digest is not None else _sha(raw or f"{case_id}:{run_id}"),
        prediction=prediction,
    )


class BoundedSelectTests(unittest.TestCase):
    def test_selects_exact_counts_and_is_deterministic(self):
        positives = [f"pos-{i:03d}" for i in range(120)]
        controls = [f"ctl-{i:03d}" for i in range(40)]
        sel_pos_1 = bounded_select(positives, AUDIT_POSITIVE_N, seed="audit")
        sel_pos_2 = bounded_select(positives, AUDIT_POSITIVE_N, seed="audit")
        sel_ctl_1 = bounded_select(controls, AUDIT_CONTROL_N, seed="audit")
        self.assertEqual(len(sel_pos_1), AUDIT_POSITIVE_N)
        self.assertEqual(len(sel_ctl_1), AUDIT_CONTROL_N)
        self.assertEqual(sel_pos_1, sel_pos_2)
        self.assertEqual(len(set(sel_pos_1)), AUDIT_POSITIVE_N)
        self.assertEqual(len(set(sel_ctl_1)), AUDIT_CONTROL_N)

    def test_different_seed_changes_sample(self):
        cases = [f"c-{i:03d}" for i in range(100)]
        a = bounded_select(cases, 10, seed="a")
        b = bounded_select(cases, 10, seed="b")
        self.assertNotEqual(a, b)


class ShadowAuditMetricTests(unittest.TestCase):
    def _cases(self):
        positives = [f"pos-{i:03d}" for i in range(AUDIT_POSITIVE_N)]
        controls = [f"ctl-{i:03d}" for i in range(AUDIT_CONTROL_N)]
        return positives, controls

    def _run(self, positives, controls, ground_truth, entries, **kwargs):
        return run_bounded_shadow_audit(
            positive_case_ids=positives,
            control_case_ids=controls,
            ground_truth=ground_truth,
            cache_lookup=_FakeCache(entries),
            fallback_extractor=kwargs.pop("fallback_extractor", lambda cid: False),
            **kwargs,
        )

    def test_passing_audit_meets_all_acceptance_metrics(self):
        positives, controls = self._cases()
        ground_truth = {cid: (i < 40) for i, cid in enumerate(positives)}  # PPV 0.80
        digest = _sha("run-4-artifact")
        entries = {}
        for cid in positives:
            entries[cid] = (_entry(cid, "run-4", True, digest=digest),)
        for cid in controls:
            entries[cid] = (_entry(cid, "run-4", False, digest=digest),)
        report = self._run(
            positives,
            controls,
            ground_truth,
            entries,
            expected_hashes={"run-4": digest},
        )
        self.assertAlmostEqual(report.ppv, 0.80)
        self.assertEqual(report.control_false_positives, 0)
        self.assertEqual(report.cache_hits, 60)
        self.assertEqual(report.cache_misses, 0)
        self.assertEqual(report.fallback_extractions, 0)
        self.assertGreaterEqual(report.cache_hit_rate, MIN_CACHE_HIT_RATE)
        self.assertTrue(report.passes())

    def test_low_ppv_fails_acceptance(self):
        positives, controls = self._cases()
        ground_truth = {cid: (i < 30) for i, cid in enumerate(positives)}  # PPV 0.60
        digest = _sha("run-4-artifact")
        entries = {}
        for cid in positives:
            entries[cid] = (_entry(cid, "run-4", True, digest=digest),)
        for cid in controls:
            entries[cid] = (_entry(cid, "run-4", False, digest=digest),)
        report = self._run(
            positives,
            controls,
            ground_truth,
            entries,
            expected_hashes={"run-4": digest},
        )
        self.assertAlmostEqual(report.ppv, 0.60)
        self.assertLess(report.ppv, MIN_PPV)
        self.assertFalse(report.passes_ppv)
        self.assertFalse(report.passes())

    def test_two_control_false_positives_fail_acceptance(self):
        positives, controls = self._cases()
        ground_truth = {cid: True for cid in positives}
        digest = _sha("run-4-artifact")
        entries = {}
        for cid in positives:
            entries[cid] = (_entry(cid, "run-4", True, digest=digest),)
        for i, cid in enumerate(controls):
            entries[cid] = (_entry(cid, "run-4", i < 2, digest=digest),)
        report = self._run(
            positives,
            controls,
            ground_truth,
            entries,
            expected_hashes={"run-4": digest},
        )
        self.assertEqual(report.control_false_positives, 2)
        self.assertFalse(report.passes_control_false_positive_budget)
        self.assertFalse(report.passes())

    def test_cache_miss_uses_fallback_and_reports_ids(self):
        positives, controls = self._cases()
        ground_truth = {cid: True for cid in positives}
        digest = _sha("run-4-artifact")
        entries = {}
        for i, cid in enumerate(positives):
            if i < 5:
                continue
            entries[cid] = (_entry(cid, "run-4", True, digest=digest),)
        for cid in controls:
            entries[cid] = (_entry(cid, "run-4", False, digest=digest),)
        fallback_calls = []

        def fallback(cid):
            fallback_calls.append(cid)
            return True

        report = self._run(
            positives,
            controls,
            ground_truth,
            entries,
            fallback_extractor=fallback,
            expected_hashes={"run-4": digest},
        )
        self.assertEqual(report.fallback_extractions, 5)
        self.assertEqual(report.cache_misses, 5)
        self.assertEqual(report.cache_hits, 55)
        self.assertGreaterEqual(report.cache_hit_rate, MIN_CACHE_HIT_RATE)
        self.assertTrue(report.passes_fallback_budget)
        self.assertEqual(len(fallback_calls), 5)
        self.assertEqual(set(report.fallback_case_ids), set(fallback_calls))
        self.assertEqual(report.fallback_run_id, "run-9")
        self.assertEqual(report.fallback_extractor_id, "B1")

    def test_more_than_five_fallbacks_fail_budget(self):
        positives, controls = self._cases()
        ground_truth = {cid: True for cid in positives}
        digest = _sha("run-4-artifact")
        entries = {}
        for i, cid in enumerate(positives):
            if i < 10:
                continue
            entries[cid] = (_entry(cid, "run-4", True, digest=digest),)
        for cid in controls:
            entries[cid] = (_entry(cid, "run-4", False, digest=digest),)
        report = self._run(
            positives,
            controls,
            ground_truth,
            entries,
            expected_hashes={"run-4": digest},
            fallback_extractor=lambda cid: True,
        )
        self.assertEqual(report.fallback_extractions, 10)
        self.assertGreater(report.fallback_extractions, MAX_FALLBACK_EXTRACTIONS)
        self.assertFalse(report.passes_fallback_budget)
        self.assertFalse(report.passes())

    def test_hash_mismatch_falls_back_to_run9_b1_and_reports(self):
        positives, controls = self._cases()
        ground_truth = {cid: True for cid in positives}
        good = _sha("run-4-artifact")
        bad = _sha("tampered")
        entries = {}
        for cid in positives:
            entries[cid] = (_entry(cid, "run-4", True, digest=bad),)
        for cid in controls:
            entries[cid] = (_entry(cid, "run-4", False, digest=good),)
        fallback_calls = []

        def fallback(cid):
            fallback_calls.append(cid)
            return True

        report = self._run(
            positives,
            controls,
            ground_truth,
            entries,
            fallback_extractor=fallback,
            expected_hashes={"run-4": good},
        )
        self.assertEqual(report.fallback_extractions, AUDIT_POSITIVE_N)
        self.assertEqual(set(report.hash_mismatch_case_ids), set(positives))
        self.assertEqual(report.fallback_run_id, "run-9")
        self.assertEqual(report.fallback_extractor_id, "B1")
        self.assertFalse(report.passes_fallback_budget)
        self.assertAlmostEqual(report.cache_hit_rate, AUDIT_CONTROL_N / 60)

    def test_hash_mismatch_skips_entry_and_uses_later_valid_entry(self):
        positives, controls = self._cases()
        ground_truth = {cid: (i < 40) for i, cid in enumerate(positives)}
        good = _sha("run-5-artifact")
        bad = _sha("run-4-artifact")
        entries = {}
        expected = {"run-5": good}
        for i, cid in enumerate(positives):
            if cid == positives[0]:
                entries[cid] = (
                    _entry(cid, "run-4", True, digest=bad),
                    _entry(cid, "run-5", True, digest=good),
                )
            else:
                entries[cid] = (_entry(cid, "run-5", True, digest=good),)
        for cid in controls:
            entries[cid] = (_entry(cid, "run-5", False, digest=good),)
        fallback_calls = []

        report = self._run(
            positives,
            controls,
            ground_truth,
            entries,
            fallback_extractor=lambda cid: fallback_calls.append(cid) or False,
            expected_hashes=expected,
        )
        self.assertIn(positives[0], report.hash_mismatch_case_ids)
        self.assertNotIn(positives[0], report.fallback_case_ids)
        self.assertEqual(report.fallback_extractions, 0)
        self.assertEqual(report.cache_hits, 60)
        self.assertTrue(report.passes())

    def test_cache_hit_rate_below_threshold_fails(self):
        positives, controls = self._cases()
        ground_truth = {cid: True for cid in positives}
        digest = _sha("run-4-artifact")
        entries = {}
        for i, cid in enumerate(positives):
            if i < 15:
                continue
            entries[cid] = (_entry(cid, "run-4", True, digest=digest),)
        for i, cid in enumerate(controls):
            if i < 5:
                continue
            entries[cid] = (_entry(cid, "run-4", False, digest=digest),)
        report = self._run(
            positives,
            controls,
            ground_truth,
            entries,
            expected_hashes={"run-4": digest},
            fallback_extractor=lambda cid: False,
        )
        self.assertEqual(report.cache_hits, 40)
        self.assertLess(report.cache_hit_rate, MIN_CACHE_HIT_RATE)
        self.assertFalse(report.passes_cache_hit_rate)
        self.assertFalse(report.passes())

    def test_rejects_wrong_sample_sizes(self):
        with self.assertRaises(ValueError):
            run_bounded_shadow_audit(
                positive_case_ids=[f"p{i}" for i in range(AUDIT_POSITIVE_N - 1)],
                control_case_ids=[f"c{i}" for i in range(AUDIT_CONTROL_N)],
                ground_truth={},
                cache_lookup=lambda cid: (),
                fallback_extractor=lambda cid: False,
            )
        with self.assertRaises(ValueError):
            run_bounded_shadow_audit(
                positive_case_ids=[f"p{i}" for i in range(AUDIT_POSITIVE_N)],
                control_case_ids=[f"c{i}" for i in range(AUDIT_CONTROL_N - 1)],
                ground_truth={},
                cache_lookup=lambda cid: (),
                fallback_extractor=lambda cid: False,
            )


if __name__ == "__main__":
    unittest.main()
