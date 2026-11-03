"""Offline unittests for the cmr_myo_small_area paired shadow audit.

All data is synthetic and in-memory; nothing is downloaded or persisted.
"""

import unittest

from experiments.cmr_myo_small_area_paired_shadow_audit import (
    CONTROL,
    POSITIVE,
    CaseRecord,
    expected_cache_hash,
    pair_by_metadata,
    run_audit,
)


def make_case(case_id, group, slice_index, phase, run, area, flags=(), bad_hash=False):
    cache_hash = expected_cache_hash(case_id, run, area)
    if bad_hash:
        cache_hash = "0" * 64
    return CaseRecord(
        case_id=case_id,
        group=group,
        slice_index=slice_index,
        phase=phase,
        source_run=run,
        cached_area=area,
        cache_hash=cache_hash,
        support_flags=tuple(flags),
    )


def build_cohort():
    """50 positives (run-13 cache) + 10 controls (run-9 cache)."""
    positives = []
    for i in range(50):
        flags = ("small_area",) if i % 2 == 0 else ()
        phase = "ED" if i % 2 else "ES"
        positives.append(
            make_case(
                "pos{:02d}".format(i), POSITIVE, i % 10, phase, 13,
                400.0 + i, flags=flags,
            )
        )
    controls = [
        make_case("ctl{:02d}".format(j), CONTROL, j, "ED", 9, 900.0 + j)
        for j in range(10)
    ]
    return positives, controls


class TestPairing(unittest.TestCase):
    def test_pairs_all_controls_one_to_one(self):
        positives, controls = build_cohort()
        pairs = pair_by_metadata(positives, controls)
        self.assertEqual(len(pairs), 10)
        self.assertEqual(len({p["control_id"] for p in pairs}), 10)
        self.assertEqual(len({p["positive_id"] for p in pairs}), 10)
        for pair in pairs:
            self.assertEqual(pair["phase"], "ED")

    def test_slice_only_fallback_pairing(self):
        pos = [make_case("p", POSITIVE, 3, "ES", 13, 100.0)]
        ctl = [make_case("c", CONTROL, 3, "ED", 9, 200.0)]
        pairs = pair_by_metadata(pos, ctl)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["positive_id"], "p")
        self.assertEqual(pairs[0]["slice_index"], 3)

    def test_unpairable_control_is_skipped(self):
        pos = [make_case("p", POSITIVE, 1, "ED", 13, 100.0)]
        ctl = [make_case("c", CONTROL, 9, "ED", 9, 200.0)]
        self.assertEqual(pair_by_metadata(pos, ctl), [])


class TestAudit(unittest.TestCase):
    def test_full_cohort_meets_acceptance(self):
        positives, controls = build_cohort()
        report = run_audit(positives, controls)
        self.assertEqual(report["total_cases"], 60)
        self.assertEqual(report["audited_cases"], 60)
        self.assertGreaterEqual(report["completion_rate"], 0.95)
        self.assertTrue(report["completion_target_met"])
        self.assertEqual(report["positive_support_flag_fraction"], 0.5)
        self.assertEqual(report["control_support_flag_fraction"], 0.0)
        self.assertAlmostEqual(report["support_flag_fraction_difference"], 0.5)
        self.assertLess(report["area_median_separation"], 0.0)
        self.assertEqual(report["pair_count"], 10)
        self.assertEqual(report["fallback_count"], 0)
        self.assertEqual(report["ledger"], [])
        self.assertTrue(report["read_only"])

    def test_hash_mismatch_uses_run9_b1_fallback_and_registers(self):
        positives, controls = build_cohort()
        positives[0] = make_case(
            "pos00", POSITIVE, 0, "ES", 5, 410.0, bad_hash=True
        )
        report = run_audit(positives, controls, fallback_extractor=lambda c: 123.0)
        self.assertEqual(report["audited_cases"], 60)
        self.assertEqual(report["fallback_count"], 1)
        entry = report["ledger"][0]
        self.assertEqual(entry["case_id"], "pos00")
        self.assertEqual(entry["reason"], "hash_mismatch")
        self.assertEqual(entry["fallback"], "run9/B1")

    def test_disallowed_run_triggers_fallback(self):
        case = make_case("x", POSITIVE, 0, "ED", 7, 100.0)
        report = run_audit([case], [], fallback_extractor=lambda c: 55.0)
        self.assertEqual(report["fallback_count"], 1)
        self.assertEqual(report["ledger"][0]["reason"], "run_not_allowed")
        self.assertEqual(report["positive_area_median"], 55.0)

    def test_missing_cache_without_fallback_lowers_completion(self):
        positives, controls = build_cohort()
        positives[0] = make_case(
            "pos00", POSITIVE, 0, "ES", 13, 400.0, bad_hash=True
        )
        report = run_audit(positives, controls)  # no fallback extractor
        self.assertEqual(report["audited_cases"], 59)
        self.assertAlmostEqual(report["completion_rate"], 59 / 60, places=4)
        self.assertEqual(report["fallback_count"], 0)
        self.assertEqual(report["ledger"][0]["fallback"], "unavailable")

    def test_read_only_inputs_unchanged(self):
        positives, controls = build_cohort()
        snapshot = list(positives) + list(controls)
        run_audit(positives, controls, fallback_extractor=lambda c: 1.0)
        for before, after in zip(snapshot, list(positives) + list(controls)):
            self.assertIs(before, after)


if __name__ == "__main__":
    unittest.main()
