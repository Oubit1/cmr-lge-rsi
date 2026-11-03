"""Offline unit tests for the class-level coverage-ledger registration helper."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments import coverage_ledger_class_entry as ledger_mod
from experiments.coverage_ledger_class_entry import (
    INVALIDATE,
    REGISTER,
    SKIP,
    backfill_run47,
    duplicate_reduction_ratio,
    entry_key,
    fingerprint,
    mi_fraction_high_entry,
    register_entry,
    rows,
    small_area_entry,
)

RULE_V1 = "cmr_myo_small_area.v1"
RULE_V2 = "cmr_myo_small_area.v2"
LGE_RULE_V3 = "lge_mi_fraction_high.v3"
CACHE_A = "cache:aaa"
CACHE_B = "cache:bbb"


class ClassEntryRegistrationTest(unittest.TestCase):
    def setUp(self):
        # Pin the local fingerprint rule so the shared-gate lookup is deterministic.
        ledger_mod.reset_shared_gate_cache(None, True)

    def tearDown(self):
        ledger_mod.reset_shared_gate_cache(None, False)

    # ------------------------------------------------------------------ identity
    def test_fingerprint_is_stable_across_key_order(self):
        first = {
            "entry_id": "cmr_myo_small_area",
            "class_id": "cmr_myo_small_area",
            "rule_version": RULE_V1,
            "cache_hash": CACHE_A,
        }
        second = {
            "cache_hash": CACHE_A,
            "class_id": "cmr_myo_small_area",
            "rule_version": RULE_V1,
            "entry_id": "cmr_myo_small_area",
        }
        self.assertEqual(fingerprint(first), fingerprint(second))

    # ---------------------------------------------------------------- skip path
    def test_unchanged_class_entry_is_skipped(self):
        entry = small_area_entry(RULE_V1, CACHE_A)
        ledger, first = register_entry({}, entry)
        self.assertEqual(first, REGISTER)

        ledger, second = register_entry(ledger, dict(entry))
        self.assertEqual(second, SKIP)
        self.assertEqual(len(rows(ledger)), 1)

    def test_skipped_entry_stays_recallable(self):
        entry = small_area_entry(RULE_V1, CACHE_A)
        ledger, _ = register_entry({}, entry)
        ledger, decision = register_entry(ledger, dict(entry))
        self.assertEqual(decision, SKIP)
        # The row is still present, so audit recall cannot drop because of the skip.
        kept = rows(ledger)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["entry_id"], "cmr_myo_small_area")
        self.assertEqual(kept[0]["cache_hash"], CACHE_A)
        self.assertIn(entry_key(entry), ledger)

    # ---------------------------------------------------------- invalidation
    def test_cache_hash_change_invalidates_ledger(self):
        ledger, _ = register_entry({}, small_area_entry(RULE_V1, CACHE_A))
        ledger, decision = register_entry(ledger, small_area_entry(RULE_V1, CACHE_B))
        self.assertEqual(decision, INVALIDATE)
        kept = rows(ledger)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["cache_hash"], CACHE_B)

    def test_rule_version_change_invalidates_ledger(self):
        ledger, _ = register_entry({}, small_area_entry(RULE_V1, CACHE_A))
        ledger, decision = register_entry(ledger, small_area_entry(RULE_V2, CACHE_A))
        self.assertEqual(decision, INVALIDATE)
        kept = rows(ledger)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["rule_version"], RULE_V2)

    def test_shared_gate_can_force_reregistration(self):
        ledger_mod.reset_shared_gate_cache(lambda entry, previous: False, True)
        entry = small_area_entry(RULE_V1, CACHE_A)
        ledger, _ = register_entry({}, entry)
        ledger, decision = register_entry(ledger, dict(entry))
        self.assertEqual(decision, INVALIDATE)

    def test_different_class_is_not_treated_as_duplicate(self):
        ledger, _ = register_entry({}, small_area_entry(RULE_V1, CACHE_A))
        ledger, decision = backfill_run47(
            ledger, [mi_fraction_high_entry(LGE_RULE_V3, CACHE_A)]
        )[0], None
        self.assertIsNone(decision)
        self.assertEqual(len(rows(ledger)), 2)

    # ---------------------------------------------------------------- backfill
    def test_backfill_run47_emits_each_open_entry_once(self):
        repeated = [mi_fraction_high_entry(LGE_RULE_V3, CACHE_A) for _ in range(3)]
        ledger, decisions = backfill_run47({}, repeated)
        verdicts = [verdict for _, verdict in decisions]
        self.assertEqual(verdicts[0], REGISTER)
        self.assertEqual(verdicts[1:], [SKIP, SKIP])
        self.assertEqual(len(rows(ledger)), 1)

    def test_backfill_keeps_existing_class_entry(self):
        ledger, _ = register_entry({}, small_area_entry(RULE_V1, CACHE_A))
        ledger, decisions = backfill_run47(
            ledger, [mi_fraction_high_entry(LGE_RULE_V3, CACHE_A)]
        )
        self.assertEqual(decisions[0][1], REGISTER)
        keys = set(ledger.keys())
        self.assertIn(("cmr_myo_small_area", "cmr_myo_small_area"), keys)
        self.assertIn(("lge_mi_fraction_high", "lge_mi_fraction_high"), keys)

    # ----------------------------------------------------------- acceptance
    def test_duplicate_rows_drop_at_least_90_percent(self):
        baseline = [dict(small_area_entry(RULE_V1, CACHE_A)) for _ in range(20)]
        baseline += [dict(mi_fraction_high_entry(LGE_RULE_V3, CACHE_A)) for _ in range(20)]

        ledger = {}
        for row in baseline:
            ledger, _ = register_entry(ledger, row)

        current = rows(ledger)
        self.assertEqual(len(current), 2)
        self.assertGreaterEqual(duplicate_reduction_ratio(baseline, current), 0.9)

    def test_reduction_ratio_with_empty_baseline_is_zero(self):
        self.assertEqual(duplicate_reduction_ratio([], []), 0.0)


if __name__ == "__main__":
    unittest.main()
