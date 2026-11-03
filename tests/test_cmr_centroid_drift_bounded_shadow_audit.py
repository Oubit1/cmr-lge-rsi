"""Offline unit tests for the bounded cmr_centroid_drift shadow audit.

No dataset, network or filesystem access is required: the cache loader, hash
probe and run-9 B1 fallback extractor are all faked in memory.
"""

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.cmr_centroid_drift_bounded_shadow_audit import (  # noqa: E402
    CONTROL_SAMPLE_SIZE,
    FALLBACK_BUDGET,
    FALLBACK_EXTRACTOR,
    FALLBACK_RUN,
    MIN_CACHE_REUSE_RATE,
    MIN_PPV,
    POSITIVE_SAMPLE_SIZE,
    RUN_IDS,
    run_shadow_audit,
    summarize_for_reviewer,
    verify_read_only_contract,
)

POSITIVES = ["P%02d" % index for index in range(POSITIVE_SAMPLE_SIZE)]
CONTROLS = ["C%02d" % index for index in range(CONTROL_SAMPLE_SIZE)]


def hashes_for_all():
    return {case_id: "hash::%s" % case_id for case_id in POSITIVES + CONTROLS}


def build_entries(*, stale=(), rescue=(), confirmed=(), review=(), flags=()):
    """Build run summaries for the 40 sampled cases.

    ``stale``   - cases whose run-4 cached hash does not match the observed hash
    ``rescue``  - cases that also carry a hash-matching run-5 summary (which can
                  rescue a stale run-4 entry so no fallback is needed)
    ``confirmed`` - cases whose summary confirms real centroid drift
    ``review``  - cases flagged by review_centroid_outlier
    ``flags``   - cases carrying a centroid_drift flag in a cached summary
    """

    stale = set(stale)
    rescue = set(rescue)
    confirmed = set(confirmed)
    review = set(review)
    flags = set(flags)
    hashes = hashes_for_all()
    entries = {}

    for case_id in POSITIVES + CONTROLS:
        summary = {
            "confirmed_drift": case_id in confirmed,
            "review_centroid_outlier": case_id in review,
            "centroid_drift_flag": case_id in flags,
        }
        # A stale case keeps a mismatched run-4 hash even when a later run
        # rescues it; the rescue only adds a hash-matching run-5 summary.
        if case_id in stale:
            stored_hash = "stale::%s" % case_id
        else:
            stored_hash = hashes[case_id]
        entries[("run-4", case_id)] = {"stored_hash": stored_hash, "summary": dict(summary)}
        if case_id in rescue:
            entries[("run-5", case_id)] = {
                "stored_hash": hashes[case_id],
                "summary": dict(summary),
            }

    return entries, hashes


def make_harness(entries, hashes, fallback_results=None, fallback_calls=None):
    def cache_loader(run_id, case_id):
        return entries.get((run_id, case_id))

    def hash_probe(case_id):
        return hashes.get(case_id)

    fallback_results = fallback_results or {}

    def fallback_extractor(case_id, run_id, extractor_id):
        if fallback_calls is not None:
            fallback_calls.append((case_id, run_id, extractor_id))
        return dict(
            fallback_results.get(
                case_id,
                {"confirmed_drift": False, "review_centroid_outlier": False},
            )
        )

    return cache_loader, hash_probe, fallback_extractor


class CmrCentroidDriftBoundedShadowAuditTest(unittest.TestCase):
    def test_full_cache_reuse_and_acceptance_passes(self):
        confirmed = POSITIVES[:24]
        entries, hashes = build_entries(confirmed=confirmed, flags=confirmed)
        cache_loader, hash_probe, fallback = make_harness(entries, hashes)

        report = run_shadow_audit(
            POSITIVES, CONTROLS, cache_loader=cache_loader, hash_probe=hash_probe,
            fallback_extractor=fallback,
        )

        self.assertEqual(report["sampling"]["n_sampled"], 40)
        self.assertEqual(report["sampling"]["positive"], POSITIVE_SAMPLE_SIZE)
        self.assertEqual(report["sampling"]["control"], CONTROL_SAMPLE_SIZE)
        self.assertEqual(report["cache_reuse_rate"], 1.0)
        self.assertEqual(report["cache_reuse_count"], 40)
        self.assertEqual(report["fallback_count"], 0)
        self.assertEqual(report["hash_mismatch_count"], 0)
        self.assertAlmostEqual(report["ppv"], 0.8, places=4)
        self.assertEqual(report["control_false_positive_rate"], 0.0)
        self.assertTrue(report["acceptance_passed"])
        self.assertTrue(report["acceptance"]["ppv_at_least_0_6"])
        self.assertTrue(report["acceptance"]["cache_reuse_at_least_0_8"])
        self.assertTrue(report["acceptance"]["fallback_within_budget"])
        self.assertEqual(len(report["evidence"]), 40)
        self.assertEqual(report["reused_runs"], list(RUN_IDS))

    def test_hash_mismatch_falls_back_to_run9_b1_only(self):
        stale = POSITIVES[:6]
        fallback_results = {
            case_id: {"confirmed_drift": True, "review_centroid_outlier": True}
            for case_id in stale
        }
        confirmed = POSITIVES[:20]
        entries, hashes = build_entries(
            stale=stale, confirmed=confirmed, review=confirmed, flags=confirmed
        )
        calls = []
        cache_loader, hash_probe, fallback = make_harness(
            entries, hashes, fallback_results=fallback_results, fallback_calls=calls
        )

        report = run_shadow_audit(
            POSITIVES, CONTROLS, cache_loader=cache_loader, hash_probe=hash_probe,
            fallback_extractor=fallback,
        )

        self.assertEqual(report["fallback_count"], 6)
        self.assertEqual(report["fallback_cases"], sorted(stale))
        self.assertEqual(report["hash_mismatch_count"], 6)
        self.assertEqual(report["cache_reuse_count"], 34)
        self.assertAlmostEqual(report["cache_reuse_rate"], 0.85, places=4)
        self.assertGreaterEqual(report["cache_reuse_rate"], MIN_CACHE_REUSE_RATE)
        self.assertLessEqual(report["fallback_count"], FALLBACK_BUDGET)
        self.assertTrue(report["acceptance_passed"])
        self.assertEqual(len(calls), 6)
        self.assertTrue(all(run_id == FALLBACK_RUN for _c, run_id, _e in calls))
        self.assertTrue(all(extractor == FALLBACK_EXTRACTOR for _c, _r, extractor in calls))
        fallback_evidence = [
            item for item in report["evidence"] if item["source"] == "fallback"
        ]
        self.assertEqual(len(fallback_evidence), 6)
        self.assertTrue(all(item["cached"] is False for item in fallback_evidence))
        self.assertTrue(all(item["run_id"] == FALLBACK_RUN for item in fallback_evidence))

    def test_hash_mismatch_rescued_by_other_cached_run_without_fallback(self):
        rescued = POSITIVES[:5]
        entries, hashes = build_entries(
            stale=rescued, rescue=rescued, confirmed=rescued, flags=rescued
        )
        calls = []
        cache_loader, hash_probe, fallback = make_harness(entries, hashes, fallback_calls=calls)

        report = run_shadow_audit(
            POSITIVES, CONTROLS, cache_loader=cache_loader, hash_probe=hash_probe,
            fallback_extractor=fallback,
        )

        self.assertEqual(report["cache_reuse_rate"], 1.0)
        self.assertEqual(report["fallback_count"], 0)
        self.assertEqual(calls, [])
        self.assertEqual(report["hash_mismatch_count"], len(rescued))
        self.assertEqual(report["hash_mismatch_cases"], sorted(rescued))

    def test_fallback_budget_exceeded_fails_acceptance_and_escalates(self):
        stale = POSITIVES[:12]
        entries, hashes = build_entries(stale=stale, confirmed=POSITIVES[:20])
        cache_loader, hash_probe, fallback = make_harness(entries, hashes)

        report = run_shadow_audit(
            POSITIVES, CONTROLS, cache_loader=cache_loader, hash_probe=hash_probe,
            fallback_extractor=fallback,
        )

        self.assertEqual(report["fallback_count"], 12)
        self.assertGreater(report["fallback_count"], FALLBACK_BUDGET)
        self.assertFalse(report["acceptance"]["fallback_within_budget"])
        self.assertFalse(report["acceptance_passed"])
        self.assertAlmostEqual(report["cache_reuse_rate"], 0.7, places=4)
        self.assertFalse(report["acceptance"]["cache_reuse_at_least_0_8"])
        escalations = " | ".join(report["escalations"])
        self.assertIn("fallback_budget_exceeded", escalations)
        self.assertIn("hash_mismatch_reported", escalations)

    def test_low_ppv_fails_acceptance(self):
        entries, hashes = build_entries(confirmed=POSITIVES[:15])
        cache_loader, hash_probe, fallback = make_harness(entries, hashes)

        report = run_shadow_audit(
            POSITIVES, CONTROLS, cache_loader=cache_loader, hash_probe=hash_probe,
            fallback_extractor=fallback,
        )

        self.assertAlmostEqual(report["ppv"], 0.5, places=4)
        self.assertLess(report["ppv"], MIN_PPV)
        self.assertFalse(report["acceptance"]["ppv_at_least_0_6"])
        self.assertFalse(report["acceptance_passed"])
        self.assertIn("ppv_below_target", " | ".join(report["escalations"]))

    def test_cooccurrence_and_duplicate_flag_rates_reported(self):
        confirmed = POSITIVES[:20]
        review = POSITIVES[:12]
        duplicated = POSITIVES[:10]
        entries, hashes = build_entries(
            rescue=duplicated,
            confirmed=confirmed,
            review=review,
            flags=duplicated,
        )
        cache_loader, hash_probe, fallback = make_harness(entries, hashes)

        report = run_shadow_audit(
            POSITIVES, CONTROLS, cache_loader=cache_loader, hash_probe=hash_probe,
            fallback_extractor=fallback,
        )

        self.assertAlmostEqual(
            report["review_centroid_outlier_cooccurrence_rate"], 0.6, places=4
        )
        self.assertEqual(report["review_centroid_outlier_cooccurrence_count"], 12)
        self.assertAlmostEqual(report["duplicate_flag_rate"], 0.25, places=4)
        self.assertEqual(report["duplicate_flag_count"], 10)
        self.assertEqual(report["duplicate_flag_cases"], sorted(duplicated))
        self.assertTrue(report["acceptance"]["review_centroid_outlier_cooccurrence_reported"])
        self.assertTrue(report["acceptance"]["duplicate_flag_rate_reported"])
        self.assertTrue(report["acceptance_passed"])

    def test_sample_cardinality_is_enforced(self):
        entries, hashes = build_entries()
        cache_loader, hash_probe, fallback = make_harness(entries, hashes)

        with self.assertRaises(ValueError):
            run_shadow_audit(
                POSITIVES[:29], CONTROLS, cache_loader=cache_loader,
                hash_probe=hash_probe, fallback_extractor=fallback,
            )
        with self.assertRaises(ValueError):
            run_shadow_audit(
                POSITIVES, CONTROLS[:9], cache_loader=cache_loader,
                hash_probe=hash_probe, fallback_extractor=fallback,
            )
        with self.assertRaises(ValueError):
            run_shadow_audit(
                POSITIVES + [POSITIVES[0]], CONTROLS, cache_loader=cache_loader,
                hash_probe=hash_probe, fallback_extractor=fallback,
            )

    def test_overlapping_samples_rejected(self):
        entries, hashes = build_entries()
        cache_loader, hash_probe, fallback = make_harness(entries, hashes)

        with self.assertRaises(ValueError):
            run_shadow_audit(
                POSITIVES, [POSITIVES[0]] + CONTROLS[1:], cache_loader=cache_loader,
                hash_probe=hash_probe, fallback_extractor=fallback,
            )

    def test_fallback_extractor_identity_is_enforced(self):
        stale = POSITIVES[:3]
        entries, hashes = build_entries(stale=stale)
        cache_loader, hash_probe, _fallback = make_harness(entries, hashes)

        def rogue_extractor(case_id, run_id, extractor_id):
            return {
                "run_id": "run-4",
                "extractor_id": extractor_id,
                "confirmed_drift": True,
            }

        with self.assertRaises(ValueError):
            run_shadow_audit(
                POSITIVES, CONTROLS, cache_loader=cache_loader, hash_probe=hash_probe,
                fallback_extractor=rogue_extractor,
            )

    def test_missing_fallback_extractor_raises(self):
        stale = POSITIVES[:3]
        entries, hashes = build_entries(stale=stale)
        cache_loader, hash_probe, _fallback = make_harness(entries, hashes)

        with self.assertRaises(RuntimeError):
            run_shadow_audit(
                POSITIVES, CONTROLS, cache_loader=cache_loader, hash_probe=hash_probe,
            )

    def test_read_only_contract_and_reviewer_summary(self):
        confirmed = POSITIVES[:22]
        entries, hashes = build_entries(confirmed=confirmed, flags=confirmed)
        cache_loader, hash_probe, fallback = make_harness(entries, hashes)

        report = run_shadow_audit(
            POSITIVES, CONTROLS, cache_loader=cache_loader, hash_probe=hash_probe,
            fallback_extractor=fallback,
        )

        contract = report["read_only_contract"]
        self.assertTrue(contract["read_only"])
        self.assertFalse(contract["writes_performed"])
        self.assertFalse(contract["labels_modified"])
        self.assertFalse(contract["production_rules_modified"])
        self.assertFalse(contract["datasets_touched"])
        self.assertFalse(contract["qc_safety_checks_weakened"])
        self.assertEqual(contract["fallback_limited_to"], "run-9/B1")
        self.assertTrue(verify_read_only_contract(report))

        digest = summarize_for_reviewer(report)
        self.assertEqual(digest["n_sampled"], 40)
        self.assertEqual(digest["fallback_budget"], FALLBACK_BUDGET)
        self.assertIn("review_centroid_outlier_cooccurrence_rate", digest)
        self.assertIn("duplicate_flag_rate", digest)
        self.assertTrue(digest["acceptance_passed"])

    def test_verify_read_only_contract_rejects_mutating_report(self):
        with self.assertRaises(AssertionError):
            verify_read_only_contract(
                {"read_only_contract": {"labels_modified": True}}
            )


if __name__ == "__main__":
    unittest.main()
