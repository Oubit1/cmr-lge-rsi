"""Offline unittests for experiments.cmr_myo_small_area_bounded_shadow_audit.

All data is synthetic and in-memory: no datasets, patient images, labels,
credentials, or runtime databases are touched.
"""

import json
import unittest

from experiments.cmr_myo_small_area_bounded_shadow_audit import (
    CONTROL_QUOTA,
    FALLBACK_EXTRACTOR_ID,
    FALLBACK_RUN_ID,
    FALLBACK_SOURCE,
    POSITIVE_QUOTA,
    CacheEntry,
    CaseRecord,
    content_hash,
    latest_reusable_cache_entry,
    run_shadow_audit,
    select_audit_sample,
    verify_acceptance,
)


def make_case(idx, positive):
    cid = ("P" if positive else "C") + "%04d" % idx
    return CaseRecord(
        case_id=cid,
        is_positive=positive,
        payload=("payload-%s" % cid).encode("utf-8"),
        label="small_area" if positive else "control",
    )


def make_cache_entry(run_id, case, match=True):
    return CacheEntry(
        run_id=run_id,
        case_id=case.case_id,
        content_hash=content_hash(case.payload) if match else "0" * 64,
        features={"myo_area_px": 42.0, "from_run": float(run_id)},
    )


class ShadowAuditTestBase(unittest.TestCase):
    def setUp(self):
        # 60 positives + 15 controls so the 50/10 quotas are fully met.
        self.positives = [make_case(i, True) for i in range(60)]
        self.controls = [make_case(i, False) for i in range(15)]
        self.cases = self.positives + self.controls
        self.sample_pos, self.sample_ctrl = select_audit_sample(self.cases)
        self.sample = sorted(
            self.sample_pos + self.sample_ctrl, key=lambda c: c.case_id
        )

    def build_manifest(self):
        manifest = {}
        for i, case in enumerate(self.sample):
            bucket = i % 5
            if bucket == 0:
                # cache hit; runs 18 and 9 both hold valid entries
                manifest.setdefault(18, {})[case.case_id] = make_cache_entry(18, case)
                manifest.setdefault(9, {})[case.case_id] = make_cache_entry(9, case)
            elif bucket == 1:
                # hash mismatch on run 13 -> B1 fallback
                manifest.setdefault(13, {})[case.case_id] = make_cache_entry(
                    13, case, match=False
                )
            elif bucket == 2:
                # only a non-whitelisted run (7) -> treated as cache miss
                manifest.setdefault(7, {})[case.case_id] = make_cache_entry(7, case)
            elif bucket == 3:
                # cache hit on run 4
                manifest.setdefault(4, {})[case.case_id] = make_cache_entry(4, case)
            # bucket == 4: no cache at all -> miss
        return manifest


class TestSampling(ShadowAuditTestBase):
    def test_quotas(self):
        self.assertEqual(len(self.sample_pos), POSITIVE_QUOTA)
        self.assertEqual(len(self.sample_ctrl), CONTROL_QUOTA)

    def test_deterministic(self):
        p2, c2 = select_audit_sample(self.cases)
        self.assertEqual(
            [c.case_id for c in p2], [c.case_id for c in self.sample_pos]
        )
        self.assertEqual(
            [c.case_id for c in c2], [c.case_id for c in self.sample_ctrl]
        )


class TestCacheReuse(ShadowAuditTestBase):
    def test_latest_whitelisted_run_preferred(self):
        case = self.sample[0]
        manifest = {
            18: {case.case_id: make_cache_entry(18, case)},
            9: {case.case_id: make_cache_entry(9, case)},
        }
        entry = latest_reusable_cache_entry(manifest, case.case_id)
        self.assertIsNotNone(entry)
        self.assertEqual(entry.run_id, 18)

    def test_non_whitelisted_run_ignored(self):
        case = self.sample[0]
        manifest = {7: {case.case_id: make_cache_entry(7, case)}}
        self.assertIsNone(latest_reusable_cache_entry(manifest, case.case_id))


class TestAuditRun(ShadowAuditTestBase):
    def setUp(self):
        super().setUp()
        self.manifest = self.build_manifest()
        self.report = run_shadow_audit(self.cases, self.manifest)

    def test_audited_total(self):
        self.assertEqual(self.report.audited_total, POSITIVE_QUOTA + CONTROL_QUOTA)
        self.assertEqual(self.report.positives_audited, POSITIVE_QUOTA)
        self.assertEqual(self.report.controls_audited, CONTROL_QUOTA)

    def test_cache_reuse_rate_recorded(self):
        expected_hits = sum(1 for i in range(len(self.sample)) if i % 5 in (0, 3))
        self.assertEqual(self.report.cache_hits, expected_hits)
        self.assertAlmostEqual(
            self.report.cache_reuse_rate, expected_hits / float(len(self.sample))
        )

    def test_hash_mismatch_fallbacks_recorded(self):
        expected = sorted(
            c.case_id for i, c in enumerate(self.sample) if i % 5 == 1
        )
        self.assertEqual(list(self.report.hash_mismatch_case_ids), expected)
        self.assertEqual(self.report.hash_mismatch_fallbacks, len(expected))
        for o in self.report.outcomes:
            if o.hash_mismatch:
                self.assertEqual(o.source, FALLBACK_SOURCE)
                self.assertEqual(o.run_id, FALLBACK_RUN_ID)

    def test_cache_misses_recorded_separately(self):
        expected = sorted(
            c.case_id for i, c in enumerate(self.sample) if i % 5 in (2, 4)
        )
        self.assertEqual(list(self.report.cache_miss_case_ids), expected)
        self.assertEqual(self.report.cache_miss_fallbacks, len(expected))

    def test_fallback_only_run9_b1(self):
        self.assertEqual(self.report.fallback_run_id, FALLBACK_RUN_ID)
        self.assertEqual(self.report.fallback_extractor_id, FALLBACK_EXTRACTOR_ID)
        for o in self.report.outcomes:
            if o.source == FALLBACK_SOURCE:
                self.assertEqual(o.run_id, 9)
                self.assertIn("extractor_run_id", o.features)

    def test_read_only_guarantees(self):
        self.assertEqual(self.report.label_modifications, 0)
        self.assertEqual(self.report.new_production_rules, 0)

    def test_report_json_serializable(self):
        payload = json.dumps(self.report.to_dict())
        self.assertIn('"rule_id": "cmr_myo_small_area"', payload)

    def test_verify_acceptance_all_passed(self):
        checks = verify_acceptance(self.report)
        self.assertTrue(checks["all_passed"], msg=str(checks))


class TestEdgeCases(unittest.TestCase):
    def test_empty_cache_all_fallback(self):
        cases = [make_case(i, True) for i in range(50)] + [
            make_case(i, False) for i in range(10)
        ]
        report = run_shadow_audit(cases, {})
        self.assertEqual(report.cache_hits, 0)
        self.assertEqual(report.cache_reuse_rate, 0.0)
        self.assertEqual(report.hash_mismatch_fallbacks, 0)
        self.assertEqual(report.cache_miss_fallbacks, 60)
        self.assertTrue(verify_acceptance(report)["all_passed"])

    def test_custom_b1_extractor_injected(self):
        cases = [make_case(i, True) for i in range(50)] + [
            make_case(i, False) for i in range(10)
        ]
        report = run_shadow_audit(cases, {}, b1_extractor=lambda c: {"custom": 1.0})
        for o in report.outcomes:
            self.assertEqual(o.source, FALLBACK_SOURCE)
            self.assertEqual(o.features, {"custom": 1.0})


if __name__ == "__main__":
    unittest.main()
