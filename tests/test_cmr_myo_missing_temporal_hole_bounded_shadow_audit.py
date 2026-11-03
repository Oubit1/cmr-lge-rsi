"""Offline tests for the bounded cmr_myo_missing_temporal_hole shadow audit."""
import os
import sys
import unittest
from dataclasses import replace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.cmr_myo_missing_temporal_hole_bounded_shadow_audit import (
    B1_EXTRACTOR,
    CACHE_RUNS,
    CONTROL_N,
    MIN_CACHE_REUSE,
    POSITIVE_N,
    MutationGuard,
    ShadowAuditCase,
    ShadowAuditViolation,
    compute_audit_report,
    compute_cache_reuse_rate,
    detect_duplicate_audit_records,
    proportion_ci,
    run_bounded_shadow_audit,
    wilson_interval,
)


def _case(
    index,
    stratum,
    predicted,
    *,
    hash_ok=True,
    covered_run=None,
    record_id=None,
    extractor_route=None,
    slice_refs=None,
):
    if covered_run is None:
        covered_run = CACHE_RUNS[index % len(CACHE_RUNS)]
    if extractor_route is None:
        extractor_route = "cache" if hash_ok else B1_EXTRACTOR
    if record_id is None:
        record_id = f"rec-{stratum}-{index:03d}"
    if slice_refs is None:
        slice_refs = (f"run-{covered_run}/slice-{index:03d}",)
    return ShadowAuditCase(
        case_id=f"{stratum}-{index:03d}",
        stratum=stratum,
        covered_run=covered_run,
        slice_evidence_refs=tuple(slice_refs),
        hash_ok=hash_ok,
        extractor_route=extractor_route,
        predicted_positive=predicted,
        label_positive=(stratum == "positive"),
        audit_record_id=record_id,
    )


class BoundedShadowAuditTests(unittest.TestCase):
    def _balanced_cases(self):
        cases = []
        for i in range(POSITIVE_N):
            cases.append(_case(i, "positive", i < 35, hash_ok=(i not in (3, 7))))
        for j in range(CONTROL_N):
            cases.append(_case(100 + j, "control", j == 0))
        return cases

    def test_bounded_audit_passes_and_reports_required_metrics(self):
        cases = self._balanced_cases()
        report = run_bounded_shadow_audit(cases)
        self.assertEqual(report["cache_reuse_rate"], 1.0)
        self.assertEqual(report["duplicate_record_count"], 0)
        self.assertEqual(report["duplicate_record_rate"], 0.0)
        self.assertEqual(report["hash_mismatch_count"], 2)
        self.assertEqual(report["hash_mismatch_routed_to_b1"], 2)
        self.assertEqual(report["non_b1_mismatch_routes"], 0)
        self.assertEqual(report["label_and_production_rule_changes"], 0)
        self.assertEqual(report["evidence_traceability_rate"], 1.0)

        metrics = report["metrics"]
        self.assertEqual(metrics["tp"], 35)
        self.assertEqual(metrics["fn"], 5)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["tn"], 9)
        self.assertAlmostEqual(metrics["support_rate"]["value"], 35 / 40)
        self.assertAlmostEqual(metrics["precision"]["value"], 35 / 36)
        self.assertAlmostEqual(metrics["false_positive_rate"]["value"], 1 / 10)
        for key in ("support_rate", "precision", "false_positive_rate"):
            ci = metrics[key]
            self.assertLessEqual(ci["low"], ci["value"])
            self.assertGreaterEqual(ci["high"], ci["value"])

    def test_duplicate_audit_record_rejected(self):
        cases = self._balanced_cases()
        duplicated = list(cases[:-1])
        duplicated.append(replace(cases[-1], audit_record_id=cases[0].audit_record_id))
        report = compute_audit_report(duplicated)
        self.assertEqual(report["duplicate_record_count"], 1)
        self.assertGreater(report["duplicate_record_rate"], 0.0)
        with self.assertRaises(ShadowAuditViolation):
            run_bounded_shadow_audit(duplicated)

    def test_cache_reuse_below_threshold_rejected(self):
        cases = []
        for i in range(50):
            stratum = "positive" if i < POSITIVE_N else "control"
            run = CACHE_RUNS[i % len(CACHE_RUNS)] if i < 47 else 100 + i
            cases.append(_case(i, stratum, True, covered_run=run))
        self.assertAlmostEqual(compute_cache_reuse_rate(cases), 47 / 50)
        self.assertLess(compute_cache_reuse_rate(cases), MIN_CACHE_REUSE)
        with self.assertRaises(ShadowAuditViolation):
            run_bounded_shadow_audit(cases)

    def test_hash_mismatch_must_route_only_to_b1(self):
        bad = ShadowAuditCase(
            case_id="bad-1",
            stratum="positive",
            covered_run=28,
            slice_evidence_refs=("run-28/slice-1",),
            hash_ok=False,
            extractor_route="cache",
            predicted_positive=True,
            label_positive=True,
            audit_record_id="bad-rec-1",
        )
        with self.assertRaises(ShadowAuditViolation):
            run_bounded_shadow_audit([bad])

    def test_mutation_guard_blocks_label_or_production_changes(self):
        cases = self._balanced_cases()
        with self.assertRaises(ShadowAuditViolation):
            run_bounded_shadow_audit(
                cases, guard=MutationGuard(labels_modified=True)
            )
        with self.assertRaises(ShadowAuditViolation):
            run_bounded_shadow_audit(
                cases, guard=MutationGuard(production_rules_modified=True)
            )

    def test_wilson_interval_bounds(self):
        lo, hi = wilson_interval(0, 10)
        self.assertEqual(lo, 0.0)
        self.assertGreater(hi, 0.0)
        self.assertLess(hi, 0.4)

        lo2, hi2 = wilson_interval(10, 10)
        self.assertGreater(lo2, 0.6)
        self.assertLess(lo2, 1.0)
        self.assertEqual(hi2, 1.0)

        self.assertEqual(wilson_interval(0, 0), (0.0, 0.0))
        ci = proportion_ci(0, 0)
        self.assertEqual(ci.value, 0.0)
        self.assertEqual(ci.denominator, 0)

    def test_slice_evidence_required(self):
        bad = ShadowAuditCase(
            case_id="no-slice",
            stratum="positive",
            covered_run=28,
            slice_evidence_refs=(),
            hash_ok=True,
            extractor_route="cache",
            predicted_positive=True,
            label_positive=True,
            audit_record_id="no-slice-rec",
        )
        with self.assertRaises(ShadowAuditViolation):
            run_bounded_shadow_audit([bad])

    def test_duplicate_detector_counts_extra_records(self):
        c1 = _case(1, "positive", True, record_id="same")
        c2 = _case(2, "positive", True, record_id="same")
        count, rate = detect_duplicate_audit_records([c1, c2])
        self.assertEqual(count, 1)
        self.assertAlmostEqual(rate, 0.5)


if __name__ == "__main__":
    unittest.main()
