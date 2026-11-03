"""Offline unit tests for the myo_topology read-only provenance shadow audit.

Everything here is synthetic and in-memory: no dataset, no cache file and no
network access is required.
"""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.myo_topology_provenance_shadow_audit import (
    CO_OCCURRENCE_FLAGS,
    DEFAULT_ARM_AGREEMENT_BAR,
    DEFAULT_ARMS,
    RUN9_B1_EXTRACTOR,
    VERIFIED_CACHE_RUNS,
    arm_agreement,
    assert_single_extractor,
    canonical_payload_hash,
    continuous_positive_runs,
    extract_b1_contract,
    freeze_slices,
    max_component_arm,
    noop_arm,
    per_case_topology,
    ring_continuity_arm,
    run_shadow_audit,
    same_slice_cooccurrence,
    severity_stratum,
    slice_key,
    verify_cache_hashes,
)


def _raw(case_id, slice_index, positive=True, severity=0.5, ring=True, components=1, share=1.0, **flags):
    record = {
        "case_id": case_id,
        "slice_index": slice_index,
        "myo_topology": positive,
        "severity": severity,
        "ring_continuous": ring,
        "n_components": components,
        "max_component_share": share,
    }
    record.update(flags)
    return record


def _raw_records():
    return [
        _raw("A", 3, True, 0.80, True, 3, 0.90, mi_multiple_components=True, mi_fraction_high=True),
        _raw("A", 4, True, 0.50, True, 2, 0.80, mi_multiple_components=True),
        _raw("A", 5, False, 0.00, False, 0, 0.00),
        _raw("B", 10, True, 0.20, False, 0, 0.00),
        _raw("B", 11, False, 0.00, False, 0, 0.00),
        _raw("C", 1, True, 0.90, True, 1, 0.30),
        _raw("C", 2, True, 0.40, True, 2, 0.70),
        _raw("C", 3, True, 0.10, False, 1, 0.20),
    ]


def _clean_records():
    return [
        _raw("P", 1, True, 0.80, True, 2, 0.95, mi_multiple_components=True),
        _raw("P", 2, True, 0.60, True, 2, 0.90),
        _raw("P", 3, False, 0.00, True, 1, 0.60),
        _raw("Q", 7, True, 0.40, True, 3, 0.99, mi_fraction_high=True),
        _raw("Q", 8, False, 0.00, True, 1, 0.70),
    ]


class MyoTopologyProvenanceShadowAuditTest(unittest.TestCase):
    def setUp(self):
        self.raw = _raw_records()
        self.audit = run_shadow_audit(records=self.raw)

    # -- contract / helper level ------------------------------------------
    def test_verified_run_tokens_cover_the_reused_cache_runs(self):
        self.assertEqual(tuple(VERIFIED_CACHE_RUNS), (4, 5, 9, 13, 14, 15, 16, 17, 18))
        self.assertEqual(
            tuple(CO_OCCURRENCE_FLAGS), ("mi_multiple_components", "mi_fraction_high")
        )
        self.assertEqual(
            sorted(DEFAULT_ARMS), ["max_component", "noop", "ring_continuity"]
        )

    def test_single_extractor_guard_refuses_a_second_tool_set(self):
        self.assertEqual(assert_single_extractor(), RUN9_B1_EXTRACTOR)
        with self.assertRaises(RuntimeError):
            assert_single_extractor(
                {RUN9_B1_EXTRACTOR: lambda payload: [], "second_tool": lambda payload: []}
            )

    def test_extract_b1_contract_normalizes_and_rejects_broken_records(self):
        payload = {
            "records": [
                {
                    "case_id": "X",
                    "slice_index": 2,
                    "positive": True,
                    "severity": 0.7,
                    "ring_closed": True,
                    "components": 2,
                    "max_component_share": 0.6,
                    "mi_fraction_high": True,
                }
            ]
        }
        records = extract_b1_contract(payload)
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["case_id"], "X")
        self.assertEqual(record["slice_index"], 2)
        self.assertTrue(record["positive"])
        self.assertTrue(record["ring_continuous"])
        self.assertEqual(record["components"], 2)
        self.assertAlmostEqual(record["max_component_share"], 0.6)
        self.assertTrue(record["flags"]["mi_fraction_high"])
        self.assertFalse(record["flags"]["mi_multiple_components"])

        with self.assertRaises(ValueError):
            extract_b1_contract({"records": [{"slice_index": 1}]})
        with self.assertRaises(ValueError):
            extract_b1_contract({"records": [{"case_id": "Y"}]})

    def test_severity_stratum_boundaries(self):
        self.assertEqual(severity_stratum(None), "none")
        self.assertEqual(severity_stratum(0.0), "none")
        self.assertEqual(severity_stratum(0.01), "low")
        self.assertEqual(severity_stratum(0.33), "moderate")
        self.assertEqual(severity_stratum(0.659), "moderate")
        self.assertEqual(severity_stratum(0.66), "high")
        self.assertEqual(severity_stratum(1.0), "high")

    # -- acceptance metric 1: per-case slice counts / rate / worst slice --
    def test_per_case_slice_counts_rates_worst_slice_and_stratum(self):
        per_case = self.audit["per_case"]
        self.assertEqual(sorted(per_case), ["A", "B", "C"])

        case_a = per_case["A"]
        self.assertEqual(case_a["slice_count"], 3)
        self.assertEqual(case_a["positive_slices"], 2)
        self.assertAlmostEqual(case_a["positive_rate"], 2.0 / 3.0, places=6)
        self.assertEqual(case_a["positive_slice_indices"], (3, 4))
        self.assertEqual(case_a["worst_slice"]["slice_index"], 3)
        self.assertAlmostEqual(case_a["worst_slice"]["severity"], 0.80)
        self.assertEqual(case_a["worst_slice"]["stratum"], "high")
        self.assertEqual(case_a["severity_stratum"], "high")

        case_b = per_case["B"]
        self.assertEqual(case_b["slice_count"], 2)
        self.assertEqual(case_b["positive_slices"], 1)
        self.assertAlmostEqual(case_b["positive_rate"], 0.5)
        self.assertEqual(case_b["worst_slice"]["slice_index"], 10)
        self.assertEqual(case_b["worst_slice"]["stratum"], "low")
        self.assertEqual(case_b["severity_stratum"], "low")

        case_c = per_case["C"]
        self.assertEqual(case_c["slice_count"], 3)
        self.assertAlmostEqual(case_c["positive_rate"], 1.0)
        self.assertEqual(case_c["positive_slice_indices"], (1, 2, 3))
        self.assertEqual(case_c["worst_slice"]["slice_index"], 1)
        self.assertEqual(case_c["worst_slice"]["stratum"], "high")

        self.assertEqual(self.audit["n_cases"], 3)
        self.assertEqual(self.audit["n_slices"], 8)
        self.assertTrue(self.audit["acceptance"]["per_case_complete"])

    # -- acceptance metric 2a: same-slice co-occurrence -------------------
    def test_same_slice_cooccurrence_with_mi_flags(self):
        co = self.audit["co_occurrence"]
        self.assertEqual(sorted(co), sorted(CO_OCCURRENCE_FLAGS))

        multiple = co["mi_multiple_components"]
        self.assertEqual(multiple["positive_slices"], 6)
        self.assertEqual(multiple["same_slice_positive_and_flag"], 2)
        self.assertAlmostEqual(multiple["cooccurrence_rate"], 2.0 / 6.0, places=6)
        self.assertEqual(multiple["cases"], ("A",))
        self.assertEqual(multiple["slice_keys"], (("A", 3), ("A", 4)))

        fraction = co["mi_fraction_high"]
        self.assertEqual(fraction["same_slice_positive_and_flag"], 1)
        self.assertAlmostEqual(fraction["cooccurrence_rate"], 1.0 / 6.0, places=6)
        self.assertEqual(fraction["slice_keys"], (("A", 3),))
        self.assertTrue(self.audit["acceptance"]["cooccurrence_computable"])

    # -- acceptance metric 2b: cross-slice continuous positive runs -------
    def test_continuous_positive_runs_are_recomputable(self):
        runs = self.audit["positive_runs"]
        self.assertEqual(runs["A"]["runs"], ((3, 4),))
        self.assertEqual(runs["A"]["run_count"], 1)
        self.assertEqual(runs["A"]["max_run_length"], 2)
        self.assertEqual(runs["B"]["runs"], ((10, 10),))
        self.assertEqual(runs["B"]["max_run_length"], 1)
        self.assertEqual(runs["C"]["runs"], ((1, 3),))
        self.assertEqual(runs["C"]["run_count"], 1)
        self.assertEqual(runs["C"]["max_run_length"], 3)

        single = continuous_positive_runs(
            extract_b1_contract(self.raw), case_id="C"
        )
        self.assertEqual(single["runs"], ((1, 3),))
        self.assertEqual(single["max_run_length"], 3)
        missing = continuous_positive_runs(extract_b1_contract(self.raw), case_id="ZZZ")
        self.assertEqual(missing["runs"], ())
        self.assertEqual(missing["max_run_length"], 0)

    # -- acceptance metric 3: read-only arms + agreement bar --------------
    def test_read_only_arms_match_the_frozen_slice_set(self):
        records = extract_b1_contract(self.raw)
        self.assertEqual(
            noop_arm(records),
            frozenset({("A", 3), ("A", 4), ("B", 10), ("C", 1), ("C", 2), ("C", 3)}),
        )
        self.assertEqual(
            ring_continuity_arm(records),
            frozenset({("A", 3), ("A", 4), ("C", 1), ("C", 2)}),
        )
        self.assertEqual(
            max_component_arm(records),
            frozenset({("A", 3), ("A", 4), ("B", 10), ("C", 2)}),
        )

    def test_arm_agreement_values_and_bar(self):
        arms = self.audit["arms"]
        self.assertEqual(sorted(arms["agreement"]), ["max_component", "ring_continuity"])

        ring = arms["agreement"]["ring_continuity"]
        self.assertAlmostEqual(ring["agreement"], 0.75)
        self.assertEqual(ring["n_slices"], 8)
        self.assertEqual(ring["disagreements"], (("B", 10), ("C", 3)))

        component = arms["agreement"]["max_component"]
        self.assertAlmostEqual(component["agreement"], 0.75)
        self.assertEqual(component["disagreements"], (("C", 1), ("C", 3)))

        self.assertFalse(arms["meets_bar"])
        self.assertFalse(self.audit["acceptance"]["arms_meet_bar"])

    def test_agreement_bar_is_met_on_metadata_consistent_slices(self):
        audit = run_shadow_audit(records=_clean_records())
        self.assertEqual(audit["n_slices"], 5)
        for entry in audit["arms"]["agreement"].values():
            self.assertEqual(entry["agreement"], 1.0)
            self.assertEqual(entry["disagreements"], ())
        self.assertTrue(audit["arms"]["meets_bar"])
        self.assertTrue(audit["acceptance"]["meets_bar"])
        self.assertLessEqual(audit["arms"]["bar"], 1.0)
        self.assertEqual(audit["arms"]["bar"], DEFAULT_ARM_AGREEMENT_BAR)

    def test_arm_agreement_requires_a_registered_baseline(self):
        with self.assertRaises(ValueError):
            arm_agreement([], arms={"ring_continuity": ring_continuity_arm})

    # -- cache hash verification / single-run-9 fallback ------------------
    def test_cache_hash_verification_reports_verified_and_mismatched_runs(self):
        payload = {"records": [_raw("W", 1)]}
        details = verify_cache_hashes({4: payload}, {4: canonical_payload_hash(payload)})
        self.assertTrue(details[4]["verified"])
        self.assertTrue(details[4]["expectation_present"])

        embedded = verify_cache_hashes({9: {"records": [], "payload_hash": "deadbeef"}})
        self.assertFalse(embedded[9]["verified"])
        self.assertEqual(embedded[9]["expected"], "deadbeef")
        self.assertEqual(embedded[9]["hash"], canonical_payload_hash({"records": [], "payload_hash": "deadbeef"}))

    def test_hash_mismatch_falls_back_to_the_single_run9_b1_extractor(self):
        cache4 = {"records": [_raw("Z", 1, True, 0.70, True, 2, 0.90)]}
        cache9 = {"records": [_raw("Z", 2, True, 0.50, True, 2, 0.80)]}
        audit = run_shadow_audit(
            caches={4: cache4, 9: cache9},
            expected_hashes={4: canonical_payload_hash(cache4)},
        )
        self.assertTrue(audit["cache"]["checked"])
        self.assertEqual(audit["cache"]["verified_run_ids"], (4,))
        self.assertEqual(audit["cache"]["mismatched_run_ids"], (9,))
        self.assertEqual(audit["cache"]["unknown_run_ids"], ())
        self.assertTrue(audit["fallback"]["used"])
        self.assertEqual(audit["fallback"]["extractor"], RUN9_B1_EXTRACTOR)
        self.assertEqual(audit["fallback"]["run_ids"], (9,))
        self.assertEqual(audit["source"], "imported_cache_summaries")
        self.assertEqual(audit["n_slices"], 2)
        self.assertFalse(audit["acceptance"]["cache_hash_ok"])
        self.assertFalse(audit["acceptance"]["meets_bar"])

    def test_unknown_run_ids_are_reported_and_skipped_not_rederived(self):
        audit = run_shadow_audit(caches={"21": {"records": [_raw("Y", 1)]}})
        self.assertEqual(audit["cache"]["unknown_run_ids"], ("21",))
        self.assertEqual(audit["cache"]["verified_run_ids"], ())
        self.assertEqual(audit["n_slices"], 0)
        self.assertFalse(audit["fallback"]["used"])
        self.assertIsNone(audit["fallback"]["extractor"])
        self.assertFalse(audit["acceptance"]["meets_bar"])

    # -- dedup + read-only guarantees -------------------------------------
    def test_duplicate_slice_records_are_flagged_and_kept_single(self):
        records = self.raw + [_raw("A", 3, True, 0.10)]
        audit = run_shadow_audit(records=records)
        self.assertEqual(audit["n_slices"], 8)
        self.assertEqual(audit["dedup"]["duplicate_count"], 1)
        self.assertFalse(audit["dedup"]["compliant"])
        self.assertFalse(audit["acceptance"]["dedup_compliant"])
        self.assertFalse(audit["acceptance"]["meets_bar"])
        duplicate = audit["dedup"]["duplicates"][0]
        self.assertEqual(duplicate["key"], ("A", 3))
        self.assertEqual(duplicate["first_position"], 0)
        self.assertEqual(duplicate["duplicate_position"], 8)
        self.assertEqual(audit["per_case"]["A"]["positive_slices"], 2)
        self.assertAlmostEqual(audit["per_case"]["A"]["worst_slice"]["severity"], 0.80)

    def test_freeze_slices_is_deterministic_and_reports_duplicates(self):
        records = extract_b1_contract(self.raw)
        frozen, duplicates = freeze_slices(records)
        self.assertEqual(len(frozen), 8)
        self.assertEqual(duplicates, ())
        self.assertEqual(
            tuple(slice_key(record) for record in frozen),
            (("A", 3), ("A", 4), ("A", 5), ("B", 10), ("B", 11), ("C", 1), ("C", 2), ("C", 3)),
        )
        again, _ = freeze_slices(list(reversed(records)))
        self.assertEqual(tuple(slice_key(record) for record in again), tuple(slice_key(record) for record in frozen))

    def test_audit_is_read_only_and_never_promotes_routing(self):
        snapshot = [dict(record) for record in self.raw]
        audit = run_shadow_audit(records=self.raw)
        self.assertEqual(self.raw, snapshot)
        self.assertFalse(audit["writes_state"])
        self.assertFalse(audit["promotes_routing"])
        self.assertEqual(audit["mode"], "read_only_import_only_shadow")
        self.assertEqual(audit["rule"], "myo_topology")
        self.assertTrue(audit["mutation_free"])
        self.assertIn("single extractor contract", " ".join(audit["notes"]))

    def test_acceptance_metrics_are_all_emitted(self):
        for key in (
            "per_case_complete",
            "cooccurrence_computable",
            "arms_meet_bar",
            "cache_hash_ok",
            "dedup_compliant",
            "meets_bar",
        ):
            self.assertIn(key, self.audit["acceptance"])
        self.assertTrue(self.audit["acceptance"]["cache_hash_ok"])
        self.assertTrue(self.audit["acceptance"]["dedup_compliant"])


if __name__ == "__main__":
    unittest.main()
