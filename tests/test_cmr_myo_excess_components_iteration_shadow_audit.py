"""Offline unit tests for the bounded read-only cmr_myo_excess_components audit.

No dataset, cache file or network access is required: every case row, cache entry
and extractor is constructed in-process.
"""

import copy
import inspect
import unittest

import experiments.cmr_myo_excess_components_iteration_shadow_audit as audit_module
from experiments.cmr_myo_excess_components_iteration_shadow_audit import (
    CACHE_HIT_RATE_FLOOR,
    CONTROL_SAMPLE_SIZE,
    CONTROL_STRATUM,
    FALLBACK_EXTRACTOR_ID,
    POSITIVE_SAMPLE_SIZE,
    POSITIVE_STRATUM,
    READER_CLASSES,
    REQUIRED_METRICS,
    canonical_case_key,
    run_bounded_shadow_audit,
    select_bounded_sample,
    summarize_reader_adjudication,
    summarize_secondary_components,
)


def make_components(primary_area=100.0, secondaries=()):
    components = [
        {"area": primary_area, "is_primary": True, "touches_main": False, "slice_ids": ("s1",)}
    ]
    for area, touches_main, levels in secondaries:
        components.append(
            {
                "area": area,
                "is_primary": False,
                "touches_main": touches_main,
                "slice_ids": tuple(levels),
            }
        )
    return components


def make_record(
    case_id,
    modality="LGE",
    issue_type="excess_components",
    stratum=POSITIVE_STRATUM,
    primary_area=100.0,
    secondaries=(),
    reader_label="pseudo_fragment",
    expected_payload_hash=None,
):
    return {
        "case_id": case_id,
        "modality": modality,
        "issue_type": issue_type,
        "stratum": stratum,
        "expected_payload_hash": expected_payload_hash or f"hash::{case_id}::{modality}",
        "components": make_components(primary_area, secondaries),
        "reader_label": reader_label,
    }


def make_cache(records, run_id="run-9"):
    return {
        record["case_id"]: {
            "modality": record["modality"],
            "payload_hash": record["expected_payload_hash"],
            "run_id": run_id,
        }
        for record in records
    }


class DedupeAndIssueTypeTest(unittest.TestCase):
    def test_duplicate_case_modality_rows_are_rejected_at_100_percent(self):
        records = [
            make_record("P-0001", issue_type="excess_components"),
            make_record("P-0001", modality="LGE", issue_type="excess_components"),
            make_record("P-0002", issue_type="excess_components"),
        ]
        report = run_bounded_shadow_audit(records, cache_index=make_cache(records))
        self.assertEqual(report["dedupe"]["duplicate_rows"], 1)
        self.assertEqual(report["dedupe"]["duplicate_rows_rejected"], 1)
        self.assertEqual(report["dedupe"]["duplicate_rejection_rate"], 1.0)
        self.assertTrue(report["gates"]["case_modality_dedupe_ok"])
        self.assertTrue(report["rows_per_case_key_unique"])
        keys = [canonical_case_key(row) for row in report["case_rows"]]
        self.assertEqual(len(keys), len(set(keys)))

    def test_unique_issue_type_lost_to_dedupe_is_reported(self):
        records = [
            make_record("P-0001", issue_type="excess_components"),
            make_record("P-0001", issue_type="myo_border_contact"),
        ]
        report = run_bounded_shadow_audit(records, cache_index={})
        self.assertEqual(report["dedupe"]["duplicate_rejection_rate"], 1.0)
        self.assertEqual(report["missing_issue_types"], ["myo_border_contact"])
        self.assertFalse(report["gates"]["issue_type_set_preserved"])
        self.assertFalse(report["all_gates_passed"])

    def test_same_case_different_modality_is_not_a_duplicate(self):
        records = [
            make_record("P-0001", modality="LGE"),
            make_record("P-0001", modality="bSSFP"),
        ]
        report = run_bounded_shadow_audit(records, cache_index={})
        self.assertEqual(report["dedupe"]["duplicate_rows"], 0)
        self.assertEqual(len(report["case_rows"]), 2)


class SecondaryComponentMetricTest(unittest.TestCase):
    def test_four_secondary_component_indicators_are_computed(self):
        components = make_components(
            100.0,
            [(20.0, True, ("s1", "s2")), (10.0, False, ("s1",)), (30.0, True, ("s1",))],
        )
        metrics = summarize_secondary_components(components)
        self.assertEqual(set(metrics), set(REQUIRED_METRICS))
        self.assertEqual(metrics["secondary_component_count"], 3)
        self.assertAlmostEqual(metrics["secondary_area_ratio"], 0.6)
        self.assertAlmostEqual(metrics["main_myo_adjacency_rate"], 2 / 3)
        self.assertAlmostEqual(metrics["cross_slice_or_temporal_persistence_rate"], 1 / 3)

    def test_persistence_uses_timepoint_levels_when_requested(self):
        components = [
            {"area": 100.0, "is_primary": True, "touches_main": False, "timepoint_ids": ("t1",)},
            {"area": 10.0, "is_primary": False, "touches_main": True, "timepoint_ids": ("t1", "t3")},
            {"area": 5.0, "is_primary": False, "touches_main": True, "timepoint_ids": ("t2",)},
        ]
        metrics = summarize_secondary_components(components, level_key="timepoint_ids")
        self.assertAlmostEqual(metrics["cross_slice_or_temporal_persistence_rate"], 0.5)
        self.assertAlmostEqual(metrics["main_myo_adjacency_rate"], 1.0)

    def test_single_primary_component_yields_zero_indicators(self):
        metrics = summarize_secondary_components(make_components(50.0, []))
        self.assertEqual(metrics["secondary_component_count"], 0)
        self.assertAlmostEqual(metrics["secondary_area_ratio"], 0.0)
        self.assertAlmostEqual(metrics["main_myo_adjacency_rate"], 0.0)
        self.assertAlmostEqual(metrics["cross_slice_or_temporal_persistence_rate"], 0.0)

    def test_non_positive_primary_area_is_rejected(self):
        with self.assertRaises(ValueError):
            summarize_secondary_components(make_components(0.0, [(5.0, True, ("s1",))]))

    def test_missing_primary_component_is_rejected(self):
        with self.assertRaises(ValueError):
            summarize_secondary_components(
                [{"area": 10.0, "is_primary": False, "touches_main": True, "slice_ids": ("s1",)}]
            )

    def test_negative_area_is_rejected(self):
        with self.assertRaises(ValueError):
            summarize_secondary_components(make_components(100.0, [(-1.0, True, ("s1",))]))

    def test_metrics_completeness_rate_is_one_for_valid_rows(self):
        records = [
            make_record("P-0001", secondaries=[(5.0, True, ("s1",))]),
            make_record("P-0002", secondaries=[]),
        ]
        report = run_bounded_shadow_audit(records, cache_index=make_cache(records))
        self.assertEqual(report["metrics"]["completeness_rate"], 1.0)
        self.assertEqual(report["metrics"]["missing"], [])
        self.assertTrue(report["gates"]["metrics_complete"])


class ReaderAdjudicationTest(unittest.TestCase):
    def test_reader_shares_total_100_basis_points(self):
        labels = (
            ["pseudo_fragment"] * 3
            + ["reasonable_anatomy"] * 2
            + ["true_segmentation_failure"] * 2
        )
        records = [make_record(f"P-{i:04d}", reader_label=label) for i, label in enumerate(labels)]
        rollup = summarize_reader_adjudication(records)
        self.assertEqual(set(rollup["counts"]), set(READER_CLASSES))
        self.assertEqual(rollup["counts"]["pseudo_fragment"], 3)
        self.assertEqual(sum(rollup["basis_points"].values()), 10000)
        self.assertAlmostEqual(round(sum(rollup["shares_pct"].values()), 2), 100.0)

    def test_three_classes_share_is_exact_for_seven_cases(self):
        labels = (
            ["pseudo_fragment"] * 3
            + ["reasonable_anatomy"] * 2
            + ["true_segmentation_failure"] * 2
        )
        rollup = summarize_reader_adjudication(
            [make_record(f"P-{i:04d}", reader_label=label) for i, label in enumerate(labels)]
        )
        self.assertEqual(rollup["basis_points"], {"pseudo_fragment": 4286, "reasonable_anatomy": 2857, "true_segmentation_failure": 2857})

    def test_unknown_reader_label_is_rejected(self):
        with self.assertRaises(ValueError):
            summarize_reader_adjudication([make_record("P-0001", reader_label="maybe_something")])


class CacheReuseTest(unittest.TestCase):
    def test_hit_rate_floor_with_single_approved_fallback_extractor(self):
        records = [make_record(f"P-{i:04d}") for i in range(10)]
        cache = make_cache(records)
        cache["P-0003"]["payload_hash"] = "stale"
        cache["P-0007"]["payload_hash"] = "stale"
        report = run_bounded_shadow_audit(records, cache_index=cache)
        self.assertEqual(report["cache"]["cache_hits"], 8)
        self.assertEqual(report["cache"]["cache_mismatches"], 2)
        self.assertEqual(report["cache"]["cache_misses"], 0)
        self.assertEqual(report["cache"]["fallback_extractions"], 2)
        self.assertAlmostEqual(report["cache"]["cache_hit_rate"], 0.8)
        self.assertGreaterEqual(report["cache"]["cache_hit_rate"], CACHE_HIT_RATE_FLOOR)
        self.assertTrue(report["gates"]["cache_hit_rate_ok"])
        self.assertEqual(report["cache"]["fallback_extractor_id"], "run-9:B1")
        fallback_rows = [row for row in report["case_rows"] if row["used_fallback_extractor"]]
        self.assertEqual(len(fallback_rows), 2)
        self.assertEqual({row["payload_source"] for row in fallback_rows}, {FALLBACK_EXTRACTOR_ID})
        self.assertEqual({row["cache_status"] for row in fallback_rows}, {"mismatch"})
        self.assertEqual({row["fallback_reason"] for row in fallback_rows}, {"payload_hash_mismatch"})

    def test_cache_miss_is_reported_and_uses_run9_b1(self):
        records = [make_record("P-0001")]
        report = run_bounded_shadow_audit(records, cache_index={})
        self.assertEqual(report["cache"]["cache_misses"], 1)
        self.assertEqual(report["cache"]["fallback_extractions"], 1)
        self.assertEqual(report["case_rows"][0]["payload_source"], FALLBACK_EXTRACTOR_ID)
        self.assertEqual(report["case_rows"][0]["fallback_reason"], "cache_miss")

    def test_cache_hit_below_floor_fails_the_gate(self):
        records = [make_record(f"P-{i:04d}") for i in range(10)]
        cache = make_cache(records)
        for case_id in ("P-0000", "P-0001", "P-0002"):
            cache[case_id]["payload_hash"] = "stale"
        report = run_bounded_shadow_audit(records, cache_index=cache)
        self.assertAlmostEqual(report["cache"]["cache_hit_rate"], 0.7)
        self.assertFalse(report["gates"]["cache_hit_rate_ok"])
        self.assertFalse(report["all_gates_passed"])

    def test_unapproved_cache_run_is_rejected(self):
        records = [make_record("P-0001")]
        with self.assertRaises(ValueError):
            run_bounded_shadow_audit(records, cache_index=make_cache(records, run_id="run-99"))

    def test_every_approved_cache_run_is_accepted(self):
        for run_id in (
            "run-4",
            "run-5",
            "run-9",
            "run-13",
            "run-14",
            "run-15",
            "run-16",
            "run-17",
            "run-18",
        ):
            records = [make_record("P-0001")]
            report = run_bounded_shadow_audit(
                records, cache_index=make_cache(records, run_id=run_id)
            )
            self.assertEqual(report["cache"]["cache_hits"], 1, run_id)
            self.assertEqual(report["case_rows"][0]["payload_source"], run_id)


class BoundedSampleTest(unittest.TestCase):
    def test_sample_selection_is_bounded_and_deterministic(self):
        records = [make_record(f"P-{i:04d}") for i in range(80)] + [
            make_record(f"C-{i:04d}", stratum=CONTROL_STRATUM) for i in range(20)
        ]
        sample, notes = select_bounded_sample(records)
        self.assertEqual(len(sample), POSITIVE_SAMPLE_SIZE + CONTROL_SAMPLE_SIZE)
        self.assertEqual(notes["positive_sampled"], 50)
        self.assertEqual(notes["control_sampled"], 10)
        self.assertEqual(notes["positive_shortfall"], 0)
        self.assertEqual(notes["control_shortfall"], 0)
        self.assertEqual(
            [row["case_id"] for row in sample[:50]], [f"P-{i:04d}" for i in range(50)]
        )

    def test_short_bounded_sample_is_flagged_not_silently_padded(self):
        records = [make_record("P-0001")]
        report = run_bounded_shadow_audit(records, cache_index=make_cache(records))
        self.assertEqual(report["sample"]["positive_sampled"], 1)
        self.assertEqual(report["sample"]["positive_shortfall"], POSITIVE_SAMPLE_SIZE - 1)
        self.assertEqual(report["sample"]["control_shortfall"], CONTROL_SAMPLE_SIZE)
        self.assertFalse(report["gates"]["bounded_sample_ok"])
        self.assertFalse(report["all_gates_passed"])

    def test_unapproved_stratum_is_rejected(self):
        with self.assertRaises(ValueError):
            select_bounded_sample([make_record("P-0001", stratum="some_other_stratum")])

    def test_record_missing_required_keys_is_rejected(self):
        record = make_record("P-0001")
        del record["reader_label"]
        with self.assertRaises(ValueError):
            run_bounded_shadow_audit([record], cache_index={})


class ReadOnlyPostureTest(unittest.TestCase):
    def test_module_source_contains_no_write_primitive(self):
        source = inspect.getsource(audit_module)
        for forbidden in (
            "os.remove",
            "os.unlink",
            "os.rename",
            "shutil.rmtree",
            "subprocess",
            "open(",
            "write(",
        ):
            self.assertNotIn(forbidden, source, msg=f"read-only module must not contain {forbidden!r}")

    def test_report_declares_zero_mutations(self):
        records = [make_record("P-0001")]
        report = run_bounded_shadow_audit(records, cache_index=make_cache(records))
        self.assertTrue(report["read_only"])
        self.assertEqual(report["label_mutations"], 0)
        self.assertEqual(report["production_rule_mutations"], 0)
        self.assertEqual(report["dataset_writes"], 0)
        self.assertEqual(report["filesystem_writes"], 0)


class BoundedAuditEndToEndTest(unittest.TestCase):
    @staticmethod
    def _cohort():
        labels = list(READER_CLASSES)
        positives = [
            make_record(
                f"P-{i:04d}",
                stratum=POSITIVE_STRATUM,
                issue_type="excess_components",
                secondaries=[(8.0, True, ("s1", "s2")), (4.0, i % 2 == 0, ("s1",))],
                reader_label=labels[i % len(labels)],
            )
            for i in range(POSITIVE_SAMPLE_SIZE + 5)
        ]
        controls = [
            make_record(
                f"C-{i:04d}",
                stratum=CONTROL_STRATUM,
                issue_type="control_reference",
                secondaries=[],
                reader_label="reasonable_anatomy",
            )
            for i in range(CONTROL_SAMPLE_SIZE + 2)
        ]
        return positives + controls

    def test_end_to_end_all_gates_pass(self):
        cohort = self._cohort()
        report = run_bounded_shadow_audit(cohort, cache_index=make_cache(cohort, run_id="run-13"))
        self.assertEqual(report["sample"]["positive_sampled"], POSITIVE_SAMPLE_SIZE)
        self.assertEqual(report["sample"]["control_sampled"], CONTROL_SAMPLE_SIZE)
        self.assertEqual(len(report["case_rows"]), POSITIVE_SAMPLE_SIZE + CONTROL_SAMPLE_SIZE)
        self.assertTrue(report["all_gates_passed"], report["gates"])
        self.assertEqual(report["metrics"]["completeness_rate"], 1.0)
        self.assertEqual(report["metrics"]["missing"], [])
        self.assertEqual(report["missing_issue_types"], [])
        self.assertEqual(set(report["issue_types_out"]), {"excess_components", "control_reference"})
        self.assertGreaterEqual(report["cache"]["cache_hit_rate"], CACHE_HIT_RATE_FLOOR)
        self.assertEqual(report["cache"]["fallback_extractions"], 0)
        self.assertEqual(sum(report["reader_adjudication"]["basis_points"].values()), 10000)
        self.assertAlmostEqual(
            round(sum(report["reader_adjudication"]["shares_pct"].values()), 2), 100.0
        )
        self.assertEqual(set(report["reader_adjudication"]["counts"]), set(READER_CLASSES))
        for row in report["case_rows"]:
            self.assertEqual(set(row["metrics"]), set(REQUIRED_METRICS))

    def test_input_records_are_not_mutated(self):
        cohort = self._cohort()
        snapshot = copy.deepcopy(cohort)
        run_bounded_shadow_audit(cohort, cache_index=make_cache(cohort))
        self.assertEqual(cohort, snapshot)

    def test_end_to_end_with_mismatches_still_meets_hit_rate_floor(self):
        cohort = self._cohort()
        cache = make_cache(cohort, run_id="run-9")
        # break 10 of 60 entries => 50/60 hit rate, still above the 0.80 floor
        for case_id in [f"P-{i:04d}" for i in range(10)]:
            cache[case_id]["payload_hash"] = "stale"
        report = run_bounded_shadow_audit(cohort, cache_index=cache)
        self.assertEqual(report["cache"]["cache_mismatches"], 10)
        self.assertEqual(report["cache"]["fallback_extractions"], 10)
        self.assertGreaterEqual(report["cache"]["cache_hit_rate"], CACHE_HIT_RATE_FLOOR)
        self.assertTrue(report["gates"]["cache_hit_rate_ok"])
        fallback_rows = [row for row in report["case_rows"] if row["used_fallback_extractor"]]
        self.assertEqual({row["payload_source"] for row in fallback_rows}, {FALLBACK_EXTRACTOR_ID})

    def test_custom_b1_extractor_is_used_only_on_mismatch(self):
        cohort = self._cohort()[:3]
        cache = make_cache(cohort)
        cache[cohort[1]["case_id"]]["payload_hash"] = "stale"
        calls = []

        def fake_b1_extractor(record):
            calls.append(record["case_id"])
            return {"payload_hash": "re-extracted::" + str(record["case_id"])}

        report = run_bounded_shadow_audit(
            cohort, cache_index=cache, b1_extractor=fake_b1_extractor
        )
        self.assertEqual(calls, [cohort[1]["case_id"]])
        self.assertEqual(report["case_rows"][1]["payload_source"], FALLBACK_EXTRACTOR_ID)


if __name__ == "__main__":
    unittest.main()
