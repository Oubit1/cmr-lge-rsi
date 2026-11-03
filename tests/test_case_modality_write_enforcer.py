"""Offline unit tests for the write-side (case_id, modality) enforcement gate."""

import unittest

from experiments.case_modality_write_enforcer import (
    CacheMissError,
    CaseModalityWriteEnforcer,
    ExtractorCache,
    InvalidRowError,
    ReusePolicyError,
    RoundMetrics,
    WriteGateError,
    WriteResult,
    compare_rounds,
    is_valid_cache_hash,
    merge_issue_types,
    merge_slice_indices,
)

VALID_HASH = "0f" * 32  # 64 hex characters


class WriteEnforcementTests(unittest.TestCase):
    def test_duplicate_rows_rejected_and_payload_merged(self):
        enforcer = CaseModalityWriteEnforcer()
        result = enforcer.ingest(
            [
                {
                    "case_id": "C001",
                    "modality": "LGE",
                    "issue_type": "myo_excess_components",
                    "slice_indices": [3],
                },
                {
                    # same key, different spelling of modality
                    "case_id": "C001",
                    "modality": "lge",
                    "issue_type": ["small_area", "myo_excess_components"],
                    "slice_indices": [5, 3],
                },
                {
                    "case_id": "C002",
                    "modality": "LGE",
                    "issue_type": "fragmentation",
                    "slice_indices": [1],
                },
            ]
        )

        self.assertEqual(result.written_row_count, 2)
        self.assertEqual(result.duplicate_rows_written, 0)
        self.assertEqual(result.duplicates_merged, 1)
        self.assertEqual(result.duplicates_rejected, 1)
        result.assert_no_duplicates()

        canonical = {row["case_id"]: row for row in result.rows}
        self.assertEqual(
            canonical["C001"]["issue_type"],
            ["myo_excess_components", "small_area"],
        )
        self.assertEqual(canonical["C001"]["slice_indices"], [3, 5])
        self.assertEqual(canonical["C001"]["modality"], "LGE")
        self.assertEqual(canonical["C002"]["issue_type"], ["fragmentation"])

    def test_second_ingest_call_still_dedups_against_store(self):
        enforcer = CaseModalityWriteEnforcer()
        enforcer.ingest(
            [{"case_id": "C001", "modality": "LGE", "issue_type": "a", "slice_indices": [0]}]
        )
        result = enforcer.ingest(
            [
                {
                    "case_id": "C001",
                    "modality": "LGE",
                    "issue_type": "b",
                    "slice_indices": [2, 0],
                }
            ]
        )
        self.assertEqual(result.written_row_count, 1)
        self.assertEqual(result.duplicate_rows_written, 0)
        self.assertEqual(result.rows[0]["issue_type"], ["a", "b"])
        self.assertEqual(result.rows[0]["slice_indices"], [0, 2])
        enforcer.assert_no_duplicate_rows()

    def test_rows_without_case_id_or_modality_are_rejected(self):
        enforcer = CaseModalityWriteEnforcer()
        result = enforcer.ingest(
            [
                {"case_id": "", "modality": "LGE", "issue_type": "a"},
                {"case_id": "C003", "modality": "   ", "issue_type": "a"},
                "not-a-mapping",
                {"case_id": "C004", "modality": "LGE", "issue_type": "b"},
            ]
        )
        self.assertEqual(result.written_row_count, 1)
        self.assertEqual(len(result.invalid_rows), 3)
        self.assertEqual([row["case_id"] for row in result.rows], ["C004"])

    def test_invalid_slice_index_is_not_silently_dropped(self):
        enforcer = CaseModalityWriteEnforcer()
        with self.assertRaises(InvalidRowError):
            enforcer.ingest(
                [
                    {
                        "case_id": "C005",
                        "modality": "LGE",
                        "issue_type": "a",
                        "slice_indices": ["3-5"],
                    }
                ]
            )

    def test_ingest_batch_is_atomic_on_invalid_payload(self):
        enforcer = CaseModalityWriteEnforcer()
        enforcer.ingest(
            [{"case_id": "C006", "modality": "LGE", "issue_type": "a", "slice_indices": [1]}]
        )
        with self.assertRaises(InvalidRowError):
            enforcer.ingest(
                [
                    {
                        "case_id": "C007",
                        "modality": "LGE",
                        "issue_type": "b",
                        "slice_indices": [1.5],
                    }
                ]
            )
        after = enforcer.ingest([])
        self.assertEqual([row["case_id"] for row in after.rows], ["C006"])

    def test_write_result_guard_flags_forged_duplicates(self):
        forged = WriteResult(
            rows=[
                {"case_id": "C1", "modality": "LGE", "issue_type": [], "slice_indices": []},
                {"case_id": "C1", "modality": "LGE", "issue_type": [], "slice_indices": []},
            ]
        )
        self.assertEqual(forged.duplicate_rows_written, 1)
        with self.assertRaises(WriteGateError):
            forged.assert_no_duplicates()

    def test_merge_helpers_preserve_order_and_dedupe(self):
        self.assertEqual(merge_issue_types(["b", "a"], ["a", "c"]), ["b", "a", "c"])
        self.assertEqual(merge_slice_indices([5, 1], [1, 3]), [1, 3, 5])


class ExtractorCacheTests(unittest.TestCase):
    def test_is_valid_cache_hash(self):
        self.assertTrue(is_valid_cache_hash(VALID_HASH))
        self.assertTrue(is_valid_cache_hash(VALID_HASH.upper()))
        self.assertFalse(is_valid_cache_hash(""))
        self.assertFalse(is_valid_cache_hash("abc123"))
        self.assertFalse(is_valid_cache_hash("z" * 64))
        self.assertFalse(is_valid_cache_hash(None))

    def test_valid_hash_is_reused_without_building_new_extractors(self):
        built = []

        def builder(payload):  # pragma: no cover - must never run
            built.append(payload)
            return object()

        cache = ExtractorCache()
        sentinel = object()
        cache.warm(VALID_HASH, sentinel)

        handles = [cache.obtain(VALID_HASH, builder) for _ in range(10)]

        self.assertTrue(all(handle.reused for handle in handles))
        self.assertFalse(any(handle.created for handle in handles))
        self.assertTrue(all(handle.extractor is sentinel for handle in handles))
        self.assertEqual(built, [])
        self.assertEqual(cache.new_extractors, 0)
        self.assertGreaterEqual(cache.reuse_rate, 0.9)
        cache.assert_reuse_policy()

    def test_valid_hash_miss_is_refused_without_building(self):
        built = []
        cache = ExtractorCache()

        with self.assertRaises(CacheMissError):
            cache.obtain(VALID_HASH, lambda payload: built.append(payload))

        self.assertEqual(built, [])
        self.assertEqual(cache.created_count, 0)
        self.assertEqual(cache.refused_count, 1)
        self.assertEqual(cache.cached_hashes(), [])

    def test_unvalidated_hash_takes_the_cold_build_path(self):
        built = []
        cache = ExtractorCache()

        handle = cache.obtain("", lambda payload: built.append(payload) or object())
        second = cache.obtain("abc", lambda payload: built.append(payload) or object())

        self.assertTrue(handle.created and not handle.reused)
        self.assertTrue(second.created and not second.reused)
        self.assertEqual(len(built), 2)
        self.assertEqual(cache.created_count, 2)
        # an unvalidated hash is never trusted as a cache key
        self.assertEqual(cache.cached_hashes(), [])

    def test_reuse_policy_violation_is_raised(self):
        cache = ExtractorCache()
        cache.obtain("", lambda payload: object())
        cache.obtain("abc", lambda payload: object())
        cache.warm(VALID_HASH, object())
        cache.obtain(VALID_HASH, None)

        self.assertLess(cache.reuse_rate, 0.9)
        with self.assertRaises(ReusePolicyError):
            cache.assert_reuse_policy()

    def test_warm_rejects_invalid_hash(self):
        cache = ExtractorCache()
        with self.assertRaises(ValueError):
            cache.warm("not-a-hash", object())

    def test_obtain_without_builder_on_miss_is_type_error(self):
        cache = ExtractorCache()
        with self.assertRaises(TypeError):
            cache.obtain("")


class RoundMetricsTests(unittest.TestCase):
    def test_cheaper_round_without_recall_regression_is_accepted(self):
        previous = RoundMetrics(tokens=1200.0, elapsed_ms=800.0, recalled_issues=12)
        current = RoundMetrics(tokens=900.0, elapsed_ms=640.0, recalled_issues=12)
        report = compare_rounds(previous, current)
        self.assertTrue(report["token_improved"])
        self.assertTrue(report["latency_improved"])
        self.assertTrue(report["recall_not_regressed"])
        self.assertTrue(report["accepted"])
        self.assertLess(report["token_delta"], 0)

    def test_recall_regression_is_not_accepted(self):
        previous = RoundMetrics(tokens=1200.0, elapsed_ms=800.0, recalled_issues=12)
        current = RoundMetrics(tokens=900.0, elapsed_ms=640.0, recalled_issues=10)
        report = compare_rounds(previous, current)
        self.assertFalse(report["recall_not_regressed"])
        self.assertFalse(report["accepted"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
