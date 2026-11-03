import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from experiments.review_centroid_outlier_shadow_audit import (
    ALLOWED_CACHE_RUNS,
    CENTROID_DRIFT_FLAG,
    IMPORT_EXTRACTOR_RUN,
    IMPORT_EXTRACTOR_STAGE,
    MAX_IMPORT_EXTRACTOR_CALLS,
    MYO_TOPOLOGY_FLAG,
    REVIEW_FLAG,
    TARGET_CONTROLS,
    TARGET_POSITIVES,
    audit_cache,
    dedupe_case_modality,
    report_to_dict,
    run_shadow_audit,
    select_cohort,
)

RUN_CYCLE = (4, 5, 9, 13, 14, 15, 16, 17, 18)


def make_case(case_id, modality='LGE', run_id=4, flags=(), slice_index=1, frame_index=0):
    return {
        'case_id': case_id,
        'modality': modality,
        'run_id': run_id,
        'flags': list(flags),
        'slice_index': slice_index,
        'frame_index': frame_index,
    }


def build_cases(positive_count=60, control_count=20):
    cases = []
    for index in range(positive_count):
        flags = [REVIEW_FLAG]
        if index % 2 == 0:
            flags.append(CENTROID_DRIFT_FLAG)
        if index % 3 == 0:
            flags.append(MYO_TOPOLOGY_FLAG)
        cases.append(make_case('P%03d' % index, 'LGE', RUN_CYCLE[index % len(RUN_CYCLE)], flags))
    for index in range(control_count):
        cases.append(make_case('C%03d' % index, 'LGE', RUN_CYCLE[index % len(RUN_CYCLE)], []))
    return cases


def make_summary(run_id, digest):
    return {'run_id': run_id, 'digest': digest, 'payload': {'run_id': run_id}}


class CohortSelectionTest(unittest.TestCase):
    def test_target_counts(self):
        positives, controls = select_cohort(build_cases())
        self.assertEqual(len(positives), TARGET_POSITIVES)
        self.assertEqual(len(controls), TARGET_CONTROLS)
        for case in positives:
            self.assertIn(REVIEW_FLAG, case.flags)

    def test_stratified_by_centroid_drift(self):
        positives, _ = select_cohort(build_cases())
        drift = [c for c in positives if CENTROID_DRIFT_FLAG in c.flags]
        non_drift = [c for c in positives if CENTROID_DRIFT_FLAG not in c.flags]
        self.assertTrue(drift)
        self.assertTrue(non_drift)
        self.assertEqual(len(drift), 15)
        self.assertEqual(len(non_drift), 15)

    def test_run_30_never_selected(self):
        cases = build_cases()
        cases.append(make_case('P000RUN30', 'LGE', 30, [REVIEW_FLAG, CENTROID_DRIFT_FLAG]))
        cases.append(make_case('C000RUN30', 'LGE', 30, []))
        positives, controls = select_cohort(cases)
        for case in list(positives) + list(controls):
            self.assertNotEqual(case.run_id, 30)
            self.assertIn(case.run_id, RUN_CYCLE)


class DedupeTest(unittest.TestCase):
    def test_single_record_per_case_modality(self):
        cases = [
            make_case('X1', 'LGE', 4, [REVIEW_FLAG]),
            make_case('X1', 'LGE', 5, [REVIEW_FLAG]),
            make_case('X1', 'LGE', 9, [REVIEW_FLAG], slice_index=None, frame_index=None),
            make_case('X1', 'T1', 4, [REVIEW_FLAG]),
        ]
        records = dedupe_case_modality(cases)
        keys = [(r.case_id, r.modality) for r in records]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(len(records), 2)
        lge = [r for r in records if r.modality == 'LGE'][0]
        self.assertTrue(lge.is_addressable())

    def test_duplicate_count_reported(self):
        cases = build_cases(positive_count=40, control_count=10)
        cases.append(make_case('P000', 'LGE', 4, [REVIEW_FLAG]))
        report = run_shadow_audit(cases)
        self.assertEqual(report.duplicate_case_modality_count, 1)
        selected_keys = [(c.case_id, c.modality) for c in report.selected_positives]
        self.assertEqual(len(selected_keys), len(set(selected_keys)))
        self.assertEqual(report.unique_case_modality_count, len(set((c.case_id, c.modality) for c in dedupe_case_modality(cases))))


class AddressabilityTest(unittest.TestCase):
    def test_full_addressability(self):
        report = run_shadow_audit(build_cases())
        self.assertEqual(report.positive_addressable_count, TARGET_POSITIVES)
        self.assertEqual(report.positive_addressability_rate, 1.0)
        self.assertNotIn('addressability_below_threshold', report.notes)

    def test_low_addressability_flagged(self):
        cases = build_cases()
        unaddressable = ('P000', 'P002', 'P004', 'P006', 'P008', 'P010')
        for case in cases:
            if REVIEW_FLAG in case['flags'] and case['case_id'] in unaddressable:
                case['slice_index'] = None
                case['frame_index'] = None
        report = run_shadow_audit(cases)
        self.assertLess(report.positive_addressability_rate, 0.95)
        self.assertIn('addressability_below_threshold', report.notes)
        self.assertEqual(report.positive_addressable_count, TARGET_POSITIVES - len(unaddressable))


class CooccurrenceTest(unittest.TestCase):
    def test_cooccurrence_and_unexplained_numbers(self):
        report = run_shadow_audit(build_cases())
        positives = report.selected_positives
        self.assertEqual(report.positive_selected_count, TARGET_POSITIVES)
        drift = sum(1 for c in positives if CENTROID_DRIFT_FLAG in c.flags)
        topology = sum(1 for c in positives if MYO_TOPOLOGY_FLAG in c.flags)
        unexplained = sum(1 for c in positives if not ({CENTROID_DRIFT_FLAG, MYO_TOPOLOGY_FLAG} & set(c.flags)))
        self.assertEqual(report.centroid_drift_cooccurrence_count, drift)
        self.assertEqual(report.centroid_drift_cooccurrence_rate, drift / TARGET_POSITIVES)
        self.assertEqual(report.myo_topology_cooccurrence_count, topology)
        self.assertEqual(report.myo_topology_cooccurrence_rate, topology / TARGET_POSITIVES)
        self.assertEqual(report.unexplained_by_existing_flags_count, unexplained)
        self.assertEqual(report.unexplained_by_existing_flags_rate, unexplained / TARGET_POSITIVES)
        self.assertEqual(report.centroid_drift_cooccurrence_count, 15)
        self.assertEqual(report.unexplained_by_existing_flags_count, 10)

    def test_report_to_dict_is_numeric(self):
        report = run_shadow_audit(build_cases())
        payload = report_to_dict(report)
        self.assertIsInstance(payload['positive_addressability_rate'], float)
        self.assertIsInstance(payload['centroid_drift_cooccurrence_rate'], float)
        self.assertIsInstance(payload['myo_topology_cooccurrence_rate'], float)
        self.assertIsInstance(payload['unexplained_by_existing_flags_rate'], float)
        self.assertIsInstance(payload['notes'], list)


class CacheHashAuditTest(unittest.TestCase):
    def test_all_matched(self):
        summaries = [make_summary(run_id, 'digest-%d' % run_id) for run_id in ALLOWED_CACHE_RUNS]
        observed = {run_id: 'digest-%d' % run_id for run_id in ALLOWED_CACHE_RUNS}
        audit = audit_cache(summaries, observed)
        self.assertEqual(audit.eligible_runs, tuple(sorted(ALLOWED_CACHE_RUNS)))
        self.assertEqual(audit.checked_count, len(ALLOWED_CACHE_RUNS))
        self.assertEqual(audit.hash_checked_rate, 1.0)
        self.assertEqual(audit.hash_match_rate, 1.0)
        self.assertEqual(audit.mismatched_count, 0)
        self.assertEqual(audit.import_extractor_calls, 0)

    def test_missing_digest_lowers_checked_rate(self):
        summaries = [make_summary(run_id, 'digest-%d' % run_id) for run_id in ALLOWED_CACHE_RUNS]
        observed = {4: 'digest-4'}
        audit = audit_cache(summaries, observed)
        self.assertLess(audit.hash_checked_rate, 1.0)
        report = run_shadow_audit(build_cases(), summaries, observed)
        self.assertIn('cache_hash_check_incomplete', report.notes)

    def test_mismatch_uses_run9_b1_at_most_once(self):
        calls = []

        def extractor(run_id, stage):
            calls.append((run_id, stage))
            return {'run_id': run_id, 'stage': stage, 'imported': True}

        summaries = [make_summary(4, 'digest-4'), make_summary(9, 'digest-9'), make_summary(9, 'digest-9b')]
        observed = {4: 'other-4', 9: 'other-9'}
        audit = audit_cache(summaries, observed, extractor)
        self.assertEqual(audit.import_extractor_calls, MAX_IMPORT_EXTRACTOR_CALLS)
        self.assertEqual(calls, [(IMPORT_EXTRACTOR_RUN, IMPORT_EXTRACTOR_STAGE)])
        self.assertEqual(audit.mismatched_count, 3)
        called = [m for m in audit.mismatches if m.import_extractor_called]
        self.assertEqual(len(called), 1)
        self.assertEqual(called[0].source_run_id, IMPORT_EXTRACTOR_RUN)
        self.assertEqual(called[0].import_extractor_run, IMPORT_EXTRACTOR_RUN)
        self.assertEqual(called[0].import_extractor_stage, IMPORT_EXTRACTOR_STAGE)
        self.assertEqual(called[0].expected_digest, 'digest-9')
        self.assertEqual(called[0].observed_digest, 'other-9')
        self.assertEqual(called[0].replacement_payload, {'run_id': 9, 'stage': 'B1', 'imported': True})
        uncalled = [m for m in audit.mismatches if not m.import_extractor_called]
        self.assertEqual(len(uncalled), 2)

    def test_non_run9_mismatch_does_not_call_extractor(self):
        calls = []

        def extractor(run_id, stage):
            calls.append((run_id, stage))
            return {'run_id': run_id, 'stage': stage}

        summaries = [make_summary(4, 'digest-4'), make_summary(13, 'digest-13')]
        observed = {4: 'other-4', 13: 'other-13'}
        audit = audit_cache(summaries, observed, extractor)
        self.assertEqual(calls, [])
        self.assertEqual(audit.import_extractor_calls, 0)
        self.assertEqual(audit.mismatched_count, 2)

    def test_disallowed_runs_are_ignored(self):
        summaries = [make_summary(30, 'digest-30'), make_summary(6, 'digest-6'), make_summary(4, 'digest-4')]
        observed = {30: 'x', 6: 'y', 4: 'digest-4'}
        audit = audit_cache(summaries, observed)
        self.assertEqual(audit.eligible_runs, (4,))
        self.assertEqual(audit.import_extractor_calls, 0)
        self.assertEqual(audit.hash_checked_rate, 1.0)

    def test_mismatch_note_reported(self):
        summaries = [make_summary(4, 'digest-4')]
        observed = {4: 'other-4'}
        report = run_shadow_audit(build_cases(), summaries, observed)
        self.assertIn('cache_hash_mismatch_reported', report.notes)
        self.assertEqual(report.cache_audit.import_extractor_calls, 0)
        self.assertEqual(report.cache_audit.mismatched_count, 1)


if __name__ == '__main__':
    unittest.main()
