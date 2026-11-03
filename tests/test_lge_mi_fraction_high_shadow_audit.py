import unittest
from experiments.lge_mi_fraction_high_shadow_audit import (
    RUN9_B1_EXTRACTOR,
    DuplicateCaseModalityError,
    MissingRun9B1EvidenceError,
    ScopeViolationError,
    adjudicate_positive,
    audit_lge_mi_fraction_high,
    compute_record_hash,
)

def _summary(mi_score=0.4, worst_slice_fraction=0.3, mi_multiple_components=False, myo_topology='compact', artifact_flags=()):
    return {
        'mi_score': mi_score,
        'worst_slice_fraction': worst_slice_fraction,
        'mi_multiple_components': mi_multiple_components,
        'myo_topology': myo_topology,
        'artifact_flags': tuple(artifact_flags),
    }

def _record(case_id, cohort, run_id='run-4', modality='LGE', content_hash=None, b1_payload=None, **summary_kwargs):
    summary = _summary(**summary_kwargs)
    return {
        'case_id': case_id,
        'modality': modality,
        'cohort': cohort,
        'run_id': run_id,
        'content_hash': content_hash or compute_record_hash(summary),
        'summary': summary,
        'b1_payload': b1_payload or {},
    }

def _baseline_records():
    records = []
    for index in range(30):
        records.append(_record(f'P{index:02d}', 'positive', run_id='run-4', mi_score=0.4 + index * 0.01, worst_slice_fraction=0.2 + index * 0.005, myo_topology='compact'))
    for index in range(10):
        records.append(_record(f'C{index:02d}', 'control', run_id='run-5', mi_score=0.1 + index * 0.005, worst_slice_fraction=0.1 + index * 0.005, myo_topology='compact'))
    return records

class LgeMiFractionHighShadowAuditTests(unittest.TestCase):
    def test_reuses_allowed_runs_and_deduplicates(self):
        report = audit_lge_mi_fraction_high(_baseline_records())
        self.assertEqual(report['positive_count'], 30)
        self.assertEqual(report['control_count'], 10)
        self.assertEqual(report['cache_reuse_rate'], 1.0)
        self.assertEqual(report['cache_reused_records'], 40)
        self.assertEqual(report['hash_mismatch_fallbacks'], 0)
        self.assertEqual(report['duplicate_case_modality_records'], 0)
        self.assertEqual(len(report['adjudications']), 30)
        self.assertTrue(all(not item['fallback_used'] for item in report['adjudications']))
        self.assertEqual(sum(report['positive_verdict_counts'].values()), 30)
        self.assertEqual(set(report['per_case_modality_counts'].values()), {1})
        self.assertIn('true_high_mi', report['positive_verdict_counts'])

    def test_hash_mismatch_uses_run9_b1_and_reports(self):
        records = _baseline_records()
        records[0] = _record('P00', 'positive', run_id='run-4', content_hash='mismatch', b1_payload=_summary(mi_score=0.55, worst_slice_fraction=0.25, myo_topology='compact'))
        report = audit_lge_mi_fraction_high(records)
        self.assertEqual(report['hash_mismatch_fallbacks'], 1)
        self.assertAlmostEqual(report['cache_reuse_rate'], 39 / 40)
        self.assertEqual(len(report['fallback_provenance']), 1)
        self.assertEqual(report['fallback_provenance'][0]['extractor'], RUN9_B1_EXTRACTOR)
        self.assertEqual(report['fallback_provenance'][0]['cached_run'], 'run-4')
        adjudication = [item for item in report['adjudications'] if item['case_id'] == 'P00'][0]
        self.assertTrue(adjudication['fallback_used'])
        self.assertEqual(adjudication['source'], RUN9_B1_EXTRACTOR)

    def test_hash_mismatch_without_b1_evidence_raises(self):
        records = _baseline_records()
        records[0] = _record('P00', 'positive', run_id='run-4', content_hash='mismatch')
        with self.assertRaises(MissingRun9B1EvidenceError):
            audit_lge_mi_fraction_high(records)

    def test_duplicate_case_modality_is_rejected(self):
        records = _baseline_records()
        records.append(_record('P00', 'positive', run_id='run-4', modality='LGE'))
        with self.assertRaises(DuplicateCaseModalityError):
            audit_lge_mi_fraction_high(records)

    def test_disallowed_cache_run_is_rejected(self):
        records = _baseline_records()
        records[0]['run_id'] = 'run-99'
        with self.assertRaises(ScopeViolationError):
            audit_lge_mi_fraction_high(records)

    def test_adjudication_separates_artifact_overestimate_and_true_high(self):
        artifact = adjudicate_positive(_summary(artifact_flags=('motion',)), 'run-4', False, 'A')
        overestimated = adjudicate_positive(_summary(mi_score=0.4, worst_slice_fraction=0.8), 'run-4', False, 'B')
        true_high = adjudicate_positive(_summary(mi_score=0.4, worst_slice_fraction=0.3), 'run-4', False, 'C')
        self.assertEqual(artifact['verdict'], 'artifact')
        self.assertEqual(overestimated['verdict'], 'overestimated')
        self.assertEqual(true_high['verdict'], 'true_high_mi')

if __name__ == '__main__':
    unittest.main()
