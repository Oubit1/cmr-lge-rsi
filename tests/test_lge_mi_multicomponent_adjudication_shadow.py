import unittest

from experiments.lge_mi_multicomponent_adjudication_shadow import (
    CaseEvidence,
    ComponentEvidence,
    classify_case,
    is_downgrade_candidate,
    resolve_cache_source,
    secondary_component_deficits,
    select_bounded_sample,
    summarize_shadow_audit,
)


def comp(area, distance=0.0, touches=True, cross=3):
    return ComponentEvidence(
        area_mm2=area,
        centroid_distance_mm=distance,
        touches_myocardium=touches,
        cross_slice_support=cross,
    )


def case(
    case_id,
    stratum='mi_multiple_components',
    components=(),
    cache_run_id='run-9',
    cache_hash='same',
    expected_cache_hash='same',
    reader_label=None,
):
    return CaseEvidence(
        case_id=case_id,
        stratum=stratum,
        cache_run_id=cache_run_id,
        components=tuple(components),
        cache_hash=cache_hash,
        expected_cache_hash=expected_cache_hash,
        reader_label=reader_label,
    )


class ClassificationTests(unittest.TestCase):
    def test_true_multifocal_requires_supported_separated_secondary(self):
        evidence = case('mi-001', components=[comp(100.0), comp(25.0, distance=12.0)])
        self.assertEqual(classify_case(evidence), 'true_multifocal')

    def test_unsupported_secondary_is_fragmentation_or_noise(self):
        evidence = case(
            'mi-002',
            components=[comp(100.0), comp(4.0, distance=2.0, touches=False, cross=0)],
        )
        self.assertEqual(classify_case(evidence), 'fragmentation_or_noise')

    def test_mixed_support_is_indeterminate(self):
        evidence = case(
            'mi-003',
            components=[
                comp(100.0),
                comp(25.0, distance=12.0),
                comp(4.0, distance=2.0, touches=False, cross=0),
            ],
        )
        self.assertEqual(classify_case(evidence), 'indeterminate')

    def test_empty_components_are_indeterminate(self):
        self.assertEqual(classify_case(case('mi-004', components=[])), 'indeterminate')


class CacheSourceTests(unittest.TestCase):
    def test_reuses_allowed_cache_when_hash_matches(self):
        source = resolve_cache_source(case('mi-001', cache_run_id='run-13'))
        self.assertTrue(source['reused_cache'])
        self.assertFalse(source['report_required'])
        self.assertIsNone(source['extractor'])

    def test_hash_mismatch_routes_to_run9_b1_and_reports(self):
        source = resolve_cache_source(
            case(
                'mi-001',
                cache_run_id='run-13',
                cache_hash='changed',
                expected_cache_hash='same',
            )
        )
        self.assertEqual(source['run_id'], 'run-9')
        self.assertEqual(source['extractor'], 'B1')
        self.assertTrue(source['report_required'])
        self.assertFalse(source['reused_cache'])

    def test_unsupported_run_routes_to_run9_b1_and_reports(self):
        source = resolve_cache_source(case('mi-001', cache_run_id='run-99'))
        self.assertEqual(source['run_id'], 'run-9')
        self.assertEqual(source['extractor'], 'B1')
        self.assertEqual(source['reason'], 'unsupported_cache_run')
        self.assertTrue(source['report_required'])


class BoundedSampleTests(unittest.TestCase):
    def test_selects_deterministic_bounded_positives_and_controls(self):
        cases = [
            case(f'mi-{index:03d}', components=[comp(100.0), comp(25.0, distance=12.0)])
            for index in range(5)
        ]
        cases.extend(
            case(f'lge-{index:03d}', stratum='lge', components=[comp(100.0)])
            for index in range(3)
        )
        selected = select_bounded_sample(cases, positive_limit=2, control_limit=1)
        self.assertEqual(
            [item.case_id for item in selected],
            ['mi-000', 'mi-001', 'lge-000'],
        )


class SecondaryDeficitTests(unittest.TestCase):
    def test_counts_secondary_components_lacking_adjacency_or_cross_slice_support(self):
        evidence = case(
            'mi-001',
            components=[
                comp(100.0),
                comp(25.0, distance=12.0),
                comp(4.0, distance=2.0, touches=False, cross=0),
            ],
        )
        deficits = secondary_component_deficits(evidence)
        self.assertEqual(deficits['secondary_component_count'], 2)
        self.assertEqual(deficits['lacking_myocardial_adjacency'], 1)
        self.assertEqual(deficits['lacking_cross_slice_support'], 1)
        self.assertEqual(deficits['lacking_myocardial_adjacency_or_cross_slice'], 1)


class SummaryTests(unittest.TestCase):
    def test_summary_reports_proportions_agreement_deficits_and_downgrade(self):
        cases = [
            case(
                'mi-001',
                components=[comp(100.0), comp(25.0, distance=12.0)],
                reader_label='true_multifocal',
            ),
            case(
                'mi-002',
                components=[comp(100.0), comp(4.0, distance=2.0, touches=False, cross=0)],
                reader_label='fragmentation_or_noise',
            ),
            case(
                'mi-003',
                components=[comp(100.0), comp(25.0, distance=12.0)],
                reader_label='true_multifocal',
            ),
            case(
                'lge-001',
                stratum='lge',
                components=[comp(100.0)],
                reader_label='fragmentation_or_noise',
            ),
        ]

        summary = summarize_shadow_audit(cases)

        self.assertTrue(summary['shadow_only'])
        self.assertEqual(summary['write_operations'], 0)
        self.assertEqual(summary['bounded_case_count'], 4)
        self.assertEqual(summary['label_counts']['true_multifocal'], 2)
        self.assertEqual(summary['label_counts']['fragmentation_or_noise'], 2)
        self.assertEqual(summary['reader_agreement']['agreement'], 1.0)
        self.assertTrue(summary['reader_agreement']['meets_floor'])
        self.assertEqual(
            summary['secondary_component_deficits']['secondary_component_count'], 3
        )
        self.assertEqual(
            summary['secondary_component_deficits']['lacking_myocardial_adjacency'], 1
        )
        self.assertEqual(
            summary['secondary_component_deficits']['lacking_cross_slice_support'], 1
        )
        self.assertEqual(summary['downgrade_estimate']['downgrade_candidate_count'], 1)
        self.assertAlmostEqual(
            summary['downgrade_estimate']['downgrade_candidate_fraction'], 1 / 3
        )


class DowngradeCandidateTests(unittest.TestCase):
    def test_fragmentation_case_is_shadow_downgrade_candidate(self):
        evidence = case(
            'mi-001',
            components=[comp(100.0), comp(4.0, distance=2.0, touches=False, cross=0)],
        )
        self.assertTrue(is_downgrade_candidate(evidence))

    def test_true_multifocal_is_not_shadow_downgrade_candidate(self):
        evidence = case('mi-001', components=[comp(100.0), comp(25.0, distance=12.0)])
        self.assertFalse(is_downgrade_candidate(evidence))

    def test_control_stratum_is_not_shadow_downgrade_candidate(self):
        evidence = case('lge-001', stratum='lge', components=[comp(100.0)])
        self.assertFalse(is_downgrade_candidate(evidence))


if __name__ == '__main__':
    unittest.main()
