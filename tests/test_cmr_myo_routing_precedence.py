'''Offline tests for CMR myocardial undersegmentation routing precedence.'''

import unittest

from experiments.cmr_myo_routing_precedence import (
    CMR_CENTROID_DRIFT_ISSUE,
    CMR_EMPTY_ISSUE,
    CMR_EXCESS_COMPONENTS_ISSUE,
    CMR_STRATIFICATION_ISSUES,
    CMR_SMALL_AREA_ISSUE,
    CROSS_MODALITY_GATING_ALLOWED,
    DEFAULT_ESCALATION_BUDGET,
    DUPLICATION_GUARDS,
    FORBIDDEN_LABEL_EDITS,
    LGE_FRACTION_HIGH_ISSUE,
    LGE_MULTIPLE_COMPONENTS_ISSUE,
    NEVER_REEXECUTE,
    ROUTE_CMR_SMALL_AREA_AUDIT,
    ROUTE_LGE_FRAGMENTATION,
    ROUTING_PRECEDENCE_TEXT,
    RUN9_ISSUE_COUNTS,
    TOOL_GUIDANCE,
    escalation_text,
    is_escalation_slice,
    is_label_edit_allowed,
    route,
    run9_precedence_order,
    select_escalation_slices,
)


class CmrRoutingPrecedenceTests(unittest.TestCase):
    def test_run9_precedence_order_matches_counts(self):
        order = run9_precedence_order()
        self.assertLess(order.index(CMR_SMALL_AREA_ISSUE), order.index(CMR_EMPTY_ISSUE))
        self.assertLess(order.index(CMR_EXCESS_COMPONENTS_ISSUE), order.index(CMR_EMPTY_ISSUE))
        self.assertLess(order.index(CMR_EMPTY_ISSUE), order.index(CMR_CENTROID_DRIFT_ISSUE))
        self.assertIn('11,105', ROUTING_PRECEDENCE_TEXT)
        self.assertIn('3,397', ROUTING_PRECEDENCE_TEXT)

    def test_small_area_routes_to_bounded_cmr_audit(self):
        decision = route('case-any', 'cmr', [CMR_SMALL_AREA_ISSUE, CMR_EMPTY_ISSUE])
        self.assertEqual(decision.route, ROUTE_CMR_SMALL_AREA_AUDIT)
        self.assertTrue(decision.read_only)
        self.assertIn(CMR_EMPTY_ISSUE, decision.stratifiers)
        self.assertEqual(decision.trigger_issues, (CMR_SMALL_AREA_ISSUE,))

    def test_excess_components_routes_to_bounded_cmr_audit(self):
        decision = route('case-any', 'cmr', [CMR_EXCESS_COMPONENTS_ISSUE])
        self.assertEqual(decision.route, ROUTE_CMR_SMALL_AREA_AUDIT)
        self.assertTrue(decision.read_only)
        self.assertIn(CMR_EXCESS_COMPONENTS_ISSUE, decision.trigger_issues)

    def test_empty_stratifies_within_bounded_cmr_audit(self):
        decision = route('case-any', 'cmr', [CMR_EMPTY_ISSUE])
        self.assertEqual(decision.route, ROUTE_CMR_SMALL_AREA_AUDIT)
        self.assertTrue(decision.read_only)
        self.assertEqual(decision.trigger_issues, ())
        self.assertIn(CMR_EMPTY_ISSUE, decision.stratifiers)

    def test_all_cmr_stratifiers_are_not_separate_routes(self):
        for issue in sorted(CMR_STRATIFICATION_ISSUES):
            decision = route('case-any', 'cmr', [issue])
            self.assertEqual(decision.route, ROUTE_CMR_SMALL_AREA_AUDIT)
            self.assertIn(issue, decision.stratifiers)
            self.assertNotEqual(decision.route, issue)

    def test_lge_fragmentation_issues_route_to_lge_rules(self):
        for issue in (LGE_MULTIPLE_COMPONENTS_ISSUE, LGE_FRACTION_HIGH_ISSUE):
            decision = route('case-any', 'lge', [issue])
            self.assertEqual(decision.route, ROUTE_LGE_FRAGMENTATION)
            self.assertEqual(decision.lge_route, ROUTE_LGE_FRAGMENTATION)

    def test_cmr_route_does_not_duplicate_lge_fragmentation_rules(self):
        decision = route('case-any', 'cmr', [CMR_SMALL_AREA_ISSUE, LGE_MULTIPLE_COMPONENTS_ISSUE])
        self.assertEqual(decision.route, ROUTE_CMR_SMALL_AREA_AUDIT)
        self.assertNotEqual(decision.route, ROUTE_LGE_FRAGMENTATION)
        self.assertIn('run-3/4 LGE fragmentation rules', DUPLICATION_GUARDS[0])
        combined_guards = ' '.join(DUPLICATION_GUARDS) + ' ' + ' '.join(NEVER_REEXECUTE)
        for item in NEVER_REEXECUTE:
            self.assertIn(item, combined_guards)

    def test_invariants_forbid_label_edits(self):
        for operation in FORBIDDEN_LABEL_EDITS:
            self.assertFalse(is_label_edit_allowed(operation))
        decision = route('case-any', 'cmr', [CMR_SMALL_AREA_ISSUE])
        self.assertEqual(decision.label_edits, ())

    def test_invariants_forbid_cross_modality_gating(self):
        self.assertFalse(CROSS_MODALITY_GATING_ALLOWED)
        decision = route('case-any', 'cmr', [CMR_SMALL_AREA_ISSUE])
        self.assertFalse(decision.cross_modality_gating)
        self.assertTrue(decision.paired_lge_report_only)

    def test_tool_guidance_requires_single_harness(self):
        self.assertIn('one cached digest', TOOL_GUIDANCE)
        self.assertIn('one consolidated extractor', TOOL_GUIDANCE)
        self.assertIn('never build a second harness', TOOL_GUIDANCE)

    def test_escalation_is_capped_by_budget(self):
        candidates = [
            {'kind': 'discordant_component_count'},
            {'kind': 'discordant_topology'},
            {'kind': 'merge'},
            {'kind': 'drop'},
            {'kind': 'unrelated'},
        ]
        selected = select_escalation_slices(candidates, budget=2)
        self.assertEqual(len(selected), 2)
        self.assertTrue(all(is_escalation_slice(item) for item in selected))

    def test_escalation_text_has_no_absolute_paths_or_case_ids(self):
        text = escalation_text(budget=DEFAULT_ESCALATION_BUDGET)
        self.assertIn(str(DEFAULT_ESCALATION_BUDGET), text)
        self.assertNotIn('/', text)
        self.assertNotIn('case-', text.lower())
        self.assertIn('pre-declared budget', text)
        self.assertIn('No new threshold tuning', text)


if __name__ == '__main__':
    unittest.main()
