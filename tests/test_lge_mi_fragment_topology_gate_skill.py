"""Offline tests for the read-only LGE MI fragment topology gate skill."""

import unittest

from experiments.lge_mi_fragment_topology_gate_skill import (
    NO_AUTO_LABEL_EDIT_CLAUSE,
    SUPPRESSION_CONDITIONS,
    additive_edit_invariants,
    audit_diff,
    heldout_checklist,
    module_hash,
    run_evaluation,
    topology_gate,
)


def _blank(h, w):
    return [[False] * w for _ in range(h)]


def _ring(h, w):
    m = _blank(h, w)
    for i in range(h):
        for j in range(w):
            if i in (0, h - 1) or j in (0, w - 1):
                m[i][j] = True
    return m


class TopologyGateTests(unittest.TestCase):
    def test_empty_mi_suppressed(self):
        res = topology_gate("c1", _blank(8, 8), _ring(8, 8))
        self.assertFalse(res.gate_passed)
        self.assertEqual(res.suppression_reason, "empty_mi")

    def test_empty_myo_suppressed(self):
        mi = _blank(8, 8)
        mi[3][3] = True
        res = topology_gate("c2", mi, _blank(8, 8))
        self.assertFalse(res.gate_passed)
        self.assertEqual(res.suppression_reason, "empty_myo")

    def test_temporal_hole_suppressed(self):
        mi = _blank(8, 8)
        mi[3][3] = True
        res = topology_gate("c3", mi, _ring(8, 8), temporal_hole=True)
        self.assertFalse(res.gate_passed)
        self.assertEqual(res.suppression_reason, "temporal_hole")

    def test_single_component_passes_gate(self):
        mi = _blank(10, 10)
        for i in range(3, 7):
            for j in range(3, 7):
                mi[i][j] = True
        res = topology_gate("c4", mi, _ring(10, 10))
        self.assertTrue(res.gate_passed)
        self.assertEqual(res.mi_components, 1)
        self.assertEqual(res.flagged_fragments, 0)

    def test_border_fragment_suppressed(self):
        mi = _blank(10, 10)
        for i in range(3, 8):
            for j in range(3, 8):
                mi[i][j] = True
        mi[0][0] = True
        res = topology_gate("c5", mi, _ring(10, 10))
        self.assertTrue(res.gate_passed)
        self.assertEqual(res.flagged_fragments, 0)

    def test_non_adjacent_fragment_flagged(self):
        mi = _blank(20, 20)
        for i in range(2, 12):
            for j in range(2, 12):
                mi[i][j] = True
        mi[17][17] = True
        myo = _blank(20, 20)
        for i in range(2, 12):
            for j in range(2, 12):
                myo[i][j] = True
        res = topology_gate("c6", mi, myo)
        self.assertTrue(res.gate_passed)
        self.assertEqual(res.flagged_fragments, 1)
        self.assertEqual(res.flagged_non_adjacent, 1)


class AdditiveEditTests(unittest.TestCase):
    def test_identical_masks_ok(self):
        m = _blank(8, 8)
        for i in range(2, 6):
            for j in range(2, 6):
                m[i][j] = True
        inv = additive_edit_invariants(m, m)
        self.assertTrue(inv["additive_edit_ok"])
        self.assertTrue(inv["components_preserved"])
        self.assertTrue(inv["holes_preserved"])

    def test_border_touch_rejected(self):
        before = _blank(8, 8)
        for i in range(2, 6):
            for j in range(2, 6):
                before[i][j] = True
        after = [row[:] for row in before]
        after[0][0] = True
        inv = additive_edit_invariants(before, after)
        self.assertFalse(inv["additive_edit_ok"])
        self.assertTrue(inv["touches_border"])

    def test_component_count_change_rejected(self):
        before = _blank(10, 10)
        for i in range(2, 6):
            for j in range(2, 6):
                before[i][j] = True
        after = [row[:] for row in before]
        after[8][8] = True
        inv = additive_edit_invariants(before, after)
        self.assertFalse(inv["components_preserved"])
        self.assertFalse(inv["additive_edit_ok"])


class SkillContractTests(unittest.TestCase):
    def test_no_auto_label_edit_clause_present(self):
        self.assertIn("never write", NO_AUTO_LABEL_EDIT_CLAUSE)

    def test_suppression_conditions_complete(self):
        for cond in ("empty_mi", "empty_myo", "border_fragment", "temporal_hole"):
            self.assertIn(cond, SUPPRESSION_CONDITIONS)

    def test_heldout_checklist_denominator(self):
        cl = heldout_checklist(20)
        self.assertTrue(cl["denominator_ok"])
        self.assertEqual(cl["target_volumes"], 20)
        cl2 = heldout_checklist(5)
        self.assertFalse(cl2["denominator_ok"])

    def test_module_hash_stable(self):
        self.assertEqual(module_hash(), module_hash())

    def test_audit_diff_zero_writes(self):
        cur = {"module_hash": "abc", "cases": {"c1": {"x": 1}}}
        d = audit_diff(None, cur)
        self.assertEqual(d["production_writes"], 0)
        self.assertEqual(d["added_cases"], ["c1"])


class RunEvaluationTests(unittest.TestCase):
    def test_run_evaluation_reports_unavailable(self):
        data = {
            "records": [
                {"case_id": "a", "before_area": 100.0, "after_area": 100.0,
                 "before_cavity": 0.2, "after_cavity": 0.2},
                {"case_id": "b", "before_area": 200.0, "after_area": 200.0,
                 "before_cavity": 0.4, "after_cavity": 0.4, "quality_delta": 0.1},
            ]
        }
        out = run_evaluation(data)
        self.assertEqual(out["n_records"], 2)
        self.assertEqual(out["additive_edit_invariants"]["area_unchanged"], 2)
        self.assertEqual(out["additive_edit_invariants"]["cavity_unchanged"], 2)
        self.assertIn("fragment_topology", out["unavailable"])
        self.assertIn("concordance", out["unavailable"])
        self.assertEqual(out["writes"]["labels"], 0)

    def test_run_evaluation_detects_area_change(self):
        data = {"records": [
            {"case_id": "a", "before_area": 100.0, "after_area": 99.0,
             "before_cavity": 0.2, "after_cavity": 0.2},
        ]}
        out = run_evaluation(data)
        self.assertEqual(out["additive_edit_invariants"]["area_unchanged"], 0)
        self.assertEqual(out["additive_edit_invariants"]["area_changed_cases"], ["a"])


if __name__ == "__main__":
    unittest.main()
