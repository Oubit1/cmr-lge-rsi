"""Offline unit tests for the narrow small-myocardial-area rule draft."""
import unittest

from experiments.cmr_myo_small_area_narrow_rule import (
    DECISION_CLEAR,
    DECISION_ESCALATE,
    DECISION_FLAG,
    MIN_CONSECUTIVE_SLICES,
    REASON_AREA_UNMEASURABLE,
    REASON_EVIDENCE_INCOMPLETE,
    REASON_TEMPORAL_ALIGNMENT,
    REASON_TEMPORAL_HOLE,
    REASON_TOUCHES_BORDER,
    STATUS_GATED,
    STATUS_NOT_SMALL,
    STATUS_SMALL,
    STATUS_UNMEASURABLE,
    agreement_rate,
    evaluate_stack,
    gate_reasons,
    longest_small_run,
    slice_area_ratio,
    slice_status,
    tally_decisions,
)


def _slice(myo, pool, aligned=True, border=False, hole=False):
    return {
        "myo_area": myo,
        "blood_pool_area": pool,
        "temporal_aligned": aligned,
        "touches_border": border,
        "temporal_hole": hole,
    }


class AreaRatioTests(unittest.TestCase):
    def test_ratio_is_myo_over_pool(self):
        self.assertAlmostEqual(slice_area_ratio(_slice(30.0, 100.0)), 0.3)

    def test_ratio_rejects_unusable_values(self):
        self.assertIsNone(slice_area_ratio(_slice(None, 100.0)))
        self.assertIsNone(slice_area_ratio(_slice(30.0, 0.0)))
        self.assertIsNone(slice_area_ratio(_slice(30.0, -5.0)))
        self.assertIsNone(slice_area_ratio(_slice(float("nan"), 100.0)))
        self.assertIsNone(slice_area_ratio(_slice(True, 100.0)))


class GateReasonTests(unittest.TestCase):
    def test_no_reasons_when_all_gates_clear(self):
        self.assertEqual(gate_reasons(_slice(30.0, 100.0)), ())

    def test_each_gate_field_maps_to_its_reason(self):
        self.assertEqual(
            gate_reasons(_slice(30.0, 100.0, aligned=False)),
            (REASON_TEMPORAL_ALIGNMENT,),
        )
        self.assertEqual(
            gate_reasons(_slice(30.0, 100.0, border=True)),
            (REASON_TOUCHES_BORDER,),
        )
        self.assertEqual(
            gate_reasons(_slice(30.0, 100.0, hole=True)),
            (REASON_TEMPORAL_HOLE,),
        )

    def test_missing_or_none_gate_field_is_incomplete_evidence(self):
        record = _slice(30.0, 100.0)
        del record["touches_border"]
        self.assertIn(REASON_EVIDENCE_INCOMPLETE, gate_reasons(record))

        record = _slice(30.0, 100.0)
        record["temporal_hole"] = None
        self.assertIn(REASON_EVIDENCE_INCOMPLETE, gate_reasons(record))

    def test_reasons_are_deduplicated(self):
        record = _slice(30.0, 100.0)
        del record["touches_border"]
        del record["temporal_hole"]
        self.assertEqual(gate_reasons(record), (REASON_EVIDENCE_INCOMPLETE,))


class SliceStatusTests(unittest.TestCase):
    def test_status_bands(self):
        self.assertEqual(slice_status(_slice(14.0, 100.0)), STATUS_SMALL)
        self.assertEqual(slice_status(_slice(15.0, 100.0)), STATUS_NOT_SMALL)
        self.assertEqual(slice_status(_slice(16.0, 100.0)), STATUS_NOT_SMALL)
        self.assertEqual(
            slice_status(_slice(14.0, 100.0, border=True)), STATUS_GATED
        )
        self.assertEqual(slice_status(_slice(14.0, None)), STATUS_UNMEASURABLE)

    def test_longest_run_counts_only_consecutive_small(self):
        self.assertEqual(longest_small_run([STATUS_SMALL] * 3), 3)
        self.assertEqual(
            longest_small_run([STATUS_SMALL, STATUS_NOT_SMALL, STATUS_SMALL]), 1
        )
        self.assertEqual(longest_small_run([STATUS_GATED, STATUS_SMALL]), 1)
        self.assertEqual(longest_small_run([]), 0)


class EvaluateStackTests(unittest.TestCase):
    def test_two_consecutive_small_slices_flag(self):
        result = evaluate_stack([
            _slice(50.0, 100.0),
            _slice(14.0, 100.0),
            _slice(12.0, 100.0),
            _slice(40.0, 100.0),
        ])
        self.assertEqual(result["decision"], DECISION_FLAG)
        self.assertFalse(result["suppressed"])
        self.assertEqual(result["reasons"], ())
        self.assertGreaterEqual(
            result["longest_small_run"], MIN_CONSECUTIVE_SLICES
        )

    def test_single_small_slice_does_not_flag(self):
        result = evaluate_stack([_slice(50.0, 100.0), _slice(10.0, 100.0)])
        self.assertEqual(result["decision"], DECISION_CLEAR)
        self.assertEqual(result["longest_small_run"], 1)

    def test_all_normal_slices_clear(self):
        result = evaluate_stack([_slice(30.0, 100.0), _slice(25.0, 100.0)])
        self.assertEqual(result["decision"], DECISION_CLEAR)
        self.assertEqual(result["reasons"], ())

    def test_alignment_failure_suppresses_and_escalates(self):
        result = evaluate_stack([
            _slice(14.0, 100.0),
            _slice(12.0, 100.0),
            _slice(11.0, 100.0, aligned=False),
        ])
        self.assertEqual(result["decision"], DECISION_ESCALATE)
        self.assertTrue(result["suppressed"])
        self.assertIn(REASON_TEMPORAL_ALIGNMENT, result["reasons"])

    def test_border_contact_suppresses_and_escalates(self):
        result = evaluate_stack([_slice(11.0, 100.0, border=True)])
        self.assertEqual(result["decision"], DECISION_ESCALATE)
        self.assertIn(REASON_TOUCHES_BORDER, result["reasons"])

    def test_temporal_hole_suppresses_and_escalates(self):
        result = evaluate_stack([_slice(11.0, 100.0, hole=True)])
        self.assertEqual(result["decision"], DECISION_ESCALATE)
        self.assertIn(REASON_TEMPORAL_HOLE, result["reasons"])

    def test_missing_gate_field_suppresses(self):
        record = _slice(11.0, 100.0)
        del record["temporal_aligned"]
        result = evaluate_stack([record])
        self.assertEqual(result["decision"], DECISION_ESCALATE)
        self.assertIn(REASON_EVIDENCE_INCOMPLETE, result["reasons"])

    def test_unmeasurable_area_suppresses_without_gate_reason(self):
        result = evaluate_stack([_slice(None, 100.0)])
        self.assertEqual(result["decision"], DECISION_ESCALATE)
        self.assertEqual(result["reasons"], (REASON_AREA_UNMEASURABLE,))

    def test_gated_slice_breaks_a_short_run(self):
        result = evaluate_stack([
            _slice(14.0, 100.0),
            _slice(14.0, 100.0, hole=True),
            _slice(14.0, 100.0),
        ])
        self.assertEqual(result["decision"], DECISION_ESCALATE)
        self.assertEqual(result["longest_small_run"], 1)


class MetricHelperTests(unittest.TestCase):
    def test_tally_counts_each_decision(self):
        results = [
            {"decision": DECISION_FLAG},
            {"decision": DECISION_CLEAR},
            {"decision": DECISION_ESCALATE},
            {"decision": DECISION_ESCALATE},
        ]
        tally = tally_decisions(results)
        self.assertEqual(tally[DECISION_FLAG], 1)
        self.assertEqual(tally[DECISION_CLEAR], 1)
        self.assertEqual(tally[DECISION_ESCALATE], 2)
        self.assertEqual(tally["total"], 4)

    def test_agreement_skips_deferred_cases(self):
        pairs = [
            (DECISION_FLAG, DECISION_FLAG),
            (DECISION_CLEAR, DECISION_FLAG),
            (DECISION_ESCALATE, DECISION_FLAG),
        ]
        self.assertAlmostEqual(agreement_rate(pairs), 0.5)
        self.assertAlmostEqual(
            agreement_rate(pairs, include_deferred=True), 1 / 3
        )

    def test_agreement_returns_none_without_comparable_pairs(self):
        self.assertIsNone(agreement_rate([(DECISION_ESCALATE, DECISION_CLEAR)]))


if __name__ == "__main__":
    unittest.main()
