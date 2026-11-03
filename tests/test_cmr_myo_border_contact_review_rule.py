"""Offline unit tests for the draft review-only rule
``R-CMR-MYO-BORDER-CONTACT-REVIEW-01`` (no downloads, no datasets).
"""

from __future__ import annotations

import unittest
from dataclasses import replace

from experiments import cmr_myo_border_contact_review_rule as rule
from experiments.cmr_myo_border_contact_review_rule import (
    ALLOWED_SUPPORT_FLAGS,
    CASE_MIN_SUPPORT_FLAGS,
    GATE_THRESHOLD_KEYS,
    PRODUCTION_WRITE_ENABLED,
    default_spec,
    evaluate_activation,
    evaluate_case,
    evaluate_slice,
    score_evidence,
    verification_record,
)


def _slice(**overrides):
    record = {
        "slice_id": "S1",
        "case_id": "C1",
        "modality": "cmr",
        "cmr_myo_touches_image_border": True,
        "support_flags": ["small_area"],
    }
    record.update(overrides)
    return record


def _case(case_id, slices, **extra):
    record = {"case_id": case_id, "modality": "cmr", "slices": list(slices)}
    record.update(extra)
    return record


def _border_slice(slice_id, flags):
    return {
        "slice_id": slice_id,
        "cmr_myo_touches_image_border": True,
        "support_flags": list(flags),
    }


class DefaultSpecTests(unittest.TestCase):
    def test_default_spec_is_review_only_and_disabled(self):
        spec = default_spec()
        self.assertFalse(spec.enabled)
        self.assertTrue(spec.review_only)
        self.assertFalse(spec.blocks_payload)
        self.assertFalse(spec.writes_production_rule)
        self.assertEqual(spec.rule_id, "R-CMR-MYO-BORDER-CONTACT-REVIEW-01")
        self.assertEqual(tuple(spec.support_flags), ALLOWED_SUPPORT_FLAGS)
        self.assertFalse(PRODUCTION_WRITE_ENABLED)
        self.assertTrue(rule.REVIEW_ONLY)

    def test_support_flags_are_limited_to_the_approved_three(self):
        self.assertEqual(
            set(ALLOWED_SUPPORT_FLAGS),
            {"small_area", "excess_components", "empty"},
        )
        self.assertEqual(CASE_MIN_SUPPORT_FLAGS, 2)

    def test_no_numeric_threshold_constants_are_defined(self):
        # Threshold *key names* may be exported; numeric threshold *values*
        # must come from the existing promotion-gate configuration.
        module_floats = [
            name
            for name in dir(rule)
            if not name.startswith("_")
            and isinstance(getattr(rule, name), float)
        ]
        self.assertEqual(module_floats, [])
        self.assertEqual(set(GATE_THRESHOLD_KEYS), {"min_slice_precision", "max_case_escalation_rate"})


class SliceRuleTests(unittest.TestCase):
    def test_baseline_slice_is_positive(self):
        decision = evaluate_slice(_slice())
        self.assertTrue(decision.positive)
        self.assertEqual(decision.support_flags, ("small_area",))
        self.assertEqual(decision.reasons, ())

    def test_non_cmr_slice_is_not_positive(self):
        decision = evaluate_slice(_slice(modality="lge"))
        self.assertFalse(decision.positive)
        self.assertIn("modality_not_cmr", decision.reasons)

    def test_slice_without_border_contact_is_not_positive(self):
        decision = evaluate_slice(_slice(cmr_myo_touches_image_border=False))
        self.assertFalse(decision.positive)
        self.assertIn("myo_does_not_touch_image_border", decision.reasons)

    def test_slice_without_support_flag_is_not_positive(self):
        decision = evaluate_slice(_slice(support_flags=[]))
        self.assertFalse(decision.positive)
        self.assertIn("no_allowed_support_flag", decision.reasons)

    def test_unsupported_flag_does_not_make_slice_positive(self):
        decision = evaluate_slice(_slice(support_flags=["temporal_hole", "centroid_drift"]))
        self.assertFalse(decision.positive)
        self.assertEqual(decision.support_flags, ())

    def test_boolean_flags_and_empty_flag_are_recognised(self):
        decision = evaluate_slice(
            _slice(support_flags=[], empty=True, small_area=False, excess_components=0)
        )
        self.assertTrue(decision.positive)
        self.assertEqual(decision.support_flags, ("empty",))

    def test_non_mapping_slice_is_not_positive(self):
        decision = evaluate_slice(None)
        self.assertFalse(decision.positive)
        self.assertEqual(decision.reasons, ("slice_record_not_mapping",))


class CaseEscalationTests(unittest.TestCase):
    def test_case_with_two_distinct_flags_escalates(self):
        case = _case(
            "C1",
            [
                _border_slice("C1-S1", ["small_area"]),
                _border_slice("C1-S2", ["empty"]),
            ],
        )
        decision = evaluate_case(case)
        self.assertTrue(decision.escalated)
        self.assertEqual(decision.positive_slice_ids, ("C1-S1", "C1-S2"))
        self.assertEqual(decision.case_support_flags, ("empty", "small_area"))
        self.assertEqual(decision.reasons, ())

    def test_case_with_one_repeated_flag_does_not_escalate(self):
        case = _case(
            "C2",
            [
                _border_slice("C2-S1", ["small_area"]),
                _border_slice("C2-S2", ["small_area"]),
            ],
        )
        decision = evaluate_case(case)
        self.assertFalse(decision.escalated)
        self.assertEqual(decision.case_support_flags, ("small_area",))
        self.assertIn("case_support_flags_below_minimum", decision.reasons)

    def test_case_without_positive_slice_does_not_escalate(self):
        case = _case(
            "C3",
            [
                {
                    "slice_id": "C3-S1",
                    "cmr_myo_touches_image_border": False,
                    "support_flags": ["small_area", "empty"],
                }
            ],
        )
        decision = evaluate_case(case)
        self.assertFalse(decision.escalated)
        self.assertEqual(decision.positive_slice_ids, ())
        self.assertIn("no_positive_slice", decision.reasons)

    def test_case_level_flag_declaration_counts_towards_escalation(self):
        case = _case(
            "C4",
            [_border_slice("C4-S1", ["small_area"])],
            support_flags=["empty"],
        )
        decision = evaluate_case(case)
        self.assertTrue(decision.escalated)
        self.assertEqual(decision.case_support_flags, ("empty", "small_area"))

    def test_unsupported_flags_never_trigger_escalation(self):
        case = _case(
            "C5",
            [
                _border_slice("C5-S1", ["temporal_hole", "small_area"]),
                _border_slice("C5-S2", ["centroid_drift"]),
            ],
        )
        decision = evaluate_case(case)
        self.assertFalse(decision.escalated)
        self.assertEqual(decision.case_support_flags, ("small_area",))

    def test_empty_case_record_is_not_escalated(self):
        decision = evaluate_case({})
        self.assertFalse(decision.escalated)
        self.assertIn("no_positive_slice", decision.reasons)


class EvidenceMetricsTests(unittest.TestCase):
    def _evidence(self):
        c1 = _case(
            "C1",
            [
                {
                    "slice_id": "C1-S1",
                    "cmr_myo_touches_image_border": True,
                    "support_flags": ["small_area"],
                    "reference_positive": True,
                },
                {
                    "slice_id": "C1-S2",
                    "cmr_myo_touches_image_border": True,
                    "support_flags": ["empty"],
                    "reference_positive": False,
                },
            ],
        )
        c2 = _case(
            "C2",
            [
                {
                    "slice_id": "C2-S1",
                    "cmr_myo_touches_image_border": True,
                    "support_flags": ["small_area"],
                    "reference_positive": True,
                }
            ],
        )
        c3 = _case(
            "C3",
            [
                {
                    "slice_id": "C3-S1",
                    "cmr_myo_touches_image_border": False,
                    "support_flags": ["small_area"],
                }
            ],
        )
        return [c1, c2, c3]

    def test_precision_and_escalation_rate_on_synthetic_a1_evidence(self):
        metrics = score_evidence(self._evidence())
        self.assertEqual(metrics.slices_scored, 4)
        self.assertEqual(metrics.slices_positive, 3)
        self.assertEqual(metrics.true_positive_slices, 2)
        self.assertEqual(metrics.false_positive_slices, 1)
        self.assertAlmostEqual(metrics.slice_precision, 2 / 3)
        self.assertEqual(metrics.cases_scored, 3)
        self.assertEqual(metrics.cases_escalated, 1)
        self.assertAlmostEqual(metrics.case_escalation_rate, 1 / 3)
        self.assertEqual(metrics.escalated_case_ids, ("C1",))

    def test_missing_reference_labels_report_unavailable_precision(self):
        case = _case(
            "C9",
            [
                {
                    "slice_id": "C9-S1",
                    "cmr_myo_touches_image_border": True,
                    "support_flags": ["small_area"],
                }
            ],
        )
        metrics = score_evidence([case])
        self.assertIsNone(metrics.slice_precision)
        self.assertIn("reference_label_missing", metrics.reasons)
        self.assertIn("slice_precision_unavailable", metrics.reasons)
        self.assertAlmostEqual(metrics.case_escalation_rate, 0.0)

    def test_empty_evidence_is_handled(self):
        metrics = score_evidence([])
        self.assertEqual(metrics.cases_scored, 0)
        self.assertEqual(metrics.slices_scored, 0)
        self.assertIsNone(metrics.slice_precision)
        self.assertIsNone(metrics.case_escalation_rate)
        self.assertIn("no_cases_scored", metrics.reasons)
        self.assertIn("no_positive_slices_observed", metrics.reasons)


class ActivationGateTests(unittest.TestCase):
    def _metrics(self, **overrides):
        """Synthetic A1 evidence that clears a 0.5 / 0.5 injected gate.

        Two cases are scored so that slice precision is 1.0 and the case
        escalation rate is exactly 0.5 (one escalating case with two distinct
        allowed flags, one non-escalating case with a single flag; the gate is
        strict, so 0.5 does not exceed ``max_case_escalation_rate`` 0.5).
        Individual metrics can still be overridden by the caller.
        """
        base = score_evidence(
            [
                _case(
                    "C1",
                    [
                        {
                            "slice_id": "C1-S1",
                            "cmr_myo_touches_image_border": True,
                            "support_flags": ["small_area"],
                            "reference_positive": True,
                        },
                        {
                            "slice_id": "C1-S2",
                            "cmr_myo_touches_image_border": True,
                            "support_flags": ["empty"],
                            "reference_positive": True,
                        },
                    ],
                ),
                _case(
                    "C2",
                    [
                        {
                            "slice_id": "C2-S1",
                            "cmr_myo_touches_image_border": True,
                            "support_flags": ["small_area"],
                            "reference_positive": True,
                        }
                    ],
                ),
            ]
        )
        return replace(base, **overrides) if overrides else base

    def test_thresholds_are_injected_and_missing_ones_block_activation(self):
        decision = evaluate_activation(self._metrics(), None)
        self.assertFalse(decision.eligible)
        self.assertFalse(decision.enabled)
        self.assertIn("missing_thresholds", decision.reasons)

    def test_missing_threshold_key_blocks_activation(self):
        decision = evaluate_activation(
            self._metrics(), {"min_slice_precision": 0.5}
        )
        self.assertFalse(decision.eligible)
        self.assertIn("missing_threshold_key:max_case_escalation_rate", decision.reasons)

    def test_non_numeric_threshold_blocks_activation(self):
        decision = evaluate_activation(
            self._metrics(),
            {"min_slice_precision": "high", "max_case_escalation_rate": 0.5},
        )
        self.assertFalse(decision.eligible)
        self.assertIn("non_numeric_threshold:min_slice_precision", decision.reasons)

    def test_metrics_above_threshold_are_eligible_but_never_enabled(self):
        metrics = self._metrics()
        # The fixture must genuinely clear the gate before the gate is asserted.
        self.assertEqual(metrics.slice_precision, 1.0)
        self.assertEqual(metrics.case_escalation_rate, 0.5)
        decision = evaluate_activation(
            metrics,
            {"min_slice_precision": 0.5, "max_case_escalation_rate": 0.5},
        )
        self.assertTrue(decision.eligible)
        self.assertFalse(decision.enabled)
        self.assertEqual(decision.reasons, ())
        self.assertEqual(decision.thresholds["min_slice_precision"], 0.5)
        self.assertEqual(decision.thresholds["max_case_escalation_rate"], 0.5)

    def test_low_precision_blocks_activation(self):
        metrics = self._metrics(slice_precision=0.25)
        decision = evaluate_activation(
            metrics, {"min_slice_precision": 0.5, "max_case_escalation_rate": 0.5}
        )
        self.assertFalse(decision.eligible)
        self.assertFalse(decision.enabled)
        self.assertIn("slice_precision_below_threshold", decision.reasons)

    def test_high_escalation_rate_blocks_activation(self):
        metrics = self._metrics(case_escalation_rate=0.9)
        decision = evaluate_activation(
            metrics, {"min_slice_precision": 0.5, "max_case_escalation_rate": 0.5}
        )
        self.assertFalse(decision.eligible)
        self.assertIn("case_escalation_rate_above_threshold", decision.reasons)

    def test_unavailable_metrics_block_activation(self):
        metrics = self._metrics(slice_precision=None, case_escalation_rate=None)
        decision = evaluate_activation(
            metrics, {"min_slice_precision": 0.5, "max_case_escalation_rate": 0.5}
        )
        self.assertFalse(decision.eligible)
        self.assertIn("slice_precision_unavailable", decision.reasons)
        self.assertIn("case_escalation_rate_unavailable", decision.reasons)


class VerificationRecordTests(unittest.TestCase):
    def _metrics(self):
        return score_evidence(
            [
                _case(
                    "C1",
                    [
                        {
                            "slice_id": "C1-S1",
                            "cmr_myo_touches_image_border": True,
                            "support_flags": ["small_area"],
                            "reference_positive": True,
                        },
                        {
                            "slice_id": "C1-S2",
                            "cmr_myo_touches_image_border": True,
                            "support_flags": ["empty"],
                            "reference_positive": False,
                        },
                    ],
                )
            ]
        )

    def test_record_is_deterministic_and_reviewable(self):
        metrics = self._metrics()
        thresholds = {"min_slice_precision": 0.5, "max_case_escalation_rate": 0.5}
        first = verification_record(metrics, thresholds)
        second = verification_record(metrics, thresholds)
        self.assertEqual(first["digest"], second["digest"])
        self.assertEqual(len(first["digest"]), 64)
        payload = first["payload"]
        self.assertEqual(payload["rule_id"], "R-CMR-MYO-BORDER-CONTACT-REVIEW-01")
        self.assertTrue(payload["review_only"])
        self.assertFalse(payload["production_rule_written"])
        self.assertFalse(payload["activation_enabled"])
        self.assertEqual(payload["metrics"]["slices_positive"], 2)
        self.assertEqual(payload["thresholds"]["max_case_escalation_rate"], 0.5)

    def test_record_without_thresholds_stays_ineligible(self):
        record = verification_record(self._metrics(), None)
        self.assertFalse(record["payload"]["activation_eligible"])
        self.assertFalse(record["payload"]["activation_enabled"])
        self.assertIn("missing_thresholds", record["payload"]["activation_reasons"])
        self.assertIsNone(record["payload"]["thresholds"])


if __name__ == "__main__":
    unittest.main()
