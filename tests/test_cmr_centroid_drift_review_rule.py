"""Offline unit tests for rule R-CMR-CENTROID-DRIFT-REVIEW-01.

All fixtures are inline: nothing is downloaded, no dataset is read and no file
is written.  Safe to run in CI / sandbox.
"""

from __future__ import annotations

import inspect
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments import cmr_centroid_drift_review_rule as rule  # noqa: E402

SUPPORT_A = rule.SUPPORT_FLAG_KEYS[0]
SUPPORT_B = rule.SUPPORT_FLAG_KEYS[1]
SUPPORT_C = rule.SUPPORT_FLAG_KEYS[2]


def _slice(modality="cmr", primary=None, supports=(), **extra):
    """Build a slice record with the given flags."""
    flags = {}
    if primary is not None:
        flags[rule.PRIMARY_FLAG] = primary
    for key in supports:
        flags[key] = True
    record = {"modality": modality, "flags": flags}
    record.update(extra)
    return record


# --- inline labelled replay fixtures -------------------------------------

CASE_TRUE_POSITIVE_1 = {
    "case_id": "c-tp-1",
    "modality": "cmr",
    "flags": {SUPPORT_A: True, SUPPORT_B: True},
    "slices": [
        _slice(primary=True, supports=(SUPPORT_A,)),
        _slice(primary=False),
        _slice(primary=True, supports=()),
    ],
}

CASE_TRUE_POSITIVE_2 = {
    "case_id": "c-tp-2",
    "modality": "cmr",
    "flags": {SUPPORT_B: True, SUPPORT_C: True},
    "slices": [
        # top-level flag lookup (no nested "flags" mapping)
        {"modality": "cmr", rule.PRIMARY_FLAG: True, SUPPORT_C: True},
        _slice(primary=True, supports=(SUPPORT_B,)),
    ],
}

CASE_NO_POSITIVE_SLICE = {
    "case_id": "c-tn-1",
    "modality": "cmr",
    "flags": {SUPPORT_A: True, SUPPORT_B: True, SUPPORT_C: True},
    "slices": [_slice(primary=False), _slice(primary=True, supports=())],
}

CASE_NON_CMR = {
    "case_id": "c-tn-2",
    "modality": "lge",
    "flags": {SUPPORT_A: True, SUPPORT_B: True},
    "slices": [_slice(modality=None, primary=True, supports=(SUPPORT_A,))],
}

# Human label says the drift is real, but only one case-level support flag is
# present, so the rule deliberately under-fires: a clean recall-bound fixture.
CASE_UNDER_FIRED = {
    "case_id": "c-fn-1",
    "modality": "cmr",
    "flags": {SUPPORT_A: True},
    "slices": [_slice(primary=True, supports=(SUPPORT_A,))],
}

REPLAY_SET = [
    {"case": CASE_TRUE_POSITIVE_1, "expected_review": True},
    {"case": CASE_TRUE_POSITIVE_2, "expected_review": True},
    {"case": CASE_NO_POSITIVE_SLICE, "expected_review": False},
    {"case": CASE_NON_CMR, "expected_review": False},
    {"case": CASE_UNDER_FIRED, "expected_review": True},
]


class RuleMetadataTests(unittest.TestCase):
    def test_metadata_carries_required_fields(self):
        meta = rule.rule_metadata()
        for key in ("issue_type", "modality", "cache_hash", "rule_version"):
            self.assertIn(key, meta)
            self.assertIsInstance(meta[key], str)
            self.assertTrue(meta[key])
        self.assertEqual(meta["issue_type"], "centroid_drift")
        self.assertEqual(meta["modality"], "cmr")
        self.assertEqual(meta["rule_id"], "R-CMR-CENTROID-DRIFT-REVIEW-01")
        self.assertEqual(meta["mode"], "review_only")

    def test_decision_record_carries_rule_metadata(self):
        decision = rule.evaluate_case(CASE_TRUE_POSITIVE_1, enabled=True)
        for key in ("issue_type", "modality", "cache_hash", "rule_version"):
            self.assertIn(key, decision)
        self.assertEqual(decision["issue_type"], rule.ISSUE_TYPE)
        self.assertEqual(decision["modality"], rule.MODALITY)
        self.assertEqual(decision["cache_hash"], rule.cache_hash())
        self.assertEqual(decision["rule_version"], rule.RULE_VERSION)

    def test_cache_hash_is_stable_and_parameter_sensitive(self):
        digest = rule.cache_hash()
        self.assertRegex(digest, r"^[0-9a-f]{64}$")
        self.assertEqual(digest, rule.cache_hash())
        self.assertEqual(digest, rule.cache_hash(rule.rule_fingerprint()))
        self.assertNotEqual(rule.cache_hash({"a": 1}), rule.cache_hash({"a": 2}))

    def test_decision_is_json_serialisable(self):
        decision = rule.evaluate_case(CASE_TRUE_POSITIVE_1, enabled=True)
        round_tripped = json.loads(json.dumps(decision, sort_keys=True))
        self.assertEqual(round_tripped, decision)


class DefaultOffReviewOnlyTests(unittest.TestCase):
    def test_rule_is_disabled_by_default(self):
        self.assertFalse(rule.DEFAULT_ENABLED)
        self.assertEqual(rule.MODE, "review_only")
        decision = rule.evaluate_case(CASE_TRUE_POSITIVE_1)  # no explicit enable
        self.assertFalse(decision["enabled"])
        self.assertFalse(decision["review_required"])
        self.assertTrue(decision["structural_escalation"])
        self.assertEqual(decision["reason"], "rule_disabled")

    def test_disabled_rule_suppresses_every_escalation(self):
        for item in REPLAY_SET:
            decision = rule.evaluate_case(item["case"], enabled=False)
            self.assertFalse(decision["review_required"])

    def test_disabled_replay_predicts_nothing(self):
        metrics = rule.replay_metrics(REPLAY_SET, enabled=False)
        self.assertEqual((metrics["tp"], metrics["fp"]), (0, 0))
        self.assertEqual(metrics["recall"], 0.0)

    def test_no_production_write_surface(self):
        self.assertFalse(rule.WRITES_PRODUCTION)
        self.assertEqual(rule.WRITE_TARGETS, ())
        decision = rule.evaluate_case(CASE_TRUE_POSITIVE_1, enabled=True)
        self.assertFalse(decision["writes_production"])
        source = inspect.getsource(rule)
        for forbidden in ("sqlite3", "subprocess", "socket", "shutil", "open(", ".write("):
            self.assertNotIn(forbidden, source)

    def test_zero_new_numeric_thresholds(self):
        self.assertEqual(dict(rule.NUMERIC_THRESHOLDS), {})
        floats = [name for name, value in vars(rule).items() if type(value) is float]
        self.assertEqual(floats, [])
        ints = {
            name: value
            for name, value in vars(rule).items()
            if type(value) is int and not name.startswith("__")
        }
        self.assertEqual(
            ints,
            {
                "MIN_SLICE_SUPPORT_FLAGS": 1,
                "MIN_POSITIVE_SLICES_PER_CASE": 1,
                "MIN_CASE_SUPPORT_FLAGS": 2,
            },
        )


class SlicePositiveTests(unittest.TestCase):
    def test_slice_positive_requires_cmr_primary_and_one_support(self):
        self.assertTrue(
            rule.is_slice_positive(_slice(primary=True, supports=(SUPPORT_A,)), "cmr")
        )
        # missing supporting flag on the same slice
        self.assertFalse(rule.is_slice_positive(_slice(primary=True, supports=()), "cmr"))
        # primary flag false
        self.assertFalse(
            rule.is_slice_positive(_slice(primary=False, supports=(SUPPORT_A,)), "cmr")
        )
        # not a cmr slice
        self.assertFalse(
            rule.is_slice_positive(
                _slice(modality="lge", primary=True, supports=(SUPPORT_A,)), "lge"
            )
        )

    def test_string_flag_values_are_interpreted(self):
        record = {rule.PRIMARY_FLAG: "true", SUPPORT_A: "yes"}
        self.assertTrue(rule.is_slice_positive(record, "cmr"))
        record_off = {rule.PRIMARY_FLAG: "false", SUPPORT_A: "1"}
        self.assertFalse(rule.is_slice_positive(record_off, "cmr"))

    def test_slice_modality_falls_back_to_case_modality(self):
        record = {rule.PRIMARY_FLAG: True, SUPPORT_A: True}
        self.assertTrue(rule.is_slice_positive(record, "cmr"))
        self.assertFalse(rule.is_slice_positive(record, "lge"))
        self.assertFalse(rule.is_slice_positive(record, None))

    def test_one_supporting_flag_is_enough_for_slice_but_not_for_case(self):
        one_flag = {
            "case_id": "c-one-support",
            "modality": "cmr",
            "flags": {SUPPORT_A: True},
            "slices": [_slice(primary=True, supports=(SUPPORT_A,))],
        }
        self.assertEqual(len(rule.positive_slice_indices(one_flag)), 1)
        self.assertFalse(rule.is_case_escalated(one_flag))
        decision = rule.evaluate_case(one_flag, enabled=True)
        self.assertEqual(decision["reason"], "insufficient_case_support_flags")


class CaseEscalationTests(unittest.TestCase):
    def test_case_escalation_requires_two_case_support_flags(self):
        self.assertTrue(rule.is_case_escalated(CASE_TRUE_POSITIVE_1))
        self.assertFalse(rule.is_case_escalated(CASE_UNDER_FIRED))

    def test_case_escalation_requires_a_positive_slice(self):
        self.assertFalse(rule.is_case_escalated(CASE_NO_POSITIVE_SLICE))
        decision = rule.evaluate_case(CASE_NO_POSITIVE_SLICE, enabled=True)
        self.assertEqual(decision["reason"], "no_positive_slice")
        self.assertEqual(decision["positive_slice_count"], 0)

    def test_non_cmr_case_is_never_escalated(self):
        self.assertFalse(rule.is_case_escalated(CASE_NON_CMR))
        decision = rule.evaluate_case(CASE_NON_CMR, enabled=True)
        self.assertEqual(decision["reason"], "case_modality_not_cmr")
        self.assertFalse(decision["review_required"])

    def test_enabled_rule_escalates_positive_case(self):
        decision = rule.evaluate_case(CASE_TRUE_POSITIVE_2, enabled=True)
        self.assertTrue(decision["review_required"])
        self.assertEqual(decision["reason"], "review_escalation")
        self.assertEqual(decision["positive_slice_indices"], [0, 1])
        self.assertEqual(decision["positive_slice_count"], 2)
        self.assertEqual(decision["case_support_flag_count"], 2)
        self.assertFalse(decision["writes_production"])

    def test_tolerates_malformed_records(self):
        for bad in (None, {}, {"modality": "cmr"}, {"slices": None}, [1, 2, 3]):
            decision = rule.evaluate_case(bad, enabled=True)
            self.assertFalse(decision["review_required"])
            self.assertIsInstance(decision["reason"], str)


class ReplayMetricTests(unittest.TestCase):
    def test_replay_precision_and_recall_meet_acceptance(self):
        metrics = rule.replay_metrics(REPLAY_SET, enabled=True)
        self.assertEqual(metrics["tp"], 2)
        self.assertEqual(metrics["fp"], 0)
        self.assertEqual(metrics["fn"], 1)
        self.assertEqual(metrics["tn"], 2)
        self.assertGreaterEqual(metrics["precision"], 0.8)
        self.assertGreaterEqual(metrics["recall"], 0.6)

    def test_replay_accepts_bare_case_records(self):
        bare = [dict(CASE_TRUE_POSITIVE_1, expected_review=True)]
        metrics = rule.replay_metrics(bare, enabled=True)
        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)

    def test_empty_replay_set_is_safe(self):
        metrics = rule.replay_metrics([], enabled=True)
        self.assertEqual(metrics["precision"], 0.0)
        self.assertEqual(metrics["recall"], 0.0)


if __name__ == "__main__":
    unittest.main()
