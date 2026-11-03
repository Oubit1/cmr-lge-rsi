import unittest

from experiments import lge_mi_multi_review_rule as rule


class TestRuleMetadata(unittest.TestCase):
    def test_disabled_by_default_and_review_only(self):
        self.assertFalse(rule.RULE_ENABLED)
        self.assertTrue(rule.REVIEW_ONLY)
        self.assertTrue(rule.PROMOTION_BLOCKED)
        self.assertIn("disabled_review_only", rule.RULE_TEXT)
        self.assertIn("review-only", rule.RULE_TEXT)
        self.assertIn("no activation path", rule.RULE_TEXT)

    def test_contraindications_present(self):
        for item in rule.CONTRAINDICATIONS:
            self.assertIn(item, rule.RULE_TEXT)

    def test_rule_references_only_existing_boolean_flags(self):
        for flag in rule.SUPPORT_FLAGS:
            self.assertIn(flag, rule.RULE_TEXT)
        self.assertIn("mi_multiple_components", rule.RULE_TEXT)
        self.assertIn("modality", rule.RULE_TEXT)


class TestSlicePositive(unittest.TestCase):
    def test_requires_lge_multi_component_and_one_support_flag(self):
        base = {"slice_index": 1, "modality": "lge", "mi_multiple_components": True}
        self.assertFalse(rule.is_slice_positive(base))
        for flag in rule.SUPPORT_FLAGS:
            with self.subTest(flag=flag):
                payload = dict(base, **{flag: True})
                self.assertTrue(rule.is_slice_positive(payload))
        self.assertFalse(
            rule.is_slice_positive(dict(base, modality="t1", mi_fraction_high=True))
        )
        self.assertFalse(
            rule.is_slice_positive(
                dict(base, mi_multiple_components=False, mi_fraction_high=True)
            )
        )

    def test_isolated_multi_component_without_support_is_not_positive(self):
        payload = {
            "slice_index": 2,
            "modality": "lge",
            "mi_multiple_components": True,
        }
        self.assertFalse(rule.is_slice_positive(payload))
        self.assertEqual(rule.evaluate_case("c1", "lge", [payload]), [])

    def test_ring_contiguity_is_not_a_gate(self):
        payload = {
            "slice_index": 1,
            "modality": "lge",
            "mi_multiple_components": True,
            "mi_fraction_high": True,
            "cmr_ring_contiguous": False,
        }
        self.assertTrue(rule.is_slice_positive(payload))


class TestCasebookEscalation(unittest.TestCase):
    def test_escalation_pair_fraction_high_and_topology(self):
        payload = {
            "slice_index": 7,
            "modality": "lge",
            "mi_multiple_components": True,
            "mi_fraction_high": True,
            "myo_topology": True,
        }
        self.assertTrue(rule.is_slice_positive(payload))
        self.assertTrue(rule.is_escalation_pair_slice(payload))
        records = rule.evaluate_case("case-1", "lge", [payload])
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["escalation_rule_id"], rule.ESCALATION_RULE_ID)
        self.assertIn(rule.ISSUE_ESCALATION, records[0]["issue_types"])
        self.assertEqual(records[0]["escalation_slice_indices"], [7])

    def test_escalation_pair_tiny_and_fraction_high(self):
        positive = {
            "slice_index": 1,
            "modality": "lge",
            "mi_multiple_components": True,
            "mi_tiny_unsupported": True,
        }
        pair = {
            "slice_index": 2,
            "modality": "lge",
            "mi_tiny_unsupported": True,
            "mi_fraction_high": True,
        }
        records = rule.evaluate_case("case-2", "lge", [positive, pair])
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["escalation_rule_id"], rule.ESCALATION_RULE_ID)
        self.assertEqual(records[0]["escalation_slice_indices"], [2])

    def test_escalation_requires_positive_slice(self):
        pair_only = {
            "slice_index": 1,
            "modality": "lge",
            "mi_fraction_high": True,
            "myo_topology": True,
        }
        self.assertTrue(rule.is_escalation_pair_slice(pair_only))
        self.assertEqual(rule.evaluate_case("case-3", "lge", [pair_only]), [])

    def test_no_escalation_without_pair(self):
        payload = {
            "slice_index": 1,
            "modality": "lge",
            "mi_multiple_components": True,
            "mi_fraction_high": True,
        }
        records = rule.evaluate_case("case-4", "lge", [payload])
        self.assertEqual(len(records), 1)
        self.assertIsNone(records[0]["escalation_rule_id"])
        self.assertNotIn(rule.ISSUE_ESCALATION, records[0]["issue_types"])


class TestRecordFormat(unittest.TestCase):
    def test_one_record_per_case_modality_with_merged_issues_and_indices(self):
        slices = [
            {
                "slice_index": 3,
                "modality": "lge",
                "mi_multiple_components": True,
                "mi_fraction_high": True,
            },
            {
                "slice_index": 5,
                "modality": "lge",
                "mi_multiple_components": True,
                "myo_topology": True,
                "mi_tiny_unsupported": True,
            },
            {
                "slice_index": 7,
                "modality": "lge",
                "mi_multiple_components": True,
                "mi_fraction_high": True,
                "myo_topology": True,
            },
        ]
        records = rule.evaluate_case("case-9", "lge", slices)
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["case_id"], "case-9")
        self.assertEqual(record["modality"], "lge")
        self.assertFalse(record["rule_enabled"])
        self.assertTrue(record["review_only"])
        self.assertTrue(record["promotion_blocked"])
        self.assertEqual(record["slice_indices"], [3, 5, 7])
        self.assertEqual(record["addressable_slice_indices"], [3, 5, 7])
        self.assertEqual(record["escalation_slice_indices"], [7])
        self.assertEqual(
            set(record["issue_types"]),
            {
                rule.ISSUE_MULTI_COMPONENT,
                rule.ISSUE_FRACTION_HIGH,
                rule.ISSUE_MYO_TOPOLOGY,
                rule.ISSUE_TINY_UNSUPPORTED,
                rule.ISSUE_ESCALATION,
            },
        )

    def test_non_lge_modality_emits_no_record(self):
        payload = {
            "slice_index": 1,
            "modality": "lge",
            "mi_multiple_components": True,
            "mi_fraction_high": True,
        }
        self.assertEqual(rule.evaluate_case("case-10", "t1", [payload]), [])

    def test_other_modality_slices_are_ignored(self):
        lge_payload = {
            "slice_index": 1,
            "modality": "lge",
            "mi_multiple_components": True,
            "mi_fraction_high": True,
        }
        t1_payload = {
            "slice_index": 2,
            "modality": "t1",
            "mi_multiple_components": True,
            "mi_fraction_high": True,
        }
        records = rule.evaluate_case("case-11", "lge", [lge_payload, t1_payload])
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["slice_indices"], [1])


if __name__ == "__main__":
    unittest.main()
