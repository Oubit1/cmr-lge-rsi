"""Offline tests for the LGE myo consumer contract manifest-flag ingestion."""

import unittest

from experiments.lge_myo_consumer_contract import (
    EXPECTED_NON_HOLDOUT,
    _dedup_issues,
    _flag_present,
    run_evaluation,
)


def _rec(cid, **extra):
    base = {
        "case_id": cid,
        "myo_components": 2,
        "primary_component_area": 100.0,
        "secondary_component_area": 20.0,
        "cross_slice_continuity_breaks": 0,
        "border_contact_proxy": False,
        "issues": [],
        "mi_multiple_components": False,
        "myo_topology": "ok",
    }
    base.update(extra)
    return base


class FlagParsingTest(unittest.TestCase):
    def test_flag_present_normalization(self):
        self.assertIsNone(_flag_present(None))
        self.assertTrue(_flag_present(True))
        self.assertFalse(_flag_present(False))
        self.assertTrue(_flag_present(["a"]))
        self.assertFalse(_flag_present([]))
        self.assertTrue(_flag_present("ok"))
        self.assertFalse(_flag_present("none"))

    def test_dedup_by_case_and_issue_type(self):
        issues = [
            {"issue_type": "fragmentation"},
            {"issue_type": "fragmentation"},
            {"issue_type": "border_clip"},
        ]
        out = _dedup_issues("case_x", issues)
        self.assertEqual(out["raw_count"], 3)
        self.assertEqual(out["dropped"], 1)
        self.assertEqual(len(out["issues"]), 2)


class RunEvaluationTest(unittest.TestCase):
    def _data(self, records, holdout=None):
        return {"records": records, "holdout_case_ids": holdout or []}

    def test_flags_move_cases_off_unavailable(self):
        records = [_rec("case_%02d" % i) for i in range(EXPECTED_NON_HOLDOUT)]
        report = run_evaluation(self._data(records))
        self.assertEqual(report["n_records"], EXPECTED_NON_HOLDOUT)
        self.assertLess(report["unavailable_count"], EXPECTED_NON_HOLDOUT)
        self.assertEqual(
            report["resolved_count"]
            + report["working_count"]
            + report["unavailable_count"],
            EXPECTED_NON_HOLDOUT,
        )
        self.assertEqual(report["unavailable_reason_fill_rate"], 1.0)
        self.assertFalse(report["label_write"])
        self.assertEqual(report["production_writes"], 0)
        self.assertEqual(report["holdout_selected"], 0)

    def test_missing_flag_stays_unavailable(self):
        rec = _rec("case_missing")
        rec.pop("myo_topology")
        report = run_evaluation(self._data([rec]))
        case = report["cases"]["case_missing"]
        self.assertEqual(case["verdict"], "unavailable")
        self.assertIn("myo_topology", case["unavailable_fields"])
        self.assertIsNone(case["myo_topology"])

    def test_holdout_excluded_and_dedup_ratio(self):
        records = [_rec("case_%02d" % i) for i in range(EXPECTED_NON_HOLDOUT)]
        records.append(_rec("case_holdout"))
        records[0]["issues"] = [
            {"issue_type": "fragmentation"},
            {"issue_type": "fragmentation"},
        ]
        report = run_evaluation(self._data(records, holdout=["case_holdout"]))
        self.assertEqual(report["holdout_excluded_count"], 1)
        self.assertEqual(report["n_records"], EXPECTED_NON_HOLDOUT)
        self.assertGreater(report["dedup_ratio"], 1.0)
        self.assertIn("cache_hit_rate", report)
        self.assertIn("s859_fixed_validation", report)

    def test_flagged_case_is_review(self):
        rec = _rec("case_flag", mi_multiple_components=True)
        report = run_evaluation(self._data([rec]))
        self.assertEqual(report["cases"]["case_flag"]["verdict"], "review")


if __name__ == "__main__":
    unittest.main()
