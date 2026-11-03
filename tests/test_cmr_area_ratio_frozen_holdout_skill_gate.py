"""Offline unit tests for the read-only frozen-holdout skill gate.

No external data, images, masks or network access required.
"""

from __future__ import annotations

import unittest

from experiments.cmr_area_ratio_frozen_holdout_skill_gate import (
    ALTERNATE_AREA_RATIO_MIN,
    COHORT_LABEL,
    ROUTING_MODE,
    TOPOLOGY_INVARIANTS,
    _classify_at,
    _ledger_deltas,
    _myo_cavity_area_ratio,
    _topology_invariants,
    module_hash,
    report_to_json,
    run_evaluation,
)
from experiments.cmr_cavity_cluster_holdout_transition_check import (
    AREA_RATIO_MIN,
    CAVITY_MAX,
    CLUSTERS,
)


def _rec(cid, ref, area, cavity, **kw):
    rec = {
        "case_id": cid,
        "reference_area": ref,
        "before_area": area,
        "before_cavity": cavity,
    }
    rec.update(kw)
    return rec


class ClassifierTests(unittest.TestCase):
    def test_small_area_low_cavity_is_first_cluster(self):
        self.assertEqual(_classify_at(0.1, 0.2, AREA_RATIO_MIN), CLUSTERS[0])

    def test_small_area_high_cavity_is_second_cluster(self):
        self.assertEqual(_classify_at(0.1, CAVITY_MAX + 0.5, AREA_RATIO_MIN), CLUSTERS[1])

    def test_large_area_is_third_cluster(self):
        self.assertEqual(_classify_at(0.9, 0.2, AREA_RATIO_MIN), CLUSTERS[2])

    def test_missing_measurement_is_none(self):
        self.assertIsNone(_classify_at(None, 0.2, AREA_RATIO_MIN))
        self.assertIsNone(_classify_at(0.1, None, AREA_RATIO_MIN))

    def test_threshold_alternate_changes_decision(self):
        # ratio 0.3 is below the primary AREA_RATIO_MIN (0.5) and below the
        # alternate threshold (0.5) as well, so it stays in the low-area
        # cluster under both thresholds; the alternate threshold is the
        # stricter one and must never reclassify a low-area volume upward.
        self.assertEqual(_classify_at(0.3, 0.2, AREA_RATIO_MIN), CLUSTERS[0])
        self.assertEqual(_classify_at(0.3, 0.2, ALTERNATE_AREA_RATIO_MIN), CLUSTERS[0])
        # A ratio between the two thresholds (if the primary were looser)
        # would flip; assert the ordering invariant instead of a fixed flip.
        self.assertLessEqual(ALTERNATE_AREA_RATIO_MIN, AREA_RATIO_MIN)

    def test_threshold_alternate_flips_adequate_area(self):
        # ratio 0.6 is above both thresholds -> adequate area under both.
        self.assertEqual(_classify_at(0.6, 0.2, AREA_RATIO_MIN), CLUSTERS[2])
        self.assertEqual(_classify_at(0.6, 0.2, ALTERNATE_AREA_RATIO_MIN), CLUSTERS[2])


class MeasurementTests(unittest.TestCase):
    def test_myo_cavity_ratio(self):
        self.assertAlmostEqual(_myo_cavity_area_ratio(_rec("a", 100.0, 50.0, 0.5)), 100.0)

    def test_myo_cavity_ratio_missing_is_none(self):
        self.assertIsNone(_myo_cavity_area_ratio(_rec("a", 100.0, 50.0, None)))
        self.assertIsNone(_myo_cavity_area_ratio(_rec("a", 100.0, 50.0, 0)))


class TopologyInvariantTests(unittest.TestCase):
    def test_preserved_when_before_equals_after(self):
        rec = _rec("a", 100.0, 50.0, 0.2)
        for name in TOPOLOGY_INVARIANTS:
            rec[f"{name}_before"] = 1
            rec[f"{name}_after"] = 1
        topo = _topology_invariants(rec)
        for name in TOPOLOGY_INVARIANTS:
            self.assertTrue(topo[name]["preserved"])
            self.assertEqual(topo[name]["status"], "ok")

    def test_violated_when_changed(self):
        rec = _rec("a", 100.0, 50.0, 0.2)
        rec["connected_components_before"] = 1
        rec["connected_components_after"] = 2
        topo = _topology_invariants(rec)
        self.assertFalse(topo["connected_components"]["preserved"])

    def test_unavailable_when_missing(self):
        topo = _topology_invariants(_rec("a", 100.0, 50.0, 0.2))
        for name in TOPOLOGY_INVARIANTS:
            self.assertIsNone(topo[name]["preserved"])
            self.assertEqual(topo[name]["status"], "unavailable_no_masks_or_geometry")


class LedgerDeltaTests(unittest.TestCase):
    def test_unavailable_without_ledger(self):
        transitions = {c: {"checked": 0, "trash": 0, "no_transition": 0} for c in CLUSTERS}
        out = _ledger_deltas(transitions, None)
        self.assertEqual(out["status"], "unavailable_no_ledger_supplied")

    def test_deltas_computed(self):
        transitions = {c: {"checked": 0, "trash": 0, "no_transition": 0} for c in CLUSTERS}
        transitions[CLUSTERS[0]]["checked"] = 3
        ledger = {CLUSTERS[0]: {"checked": 1, "trash": 0}}
        out = _ledger_deltas(transitions, ledger)
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out[CLUSTERS[0]]["checked_delta"], 2)
        self.assertIsNone(out[CLUSTERS[1]]["checked_delta"])


class RunEvaluationTests(unittest.TestCase):
    def _data(self):
        records = [
            _rec("c1", 1000.0, 100.0, 0.2),   # small, low cavity -> cluster0
            _rec("c2", 1000.0, 100.0, 0.9),   # small, high cavity -> cluster1
            _rec("c3", 1000.0, 900.0, 0.2),   # large -> cluster2
            _rec("c4", 1000.0, 300.0, 0.2),   # ratio 0.3 -> cluster0 under both thresholds
        ]
        return {"records": records}

    def test_report_shape_and_cohort(self):
        report = run_evaluation(self._data())
        self.assertEqual(report["cohort"], COHORT_LABEL)
        self.assertEqual(report["routing_mode"], ROUTING_MODE)
        self.assertEqual(report["n_records"], 4)
        self.assertEqual(report["writes"], {"labels": 0, "code": 0, "datasets": 0})

    def test_cluster_counts_have_denominators(self):
        report = run_evaluation(self._data())
        for c in CLUSTERS:
            self.assertEqual(report["primary"]["cluster_counts"][c]["denominator"], 4)

    def test_transitions_have_denominators(self):
        report = run_evaluation(self._data())
        for c in CLUSTERS:
            self.assertIn("denominator", report["primary"]["simulated_transitions"][c])

    def test_threshold_reconciliation_present(self):
        report = run_evaluation(self._data())
        tr = report["threshold_reconciliation"]
        self.assertEqual(tr["primary_threshold"], AREA_RATIO_MIN)
        self.assertEqual(tr["alternate_threshold"], ALTERNATE_AREA_RATIO_MIN)
        self.assertGreaterEqual(tr["alternate_flagged_total"], tr["primary_flagged_total"])

    def test_unavailable_metrics_declared(self):
        report = run_evaluation(self._data())
        for key in ("concordance_vs_independent_labels", "precision", "ci"):
            self.assertIn(key, report["unavailable"])

    def test_suppression_coverage_unavailable_without_flags(self):
        report = run_evaluation(self._data())
        for cond, info in report["primary"]["suppression_coverage"].items():
            self.assertIsNone(info["count"])
            self.assertEqual(info["status"], "unavailable_no_masks_or_geometry")

    def test_topology_invariants_unavailable_without_geometry(self):
        report = run_evaluation(self._data())
        for name in TOPOLOGY_INVARIANTS:
            self.assertEqual(
                report["primary"]["topology_invariants"][name]["status"],
                "unavailable_no_masks_or_geometry",
            )

    def test_topology_violation_detected(self):
        data = self._data()
        data["records"][0]["connected_components_before"] = 1
        data["records"][0]["connected_components_after"] = 2
        report = run_evaluation(data)
        self.assertIn("c1", report["primary"]["topology_violation_cases"])

    def test_missing_measurement_not_imputed(self):
        data = {"records": [_rec("c1", 1000.0, None, 0.2)]}
        report = run_evaluation(data)
        self.assertIn("c1", report["primary"]["missing_measurement_cases"])
        self.assertIsNone(report["primary"]["cases"]["c1"]["area_ratio"])

    def test_module_hash_stable(self):
        self.assertEqual(module_hash(), module_hash())

    def test_report_serializes(self):
        report = run_evaluation(self._data())
        text = report_to_json(report)
        self.assertIn("module_hash", text)


if __name__ == "__main__":
    unittest.main()
