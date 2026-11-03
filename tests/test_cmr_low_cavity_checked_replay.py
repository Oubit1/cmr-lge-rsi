"""Offline unit tests for the S787 read-only checked replay.

No external data, images, masks or network access. Verifies per-cluster
simulated counts sum to the checked denominator, reconciliation against the
S783/S784 baselines, monitoring-grade agreement labelling, and that
concordance/precision/CI/topology stay unavailable.
"""

import unittest

from experiments.cmr_low_cavity_checked_replay import (
    EXPECTED_CHECKED,
    run_evaluation,
)
from experiments.cmr_low_cavity_discriminator_scan import (
    NO_TRANSITION_BASELINE_S783,
)


def _rec(cid, area, cavity, checked_label=None):
    rec = {
        "case_id": cid,
        "reference_area": 1000.0,
        "before_area": area,
        "before_cavity": cavity,
        "after_area": area,
        "after_cavity": cavity,
    }
    if checked_label is not None:
        rec["checked_label"] = checked_label
    return rec


def _cohort():
    # 20 stratified checked volumes: mix of adequate_area_low_cavity and
    # low_area clusters, all with cavity <= CAVITY_MAX so they are routed.
    records = []
    for i in range(10):
        records.append(_rec("adeq_%02d" % i, 600.0 + i, 0.2, "checked"))
    for i in range(10):
        records.append(_rec("low_%02d" % i, 100.0 + i, 0.2, "trash"))
    return {"records": records}


class CheckedReplayTest(unittest.TestCase):
    def test_counts_sum_to_checked_denominator(self):
        report = run_evaluation(_cohort())
        self.assertEqual(report["n_records"], EXPECTED_CHECKED)
        self.assertEqual(report["simulated_total"], EXPECTED_CHECKED)
        self.assertTrue(report["simulated_total_matches_checked"])
        per_cluster = sum(
            v["checked"] + v["trash"] + v["no_transition"]
            for v in report["simulated_transitions"].values()
        )
        self.assertEqual(per_cluster, EXPECTED_CHECKED)

    def test_reconciliation_against_s783_baseline(self):
        report = run_evaluation(_cohort())
        recon = report["reconciliation"]
        self.assertEqual(recon["status"], "ok")
        self.assertEqual(recon["s783_no_transition_baseline"], NO_TRANSITION_BASELINE_S783)
        self.assertEqual(
            recon["no_transition_delta_vs_s783"],
            report["simulated_no_transition_total"] - NO_TRANSITION_BASELINE_S783,
        )

    def test_reconciliation_unavailable_on_denominator_mismatch(self):
        data = _cohort()
        data["records"] = data["records"][:5]
        report = run_evaluation(data)
        self.assertEqual(
            report["reconciliation"]["status"],
            "unavailable_checked_denominator_mismatch",
        )
        self.assertIsNone(report["reconciliation"]["no_transition_delta_vs_s783"])

    def test_agreement_is_monitoring_grade_only(self):
        report = run_evaluation(_cohort())
        agreement = report["agreement_vs_checked_labels"]
        self.assertEqual(agreement["status"], "monitoring_grade_only")
        self.assertEqual(agreement["denominator"], EXPECTED_CHECKED)
        self.assertIsNotNone(agreement["rate"])

    def test_unavailable_metrics_are_not_imputed(self):
        report = run_evaluation(_cohort())
        unavailable = report["unavailable"]
        for key in (
            "concordance_vs_independent_labels",
            "precision",
            "ci",
            "topology_flags",
        ):
            self.assertIn(key, unavailable)
        self.assertEqual(report["writes"], {"labels": 0, "code": 0, "datasets": 0})
        self.assertEqual(report["audit_diff"]["production_writes"], 0)

    def test_empty_input_is_safe(self):
        report = run_evaluation({"records": []})
        self.assertEqual(report["simulated_total"], 0)
        self.assertEqual(
            report["agreement_vs_checked_labels"]["status"],
            "unavailable_no_checked_labels",
        )


if __name__ == "__main__":
    unittest.main()
