import unittest
from experiments.lge_s1_stratum_orchestrator import LGEStratumOrchestrator

class TestLGEStratumOrchestrator(unittest.TestCase):

    def setUp(self):
        self.orchestrator = LGEStratumOrchestrator()
        # Create a mock cached digest with some keys representing slice/component data
        self.mock_digest = {f"slice_{i}": {"centroid": [i, i, i], "area": 100} for i in range(10)}
        self.valid_arms = ["no_op", "largest_component_retention", "cross_slice_continuity"]

    def test_execute_all_arms(self):
        """Test that all valid arms are executed and reported."""
        report = self.orchestrator.execute(self.mock_digest, self.valid_arms)

        self.assertEqual(report["status"], "completed")
        self.assertEqual(len(report["arms_executed"]), 3)
        for arm in self.valid_arms:
            self.assertIn(arm, report["arms_executed"])
            self.assertIn(arm, report["arm_results"])

    def test_review_centroid_outlier_persistence(self):
        """Test that persistence rates are reported for each arm."""
        report = self.orchestrator.execute(self.mock_digest, self.valid_arms)

        for arm in self.valid_arms:
            result = report["arm_results"][arm]
            self.assertIn("review_centroid_outlier_persistence_rate", result)
            self.assertIsInstance(result["review_centroid_outlier_persistence_rate"], float)
            self.assertGreaterEqual(result["review_centroid_outlier_persistence_rate"], 0.0)
            self.assertLessEqual(result["review_centroid_outlier_persistence_rate"], 1.0)

    def test_phenotype_attributions(self):
        """Test that all required phenotypes are attributed."""
        report = self.orchestrator.execute(self.mock_digest, self.valid_arms)

        required_phenotypes = [
            "mi_multiple_components", "mi_fraction_high", "myo_topology",
            "cmr_centroid_drift", "small_area", "border_touching",
            "excess_component", "topology_break"
        ]

        for p in required_phenotypes:
            self.assertIn(p, report["phenotype_attributions"])
            self.assertIsInstance(report["phenotype_attributions"][p], float)
            self.assertGreaterEqual(report["phenotype_attributions"][p], 0.0)
            self.assertLessEqual(report["phenotype_attributions"][p], 1.0)

    def test_cross_slice_continuity_metrics(self):
        """Test that continuity agreement rates are reported before and after each arm."""
        report = self.orchestrator.execute(self.mock_digest, self.valid_arms)

        for arm in self.valid_arms:
            cont_metrics = report["continuity_metrics"][arm]
            self.assertIn("before_agreement_rate", cont_metrics)
            self.assertIn("after_agreement_rate", cont_metrics)
            self.assertIsInstance(cont_metrics["before_agreement_rate"], float)
            self.assertIsInstance(cont_metrics["after_agreement_rate"], float)

    def test_cross_modality_metric_only(self):
        """Test that cross-modality agreement is reported but does not affect status/gating."""
        report = self.orchestrator.execute(self.mock_digest, self.valid_arms)

        self.assertIsNotNone(report["cross_modality_metric"])
        self.assertIsInstance(report["cross_modality_metric"], float)
        # Verify status is completed regardless of cross-modality value
        self.assertEqual(report["status"], "completed")
        # Verify note indicates metric-only nature
        self.assertIn("metric only", report["notes"].lower())

    def test_invalid_arm_raises_error(self):
        """Test that providing an invalid arm raises a ValueError."""
        with self.assertRaises(ValueError):
            self.orchestrator.execute(self.mock_digest, ["invalid_arm"])

    def test_empty_arms_raises_error(self):
        """Test that providing no arms raises a ValueError."""
        with self.assertRaises(ValueError):
            self.orchestrator.execute(self.mock_digest, [])

if __name__ == "__main__":
    unittest.main()
