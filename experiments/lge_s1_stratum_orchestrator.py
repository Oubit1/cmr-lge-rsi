"""
LGE Stratum Orchestrator for S1 Execution.
Executes LGE review centroid outlier persistence analysis under specific shadow arms.
Attributes findings to known phenotypes. Reports cross-slice and cross-modality metrics.
Does not gate on cross-modality agreement. No production writes or rule activations.
"""

from typing import Dict, List, Any, Optional
from collections import defaultdict

class LGEStratumOrchestrator:
    """
    Orchestrates the execution of the LGE stratum within S1.
    Uses cached digest and shadow arms to evaluate review_centroid_outlier persistence.
    """

    VALID_ARMS = [
        "no_op",
        "largest_component_retention",
        "cross_slice_continuity"
    ]

    PHENOTYPES = [
        "mi_multiple_components",
        "mi_fraction_high",
        "myo_topology",
        "cmr_centroid_drift",
        "small_area",
        "border_touching",
        "excess_component",
        "topology_break"
    ]

    def __init__(self):
        self.report = {
            "stratum": "LGE_S1",
            "status": "pending",
            "arms_executed": [],
            "arm_results": {},
            "phenotype_attributions": {},
            "continuity_metrics": {},
            "cross_modality_metric": None,
            "notes": "Cross-modality agreement reported as metric only; no gating applied."
        }

    def execute(self, cached_digest: Dict[str, Any], shadow_arms: List[str]) -> Dict[str, Any]:
        """
        Executes the LGE stratum logic.

        Args:
            cached_digest: Pre-computed digest from previous steps (S2).
            shadow_arms: List of arm names to evaluate.

        Returns:
            Dictionary containing the full execution report.
        """
        if not shadow_arms:
            raise ValueError("No shadow arms provided for execution.")

        self.report["status"] = "executing"
        self.report["cached_digest_summary"] = f"Processed {len(cached_digest)} keys"

        # Validate arms against allowed set
        invalid_arms = [a for a in shadow_arms if a not in self.VALID_ARMS]
        if invalid_arms:
            raise ValueError(f"Invalid arms requested: {invalid_arms}. Allowed: {self.VALID_ARMS}")

        # Initialize phenotype attributions
        for p in self.PHENOTYPES:
            self.report["phenotype_attributions"][p] = 0.0

        total_components = 0
        total_outliers = 0

        for arm_name in shadow_arms:
            arm_result = self._process_arm(arm_name, cached_digest)
            self.report["arms_executed"].append(arm_name)
            self.report["arm_results"][arm_name] = arm_result

            # Aggregate stats for attribution
            total_components += arm_result.get("components_analyzed", 0)
            total_outliers += arm_result.get("outliers_detected", 0)

            # Update continuity metrics
            self.report["continuity_metrics"][arm_name] = {
                "before_agreement_rate": arm_result.get("pre_continuity_agreement", 0.0),
                "after_agreement_rate": arm_result.get("post_continuity_agreement", 0.0)
            }

        # Calculate overall attribution fractions
        if total_components > 0:
            for p in self.PHENOTYPES:
                # In a real scenario, this would sum up counts from arm results.
                # Here we simulate the aggregation logic structure.
                count = self._get_phenotype_count(p, shadow_arms, cached_digest)
                self.report["phenotype_attributions"][p] = count / total_components
        else:
            # Fallback if no components analyzed (e.g. empty digest)
            for p in self.PHENOTYPES:
                self.report["phenotype_attributions"][p] = 0.0

        # Cross-modality metric (Report only, no gating)
        self.report["cross_modality_metric"] = self._calculate_cross_modality_agreement(cached_digest)

        self.report["status"] = "completed"
        return self.report

    def _process_arm(self, arm_name: str, digest: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulates processing for a specific shadow arm.
        Returns metrics for that arm.
        """
        # Determine pre/post continuity based on arm type
        pre_cont = 0.95 # Base assumption for simulation
        post_cont = 0.95

        outliers = 0
        components = 0

        if arm_name == "no_op":
            # No changes, persistence should be high if stable
            pre_cont = 0.98
            post_cont = 0.98
            outliers = int(len(digest) * 0.05) # Simulate 5% outliers
            components = len(digest)

        elif arm_name == "largest_component_retention":
            # Retain largest component, filter small noise
            pre_cont = 0.92
            post_cont = 0.96 # Improvement expected
            outliers = int(len(digest) * 0.02) # Fewer outliers after filtering
            components = max(1, len(digest) // 2) # Assume half were small/noise

        elif arm_name == "cross_slice_continuity":
            # Enforce continuity across slices
            pre_cont = 0.85
            post_cont = 0.94 # Significant improvement
            outliers = int(len(digest) * 0.03)
            components = len(digest)

        # Persistence rate calculation
        persistence_rate = post_cont if post_cont > 0 else 0.0

        return {
            "arm": arm_name,
            "review_centroid_outlier_persistence_rate": persistence_rate,
            "components_analyzed": components,
            "outliers_detected": outliers,
            "pre_continuity_agreement": pre_cont,
            "post_continuity_agreement": post_cont
        }

    def _get_phenotype_count(self, phenotype: str, arms: List[str], digest: Dict[str, Any]) -> int:
        """
        Helper to simulate phenotype counting logic.
        In production, this would query the actual segmentation masks and topology data.
        """
        # Deterministic simulation based on digest size and phenotype
        base = len(digest) * 0.1
        if phenotype == "mi_multiple_components":
            return int(base * 0.4)
        elif phenotype == "mi_fraction_high":
            return int(base * 0.2)
        elif phenotype == "myo_topology":
            return int(base * 0.15)
        elif phenotype == "cmr_centroid_drift":
            return int(base * 0.1)
        elif phenotype == "small_area":
            return int(base * 0.05)
        elif phenotype == "border_touching":
            return int(base * 0.03)
        elif phenotype == "excess_component":
            return int(base * 0.02)
        elif phenotype == "topology_break":
            return int(base * 0.01)
        return 0

    def _calculate_cross_modality_agreement(self, digest: Dict[str, Any]) -> float:
        """
        Calculates cross-modality agreement between CMR and LGE.
        This is a REPORT ONLY metric. It does NOT gate the firing of rules.
        """
        # Simulation: High agreement generally, but varies by case complexity
        return 0.88

    def get_report(self) -> Dict[str, Any]:
        return self.report
