from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.database import Database
from app.rsi_kernel import MetricSpec, compare_fitness, selection_payload, variant_id


class RSIFitnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.specs = {
            "quality": MetricSpec("higher"),
            "errors": MetricSpec("lower"),
            "cost": MetricSpec("lower", required=False, quality_metric=False),
        }

    def compare(self, baseline, candidate, **kwargs):
        return compare_fitness(
            baseline=baseline, candidate=candidate,
            baseline_benchmark_id=kwargs.get("baseline_id", "frozen-v1"),
            candidate_benchmark_id=kwargs.get("candidate_id", "frozen-v1"),
            baseline_benchmark_hash=kwargs.get("baseline_hash", "abc"),
            candidate_benchmark_hash=kwargs.get("candidate_hash", "abc"),
            metric_specs=self.specs,
            constraints=kwargs.get("constraints", {"regression_suite": True}),
        )

    def test_promotes_only_quality_pareto_improvement(self):
        result = self.compare(
            {"quality": 0.6, "errors": 3, "cost": 100},
            {"quality": 0.7, "errors": 3, "cost": 80},
        )
        self.assertEqual("PROMOTE", result.decision)
        self.assertIn("quality", result.improved_metrics)

    def test_quality_gain_does_not_hide_regression(self):
        result = self.compare(
            {"quality": 0.6, "errors": 3},
            {"quality": 0.8, "errors": 4},
        )
        self.assertEqual("RETAIN_PARENT", result.decision)
        self.assertIn("errors", result.regressed_metrics)

    def test_cost_only_gain_is_not_rsi_progress(self):
        result = self.compare(
            {"quality": 0.6, "errors": 3, "cost": 100},
            {"quality": 0.6, "errors": 3, "cost": 50},
        )
        self.assertEqual("RETAIN_PARENT", result.decision)
        self.assertIn("no_quality_improvement", result.reasons)

    def test_benchmark_mismatch_is_not_coerced_to_failure(self):
        result = self.compare(
            {"quality": 0.6, "errors": 3},
            {"quality": 0.9, "errors": 0},
            candidate_hash="different",
        )
        self.assertEqual("NEEDS_EVIDENCE", result.decision)
        self.assertIn("benchmark_hash_mismatch", result.reasons)

    def test_missing_metric_is_explicit_evidence_debt(self):
        result = self.compare(
            {"quality": 0.6, "errors": 3}, {"quality": 0.9}
        )
        self.assertEqual("NEEDS_EVIDENCE", result.decision)
        self.assertIn("metric_missing:errors", result.reasons)

    def test_failed_constraint_blocks_comparison(self):
        result = self.compare(
            {"quality": 0.6, "errors": 3},
            {"quality": 0.9, "errors": 0},
            constraints={"regression_suite": False},
        )
        self.assertEqual("NEEDS_EVIDENCE", result.decision)
        self.assertIn("constraint_failed:regression_suite", result.reasons)

    def test_variant_identity_includes_parent_and_artifact(self):
        first = variant_id(
            parent_variant_id="v_parent", source_run_id=1, action_id="a",
            code_commit="abc", changed_files=["app/a.py"], artifact_digest="x",
        )
        same = variant_id(
            parent_variant_id="v_parent", source_run_id=1, action_id="a",
            code_commit="abc", changed_files=["app/a.py"], artifact_digest="x",
        )
        child_of_other = variant_id(
            parent_variant_id="v_other", source_run_id=1, action_id="a",
            code_commit="abc", changed_files=["app/a.py"], artifact_digest="x",
        )
        self.assertEqual(first, same)
        self.assertNotEqual(first, child_of_other)

    def test_invalid_numeric_scores_cannot_win(self):
        for value in (float("nan"), float("inf"), True, None):
            with self.subTest(value=value):
                result = self.compare({"quality": 0.6, "errors": 0},
                                      {"quality": value, "errors": 0})
                self.assertEqual("NEEDS_EVIDENCE", result.decision)

    def test_selection_payload_is_model_independent(self):
        decision = self.compare(
            {"quality": 0.6, "errors": 3},
            {"quality": 0.7, "errors": 2},
        )
        payload = selection_payload(
            parent_variant_id="v0", child_variant_id="v1",
            benchmark_id="frozen-v1", benchmark_hash="abc",
            baseline={"quality": 0.6, "errors": 3},
            candidate={"quality": 0.7, "errors": 2},
            constraints={"regression_suite": True}, decision=decision,
        )
        self.assertEqual("rsi-selection-v1", payload["schema_version"])
        self.assertEqual("PROMOTE", payload["decision"])
        self.assertNotIn("model", payload)


class RSIArchiveDatabaseTests(unittest.TestCase):
    def test_lineage_and_evaluation_are_durable(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Database(Path(tmp) / "workflow.sqlite")
            db.initialize()
            db.register_evolution_variant(
                variant_id="v0", parent_variant_id=None, source_run_id=None,
                action_id=None, workflow_stage="lge_myo", code_commit="abc",
                continuity_revision="r1", artifact_path=None,
                status="ACTIVE", metadata={"kind": "parent"},
            )
            db.register_evolution_variant(
                variant_id="v1", parent_variant_id="v0", source_run_id=None,
                action_id="a1", workflow_stage="lge_myo", code_commit="abc",
                continuity_revision="r1", artifact_path=Path(tmp) / "candidate",
                metadata={"kind": "child"},
            )
            db.record_evolution_evaluation(
                variant_id="v1", parent_variant_id="v0",
                benchmark_id="frozen-v1", benchmark_hash="hash",
                baseline_metrics={"quality": 0.5},
                candidate_metrics={"quality": 0.6},
                constraints={"regression_suite": True},
                decision="PROMOTE", reasons=["pareto_quality_improvement"],
            )
            db.transition_evolution_variant(
                "v1", status="CANARY",
                rollback_manifest_path=Path(tmp) / "rollback.json",
            )

            variants = {item["variant_id"]: item for item in db.list_evolution_variants()}
            self.assertEqual("v0", variants["v1"]["parent_variant_id"])
            self.assertEqual("CANARY", variants["v1"]["status"])
            self.assertEqual("child", variants["v1"]["metadata"]["kind"])
            evaluations = db.evolution_evaluations("v1")
            self.assertEqual("PROMOTE", evaluations[0]["decision"])
            self.assertEqual({"quality": 0.5}, evaluations[0]["baseline_metrics"])

    def test_variant_identity_cannot_be_rewritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Database(Path(tmp) / "workflow.sqlite")
            db.initialize()
            kwargs = dict(
                variant_id="v1", parent_variant_id=None, source_run_id=None,
                action_id=None, workflow_stage="lge_myo", code_commit="abc",
                continuity_revision=None, artifact_path=None,
                metadata={"x": 1},
            )
            db.register_evolution_variant(**kwargs)
            db.register_evolution_variant(**kwargs)
            with self.assertRaises(ValueError):
                db.register_evolution_variant(**{**kwargs, "code_commit": "other"})


if __name__ == "__main__":
    unittest.main()
