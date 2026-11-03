"""Synthetic-only control tests; no roles, providers, clinical data or real models."""
from __future__ import annotations

import ast
import copy
import json
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np

from app import segmentation_cycle as cycle
from app import segmentation_registry as registry
from app import segmentation_training as training
from app.database import Database
from tests import test_segmentation_training as fixtures

binding = fixtures.binding


class CycleTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.SegmentationTrainingTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.base = fixture.root
        self.target = getattr(self, "target", "lge_myo")
        self.root = self.base / "project"
        self.reg = self.base / "registry"
        self.job = fixture.make_job(self.target)
        architecture = Path(self.job["vendor"]["root"]) / "synthetic_model.py"
        architecture.write_text("# synthetic architecture identity\n")
        self.job["vendor"]["files"]["synthetic_model.py"] = training.sha256_file(architecture)
        config_path = Path(self.job["config"]["path"])
        config = json.loads(config_path.read_text())
        config = {k: config[k] for k in (("model", "training", "visualization", "roi", "loss")
                                       if self.target == "lge_mi" else ("model", "training", "visualization"))}
        config_path.write_text(json.dumps(config))
        self.job["config"] = binding(config_path)
        self.job_path = self.write_json("job.json", self.job)
        manifest = self.write_json("data.json", cycle.dataset_manifest_for_job(self.job))
        baseline_weights = self.base / "baseline.pt"
        baseline_weights.write_bytes(b"synthetic-baseline")
        self.baseline = registry.register_baseline(self.reg, target=self.target, weights=baseline_weights,
            config=config_path, architecture=architecture, dataset_manifest=manifest, owner="baseline-curator")
        registry.select_baseline(self.reg, self.baseline["id"])
        self.rows = []
        for i in range(2):
            image = self.base / ("evaluation-image-" + str(i) + ".nii")
            reference = self.base / ("reference-" + str(i) + ".nii")
            nib.save(nib.Nifti1Image(np.full((2, 2), i + 3, dtype=np.uint8), np.eye(4)), image)
            nib.save(nib.Nifti1Image(self.mask(i), np.eye(4)), reference)
            self.rows.append({"case_id": "case-" + str(i), "patient_id": "external-patient-" + str(i),
                              "image": binding(image), "reference": binding(reference)})
        cohort = self.write_json("cohort.json", {"schema": "segmentation-cycle-cohort-v1", "target": self.target, "cases": self.rows})
        definition = {"target": self.target, "dataset_sha256": binding(manifest)["sha256"],
            "benchmark_sha256": binding(cohort)["sha256"], "metrics": {
                "dice": {"kind": "segmentation", "direction": "max", "min_delta": 0.1, "max_regression": 0},
                "coverage": {"kind": "segmentation", "direction": "max", "min_delta": 0, "max_regression": 0},
                "empty_reference_fp_rate": {"kind": "segmentation", "direction": "min", "min_delta": 0, "max_regression": 0}}}
        policy = registry.approve_policy(self.reg, definition, approver="synthetic-policy-reviewer")
        owner = {"cmr_myo": "cmr_specialist", "lge_myo": "lge_myo_specialist", "lge_mi": "lge_mi_specialist"}[self.target]
        self.plan = {"schema": "segmentation-cycle-v1", "run_id": 1, "action_id": "train/synthetic",
            "owner": owner, "owner_role": owner, "target": self.target,
            "registry_root": str(self.reg), "parent_id": self.baseline["id"], "policy_id": policy["id"],
            "job": binding(self.job_path), "architecture": binding(architecture),
            "dataset_manifest": binding(manifest), "cohort": binding(cohort)}
        self.approvals, self.controls, self.predictions, self.evaluations = [], [], [], []
        self.allow_execute, self.allow_activate = True, True
        self.controller = cycle.TrustedController(self.authorize, self.control, self.predict, self.evaluate)

    def write_json(self, name, value):
        path = self.base / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    @staticmethod
    def mask(index):
        return np.array([[1, 0], [0, 0]] if index == 0 else [[0, 0], [0, 0]], dtype=np.uint8)

    def authorize(self, request):
        self.approvals.append(request)
        if not (self.allow_execute if request.phase == "execute" else self.allow_activate):
            return None
        return cycle.StoredApproval("synthetic-stored-review:" + request.phase, "synthetic-reviewer",
                                    request.plan_sha256, request.phase, request.subject_sha256)

    def control(self, request):
        self.controls.append(request)
        return cycle.ControlDecision(cycle.Control.RUN)

    def predict(self, request, checkpoint):
        checkpoint()
        self.predictions.append(copy.deepcopy(request))
        self.assertTrue(all(set(r) == {"case_id", "patient_id", "image"} for r in request.cases))
        request.output_dir.mkdir()
        outputs = {}
        for i, row in enumerate(request.cases):
            path = request.output_dir / (str(i) + ".nii")
            values = np.zeros((2, 2), dtype=np.uint8) if request.model_id == self.baseline["id"] else self.mask(i)
            nib.save(nib.Nifti1Image(values, np.eye(4)), path)
            outputs[row["case_id"]] = binding(path)
        checkpoint()
        return cycle.PredictionReceipt(request.model_id, request.cohort_sha256, outputs)

    def evaluate(self, request):
        self.evaluations.append(request)
        return cycle.IndependentEvidence("synthetic-stored-evidence", "synthetic-evaluator", request.plan_sha256,
            request.report_sha256, "synthetic-reference-provenance", "synthetic-parent-exposure",
            "synthetic-split-review", True, True)

    def receipt(self, job, *, execute):
        self.assertTrue(execute)
        checked = training.validate_job(job)
        directory = Path(checked["run_dir"])
        artifacts = directory / "artifacts"
        artifacts.mkdir(exist_ok=True)
        checkpoint = artifacts / "synthetic.pt"
        checkpoint.write_bytes(b"synthetic-completed-training")
        effective = copy.deepcopy(checked["config"])
        if job["target"] == "lge_mi":
            dependency = directory / "inputs/myo_weights" / Path(job["myo_weights"]["path"]).name
            dependency.parent.mkdir(parents=True, exist_ok=True)
            dependency.write_bytes(Path(job["myo_weights"]["path"]).read_bytes())
            effective["myocardium_model"] = {"weights": str(dependency), "sha256": job["myo_weights"]["sha256"]}
        training._atomic_json(directory / "effective_config.json", effective)
        r = {"status": "TRAINED", "job_sha256": checked["job_sha256"], "checkpoint": str(checkpoint),
            "checkpoint_sha256": training.sha256_file(checkpoint), "config_sha256": job["config"]["sha256"],
            "projected_config_sha256": cycle._hash(checked["config"]),
            "vendor_sources_sha256": cycle._hash(job["vendor"]["files"]),
            "parent_weights_sha256": None, "myo_weights_sha256": job.get("myo_weights", {}).get("sha256")}
        training._atomic_json(directory / "status.json", r)
        return r

    def run_mock(self, **kwargs):
        with patch.object(training, "run_job", side_effect=self.receipt) as train:
            result = cycle.run_cycle(self.root, self.plan, controller=kwargs.pop("controller", self.controller),
                                     execute=True, **kwargs)
        return result, train.call_count

    def test_dry_run_is_read_only_and_never_calls_capabilities(self):
        before = sorted(str(p) for p in self.base.rglob("*"))
        with patch.object(training, "run_job", side_effect=AssertionError("launch")):
            result = cycle.run_cycle(self.root, self.plan, controller=self.controller)
        self.assertEqual(result["stage"], "PREFLIGHT")
        self.assertEqual(before, sorted(str(p) for p in self.base.rglob("*")))
        self.assertEqual(self.approvals + self.predictions + self.controls, [])

    def test_typed_plan_and_json_plan_have_identical_bindings(self):
        plan = cycle.CyclePlan(**self.plan)
        self.assertEqual(plan.to_dict(), self.plan)
        self.assertEqual(cycle.plan_sha256(plan), cycle.plan_sha256(self.plan))
        self.assertEqual(cycle.implementation_sha256(plan), cycle.implementation_sha256(self.plan))
        result = cycle.run_cycle(self.root, plan, controller=self.controller)
        self.assertEqual(result["stage"], "PREFLIGHT")

    def test_complete_cycle_measures_predictions_and_activates_once(self):
        result, calls = self.run_mock()
        self.assertEqual(result["stage"], "ACTIVATED", result.get("reason"))
        self.assertEqual(calls, 1)
        self.assertEqual(len(self.predictions), 2)
        self.assertEqual(self.predictions[0].cases, self.predictions[1].cases)
        candidate = registry.resolve_candidate(self.reg, result["candidate_id"])
        self.assertEqual(candidate["training_record"]["vendor_sources_sha256"], cycle._hash(self.job["vendor"]["files"]))
        self.assertEqual(candidate["training_record"]["artifact_hashes"]["dataset_manifest"], self.plan["dataset_manifest"]["sha256"])
        self.assertEqual(result["report"]["metrics"]["dice"]["candidate"], 1)
        self.assertEqual(result["report"]["metrics"]["dice"]["parent"], 0.5)
        again, calls = self.run_mock()
        self.assertEqual(again, result)
        self.assertEqual(calls, 0)
        self.assertEqual(len(self.predictions), 2)
        stages = list((self.root / "runtime/segmentation_cycles").glob("*/stages/*.json"))
        self.assertGreater(len(stages), 5)

    def test_actual_existing_vendor_adapter_on_synthetic_files(self):
        result = cycle.run_cycle(self.root, self.plan, controller=self.controller, execute=True)
        self.assertEqual(result["stage"], "ACTIVATED", result.get("reason"))
        self.assertEqual(result["training"]["status"], "TRAINED")
        candidate = registry.resolve_candidate(self.reg, result["candidate_id"])
        config = json.loads(Path(candidate["artifacts"]["config"]["path"]).read_text())
        self.assertEqual(config["training"]["epochs"], self.job["resources"]["max_epochs"])
        self.assertEqual(set(config), {"model", "training", "visualization"})
        self.assertNotEqual(candidate["artifacts"]["config"]["sha256"], self.job["config"]["sha256"])

    def test_actual_mi_training_registers_pinned_effective_dependency(self):
        fixture = CycleTests()
        fixture.target = "lge_mi"
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        result = cycle.run_cycle(fixture.root, fixture.plan, controller=fixture.controller, execute=True)
        self.assertEqual(result["stage"], "ACTIVATED", result.get("reason"))
        model = registry.resolve_candidate(fixture.reg, result["candidate_id"])
        config = json.loads(Path(model["artifacts"]["config"]["path"]).read_text())
        dependency = config["myocardium_model"]
        self.assertEqual(dependency["sha256"], fixture.job["myo_weights"]["sha256"])
        self.assertNotEqual(dependency["weights"], fixture.job["myo_weights"]["path"])
        self.assertEqual(training.sha256_file(dependency["weights"]), dependency["sha256"])
        self.assertEqual(model["training_record"]["artifact_hashes"]["config"], model["artifacts"]["config"]["sha256"])

    def test_effective_config_tampering_prevents_registration(self):
        original = self.receipt
        def corrupt(job, *, execute):
            result = original(job, execute=execute)
            directory = Path(training.validate_job(job)["run_dir"])
            config = training.read_json(directory / "effective_config.json")
            config["training"]["epochs"] += 1
            training._atomic_json(directory / "effective_config.json", config)
            return result
        with patch.object(training, "run_job", side_effect=corrupt):
            result = cycle.run_cycle(self.root, self.plan, controller=self.controller, execute=True)
        self.assertEqual(result["stage"], "FAILED")
        self.assertIsNone(result["candidate_id"])

    def test_board_latest_has_real_stage_and_only_aggregate_fields(self):
        result, _ = self.run_mock(controller=replace(self.controller, predict=None))
        latest = training.read_json(self.root / "runtime/segmentation_training/latest.json")
        self.assertEqual(latest["status"], result["stage"])
        self.assertEqual(latest["target"], self.target)
        self.assertEqual(latest["job_id"], result["job_sha256"])
        self.assertGreaterEqual(latest["elapsed_seconds"], 0)
        self.assertEqual(set(latest), {"schema", "stage", "status", "target", "job_id", "elapsed_seconds"})
        from app.segmentation_status import status
        self.assertEqual(status(self.root)["training"]["status"], "AWAITING_EVALUATION")

    def test_missing_evaluator_or_prediction_is_explicit_and_resumes_without_retraining(self):
        result, calls = self.run_mock(controller=replace(self.controller, predict=None))
        self.assertEqual(result["stage"], "AWAITING_EVALUATION")
        self.assertIn("trusted_prediction_adapter", result["missing_evidence"])
        result, calls = self.run_mock(controller=replace(self.controller, evaluate=None))
        self.assertEqual(calls, 0)
        self.assertEqual(result["stage"], "AWAITING_EVALUATION")
        result, calls = self.run_mock()
        self.assertEqual(result["stage"], "ACTIVATED")
        self.assertEqual((calls, len(self.predictions)), (0, 2))

    def test_approval_wait_is_resumable_and_not_an_owner_boolean(self):
        self.allow_execute = False
        result, calls = self.run_mock()
        self.assertEqual((result["stage"], calls), ("AWAITING_APPROVAL", 0))
        self.allow_execute = True
        self.allow_activate = False
        result, calls = self.run_mock()
        self.assertEqual(result["stage"], "AWAITING_ACTIVATION_APPROVAL")
        self.assertEqual(registry.resolve_active(self.reg, "lge_myo")["candidate_id"], self.baseline["id"])
        self.allow_activate = True
        result, calls = self.run_mock()
        self.assertEqual((result["stage"], calls), ("ACTIVATED", 0))

    def test_self_approval_boolean_and_wrong_plan_receipt_fail_closed(self):
        for value in (True, {"approved": True}, cycle.StoredApproval("review", self.plan["owner"],
                cycle.plan_sha256(self.plan), "execute", cycle.plan_sha256(self.plan)),
                cycle.StoredApproval("review", "reviewer", "0" * 64, "execute", "0" * 64)):
            with self.subTest(value=value):
                root = self.base / ("negative-" + str(len(self.approvals)))
                self.approvals.append(value)
                with patch.object(training, "run_job") as train:
                    result = cycle.run_cycle(root, self.plan,
                        controller=replace(self.controller, authorize=lambda r: value), execute=True)
                self.assertEqual(result["stage"], "FAILED")
                train.assert_not_called()

    def test_failed_training_never_retries_and_does_not_register(self):
        with patch.object(training, "run_job", return_value={"status": "FAILED", "reason": "synthetic-failure"}) as train:
            first = cycle.run_cycle(self.root, self.plan, controller=self.controller, execute=True)
            second = cycle.run_cycle(self.root, self.plan, controller=self.controller, execute=True)
        self.assertEqual(first, second)
        self.assertEqual(train.call_count, 1)
        self.assertIsNone(first["candidate_id"])

    def test_pause_at_launch_then_resume_and_budget_file_fail_closed(self):
        operator = self.root / "runtime/control/operator_state.json"
        operator.parent.mkdir(parents=True)
        operator.write_text('{"paused":true}')
        result, calls = self.run_mock()
        self.assertEqual((result["stage"], calls), ("PAUSED", 0))
        operator.write_text('{"paused":false}')
        budget = operator.with_name("run_budget.json")
        budget.write_text('{"active":false}')
        result, calls = self.run_mock()
        self.assertEqual((result["stage"], calls), ("PAUSED", 0))
        budget.unlink()
        result, calls = self.run_mock()
        self.assertEqual((result["stage"], calls), ("ACTIVATED", 1))

    def test_polling_pause_cancels_attempt_without_retry(self):
        def control(request):
            return cycle.ControlDecision(cycle.Control.PAUSE if request.phase == "training_poll" else cycle.Control.RUN,
                                         "synthetic-budget-exhausted")
        def slow_train(job, *, execute):
            run_dir = Path(training.validate_job(job)["run_dir"])
            for _ in range(100):
                if (run_dir / "CANCEL").exists():
                    return {"status": "CANCELLED", "reason": "cancel_marker"}
                threading.Event().wait(0.02)
            self.fail("training was not cancelled")
        with patch.object(training, "run_job", side_effect=slow_train) as train:
            result = cycle.run_cycle(self.root, self.plan, controller=replace(self.controller, control=control), execute=True)
            again = cycle.run_cycle(self.root, self.plan, controller=self.controller, execute=True)
        self.assertEqual((result["stage"], train.call_count), ("CANCELLED", 1))
        self.assertEqual(result, again)

    def test_crash_after_launch_intent_never_automatically_retries(self):
        with patch.object(cycle, "_train", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                cycle.run_cycle(self.root, self.plan, controller=self.controller, execute=True)
        result, calls = self.run_mock()
        self.assertEqual((result["stage"], calls), ("FAILED", 0))
        self.assertIn("no_retry", result["reason"])

    def test_recover_completed_receipt_after_controller_crash(self):
        original = cycle._train
        def crash(checked, checkpoint):
            original(checked, checkpoint)
            raise KeyboardInterrupt
        with patch.object(training, "run_job", side_effect=self.receipt), patch.object(cycle, "_train", side_effect=crash):
            with self.assertRaises(KeyboardInterrupt):
                cycle.run_cycle(self.root, self.plan, controller=self.controller, execute=True)
        result, calls = self.run_mock()
        self.assertEqual((result["stage"], calls), ("ACTIVATED", 0))

    def test_manifest_must_bind_split_and_changed_plan_cannot_reuse_action(self):
        path = Path(self.plan["dataset_manifest"]["path"])
        manifest = json.loads(path.read_text())
        manifest["splits"]["test"] = []
        path.write_text(json.dumps(manifest))
        self.plan["dataset_manifest"] = binding(path)
        with self.assertRaisesRegex(cycle.CycleError, "exact_records_and_splits"):
            self.run_mock()

    def test_plan_drift_and_unknown_self_approval_fields_are_rejected(self):
        self.allow_execute = False
        self.run_mock()
        changed = dict(self.plan, owner="other-owner")
        with self.assertRaisesRegex(cycle.CycleError, "another_plan"):
            cycle.run_cycle(self.root, changed, controller=self.controller, execute=True)
        for field in ("approved", "approval", "metrics", "auto_retry"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                cycle.run_cycle(self.root, dict(self.plan, **{field: True}), controller=self.controller)

    def test_candidate_training_exposure_is_checked_without_claiming_parent_exposure(self):
        self.rows[0]["patient_id"] = self.job["splits"]["validation"][0]
        path = Path(self.plan["cohort"]["path"])
        path.write_text(json.dumps({"schema": "segmentation-cycle-cohort-v1", "target": "lge_myo", "cases": self.rows}))
        self.plan["cohort"] = binding(path)
        with self.assertRaisesRegex(cycle.CycleError, "cohort_overlap"):
            self.run_mock()

    def test_missing_parent_exposure_and_disjointness_are_not_invented(self):
        def evidence(request):
            return replace(self.evaluate(request), parent_exposure_evidence_id="",
                           training_evaluation_patient_disjoint=False)
        result, calls = self.run_mock(controller=replace(self.controller, evaluate=evidence))
        self.assertEqual(result["stage"], "AWAITING_EVALUATION")
        self.assertCountEqual(result["missing_evidence"], ["parent_exposure_evidence_id", "training_evaluation_patient_disjoint"])
        self.assertNotIn("assessment", result)

    def test_owner_and_stale_evaluator_evidence_cannot_promote(self):
        def evidence(request):
            return replace(self.evaluate(request), actor=self.plan["owner"])
        result, _ = self.run_mock(controller=replace(self.controller, evaluate=evidence))
        self.assertEqual(result["stage"], "FAILED")
        self.assertIn("not_independent", result["reason"])

    def test_stale_evidence_report_binding_fails(self):
        def evidence(request):
            return replace(self.evaluate(request), report_sha256="0" * 64)
        result, _ = self.run_mock(controller=replace(self.controller, evaluate=evidence))
        self.assertEqual(result["stage"], "FAILED")
        self.assertIn("evidence_binding_mismatch", result["reason"])

    def test_evaluator_cannot_supply_metrics_or_mutate_computed_report(self):
        def evidence(request):
            request.report["metrics"]["dice"]["candidate"] = 900
            return self.evaluate(request)
        result, _ = self.run_mock(controller=replace(self.controller, evaluate=evidence))
        self.assertEqual(result["stage"], "ACTIVATED")
        self.assertEqual(result["report"]["metrics"]["dice"]["candidate"], 1)

    def test_prediction_change_during_independent_evidence_lookup_is_rejected(self):
        def evidence(request):
            Path(request.report["manifest"]["cases"][0]["candidate"]["path"]).write_bytes(b"changed")
            return self.evaluate(request)
        result, _ = self.run_mock(controller=replace(self.controller, evaluate=evidence))
        self.assertEqual(result["stage"], "FAILED")
        self.assertNotIn("assessment", result)

    def test_prediction_change_during_activation_approval_is_rejected(self):
        def authorize(request):
            if request.phase == "activate":
                path = self.predictions[-1].output_dir / "0.nii"
                path.write_bytes(b"changed-after-assessment")
            return self.authorize(request)
        result, _ = self.run_mock(controller=replace(self.controller, authorize=authorize))
        self.assertEqual(result["stage"], "FAILED")
        self.assertEqual(registry.resolve_active(self.reg, "lge_myo")["candidate_id"], self.baseline["id"])

    def test_wrong_prediction_identity_or_omitted_cases_fail_without_retry(self):
        def predict(request, checkpoint):
            receipt = self.predict(request, checkpoint)
            return replace(receipt, model_id="0" * 64)
        result, _ = self.run_mock(controller=replace(self.controller, predict=predict))
        self.assertEqual(result["stage"], "FAILED")
        self.assertIn("prediction_model", result["reason"])
        again, calls = self.run_mock()
        self.assertEqual((again, calls), (result, 0))

    def test_omitted_prediction_case_is_not_an_improvement(self):
        def predict(request, checkpoint):
            receipt = self.predict(request, checkpoint)
            receipt.predictions.pop("case-1")
            return receipt
        result, _ = self.run_mock(controller=replace(self.controller, predict=predict))
        self.assertEqual(result["stage"], "FAILED")
        self.assertIn("incomplete_prediction_cohort", result["reason"])

    def test_no_segmentation_gain_retains_parent_without_activation_approval(self):
        def predict(request, checkpoint):
            receipt = self.predict(request, checkpoint)
            for output in receipt.predictions.values():
                path = Path(output["path"])
                nib.save(nib.Nifti1Image(np.zeros((2, 2), dtype=np.uint8), np.eye(4)), path)
                output.update(binding(path))
            return receipt
        result, _ = self.run_mock(controller=replace(self.controller, predict=predict))
        self.assertEqual(result["stage"], "RETAIN_PARENT")
        self.assertFalse(result["assessment"]["passed"])
        self.assertFalse(any(r.phase == "activate" for r in self.approvals))
        again, calls = self.run_mock()
        self.assertEqual((again, calls), (result, 0))

    def test_prediction_poll_pause_is_terminal_no_automatic_retry(self):
        def control(request):
            return cycle.ControlDecision(cycle.Control.PAUSE if request.phase.endswith("prediction_poll") else cycle.Control.RUN)
        result, _ = self.run_mock(controller=replace(self.controller, control=control))
        self.assertEqual(result["stage"], "CANCELLED")
        self.assertEqual(self.predictions, [])
        again, calls = self.run_mock()
        self.assertEqual((again, calls), (result, 0))

    def test_pause_before_register_preserves_completed_training_for_resume(self):
        def control(request):
            return cycle.ControlDecision(cycle.Control.PAUSE if request.phase == "register" else cycle.Control.RUN)
        result, _ = self.run_mock(controller=replace(self.controller, control=control))
        self.assertEqual(result["stage"], "PAUSED")
        self.assertEqual(result["training"]["status"], "TRAINED")
        result, calls = self.run_mock()
        self.assertEqual((result["stage"], calls), ("ACTIVATED", 0))

    def test_changed_predictions_fail_on_resume(self):
        result, _ = self.run_mock(controller=replace(self.controller, evaluate=None))
        path = Path(next(iter(result["predictions"]["candidate"]["predictions"].values()))["path"])
        path.write_bytes(b"changed")
        result, calls = self.run_mock()
        self.assertEqual((result["stage"], calls), ("FAILED", 0))
        self.assertIn("SHA-256", result["reason"])

    def test_false_vendor_provenance_prevents_registration(self):
        original = self.receipt
        def bad(job, *, execute):
            return dict(original(job, execute=execute), vendor_sources_sha256="0" * 64)
        with patch.object(training, "run_job", side_effect=bad):
            result = cycle.run_cycle(self.root, self.plan, controller=self.controller, execute=True)
        self.assertEqual(result["stage"], "FAILED")
        self.assertIsNone(result["candidate_id"])

    def test_selector_conflict_retains_new_parent(self):
        self.allow_activate = False
        result, _ = self.run_mock()
        # Explicitly activate the candidate outside this action, then rollback;
        # a different current model must cause the final CAS to fail.
        registry.activate(self.reg, result["candidate_id"], result["assessment"], self.baseline["id"])
        current = registry.resolve_active(self.reg, "lge_myo")
        with patch.object(registry, "activate", side_effect=registry.ConcurrentUpdate("synthetic parent changed")):
            # Rollback makes activation necessary, patched CAS models a concurrent change.
            registry.rollback(self.reg, "lge_myo", expected_active_id=current["candidate_id"])
            self.allow_activate = True
            result, _ = self.run_mock()
        self.assertEqual(result["stage"], "FAILED")
        self.assertIn("ConcurrentUpdate", result["reason"])

    def test_crash_after_activation_reconciles_without_second_pointer_swap(self):
        original = registry.activate
        def activate(*args, **kwargs):
            original(*args, **kwargs)
            raise KeyboardInterrupt
        with patch.object(registry, "activate", side_effect=activate):
            with self.assertRaises(KeyboardInterrupt):
                self.run_mock()
        with patch.object(registry, "activate", side_effect=AssertionError("duplicate activation")):
            result, calls = self.run_mock()
        self.assertEqual((result["stage"], calls), ("ACTIVATED", 0))

    def test_action_lock_excludes_concurrent_controller(self):
        def control(request):
            if request.phase == "training_launch":
                with self.assertRaises(training.JobBusy):
                    cycle.run_cycle(self.root, self.plan, controller=self.controller, execute=True)
            return cycle.ControlDecision(cycle.Control.RUN)
        result, _ = self.run_mock(controller=replace(self.controller, control=control))
        self.assertEqual(result["stage"], "ACTIVATED")

    def test_python39_grammar(self):
        ast.parse(Path(cycle.__file__).read_text(), feature_version=(3, 9))

    def make_db(self):
        db = Database(self.base / "synthetic-governance.sqlite")
        db.initialize()
        dataset = db.register_dataset("synthetic", self.base)
        run = db.create_run(dataset, "audit", {})
        self.assertEqual(run, self.plan["run_id"])
        action = {"action_id": self.plan["action_id"], "owner_role": self.plan["owner_role"],
                  "action_class": "segmentation_training", "review_approved": True,
                  "segmentation_cycle_plan": self.plan}
        db.save_maintenance_action(run, action, status="APPROVED")
        return db

    def post(self, db, sender, kind, payload):
        db.post_agent_message(self.plan["run_id"], thread_id=f"run-{self.plan['run_id']}:{self.plan['action_id']}",
                              sender_role=sender, recipient_role=self.plan["owner_role"], message_type=kind, payload=payload)

    def test_existing_db_requires_original_handoffs_and_exact_implementation_hash(self):
        db = self.make_db()
        controller = replace(self.controller, authorize=cycle.DatabaseAuthority(db))
        result, calls = self.run_mock(controller=controller)
        self.assertEqual((result["stage"], calls), ("AWAITING_APPROVAL", 0))
        self.post(db, "supervisor", "assigned_action", {})
        self.post(db, "independent_reviewer", "action_review", {"decision": "accept_experiment"})
        self.post(db, "independent_reviewer", "implementation_review", {
            "decision": "accept_experiment", "implementation_sha256": "0" * 64})
        result, calls = self.run_mock(controller=controller)
        self.assertEqual((result["stage"], calls), ("AWAITING_APPROVAL", 0))
        self.post(db, "supervisor", "assigned_action", {
            "action_class": "segmentation_training", "segmentation_cycle_plan": self.plan})
        self.post(db, "independent_reviewer", "action_review", {
            "decision": "accept_experiment", "segmentation_plan_sha256": cycle.implementation_sha256(self.plan)})
        self.post(db, "independent_reviewer", "implementation_review", {
            "decision": "accept_experiment", "implementation_sha256": cycle.implementation_sha256(self.plan)})
        result, calls = self.run_mock(controller=controller)
        self.assertEqual((result["stage"], calls), ("AWAITING_ACTIVATION_APPROVAL", 1))
        approval = {"decision": "accept_experiment", "plan_sha256": cycle.plan_sha256(self.plan),
            "candidate_id": result["candidate_id"], "assessment_id": result["assessment"]["assessment_id"],
            "policy_id": self.plan["policy_id"]}
        self.post(db, self.plan["owner_role"], "segmentation_activation_review", approval)
        result, calls = self.run_mock(controller=controller)
        self.assertEqual((result["stage"], calls), ("AWAITING_ACTIVATION_APPROVAL", 0))
        self.post(db, "independent_reviewer", "segmentation_activation_review", approval)
        result, calls = self.run_mock(controller=controller)
        self.assertEqual((result["stage"], calls), ("ACTIVATED", 0))

    def test_latest_db_review_revocation_overrides_old_exact_approval(self):
        db = self.make_db()
        self.post(db, "supervisor", "assigned_action", {
            "action_class": "segmentation_training", "segmentation_cycle_plan": self.plan})
        self.post(db, "independent_reviewer", "action_review", {
            "decision": "accept_experiment", "segmentation_plan_sha256": cycle.implementation_sha256(self.plan)})
        self.post(db, "independent_reviewer", "implementation_review", {
            "decision": "accept_experiment", "implementation_sha256": cycle.implementation_sha256(self.plan)})
        controller = replace(self.controller, authorize=cycle.DatabaseAuthority(db))
        result, _ = self.run_mock(controller=controller)
        self.assertEqual(result["stage"], "AWAITING_ACTIVATION_APPROVAL")
        self.post(db, "independent_reviewer", "implementation_review", {
            "decision": "reject", "implementation_sha256": cycle.implementation_sha256(self.plan)})
        result, calls = self.run_mock(controller=controller)
        self.assertEqual((result["stage"], calls), ("AWAITING_APPROVAL", 0))


if __name__ == "__main__":
    unittest.main()
