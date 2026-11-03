"""Synthetic host/CLI tests, including real subprocesses and fake-vendor training."""
from __future__ import annotations

import ast
import copy
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from app import segmentation_cycle as cycle
from app import segmentation_host as host
from app import segmentation_registry as registry
from app import segmentation_training as training
from tests import test_segmentation_cycle as fixtures


CLI = Path(__file__).resolve().parents[1] / "scripts/run_segmentation_cycle.py"

FAKE_PREDICT = r'''
if __name__ == "__main__":
    import argparse
    import numpy as np
    import nibabel as nib
    p = argparse.ArgumentParser()
    p.add_argument("--config")
    p.add_argument("action", choices=["predict"])
    p.add_argument("--weights")
    p.add_argument("--input")
    p.add_argument("--output-dir")
    p.add_argument("--device")
    p.add_argument("--myo-weights")
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text())
    mode = config["model"].get("host_test_mode", 0)
    assert "SYNTHETIC_PROVIDER_API_KEY" not in os.environ
    inputs = list(Path(args.input).iterdir())
    assert inputs and all(x.name.endswith((".nii", ".nii.gz")) for x in inputs)
    assert all(len(x.name.split(".")[0]) == 24 for x in inputs)
    output = Path(args.output_dir)
    output.mkdir()
    (output / "observations.json").write_text(json.dumps({"input_names": sorted(x.name for x in inputs),
        "device": args.device, "config": config}))
    if mode == 1:
        raise RuntimeError("synthetic execution failure")
    if mode == 2:
        import sys
        code = 'import pathlib,sys,time; p=pathlib.Path(sys.argv[1]);\nwhile True:\n p.write_text(str(time.monotonic())); time.sleep(.02)'
        child = subprocess.Popen([sys.executable, "-c", code, str(output / "heartbeat")],
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        while not (output / "heartbeat").exists():
            time.sleep(.02)
        time.sleep(60)
    mask_dir = output / ("predictions" if mode == 7 else "masks")
    mask_dir.mkdir()
    if mode == 8:
        (output / "predictions").mkdir()
    baseline = Path(args.weights).read_bytes() == b"synthetic-baseline"
    for i, image in enumerate(sorted(inputs)):
        if mode == 3 and i == 0:
            continue
        values = np.zeros((2, 2), dtype=np.uint8)
        if not baseline and np.all(np.asarray(nib.load(image).dataobj) == 3):
            values[0, 0] = 1
        nib.save(nib.Nifti1Image(values, np.eye(4)), mask_dir / image.name)
        if mode == 5 and i == 0:
            nib.save(nib.Nifti1Image(values, np.eye(4)), mask_dir / (image.name.split(".")[0] + ".nii.gz"))
    if mode == 4:
        nib.save(nib.Nifti1Image(np.zeros((2, 2), dtype=np.uint8), np.eye(4)), mask_dir / "unmapped.nii")
    if mode == 6:
        inputs[0].write_bytes(b"changed input snapshot")
'''


class HostTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.CycleTests()
        self.f.target = getattr(self, "target", "lge_myo")
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        f = self.f
        folder = Path(f.job["vendor"]["root"])
        entry = folder / training.VENDORS[f.target][1]
        # Reuse the existing synthetic trainer; only its __main__ is inference.
        entry.write_text(entry.read_text() + FAKE_PREDICT, encoding="utf-8")
        name = {"lge_myo": "lge_seg/model.py", "cmr_myo": "cine_seg/model.py",
                "lge_mi": "lge_mi_prior_seg/model.py"}[f.target]
        architecture = folder / name
        architecture.parent.mkdir(exist_ok=True)
        architecture.write_bytes(Path(f.plan["architecture"]["path"]).read_bytes())
        f.job["vendor"]["files"] = {p.relative_to(folder).as_posix(): training.sha256_file(p) for p in folder.rglob("*.py")}
        f.write_json("job.json", f.job)
        f.plan["job"] = fixtures.binding(f.job_path)
        f.plan["architecture"] = fixtures.binding(architecture)
        role = {"cmr_myo": "cmr_specialist", "lge_myo": "lge_myo_specialist", "lge_mi": "lge_mi_specialist"}[f.target]
        f.plan.update(owner=role, owner_role=role)
        self.db = f.make_db()
        self.db_path = self.db.path
        self.plan = f.plan
        self.bundles = {}
        self.bind_sources()
        self.grant_budget()

    def predictor(self, **kwargs):
        return host.RegistryPredictor(self.plan, **kwargs)

    def bind_sources(self):
        model = registry.resolve_candidate(self.f.reg, self.plan["parent_id"])
        if model["candidate_id"] not in self.bundles:
            root = self.f.base / ("b" + model["candidate_id"][:16])
            for relative in self.f.job["vendor"]["files"]:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(Path(self.f.job["vendor"]["root"]) / relative, path)
            files = {p.relative_to(root).as_posix(): training.sha256_file(p) for p in root.rglob("*.py")}
            self.bundles[model["candidate_id"]] = (root, files)
        root, files = self.bundles[model["candidate_id"]]
        return host.freeze_model_sources(self.f.reg, model["candidate_id"], vendor_root=root)

    def grant_budget(self):
        folder = self.f.root / "runtime/control"
        folder.mkdir(parents=True, exist_ok=True)
        self.budget = folder / "run_budget.json"
        self.budget.write_text(json.dumps({"active": True, "status": "RUNNING", "deadline": time.time() + 600,
                                         "limits": {"analyses": 3, "requests": 30}, "used": {"analyses": 1, "requests": 2}}))
        self.operator = folder / "operator_state.json"
        self.operator.write_text('{"paused":false}')

    def control_request(self):
        return cycle.ControlRequest(self.f.root, cycle.plan_sha256(self.plan), self.plan["run_id"], self.plan["action_id"], "launch")

    def approve_plan(self):
        self.f.post(self.db, "supervisor", "assigned_action", {
            "action_class": "segmentation_training", "segmentation_cycle_plan": self.plan})
        self.f.post(self.db, "independent_reviewer", "action_review", {
            "decision": "accept_experiment", "segmentation_plan_sha256": cycle.implementation_sha256(self.plan)})
        self.f.post(self.db, "independent_reviewer", "implementation_review", {
            "decision": "accept_experiment", "implementation_sha256": cycle.implementation_sha256(self.plan)})

    def prediction_request(self, name="predictions"):
        model = registry.resolve_candidate(self.f.reg, self.plan["parent_id"])
        return cycle.PredictionRequest(model["candidate_id"], self.plan["target"], self.f.reg, model["artifacts"],
            self.plan["cohort"]["sha256"], [{k: r[k] for k in ("case_id", "patient_id", "image")} for r in self.f.rows],
            self.f.base / name)

    def set_mode(self, mode):
        f = self.f
        config = training.read_json(f.job["config"]["path"])
        config["model"]["host_test_mode"] = mode
        Path(f.job["config"]["path"]).write_text(json.dumps(config))
        f.job["config"] = fixtures.binding(Path(f.job["config"]["path"]))
        f.write_json("job.json", f.job)
        self.plan["job"] = fixtures.binding(f.job_path)
        baseline = registry.register_baseline(f.reg, target=f.target, weights=f.base / "baseline.pt",
            config=Path(f.job["config"]["path"]), architecture=Path(self.plan["architecture"]["path"]),
            dataset_manifest=Path(self.plan["dataset_manifest"]["path"]), owner="synthetic-custodian")
        self.plan["parent_id"] = baseline["id"]
        self.bind_sources()

    def post(self, kind, payload, sender="independent_evaluator"):
        return self.db.post_agent_message(self.plan["run_id"],
            thread_id=f"run-{self.plan['run_id']}:{self.plan['action_id']}", sender_role=sender,
            recipient_role=self.plan["owner_role"], message_type=kind, payload=payload)

    def evaluation_request(self):
        # A synthetic stored report fixture; real cycle tests below compute this.
        report = {"reference_sha256": "1" * 64}
        return cycle.EvaluationRequest(copy.deepcopy(self.plan), cycle.plan_sha256(self.plan), "2" * 64,
                                       cycle._hash(report), report)

    def evidence_messages(self, request, *, sender="independent_evaluator", exposure="ESTABLISHED"):
        p = self.plan
        common = {"plan_sha256": request.plan_sha256, "status": "VERIFIED", "cohort_sha256": p["cohort"]["sha256"]}
        reference = self.post("segmentation_reference_provenance", dict(common,
            reference_sha256=request.report["reference_sha256"]), sender)
        parent = self.post("segmentation_parent_exposure", dict(common,
            parent_id=p["parent_id"], training_evaluation_patient_disjoint=exposure), sender)
        split = self.post("segmentation_split_evidence", dict(common,
            dataset_sha256=p["dataset_manifest"]["sha256"], validation_test_patient_disjoint="ESTABLISHED",
            training_evaluation_patient_disjoint="ESTABLISHED"), sender)
        body = {"schema": "segmentation-evaluation-evidence-v1", "status": "VERIFIED",
            "plan_sha256": request.plan_sha256, "report_sha256": request.report_sha256,
            "parent_id": p["parent_id"], "candidate_id": request.candidate_id, "policy_id": p["policy_id"], "target": p["target"],
            "reference_provenance_message_id": reference, "parent_exposure_message_id": parent, "split_evidence_message_id": split}
        self.post("segmentation_evaluation_evidence", body, sender)
        return body

    def test_predict_real_subprocess_uses_stable_image_only_mapping_and_no_activation(self):
        before = registry.resolve_active(self.f.reg, self.f.target)
        request = self.prediction_request()
        count = []
        with patch.dict(os.environ, {"SYNTHETIC_PROVIDER_API_KEY": "synthetic-do-not-forward"}):
            receipt = self.predictor()(request, lambda: count.append(1))
        self.assertGreater(len(count), 3)
        self.assertEqual(set(receipt.predictions), {r["case_id"] for r in self.f.rows})
        inputs = request.output_dir / "inputs"
        self.assertEqual(len(list(inputs.iterdir())), len(self.f.rows))
        for row in self.f.rows:
            path = inputs / (cycle._hash({"case_id": row["case_id"]})[:24] + ".nii")
            self.assertEqual(training.sha256_file(path), row["image"]["sha256"])
            self.assertNotIn(row["patient_id"], path.name)
        for value in receipt.predictions.values():
            self.assertEqual(training.sha256_file(value["path"]), value["sha256"])
        second = self.predictor()(self.prediction_request("again"), lambda: None)
        self.assertEqual({k: v["sha256"] for k, v in receipt.predictions.items()},
                         {k: v["sha256"] for k, v in second.predictions.items()})
        self.assertEqual(registry.resolve_active(self.f.reg, self.f.target), before)

    def test_cmr_current_and_legacy_layouts_and_ambiguity(self):
        fixture = HostTests()
        fixture.target = "cmr_myo"
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        result = fixture.predictor()(fixture.prediction_request(), lambda: None)
        self.assertTrue(all(Path(v["path"]).parent.name == "masks" for v in result.predictions.values()))
        fixture.set_mode(7)
        result = fixture.predictor()(fixture.prediction_request("legacy"), lambda: None)
        self.assertTrue(all(Path(v["path"]).parent.name == "predictions" for v in result.predictions.values()))
        fixture.set_mode(8)
        with self.assertRaisesRegex(host.HostError, "ambiguous"):
            fixture.predictor()(fixture.prediction_request("ambiguous"), lambda: None)

    def test_subprocess_failure_partial_extra_and_mutated_outputs_fail_closed(self):
        for mode, error in ((1, "worker_failed"), (3, "incomplete"), (4, "unexpected"), (5, "duplicate"), (6, "SHA-256")):
            with self.subTest(mode=mode):
                self.set_mode(mode)
                with self.assertRaisesRegex(ValueError, error):
                    self.predictor()(self.prediction_request("mode-" + str(mode)), lambda: None)

    def test_deadline_is_bounded_and_existing_output_never_retried(self):
        self.set_mode(2)
        request = self.prediction_request()
        started = time.monotonic()
        with self.assertRaisesRegex(host.HostError, "deadline"):
            self.predictor(max_seconds=0.2)(request, lambda: None)
        self.assertLess(time.monotonic() - started, 5)
        with self.assertRaises(FileExistsError):
            self.predictor()(request, lambda: None)

    def test_poll_cancellation_kills_real_descendant(self):
        self.set_mode(2)
        request = self.prediction_request()
        heartbeat = request.output_dir / "outputs/heartbeat"
        controller = cycle.TrustedController(lambda r: None, host.GlobalControl(self.f.root, self.plan))
        def checkpoint():
            if heartbeat.exists():
                self.operator.write_text('{"paused":true}')
            cycle._checkpoint(self.f.root, self.plan, cycle.plan_sha256(self.plan), controller, "prediction_poll")
        with self.assertRaises(cycle._Interrupted):
            self.predictor()(request, checkpoint)
        before = heartbeat.read_bytes()
        time.sleep(0.2)
        self.assertEqual(heartbeat.read_bytes(), before)

    def test_request_mismatch_and_changed_source_never_launch(self):
        request = self.prediction_request()
        request.cases[0]["image"] = request.cases[1]["image"]
        with patch.object(host, "_run_process") as process:
            with self.assertRaisesRegex(host.HostError, "cohort_mismatch"):
                self.predictor()(request, lambda: None)
        process.assert_not_called()
        self.assertFalse(request.output_dir.exists())

    def test_parent_source_bundle_missing_is_waiting_without_current_vendor_fallback(self):
        self.new_unbound_baseline()
        self.approve_plan()
        result = self.cli("--execute")
        self.assertEqual(result.returncode, 3, result.stdout + result.stderr)
        self.assertEqual(json.loads(result.stdout)["stage"], "AWAITING_MODEL_SOURCES")
        self.assertFalse(Path(self.f.job["output_dir"]).exists())

    def new_unbound_baseline(self):
        model = registry.register_baseline(self.f.reg, target=self.f.target, weights=self.f.base / "baseline.pt",
            config=Path(self.f.job["config"]["path"]), architecture=Path(self.plan["architecture"]["path"]),
            dataset_manifest=Path(self.plan["dataset_manifest"]["path"]), owner="another-baseline-custodian")
        self.plan["parent_id"] = model["id"]
        return model

    def test_baseline_sources_are_mechanical_once_per_model_not_medical_approval(self):
        model = self.new_unbound_baseline()
        before = self.db.agent_inbox(self.plan["run_id"], self.plan["owner_role"])
        vendor = Path(self.f.job["vendor"]["root"])
        controller = host.make_controller(self.f.root, self.plan, database=self.db_path, baseline_vendor=vendor)
        self.assertFalse((self.f.reg / "source_bundles" / model["id"]).exists())
        controller.predict(self.prediction_request(), lambda: None)
        body = training.read_json(self.f.reg / "source_bundles" / model["id"] / "bundle.json")
        self.assertEqual(body["status"], "BASELINE_UNMEASURED")
        self.assertEqual(self.db.agent_inbox(self.plan["run_id"], self.plan["owner_role"]), before)
        # Subsequent plans require neither the source-path argument nor DB rows.
        controller = host.make_controller(self.f.root, self.plan, database=self.db_path)
        self.assertEqual(controller.predict.sources.resolve(registry.resolve_candidate(self.f.reg, model["id"]))["sha256"],
                         body["vendor_sources_sha256"])

    def test_full_parent_vendor_manifest_is_checked_before_launch(self):
        root = self.f.reg / "source_bundles" / self.plan["parent_id"] / "vendor"
        entry = root / training.VENDORS[self.f.target][1]
        entry.chmod(0o600)
        entry.write_text(entry.read_text() + "\n# changed preprocessing\n")
        with patch.object(host, "_run_process") as process:
            with self.assertRaisesRegex(host.HostError, "source_bundle_changed"):
                self.predictor()(self.prediction_request(), lambda: None)
        process.assert_not_called()

    def test_source_binding_must_match_model_config_as_well_as_architecture(self):
        path = self.f.reg / "source_bundles" / self.plan["parent_id"] / "bundle.json"
        body = training.read_json(path)
        body["artifact_hashes"]["config"] = "0" * 64
        body.pop("id")
        path.chmod(0o600)
        training._atomic_json(path, registry._record(body))
        with self.assertRaisesRegex(host.HostError, "model_binding_mismatch"):
            host.make_controller(self.f.root, self.plan, database=self.db_path)

    def test_original_image_change_during_prediction_invalidates_all_outputs(self):
        request = self.prediction_request()
        source = Path(request.cases[0]["image"]["path"])
        def checkpoint():
            if (request.output_dir / "outputs/observations.json").exists():
                source.write_bytes(b"synthetic concurrent source change")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.predictor()(request, checkpoint)

    def test_host_walltime_cannot_increase_approved_job_limit(self):
        predictor = self.predictor(max_seconds=1e6)
        self.assertEqual(predictor.max_seconds, self.f.job["resources"]["max_seconds"])
        for bad in (0, -1, True, float("inf"), float("nan")):
            with self.subTest(bad=bad), self.assertRaises(host.HostError):
                self.predictor(max_seconds=bad)

    def test_controls_are_read_only_fail_closed_and_do_not_grant_budget(self):
        control = host.GlobalControl(self.f.root, self.plan)
        request = self.control_request()
        before = self.budget.read_bytes()
        self.assertEqual(control(request).action, cycle.Control.RUN)
        self.assertEqual(self.budget.read_bytes(), before)
        for changes in ({"active": False}, {"status": "EXHAUSTED"}, {"deadline": 0},
                        {"deadline": float("nan")}, {"used": {"analyses": 3, "requests": 2}},
                        {"used": {"analyses": 1, "requests": 30}}, {"used": {"analyses": True, "requests": 2}}):
            value = json.loads(before)
            value.update(changes)
            self.budget.write_text(json.dumps(value))
            previous = self.budget.read_bytes()
            with self.subTest(changes=changes):
                self.assertEqual(control(request).action, cycle.Control.PAUSE)
                self.assertEqual(self.budget.read_bytes(), previous)
        self.budget.unlink()
        self.assertEqual(control(request).action, cycle.Control.PAUSE)
        self.assertFalse(self.budget.exists())
        self.grant_budget()
        self.operator.write_text('{"paused":true}')
        self.assertEqual(control(request).action, cycle.Control.PAUSE)

    def test_database_connection_is_read_only_and_missing_db_is_not_created(self):
        db = host.ReadOnlyDatabase(self.db_path)
        with db.connect() as connection:
            with self.assertRaises(sqlite3.OperationalError):
                connection.execute("DELETE FROM agent_messages")
        missing = self.f.base / "missing/db.sqlite"
        with self.assertRaises(host.HostError):
            host.ReadOnlyDatabase(missing)
        self.assertFalse(missing.parent.exists())

    def test_evidence_requires_independent_records_and_known_baseline_exposure(self):
        request = self.evaluation_request()
        reader = host.DatabaseEvidence(host.ReadOnlyDatabase(self.db_path))
        self.assertIsNone(reader(request))
        self.evidence_messages(request, sender=self.plan["owner_role"])
        self.assertIsNone(reader(request))
        self.evidence_messages(request, exposure="UNKNOWN")
        self.assertIsNone(reader(request))
        self.assertEqual(reader.last_lookup.status, "WAITING")
        self.assertIn("parent_training_evaluation_patient_disjoint_not_established", reader.last_lookup.missing_evidence)
        self.evidence_messages(request)
        result = reader(request)
        self.assertIsInstance(result, cycle.IndependentEvidence)
        self.assertEqual(reader.last_lookup.status, "EVIDENCE_VERIFIED")
        self.assertTrue(result.parent_exposure_evidence_id.startswith("db:agent_messages:"))

    def test_exact_report_binding_and_latest_revocation_cannot_fall_back(self):
        request = self.evaluation_request()
        reader = host.DatabaseEvidence(host.ReadOnlyDatabase(self.db_path))
        body = self.evidence_messages(request)
        self.assertIsNotNone(reader(request))
        self.post("segmentation_evaluation_evidence", dict(body, report_sha256="0" * 64))
        self.assertIsNone(reader(request))
        self.post("segmentation_evaluation_evidence", body)
        self.assertIsNotNone(reader(request))
        self.post("segmentation_parent_exposure", {"status": "UNKNOWN", "plan_sha256": request.plan_sha256})
        self.assertIsNone(reader(request))
        self.assertIn("superseded", reader.last_lookup.missing_evidence[0])

    def test_owner_boolean_evidence_and_cross_thread_references_are_not_authority(self):
        request = self.evaluation_request()
        body = self.evidence_messages(request)
        unrelated = self.db.post_agent_message(self.plan["run_id"], thread_id="another-action", sender_role="independent_evaluator",
            recipient_role=self.plan["owner_role"], message_type="segmentation_parent_exposure",
            payload={"status": "VERIFIED", "plan_sha256": request.plan_sha256, "training_evaluation_patient_disjoint": True})
        body["parent_exposure_message_id"] = unrelated
        self.post("segmentation_evaluation_evidence", body)
        reader = host.DatabaseEvidence(host.ReadOnlyDatabase(self.db_path))
        self.assertIsNone(reader(request))

    def test_claim_flags_without_independent_provenance_records_do_not_verify(self):
        request = self.evaluation_request()
        body = self.evidence_messages(request)
        body.update(reference_provenance_message_id=True, parent_exposure_message_id=True,
                    validation_test_patient_disjoint=True, training_evaluation_patient_disjoint=True)
        self.post("segmentation_evaluation_evidence", body)
        reader = host.DatabaseEvidence(host.ReadOnlyDatabase(self.db_path))
        self.assertIsNone(reader(request))
        self.assertEqual(reader.last_lookup.status, "WAITING")

    def test_evidence_report_or_plan_mutation_does_not_match(self):
        request = self.evaluation_request()
        self.evidence_messages(request)
        reader = host.DatabaseEvidence(host.ReadOnlyDatabase(self.db_path))
        request.report["reference_sha256"] = "9" * 64
        self.assertIsNone(reader(request))
        self.assertEqual(reader.last_lookup.missing_evidence, ("evaluation_request_binding_mismatch",))

    def test_full_host_cycle_actual_synthetic_training_and_prediction_then_stored_reviews(self):
        # The parent retains its independently stored old source bundle while
        # this approved job introduces a different architecture/source hash.
        architecture = Path(self.plan["architecture"]["path"])
        architecture.write_text("# synthetic next architecture\n")
        self.f.job["vendor"]["files"][architecture.relative_to(self.f.job["vendor"]["root"]).as_posix()] = training.sha256_file(architecture)
        self.f.write_json("job.json", self.f.job)
        self.plan["job"] = fixtures.binding(self.f.job_path)
        self.plan["architecture"] = fixtures.binding(architecture)
        self.db.save_maintenance_action(self.plan["run_id"], {"action_id": self.plan["action_id"],
            "owner_role": self.plan["owner_role"], "action_class": "segmentation_training", "review_approved": True,
            "segmentation_cycle_plan": self.plan}, status="APPROVED")
        self.bind_sources()
        self.approve_plan()
        budget_before = self.budget.read_bytes()
        controller = host.make_controller(self.f.root, self.plan, database=self.db_path)
        command = self.cli("--execute")
        self.assertEqual(command.returncode, 3, command.stdout + command.stderr)
        self.assertEqual(json.loads(command.stdout)["stage"], "AWAITING_EVALUATION")
        journals = list((self.f.root / "runtime/segmentation_cycles").glob("*/latest.json"))
        self.assertEqual(len(journals), 1)
        result = training.read_json(journals[0])
        self.assertEqual(result["stage"], "AWAITING_EVALUATION", result.get("reason"))
        self.assertEqual(result["report"]["metrics"]["dice"]["candidate"], 1)
        self.assertEqual(result["report"]["metrics"]["dice"]["parent"], 0.5)
        candidate = registry.resolve_candidate(self.f.reg, result["candidate_id"])
        parent = registry.resolve_candidate(self.f.reg, self.plan["parent_id"])
        self.assertNotEqual(parent["artifacts"]["architecture"]["sha256"], candidate["artifacts"]["architecture"]["sha256"])
        # A historical trained parent can resolve the prior immutable receipt
        # without either a current-plan binding or new source/medical messages.
        alternate = self.f.base / "alternate-registry"
        registry.register_baseline(alternate, target=self.f.target, owner="baseline-curator",
            **{k: Path(v["path"]) for k, v in parent["artifacts"].items()})
        archived = registry.register_candidate(alternate, target=self.f.target, parent_id=parent["candidate_id"],
            owner=self.plan["owner"], training_record=candidate["training_record"],
            **{k: Path(v["path"]) for k, v in candidate["artifacts"].items()})
        prior = registry.resolve_candidate(alternate, archived["id"])
        sources = host.ModelSources(alternate, baseline_id=prior["candidate_id"]).resolve(prior)
        self.assertEqual(sources["sha256"], prior["training_record"]["vendor_sources_sha256"])
        self.assertEqual(sources["root"].name, "vendor")
        self.assertFalse((alternate / "source_bundles").exists())
        for role, model in (("parent", parent), ("candidate", candidate)):
            copied = journals[0].parent / (role + "_predictions") / "e/automated_annotation_of_lge/lge_seg/model.py"
            self.assertEqual(training.sha256_file(copied), model["artifacts"]["architecture"]["sha256"])
        self.assertEqual(registry.resolve_active(self.f.reg, self.f.target)["candidate_id"], self.plan["parent_id"])
        request = cycle.EvaluationRequest(self.plan, cycle.plan_sha256(self.plan), result["candidate_id"], result["report_sha256"], result["report"])
        body = self.evidence_messages(request)
        # Known exposure and measured comparison do not manufacture a final
        # test partition. The metrics remain visible while promotion waits.
        split_id = self.post("segmentation_split_evidence", {
            "status": "VERIFIED", "plan_sha256": request.plan_sha256,
            "dataset_sha256": self.plan["dataset_manifest"]["sha256"], "cohort_sha256": self.plan["cohort"]["sha256"],
            "training_evaluation_patient_disjoint": "ESTABLISHED", "validation_test_patient_disjoint": "NOT_ESTABLISHED"})
        self.post("segmentation_evaluation_evidence", dict(body, split_evidence_message_id=split_id))
        public = host.public_result(result, controller, self.plan)
        self.assertEqual(public["comparison_status"], "MEASURED")
        self.assertEqual(public["comparison_metrics"]["dice"]["candidate"], 1)
        self.assertEqual(public["evidence_status"], "WAITING")
        self.assertEqual(public["evidence_missing"], ["validation_test_patient_disjoint_not_established"])
        self.assertEqual(public["untouched_final_test_status"], "NOT_ESTABLISHED_BY_THIS_COMPARISON")
        self.evidence_messages(request)
        with patch.object(training, "run_job", side_effect=AssertionError("retraining")), patch.object(host, "_run_process", side_effect=AssertionError("reprediction")):
            result = cycle.run_cycle(self.f.root, self.plan, controller=controller, execute=True)
            self.assertEqual(result["stage"], "AWAITING_ACTIVATION_APPROVAL", result.get("reason"))
            self.f.post(self.db, "independent_reviewer", "segmentation_activation_review", {
                "decision": "accept_experiment", "plan_sha256": cycle.plan_sha256(self.plan),
                "assessment_id": result["assessment"]["assessment_id"], "candidate_id": result["candidate_id"],
                "policy_id": self.plan["policy_id"]})
            result = cycle.run_cycle(self.f.root, self.plan, controller=controller, execute=True)
        self.assertEqual(result["stage"], "ACTIVATED", result.get("reason"))
        self.assertEqual(self.budget.read_bytes(), budget_before)
        public = host.public_result(result, controller, self.plan)
        self.assertEqual(public["evidence_status"], "EVIDENCE_VERIFIED")
        serialized = json.dumps(public)
        for row in self.f.rows:
            self.assertNotIn(row["patient_id"], serialized)
            self.assertNotIn(row["reference"]["path"], serialized)

    def cli(self, *options):
        path = self.f.write_json("cycle-plan.json", self.plan)
        return subprocess.run([sys.executable, "-B", str(CLI), "--root", str(self.f.root), "--plan", str(path),
            "--database", str(self.db_path), *options], capture_output=True, text=True, timeout=30,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)

    def test_cli_preflight_no_mutations_and_execute_honors_pause(self):
        path = self.f.write_json("cycle-plan.json", self.plan)
        before = sorted(str(p) for p in self.f.base.rglob("*"))
        result = self.cli()
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertEqual(json.loads(result.stdout)["stage"], "PREFLIGHT")
        # mode=ro preserves live WAL review visibility; SQLite can create its
        # own coordination sidecars, but no controller/model/data artifacts.
        sidecars = {str(self.db_path) + suffix for suffix in ("-wal", "-shm")}
        self.assertEqual(sorted(str(p) for p in self.f.base.rglob("*") if str(p) not in sidecars),
                         [p for p in before if p not in sidecars])
        self.operator.write_text('{"paused":true}')
        result = self.cli("--execute")
        self.assertEqual(result.returncode, 3, result.stderr + result.stdout)
        self.assertEqual(json.loads(result.stdout)["stage"], "PAUSED")
        self.assertFalse(Path(self.f.job["output_dir"]).exists())

    def test_cli_validation_errors_do_not_expose_paths(self):
        self.plan["approved"] = True
        result = self.cli()
        self.assertEqual(result.returncode, 2)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["stage"], "ERROR")
        self.assertNotIn(str(self.f.base), result.stdout)

    def test_python39_grammar(self):
        for path in (Path(host.__file__), CLI, Path(__file__)):
            ast.parse(path.read_text(encoding="utf-8"), feature_version=(3, 9))


if __name__ == "__main__":
    unittest.main()
