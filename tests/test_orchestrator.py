from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from app.database import Database
from app.orchestrator import WorkflowOrchestrator


ROOT = Path(__file__).resolve().parents[1]


def load_script(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OrchestratorTests(unittest.TestCase):
    def test_audit_command_is_non_mutating_and_skips_annotation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            dataset.mkdir()
            skill = root / "skill"
            (skill / "scripts").mkdir(parents=True)
            (skill / "scripts/run_full_qc_pipeline.py").write_text("", encoding="utf-8")
            python = root / "python"
            python.write_text("", encoding="utf-8")
            db = Database(root / "workflow.sqlite")
            db.initialize()
            orch = WorkflowOrchestrator(db, root, skill, python)
            run_id = orch.submit(dataset_root=dataset, raw_input=None, execute_mode="audit",
                                 device="auto", workers=4, run_annotation=False)
            command = orch.build_pipeline_command(db.get_run(run_id))
            self.assertIn("--audit-only", command)
            self.assertIn("--skip-annotation", command)


class OrchestratorRegistryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="synthetic orchestrator ")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name).resolve()
        self.root = self.base / "project"
        self.root.mkdir()
        self.dataset = self.base / "dataset"
        self.dataset.mkdir()
        self.skill = self.base / "external skill"
        self.registry = self.base / "selected registry"
        self.registry.mkdir()
        self.settings = self.root / WorkflowOrchestrator.MODEL_CONFIG
        self.trusted = {"model_registry": str(self.registry), "orchestrator_root": str(ROOT)}
        self.db = Database(self.root / "workflow.sqlite")
        self.db.initialize()
        self.orch = WorkflowOrchestrator(self.db, self.root, self.skill, Path(sys.executable))

    def configure(self, value=None):
        self.settings.parent.mkdir(parents=True, exist_ok=True)
        self.settings.write_text(json.dumps(self.trusted if value is None else value), encoding="utf-8")

    def submit(self, **overrides):
        args = dict(dataset_root=self.dataset, raw_input=None, execute_mode="apply",
                    device="cpu", workers=2, run_annotation=True)
        return self.orch.submit(**{**args, **overrides})

    def test_trusted_registry_is_persisted_and_forwarded_after_controller_restart(self):
        self.configure()
        run_id = self.submit()
        run = self.db.get_run(run_id)
        self.assertEqual(run["config"]["segmentation_inference"], self.trusted)
        restarted = WorkflowOrchestrator(self.db, self.root, self.skill, Path(sys.executable))
        command = restarted.build_pipeline_command(run)
        self.assertEqual(command[-4:], ["--model-registry", str(self.registry), "--orchestrator-root", str(ROOT)])

    def test_unconfigured_and_preexisting_runs_keep_exact_legacy_command(self):
        run_id = self.submit()
        run = self.db.get_run(run_id)
        expected = [str(Path(sys.executable).resolve()), str(self.skill / "scripts/run_full_qc_pipeline.py"),
                    "--root", str(self.dataset), "--device", "cpu", "--workers", "2"]
        self.assertEqual(self.orch.build_pipeline_command(run), expected)
        self.assertNotIn("segmentation_inference", run["config"])
        # Runtime opt-in applies to new submissions, not already queued legacy jobs.
        self.configure()
        self.assertEqual(self.orch.build_pipeline_command(run), expected)
        run["config"].update(model_registry="owner-controlled", orchestrator_root="owner-controlled",
                             extra_args=["--threshold", "0.9"])
        self.assertEqual(self.orch.build_pipeline_command(run), expected)

    def test_audit_and_skip_annotation_do_not_consume_model_configuration(self):
        self.configure({"malformed": "unused by audit"})
        for overrides in ({"execute_mode": "audit"}, {"run_annotation": False}):
            with self.subTest(overrides=overrides):
                run = self.db.get_run(self.submit(**overrides))
                self.assertNotIn("segmentation_inference", run["config"])
                command = self.orch.build_pipeline_command(run)
                self.assertNotIn("--model-registry", command)
                self.assertIn("--audit-only" if overrides.get("execute_mode") == "audit" else "--skip-annotation", command)

    def test_invalid_or_extra_trusted_parameters_fail_before_registering_a_run(self):
        for value in ([], {}, {"model_registry": str(self.registry)},
                      {**self.trusted, "command": "owner-selected-command"},
                      {**self.trusted, "model_registry": "relative/registry"},
                      {**self.trusted, "model_registry": str(self.base / "missing")},
                      {**self.trusted, "orchestrator_root": str(self.registry)}):
            with self.subTest(value=value):
                self.configure(value)
                with self.assertRaises(ValueError):
                    self.submit()
                self.assertEqual(self.db.list_runs(), [])
        self.settings.write_text("{invalid JSON", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.submit()
        self.assertEqual(self.db.list_runs(), [])

    def test_payload_cannot_override_trusted_registry_or_add_parameters(self):
        self.configure()
        run_id = self.submit()
        for replacement in ({**self.trusted, "model_registry": str(self.base)},
                            {**self.trusted, "extra_args": ["--threshold", "0.9"]}, None):
            with self.subTest(replacement=replacement):
                run = self.db.get_run(run_id)
                run["config"]["segmentation_inference"] = replacement
                with self.assertRaisesRegex(ValueError, "configuration changed"):
                    self.orch.build_pipeline_command(run)
        self.settings.unlink()
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            self.orch.build_pipeline_command(self.db.get_run(run_id))

    def test_changed_trusted_config_fails_without_child_or_blocking_queue(self):
        self.configure()
        run_id = self.submit()
        other_registry = self.base / "other registry"
        other_registry.mkdir()
        self.configure({**self.trusted, "model_registry": str(other_registry)})
        with mock.patch.object(self.orch.runner, "run") as runner:
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                self.orch.execute(run_id)
            runner.assert_not_called()
        self.assertEqual(self.db.get_run(run_id)["status"], "ERROR")
        self.assertEqual(self.db.get_run(run_id)["current_step"], "pipeline_configuration")
        self.assertIsNone(self.db.next_queued_run_id())
        self.assertFalse((self.root / "artifacts").exists())

    def test_orchestrator_pipeline_annotation_reaches_selected_model_factory(self):
        from scripts import segment_from_registry
        pipeline = load_script("orchestrated_registry_pipeline", "skills/cmr-lge-label-qc-workflow/scripts/run_full_qc_pipeline.py")
        annotation = load_script("orchestrated_registry_annotation", "skills/cmr-lge-label-qc-workflow/tools/auto_annotate_dataset.py")
        self.configure()
        run = self.db.get_run(self.submit())
        command = self.orch.build_pipeline_command(run)
        config = self.registry / "config"
        config.write_text('{"training": {}}', encoding="utf-8")

        def predict_command(registry, vendor, target, images, output, *, device):
            self.assertEqual(registry, self.registry)
            self.assertEqual(vendor, self.skill / "vendor")
            self.assertEqual(images, self.dataset / "images" / ("cmr" if target == "cmr_myo" else "lge"))
            self.assertEqual(device, "cpu")
            self.assertFalse(output.exists())
            return [sys.executable, str(vendor / "synthetic.py"), "--config", str(config), "predict",
                    "--weights", str(self.registry / "selected-weights"), "--output-dir", str(output)], {
                "model_id": "activated-version-2", "target": target, "weights_sha256": "a" * 64,
                "dataset_sha256": "b" * 64, "status": "PLANNED", "production_labels_written": False}

        def step(run_dir, name, args, env, **kwargs):
            if name == "01_annotation":
                annotation.main(args[1:])
            else:
                self.assertNotIn("--model-registry", args)
                self.assertNotIn("--orchestrator-root", args)

        with mock.patch.object(pipeline, "HERE", self.skill), \
                mock.patch.object(pipeline, "TOOLS", self.skill / "tools"), \
                mock.patch.object(pipeline, "run_step", side_effect=step), \
                mock.patch.object(annotation, "HERE", self.skill), \
                mock.patch.object(annotation, "has_nifti", return_value=True), \
                mock.patch.object(annotation, "run") as prediction, \
                mock.patch.object(annotation, "copy_masks", return_value=0), \
                mock.patch.object(segment_from_registry, "predict_command", side_effect=predict_command) as factory, \
                mock.patch.object(annotation.subprocess, "run", side_effect=AssertionError("real child forbidden")), \
                mock.patch.object(sys, "argv", command[1:]), redirect_stdout(io.StringIO()):
            pipeline.main()
        self.assertEqual([call.args[2] for call in factory.call_args_list], ["cmr_myo", "lge_myo", "lge_mi"])
        self.assertEqual(prediction.call_count, 3)
        summaries = list(self.dataset.glob("label_iteration/annotation_runs/*/annotation_summary.json"))
        self.assertEqual(len(summaries), 1)
        receipts = json.loads(summaries[0].read_text(encoding="utf-8"))["model_receipts"]
        self.assertTrue(all(receipt["model_id"] == "activated-version-2" for receipt in receipts.values()))
        self.assertFalse((self.dataset / "labels").exists())


if __name__ == "__main__":
    unittest.main()
