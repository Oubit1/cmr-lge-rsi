"""Initial annotation wiring, using synthetic files and no model execution."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from app import segmentation_registry as registry
from scripts import segment_from_registry


PREDICT_COMMAND = segment_from_registry.predict_command
ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "skills/cmr-lge-label-qc-workflow/tools/auto_annotate_dataset.py"
SPEC = importlib.util.spec_from_file_location("registry_initial_annotation", SCRIPT)
annotation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(annotation)
TARGETS = ("cmr_myo", "lge_myo", "lge_mi")
LABELS = {"cmr_myo": "cmr/cmr_Myo_mask", "lge_myo": "lge_checked/lge_Myo_labels",
          "lge_mi": "lge_checked/lge_MI_labels"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class AnnotationRegistryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="synthetic annotation ")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name).resolve()
        self.dataset = self.base / "dataset"
        self.registry = self.base / "registry"
        self.skill = self.base / "external skill"
        self.receipts = {}
        for modality in ("cmr", "lge"):
            self.write(self.dataset / "images" / modality / "synthetic/1.nii.gz")
        for target in TARGETS:
            self.write(self.registry / target / "config", b'{"training": {"epochs": 1}}')
            weights = self.registry / target / "weights"
            self.write(weights, f"synthetic selected weights {target}".encode())
            self.write(self.skill / "vendor" / segment_from_registry.ENTRYPOINTS[target], b"# synthetic entry")
            self.receipts[target] = {
                "target": target, "model_id": hashlib.sha256(target.encode()).hexdigest(),
                "dataset_sha256": hashlib.sha256(b"synthetic training manifest").hexdigest(),
                "weights_sha256": sha(weights), "status": "PLANNED", "production_labels_written": False,
            }
        self.patch(annotation, "HERE", self.skill)
        self.patch(annotation, "VENDOR", self.skill / "vendor")
        self.patch(annotation, "MODELS", self.skill / "legacy models")
        self.factory = self.patch(segment_from_registry, "predict_command", side_effect=self.plan)
        self.runner = self.patch(annotation, "run", side_effect=self.predict)
        self.copy = self.patch(annotation, "copy_masks", wraps=annotation.copy_masks)
        self.child = self.patch(annotation.subprocess, "run", side_effect=AssertionError("unexpected real execution"))

    def patch(self, obj, name, *args, **kwargs):
        patcher = mock.patch.object(obj, name, *args, **kwargs)
        result = patcher.start()
        self.addCleanup(patcher.stop)
        return result

    @staticmethod
    def write(path, data=b"synthetic bytes, not medical data"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    @staticmethod
    def snapshot(root):
        return {str(p.relative_to(root)): None if p.is_dir() else p.read_bytes() for p in root.rglob("*")}

    def plan(self, registry_path, vendor, target, images, output, *, device):
        self.assertEqual(registry_path, self.registry)
        self.assertEqual(vendor, self.skill / "vendor")
        self.assertEqual(images, self.dataset / "images" / ("cmr" if target == "cmr_myo" else "lge"))
        self.assertEqual(output.name, target)
        self.assertFalse(output.exists())
        self.assertFalse(output.parent.exists())
        return [sys.executable, str(vendor / segment_from_registry.ENTRYPOINTS[target]),
                "--config", str(self.registry / target / "config"), "predict",
                "--weights", str(self.registry / target / "weights"), "--input", str(images),
                "--output-dir", str(output), "--device", device], dict(self.receipts[target])

    def predict(self, command, cwd, dry_run):
        self.assertEqual(cwd, Path(command[1]).parent)
        if dry_run:
            return
        output = Path(command[command.index("--output-dir") + 1])
        subdir = "predictions" if output.name == "cmr_myo" else "masks"
        self.write(output / subdir / "synthetic/1.nii.gz", f"synthetic prediction {output.name}".encode())

    def invoke(self, *flags, use_registry=True):
        args = ["--root", str(self.dataset)]
        if use_registry:
            args += ["--model-registry", str(self.registry), "--orchestrator-root", str(ROOT)]
        stream = io.StringIO()
        with redirect_stdout(stream):
            annotation.main([*args, *flags])
        return json.loads(stream.getvalue())

    def assert_no_outputs(self):
        self.assertFalse((self.dataset / "label_iteration").exists())
        self.assertFalse((self.dataset / "labels").exists())
        self.runner.assert_not_called()
        self.copy.assert_not_called()
        self.child.assert_not_called()

    def test_selected_commands_receipts_and_original_initial_copy_destinations(self):
        original_copy = annotation.copy_masks._mock_wraps

        def copy_after_predictions(*args):
            self.assertEqual(self.runner.call_count, 3)
            return original_copy(*args)

        self.copy.side_effect = copy_after_predictions
        result = self.invoke("--device", "cpu", "--threshold", "0.5", "--classification-gate-threshold", "0.1")
        self.assertEqual([call.args[2] for call in self.factory.call_args_list], list(TARGETS))
        self.assertEqual(result["copied_masks"], dict.fromkeys(TARGETS, 1))
        self.assertEqual(result["model_registry"], str(self.registry))
        self.assertEqual(result["orchestrator_root"], str(ROOT))
        for target, call in zip(TARGETS, self.runner.call_args_list):
            command, cwd, dry_run = call.args
            self.assertFalse(dry_run)
            self.assertEqual(call.args[0][command.index("--device") + 1], "cpu")
            self.assertEqual(command[command.index("--config") + 1], str(self.registry / target / "config"))
            self.assertLess(command.index("--config"), command.index("predict"))
            self.assertEqual(result["weights"][target], str(self.registry / target / "weights"))
            for flag in ("--threshold", "--classification-gate-threshold", "--visualize"):
                self.assertNotIn(flag, command)
            expected_receipt = dict(self.receipts[target], config_sha256=sha(self.registry / target / "config"))
            self.assertEqual(result["model_receipts"][target], expected_receipt)
            label = self.dataset / "labels" / LABELS[target] / "synthetic/1.nii.gz"
            self.assertEqual(label.read_bytes(), f"synthetic prediction {target}".encode())
        summary_path = Path(result["run_root"]) / "annotation_summary.json"
        self.assertEqual(json.loads(summary_path.read_text(encoding="utf-8")), result)
        self.assertTrue(all(p.read_bytes() == b"synthetic bytes, not medical data"
                            for p in (self.dataset / "images").rglob("*.nii.gz")))
        self.child.assert_not_called()

    def test_dry_run_plans_selected_models_without_any_files_or_copy(self):
        before = self.snapshot(self.base)
        result = self.invoke("--dry-run")
        self.assertTrue(result["dry_run"])
        self.assertEqual(result["commands"], 3)
        self.assertEqual(set(result["model_receipts"]), set(TARGETS))
        self.assertEqual(self.snapshot(self.base), before)
        self.assertTrue(all(call.args[2] is True for call in self.runner.call_args_list))
        self.copy.assert_not_called()
        self.child.assert_not_called()

    def test_missing_or_corrupt_active_model_fails_before_any_execution(self):
        for error in (ValueError("no_selected_model"), registry.RegistryError("artifact hash/size mismatch")):
            with self.subTest(error=str(error)):
                def fail_second(*args, **kwargs):
                    if args[2] == "lge_myo":
                        raise error
                    return self.plan(*args, **kwargs)

                self.factory.side_effect = fail_second
                with self.assertRaisesRegex(ValueError, str(error)):
                    self.invoke()
                self.assert_no_outputs()

    def test_prediction_failure_does_not_copy_even_completed_targets(self):
        sentinel = self.dataset / "labels/cmr/cmr_Myo_mask/synthetic/1.nii.gz"
        self.write(sentinel, b"existing initial label")
        before = self.snapshot(self.dataset / "labels")

        def fail_second(command, cwd, dry_run):
            self.predict(command, cwd, dry_run)
            if Path(command[command.index("--output-dir") + 1]).name == "lge_myo":
                raise subprocess.CalledProcessError(7, command)

        self.runner.side_effect = fail_second
        with self.assertRaises(subprocess.CalledProcessError):
            self.invoke()
        self.assertEqual(self.runner.call_count, 2)
        self.copy.assert_not_called()
        self.child.assert_not_called()
        self.assertEqual(self.snapshot(self.dataset / "labels"), before)
        self.assertEqual(list(self.dataset.rglob("annotation_summary.json")), [])

    def test_skipped_targets_do_not_resolve_or_copy(self):
        result = self.invoke("--skip-cmr", "--skip-lge-mi")
        self.assertEqual([call.args[2] for call in self.factory.call_args_list], ["lge_myo"])
        self.assertEqual(set(result["model_receipts"]), {"lge_myo"})
        self.assertEqual(result["copied_masks"], {"lge_myo": 1})
        self.assertEqual(self.copy.call_count, 1)

    def test_registry_requires_explicit_existing_orchestrator_and_script(self):
        invalid_roots = (None, self.base / "absent", self.registry,
                         self.registry / "cmr_myo/config")
        for orchestrator in invalid_roots:
            with self.subTest(root=orchestrator), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                flags = ["--model-registry", str(self.registry)]
                if orchestrator is not None:
                    flags += ["--orchestrator-root", str(orchestrator)]
                self.invoke(*flags, use_registry=False)
            self.assert_no_outputs()
        self.factory.assert_not_called()

    def test_missing_registry_rejected_before_factory_or_output(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.invoke("--model-registry", str(self.base / "missing registry"))
        self.factory.assert_not_called()
        self.assert_no_outputs()

    def test_registry_rejects_custom_or_nonfinite_thresholds(self):
        for flag, value in (("--threshold", "0.6"), ("--classification-gate-threshold", "0.2"),
                            ("--threshold", "nan"), ("--classification-gate-threshold", "inf")):
            with self.subTest(flag=flag, value=value), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.invoke(flag, value)
            self.assert_no_outputs()
        self.factory.assert_not_called()

    def test_registry_runs_have_distinct_new_private_directories(self):
        first = self.invoke()
        before = self.snapshot(Path(first["run_root"]))
        second = self.invoke()
        self.assertNotEqual(first["run_root"], second["run_root"])
        self.assertEqual(self.snapshot(Path(first["run_root"])), before)
        for result in (first, second):
            self.assertEqual(Path(result["run_root"]).parent, self.dataset / "label_iteration/annotation_runs")

    def test_existing_registry_run_is_never_reused(self):
        self.patch(annotation, "datetime").now.return_value.strftime.return_value = "fixed_stamp"
        self.patch(annotation, "uuid4").return_value.hex = "fixed_nonce"
        existing = self.dataset / "label_iteration/annotation_runs/fixed_stamp_fixed_nonce"
        self.write(existing / "cmr_myo/predictions/stale.nii.gz", b"stale prediction")
        before = self.snapshot(self.dataset)
        with self.assertRaisesRegex(ValueError, "new private output"):
            self.invoke()
        self.assertEqual(self.snapshot(self.dataset), before)
        self.factory.assert_not_called()
        self.runner.assert_not_called()
        self.copy.assert_not_called()

    def test_legacy_mode_keeps_weights_threshold_overrides_and_copy_layout(self):
        result = self.invoke("--threshold", "0.63", "--classification-gate-threshold", "0.24", use_registry=False)
        self.factory.assert_not_called()
        self.assertEqual(result["copied_masks"], dict.fromkeys(TARGETS, 1))
        self.assertNotIn("model_receipts", result)
        for target, call in zip(TARGETS, self.runner.call_args_list):
            command = call.args[0]
            self.assertEqual(command[command.index("--threshold") + 1], "0.63")
            self.assertIn("--visualize", command)
            self.assertNotIn("--config", command)
            self.assertEqual(command[command.index("--weights") + 1], result["weights"][target])
        mi_command = self.runner.call_args_list[-1].args[0]
        self.assertEqual(mi_command[mi_command.index("--classification-gate-threshold") + 1], "0.24")
        self.assertEqual(mi_command[mi_command.index("--myo-weights") + 1], result["weights"]["lge_myo"])

    def test_legacy_dry_run_does_not_load_orchestrator_or_create_output(self):
        before = self.snapshot(self.base)
        loader = self.patch(annotation, "registry_predictor", side_effect=AssertionError("legacy imported registry"))
        result = self.invoke("--dry-run", use_registry=False)
        self.assertTrue(result["dry_run"])
        self.assertEqual(self.snapshot(self.base), before)
        loader.assert_not_called()
        self.copy.assert_not_called()

    def register_synthetic_models(self):
        def writable():
            for path in self.registry.rglob("*"):
                if path.is_file():
                    path.chmod(0o600)

        self.addCleanup(writable)
        selected = {}
        architectures = {"cmr_myo": "cine_seg/model.py", "lge_myo": "lge_seg/model.py",
                         "lge_mi": "lge_mi_prior_seg/model.py"}
        for target in TARGETS:
            architecture = (self.skill / "vendor" / segment_from_registry.ENTRYPOINTS[target]).parent / architectures[target]
            self.write(architecture, b"# synthetic architecture, never loaded")
            manifest = self.registry / target / "dataset_manifest"
            self.write(manifest, b"synthetic training provenance")
            if target == "lge_mi":
                pinned_myo = self.base / "pinned myo/weights"
                self.write(pinned_myo, b"synthetic MI-specific Myo dependency")
                self.write(self.registry / target / "config", json.dumps({
                    "training": {"epochs": 1},
                    "myocardium_model": {"weights": str(pinned_myo), "sha256": sha(pinned_myo)},
                }).encode())
            candidate = registry.register_baseline(
                self.registry, target=target, owner="synthetic custodian",
                weights=self.registry / target / "weights", config=self.registry / target / "config",
                architecture=architecture, dataset_manifest=manifest)
            registry.select_baseline(self.registry, candidate["id"])
            selected[target] = registry.resolve_active(self.registry, target)
        self.factory.side_effect = PREDICT_COMMAND
        return selected

    def test_actual_factory_binds_selected_versions_configs_and_mi_dependency(self):
        selected = self.register_synthetic_models()
        result = self.invoke()
        for target, call in zip(TARGETS, self.runner.call_args_list):
            model = selected[target]
            command = call.args[0]
            self.assertEqual(command[command.index("--config") + 1], model["artifacts"]["config"]["path"])
            self.assertEqual(command[command.index("--weights") + 1], model["artifacts"]["weights"]["path"])
            receipt = result["model_receipts"][target]
            self.assertEqual(receipt["model_id"], model["candidate_id"])
            self.assertEqual(receipt["config_sha256"], model["artifacts"]["config"]["sha256"])
            self.assertEqual(receipt["weights_sha256"], model["artifacts"]["weights"]["sha256"])
            self.assertEqual(receipt["dataset_sha256"], model["dataset_sha256"])
        command = self.runner.call_args_list[-1].args[0]
        self.assertEqual(command[command.index("--myo-weights") + 1], str(self.base / "pinned myo/weights"))
        self.assertNotEqual(command[command.index("--myo-weights") + 1], result["weights"]["lge_myo"])
        self.child.assert_not_called()

    def test_actual_corrupt_active_artifact_never_falls_back_or_copies(self):
        selected = self.register_synthetic_models()
        weights = Path(selected["lge_mi"]["artifacts"]["weights"]["path"])
        weights.chmod(0o600)
        weights.write_bytes(b"synthetic corruption")
        with self.assertRaisesRegex(registry.RegistryError, "artifact hash/size mismatch"):
            self.invoke()
        self.assert_no_outputs()

    def test_actual_missing_active_model_never_falls_back_or_copies(self):
        self.factory.side_effect = PREDICT_COMMAND
        with self.assertRaisesRegex(ValueError, "no_selected_model"):
            self.invoke()
        self.assert_no_outputs()


class ExternalSkillImportTests(unittest.TestCase):
    def test_external_skill_uses_supplied_root_without_pythonpath_or_import_output(self):
        with tempfile.TemporaryDirectory(prefix="external annotation ") as directory:
            base = Path(directory).resolve()
            script = base / "external skill/tools/auto_annotate_dataset.py"
            script.parent.mkdir(parents=True)
            shutil.copyfile(SCRIPT, script)
            orchestrator = base / "supplied orchestrator"
            factory = orchestrator / "scripts/segment_from_registry.py"
            AnnotationRegistryTests.write(factory, b'''
def predict_command(registry, vendor, target, images, output, *, device):
    from scripts.synthetic_dependency import selected_version
    assert not output.exists()
    return ["synthetic-python", str(vendor / "entry.py"), "--config", str(registry / "config"),
            "predict", "--weights", str(registry / "weights"), "--output-dir", str(output)], {
        "target": target, "model_id": selected_version, "weights_sha256": "synthetic"}
''')
            AnnotationRegistryTests.write(orchestrator / "scripts/synthetic_dependency.py",
                                          b'selected_version = "supplied-root-selected-version"\n')
            dataset = base / "dataset"
            AnnotationRegistryTests.write(dataset / "images/lge/synthetic.nii.gz")
            AnnotationRegistryTests.write(base / "registry/config", b'{"training": {}}')
            before = AnnotationRegistryTests.snapshot(base)
            env = dict(os.environ)
            env.pop("PYTHONPATH", None)
            # Deliberately allow bytecode normally: lazy imports must stay read-only.
            env.pop("PYTHONDONTWRITEBYTECODE", None)
            result = subprocess.run([sys.executable, str(script), "--root", str(dataset),
                                     "--model-registry", str(base / "registry"),
                                     "--orchestrator-root", str(orchestrator), "--skip-cmr", "--skip-lge-mi",
                                     "--dry-run"], cwd=str(base), env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("supplied-root-selected-version", result.stdout)
            self.assertEqual(AnnotationRegistryTests.snapshot(base), before)

    def test_cached_factory_from_another_root_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="wrong annotation root ") as directory:
            other = Path(directory).resolve()
            AnnotationRegistryTests.write(other / "scripts/segment_from_registry.py", b"# other checkout")
            before = sys.path[:]
            with self.assertRaisesRegex(ValueError, "different --orchestrator-root"):
                with annotation.registry_predictor(other):
                    self.fail("must not use a different checkout's factory")
            self.assertEqual(sys.path, before)


if __name__ == "__main__":
    unittest.main()
