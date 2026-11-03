from __future__ import annotations

import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock


SCRIPT = (Path(__file__).resolve().parents[1] / "skills/cmr-lge-label-qc-workflow"
          / "scripts/run_full_qc_pipeline.py")
SPEC = importlib.util.spec_from_file_location("full_qc_pipeline_safety", SCRIPT)
pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pipeline)


class SegmentationPipelineSafetyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="synthetic qc ")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name).resolve()
        patcher = mock.patch.object(pipeline.subprocess, "run", side_effect=self.fake_child)
        self.child = patcher.start()
        self.addCleanup(patcher.stop)

    def fake_child(self, command, *, stdout, stderr, env):
        self.assertEqual(command[0], sys.executable)
        self.assertEqual(stderr, subprocess.STDOUT)
        self.assertEqual(env["PYTHONUNBUFFERED"], "1")
        # Simulate a child's direct writes to stdout, including header ordering.
        os.write(stdout.fileno(), f"synthetic output: {Path(command[1]).name}\n".encode())
        return subprocess.CompletedProcess(command, 0)

    def invoke(self, root, *flags):
        with mock.patch.object(sys, "argv", [str(SCRIPT), "--root", str(root), *flags]):
            with redirect_stdout(io.StringIO()):
                pipeline.main()

    def write(self, path, content=b"synthetic fixture"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    def snapshot(self, root):
        return {str(path.relative_to(root)): None if path.is_dir() else path.read_bytes()
                for path in root.rglob("*")}

    def commands(self):
        return [call.args[0] for call in self.child.call_args_list]

    def run_dir(self, root):
        runs = list((root / "label_iteration/full_pipeline_runs").iterdir())
        self.assertEqual(len(runs), 1)
        return runs[0]

    def summary(self, root):
        return json.loads((self.run_dir(root) / "summary.json").read_text(encoding="utf-8"))

    def test_audit_only_skips_annotation_and_all_label_writes(self):
        modes = {
            "run_lge_qc_workflow.py": "audit",
            "cmr_qc.py": "audit",
            "casebook_qc.py": "dry-run",
            "repair_cmr_temporal_missing_holes.py": "dry-run",
            "repair_cmr_temporal_open_ring.py": "dry-run",
            "suppress_supported_cmr_small_area.py": "dry-run",
            "myo_stack_trajectory_qc.py": "dry-run",
            "finalize_paired_dataset.py": "audit",
        }
        for no_render in (False, True):
            for skip_annotation in (False, True):
                with self.subTest(no_render=no_render, skip_annotation=skip_annotation):
                    self.child.reset_mock()
                    root = self.base / f"audit_{no_render}_{skip_annotation}"
                    for index, label_dir in enumerate((
                        "cmr/Myo", "lge_checked/lge_Myo_labels", "lge_checked/lge_MI_labels",
                    )):
                        self.write(root / "labels" / label_dir / "synthetic/1.nii.gz",
                                   f"synthetic label {index}".encode())
                    # Exercise optional render paths without loading any medical images.
                    self.write(root / "label_iteration/myo_trajectory_runs/synthetic/trajectory_candidates.csv")
                    for modality in ("cmr", "lge"):
                        for group in ("boundary", "middle"):
                            self.write(root / "labels/check" / f"{modality}_{group}_issue_slices/issue_manifest.csv")
                    self.write(root / "labels/check/cmr_issue_slices/synthetic.png")
                    before = self.snapshot(root / "labels")
                    flags = ["--audit-only"]
                    if no_render:
                        flags.append("--no-render")
                    if skip_annotation:
                        flags.append("--skip-annotation")
                    self.invoke(root, *flags)

                    names = [Path(command[1]).name for command in self.commands()]
                    self.assertNotIn("auto_annotate_dataset.py", names)
                    self.assertNotIn("prepare_dataset.py", names)
                    self.assertTrue(set(modes).issubset(names))
                    for command in self.commands():
                        name = Path(command[1]).name
                        if name in modes:
                            self.assertEqual(command[command.index("--mode") + 1], modes[name])
                        else:
                            self.assertIn(name, {"isolate_boundary_issue_slices.py",
                                                "render_check_panels.py", "render_myo_trajectory_candidates.py"})
                        self.assertNotIn("--remove-source-cmr", command)

                    self.assertEqual(names.count("render_check_panels.py"), 0 if no_render else 4)
                    self.assertEqual(names.count("render_myo_trajectory_candidates.py"), 0 if no_render else 1)
                    self.assertEqual(self.snapshot(root / "labels"), before)
                    # A copied label outside labels/ must also fail the regression.
                    self.assertEqual(len(list(root.rglob("*.nii.gz"))), 3)
                    summary = self.summary(root)
                    self.assertTrue(summary["audit_only"])
                    self.assertNotIn("annotation", summary["steps"])
                    self.assertNotIn("prepare_dataset", summary["steps"])

    def test_selected_model_registry_is_forwarded_to_annotation_only(self):
        root = self.base / "fresh"
        registry = self.base / "models"
        controller = self.base / "controller"
        self.invoke(root, "--model-registry", str(registry), "--orchestrator-root", str(controller),
                    "--max-rounds", "1", "--no-render")
        calls = [c for c in self.commands() if "--model-registry" in c]
        self.assertEqual(len(calls), 1)
        self.assertEqual(Path(calls[0][1]).name, "auto_annotate_dataset.py")
        self.assertEqual(calls[0][calls[0].index("--model-registry") + 1], str(registry.resolve()))

    def test_audit_only_rejects_intake_before_any_files_or_children(self):
        source = self.base / "raw input"
        self.write(source / "lge/synthetic_label.nii.gz", b"synthetic source label")
        before = self.snapshot(source)
        root = self.base / "unused output"
        error = io.StringIO()
        with redirect_stderr(error), self.assertRaises(SystemExit) as raised:
            self.invoke(root, "--audit-only", "--input", str(source))
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--input cannot be combined with --audit-only", error.getvalue())
        self.child.assert_not_called()
        self.assertFalse(root.exists())
        self.assertEqual(self.snapshot(source), before)

    def test_audit_only_rejects_finalization_before_any_files_or_children(self):
        root = self.base / "unused output"
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            self.invoke(root, "--audit-only", "--finalize-dataset")
        self.assertEqual(raised.exception.code, 2)
        self.child.assert_not_called()
        self.assertFalse(root.exists())

    def test_intake_precedes_output_logs_and_preserves_them_afterward(self):
        for existing_empty_root in (False, True):
            with self.subTest(existing_empty_root=existing_empty_root):
                self.child.reset_mock()
                source = self.base / f"raw input {existing_empty_root}"
                self.write(source / "lge/synthetic.nii.gz")
                source_before = self.snapshot(source)
                root = self.base / f"processed output {existing_empty_root}"
                if existing_empty_root:
                    root.mkdir()

                def child(command, **kwargs):
                    if Path(command[1]).name == "prepare_dataset.py":
                        self.assertEqual(command[2:], ["--input", str(source), "--output", str(root)])
                        self.assertFalse(root.exists() and any(root.iterdir()))
                        self.assertFalse((root / "label_iteration").exists())
                        self.write(root / "label_iteration/intake_manifest.json", b'{"synthetic": true}')
                    else:
                        self.assertTrue((self.run_dir(root) / "00_prepare_dataset.log").is_file())
                        self.assertTrue((self.run_dir(root) / "command.json").is_file())
                    return self.fake_child(command, **kwargs)

                self.child.side_effect = child
                self.invoke(root, "--input", str(source), "--no-render", "--max-rounds", "1")
                names = [Path(command[1]).name for command in self.commands()]
                self.assertEqual(names[:2], ["prepare_dataset.py", "auto_annotate_dataset.py"])
                run_dir = self.run_dir(root)
                log = (run_dir / "00_prepare_dataset.log").read_text(encoding="utf-8")
                self.assertTrue(log.startswith("$ " + sys.executable))
                self.assertIn(str(source), log)
                self.assertIn("synthetic output: prepare_dataset.py", log)
                self.assertEqual(len(list(run_dir.glob("*.log"))), len(names))
                command_record = json.loads((run_dir / "command.json").read_text(encoding="utf-8"))
                self.assertEqual(command_record["input"], str(source))
                self.assertEqual(self.summary(root)["steps"][:2], ["prepare_dataset", "annotation"])
                self.assertEqual(self.snapshot(source), source_before)

    def test_failed_intake_keeps_diagnostics_and_stops_pipeline(self):
        source = self.base / "raw input"
        self.write(source / "lge/synthetic.nii.gz")
        root = self.base / "partial output"

        def fail_intake(command, **kwargs):
            self.assertEqual(Path(command[1]).name, "prepare_dataset.py")
            self.assertFalse(root.exists())
            self.write(root / "label_iteration/intake_manifest.json", b'{"issues": ["synthetic failure"]}')
            os.write(kwargs["stdout"].fileno(), b"synthetic intake failed\n\xff\xfe\n")
            return subprocess.CompletedProcess(command, 7)

        self.child.side_effect = fail_intake
        with self.assertRaises(RuntimeError) as raised:
            self.invoke(root, "--input", str(source))
        self.child.assert_called_once()
        run_dir = self.run_dir(root)
        log = run_dir / "00_prepare_dataset.log"
        self.assertIn(str(log), str(raised.exception))
        self.assertTrue(log.read_bytes().endswith(b"synthetic intake failed\n\xff\xfe\n"))
        self.assertTrue((run_dir / "command.json").is_file())
        self.assertTrue((root / "label_iteration/intake_manifest.json").is_file())
        self.assertFalse((run_dir / "summary.json").exists())

    def test_intake_launch_error_keeps_command_and_log(self):
        source = self.base / "raw input"
        source.mkdir()
        root = self.base / "failed launch"
        self.child.side_effect = OSError("synthetic launch failure")
        with self.assertRaisesRegex(OSError, "synthetic launch failure"):
            self.invoke(root, "--input", str(source))
        self.child.assert_called_once()
        run_dir = self.run_dir(root)
        self.assertTrue((run_dir / "command.json").is_file())
        self.assertIn("prepare_dataset.py", (run_dir / "00_prepare_dataset.log").read_text(encoding="utf-8"))
        self.assertFalse((run_dir / "summary.json").exists())

    def test_intake_refuses_nonempty_destinations_without_modifying_them(self):
        source = self.base / "raw input"
        self.write(source / "lge/synthetic.nii.gz", b"synthetic source")
        destinations = [source]
        for entry in ("labels/synthetic.nii.gz", "label_iteration/old.log", "empty_directory"):
            root = self.base / f"existing output {len(destinations)}"
            if entry == "empty_directory":
                (root / entry).mkdir(parents=True)
            else:
                self.write(root / entry, b"must remain untouched")
            destinations.append(root)
        for root in destinations:
            with self.subTest(root=root.name):
                before = self.snapshot(root)
                error = io.StringIO()
                with redirect_stderr(error), self.assertRaises(SystemExit) as raised:
                    self.invoke(root, "--input", str(source))
                self.assertEqual(raised.exception.code, 2)
                self.assertIn("refusing to overwrite non-empty output", error.getvalue())
                self.child.assert_not_called()
                self.assertEqual(self.snapshot(root), before)

    def test_apply_mode_retains_annotation_and_explicit_skip_behavior(self):
        for skip_annotation in (False, True):
            with self.subTest(skip_annotation=skip_annotation):
                self.child.reset_mock()
                root = self.base / f"apply_{skip_annotation}"
                flags = ["--no-render", "--max-rounds", "1"]
                if skip_annotation:
                    flags.append("--skip-annotation")
                self.invoke(root, *flags)
                names = [Path(command[1]).name for command in self.commands()]
                self.assertEqual(names.count("auto_annotate_dataset.py"), 0 if skip_annotation else 1)
                self.assertEqual("annotation" in self.summary(root)["steps"], not skip_annotation)
                self.assertNotIn("prepare_dataset.py", names)
                triage = [command for command in self.commands()
                          if Path(command[1]).name == "isolate_boundary_issue_slices.py"]
                self.assertTrue(triage)
                self.assertTrue(all("--remove-source-cmr" in command for command in triage))


if __name__ == "__main__":
    unittest.main()
