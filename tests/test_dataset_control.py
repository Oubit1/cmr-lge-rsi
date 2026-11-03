from __future__ import annotations

import ast
import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import nibabel as nib
import numpy as np

from app import dataset_control as control
from app import paired_dataset as paired
from app.control_checkpoint import save_journal
from scripts.dataset_release import run


class DatasetControlTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="synthetic-control-")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name).resolve()
        self.root = self.base / "project"
        self.dataset = self.base / "dataset"
        self.root.mkdir()
        self.dataset.mkdir()
        self.folder = self.root / "runtime/dataset_control"
        self.addCleanup(self.writable)
        network = patch("socket.socket.connect", side_effect=AssertionError("network forbidden"))
        self.network = network.start()
        self.addCleanup(network.stop)
        self.process = Mock(pid=43210)
        self.process.wait.return_value = 0
        self.process.poll.return_value = None
        self.popen = Mock(return_value=self.process)
        self.live = {43210: "synthetic-birth"}

    def writable(self):
        for directory, folders, files in os.walk(self.base):
            Path(directory).chmod(0o700)
            for name in folders + files:
                path = Path(directory) / name
                if not path.is_symlink():
                    path.chmod(0o700 if path.is_dir() else 0o600)

    def launch(self, action="audit", **kwargs):
        with patch.object(control, "_process_start", side_effect=self.live.get), patch.object(control, "_reap"):
            return control.launch(self.root, self.dataset, action, popen=self.popen, **kwargs)

    def status(self):
        with patch.object(control, "_process_start", side_effect=self.live.get):
            return control.status(self.root)

    def job(self):
        return control._current(self.folder)

    def save_job(self, job):
        save_journal(control._job_path(self.folder, job["job_id"]), job)

    def bundle(self, key="SYNTHETIC_PRIVATE/level.nii.gz"):
        layout = {"cmr_image": "images/cmr", "lge_image": "images/lge",
                  "cmr_myo": "labels/cmr_checked/cmr_Myo_mask",
                  "lge_myo": "labels/lge_checked/lge_Myo_labels",
                  "lge_mi": "labels/lge_checked/lge_MI_labels"}
        paths = {}
        for role, directory in layout.items():
            path = self.dataset / directory / key
            path.parent.mkdir(parents=True, exist_ok=True)
            shape = (12, 13, 1, 3) if role.startswith("cmr") else (12, 13, 1)
            data = np.ones(shape, dtype=np.float32) * (17 if role.endswith("image") else 1)
            nib.save(nib.Nifti1Image(data, np.eye(4)), path)
            paths[role] = path
        return key, paths

    def snapshot(self, root=None):
        root = root or self.dataset
        return {p.relative_to(root).as_posix(): (p.read_bytes(), p.stat().st_mtime_ns)
                for p in root.rglob("*") if p.is_file()}

    def evidence(self, key, paths):
        hashes = {role: hashlib.sha256(path.read_bytes()).hexdigest() for role, path in paths.items()}
        rows = []
        for target in ("cmr_myo", "lge_myo", "lge_mi"):
            bindings = {"image": hashes[target[:3] + "_image"], "label": hashes[target]}
            if target.startswith("lge"):
                other = "mi" if target == "lge_myo" else "myo"
                bindings[other] = hashes["lge_" + other]
            rows.append({"dataset": str(self.dataset), "key": key, "target": target,
                         "status": "ACCEPTED", "scope": "whole_volume", "hashes": bindings,
                         "provenance": {"kind": "human", "record_id": "synthetic-" + target}})
        path = self.root / "runtime/dataset_acceptance/receipts.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def perform(self, action, job_id=None):
        phases = []
        result = control._perform(self.root, self.dataset, action,
                                  job_id or "20261103T123456123456Z_0123456789abcdef", phases.append)
        return result, phases

    def test_idle_status_does_not_create_runtime_or_scan_dataset(self):
        with patch.object(control.subprocess, "Popen", side_effect=AssertionError("spawn forbidden")):
            result = control.status(self.root)
        self.assertEqual(result["status"], "IDLE")
        self.assertFalse((self.root / "runtime").exists())

    def test_launch_returns_without_waiting_and_uses_fixed_worker_arguments(self):
        result = self.launch()
        self.assertTrue(result["accepted"])
        self.assertEqual(result["status"], "STARTING")
        self.assertTrue(result["busy"])
        self.assertEqual(result["progress"], {"phase": "QUEUED"})
        args, options = self.popen.call_args
        self.assertEqual(args[0], [sys.executable, "-u", "-m", "app.dataset_control", "--worker",
                                  str(self.root), str(self.dataset), "audit", result["job_id"]])
        self.assertEqual(options["stdin"], subprocess.DEVNULL)
        self.assertFalse(options.get("shell", False))
        if os.name == "nt":
            self.assertTrue(options["creationflags"] & subprocess.CREATE_NO_WINDOW)
        else:
            self.assertTrue(options["start_new_session"])
        self.process.wait.assert_not_called()
        self.assertEqual(self.job()["dataset"], str(self.dataset))
        self.assertTrue((control._job_path(self.folder, result["job_id"]).parent / "worker.log").is_file())
        self.assertNotIn(str(self.dataset), json.dumps(result))
        self.assertNotIn("pid", json.dumps(result))

    def test_running_job_rejects_every_action_and_new_controller_instance(self):
        first = self.launch()
        for action in ("audit", "validate", "freeze"):
            result = self.launch(action)
            self.assertFalse(result["accepted"])
            self.assertEqual(result["job_id"], first["job_id"])
            self.assertEqual(result["error_code"], "BUSY")
        self.popen.assert_called_once()

    def test_nonblocking_lock_serializes_threads(self):
        control._private_dir(self.folder)
        outputs = []
        with control._lock(self.folder):
            thread = threading.Thread(target=lambda: outputs.append(self.launch()))
            thread.start()
            thread.join(2)
            self.assertFalse(thread.is_alive(), "HTTP launch waited for a busy lock")
            self.assertEqual(control.status(self.root)["status"], "BUSY")
        self.assertEqual(outputs[0]["error_code"], "BUSY")
        self.popen.assert_not_called()

    def test_lock_serializes_separate_processes(self):
        control._private_dir(self.folder)
        command = [sys.executable, "-c", "from pathlib import Path; from app.dataset_control import launch; "
                   "import json,sys; print(json.dumps(launch(Path(sys.argv[1]),Path(sys.argv[2]),'audit')))",
                   str(self.root), str(self.dataset)]
        with control._lock(self.folder):
            process = subprocess.run(command, cwd=Path(control.__file__).parents[1], capture_output=True,
                                     text=True, timeout=15, **({"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}))
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout)["error_code"], "BUSY")
        self.assertFalse((self.folder / "current.json").exists())

    def test_invalid_action_and_configuration_do_not_spawn_or_write_dataset(self):
        for action in ("../freeze", "train", {"path": str(self.dataset)}, None):
            self.assertEqual(self.launch(action)["error_code"], "INVALID_ACTION")
        for walltime in (0, 1801, True, "1800"):
            self.assertEqual(self.launch(walltime=walltime)["error_code"], "INVALID_CONFIGURATION")
        result = control.launch(self.dataset, self.dataset, "audit", popen=self.popen)
        self.assertEqual(result["error_code"], "INVALID_CONFIGURATION")
        self.assertEqual(list(self.dataset.iterdir()), [])
        self.popen.assert_not_called()

    def test_source_output_overlap_and_junction_are_rejected(self):
        nested = self.root / "runtime/dataset_releases/source"
        nested.mkdir(parents=True)
        result = control.launch(self.root, nested, "freeze", popen=self.popen)
        self.assertEqual(result["error_code"], "INVALID_CONFIGURATION")
        original = Path.lstat
        def reparse(path, *args, **kwargs):
            info = original(path, *args, **kwargs)
            return Mock(st_mode=info.st_mode, st_file_attributes=0x400) if path == self.dataset else info
        with patch.object(Path, "lstat", reparse):
            self.assertEqual(self.launch()["error_code"], "INVALID_CONFIGURATION")
        self.popen.assert_not_called()

    def test_pid_reuse_is_not_liveness_and_status_never_relaunches(self):
        first = self.launch()
        self.live[43210] = "new-process-same-pid"
        for _ in range(3):
            result = self.status()
            self.assertEqual(result["status"], "FAILED")
            self.assertEqual(result["error_code"], "WORKER_EXITED")
            self.assertFalse(result["busy"])
            self.assertEqual(result["job_id"], first["job_id"])
        self.popen.assert_called_once()
        self.assertEqual(self.job()["status"], "FAILED")

    def test_unknown_process_identity_blocks_launch_without_claiming_death(self):
        self.launch()
        with patch.object(control, "_process_start", side_effect=PermissionError("PRIVATE_PATH")):
            for result in (control.status(self.root), control.launch(self.root, self.dataset, "freeze", popen=self.popen)):
                self.assertEqual(result["error_code"], "PROCESS_UNVERIFIABLE")
                self.assertTrue(result["busy"])
                self.assertNotIn("PRIVATE_PATH", json.dumps(result))
        self.popen.assert_called_once()

    def test_orphan_executor_blocks_launch_even_when_supervisor_is_dead(self):
        self.launch()
        job = self.job()
        job["executor"] = {"pid": 43211, "start": "child-birth"}
        self.save_job(job)
        self.live = {43211: "child-birth"}
        result = self.status()
        self.assertEqual(result["error_code"], "SUPERVISOR_EXITED")
        self.assertTrue(result["busy"])
        self.assertFalse(self.launch("freeze")["accepted"])
        self.live.clear()
        self.assertFalse(self.status()["busy"])
        self.popen.assert_called_once()

    def test_terminal_job_still_blocks_until_owned_processes_exit(self):
        self.launch()
        job = self.job()
        control._finish(self.folder, job, "SUCCEEDED")
        self.assertTrue(self.status()["busy"])
        self.assertFalse(self.launch()["accepted"])
        self.live.clear()
        self.assertFalse(self.status()["busy"])

    def test_spawn_failure_is_durable_and_private(self):
        self.popen.side_effect = OSError("SYNTHETIC_PRIVATE /private/data/level.nii")
        result = self.launch()
        self.assertFalse(result["accepted"])
        self.assertEqual(result["error_code"], "LAUNCH_FAILED")
        self.assertEqual(self.status()["status"], "FAILED")
        self.assertNotIn("SYNTHETIC_PRIVATE", json.dumps(result))
        log = control._job_path(self.folder, result["job_id"]).parent / "worker.log"
        self.assertIn("SYNTHETIC_PRIVATE", log.read_text())
        self.popen.assert_called_once()

    def test_identity_capture_failure_kills_only_owned_child(self):
        self.live.clear()
        result = self.launch()
        self.assertEqual(result["error_code"], "LAUNCH_FAILED")
        self.process.kill.assert_called_once()

    def test_corrupt_journal_never_resets_or_relaunches(self):
        self.launch()
        (self.folder / "current.json").write_text('{"job_id":"../private"}')
        self.assertEqual(self.status()["error_code"], "JOURNAL_UNREADABLE")
        self.assertEqual(self.launch()["error_code"], "JOURNAL_UNREADABLE")
        self.popen.assert_called_once()

    def test_public_projection_drops_paths_dynamic_keys_and_invalid_counters(self):
        self.launch()
        job = self.job()
        job.update(status="SUCCEEDED", phase="SYNTHETIC_PRIVATE", error_code="/private/failure",
                   result={"audit": {"keys": 7, "blocked_pairs": True, "releasable_pairs": -1,
                                     "patient": "SYNTHETIC_PRIVATE", "paths": ["/private"]},
                           "validation": {"attempted": 3, "calibrations": {"secret": "/private"},
                                          "counts": {"accepted_pairs": 2, "SYNTHETIC_PRIVATE": 4}},
                           "release": {"release_id": job["job_id"], "pair_count": 2,
                                       "manifest_sha256": "a" * 64, "path": "/private"}, "rows": ["private"]})
        self.save_job(job)
        result = self.status()
        self.assertEqual(result["result"]["audit"], {"keys": 7})
        self.assertEqual(result["result"]["validation"], {"attempted": 3, "counts": {"accepted_pairs": 2}})
        for forbidden in ("SYNTHETIC_PRIVATE", "/private", "calibrations", "paths", "pid"):
            self.assertNotIn(forbidden, json.dumps(result))

    def test_worker_rejects_unclaimed_entry_without_running_domain_code(self):
        result = self.launch()
        with patch.object(control, "_perform") as perform:
            with self.assertRaisesRegex(ValueError, "identity_mismatch"):
                control._execute(self.root, self.dataset, "audit", result["job_id"])
        perform.assert_not_called()

    def test_supervisor_deadline_kills_owned_executor_and_persists_timeout(self):
        result = self.launch(walltime=1)
        child = Mock(pid=43211)
        child.wait.side_effect = [subprocess.TimeoutExpired("synthetic", 1), -9]
        self.live[43211] = "child-birth"
        with patch.object(control.os, "getpid", return_value=43210), \
                patch.object(control, "_process_start", side_effect=self.live.get), \
                patch.object(control.subprocess, "Popen", return_value=child):
            code = control._worker(self.root, self.dataset, "audit", result["job_id"])
        self.assertEqual(code, 124)
        child.kill.assert_called_once()
        self.assertEqual(self.job()["status"], "TIMED_OUT")
        self.assertEqual(self.job()["error_code"], "TIME_LIMIT")
        self.assertLessEqual(child.wait.call_args_list[0].kwargs["timeout"], 1)

    def test_worker_exit_without_terminal_receipt_is_explicit_failure(self):
        result = self.launch()
        child = Mock(pid=43211)
        child.wait.return_value = 0
        self.live[43211] = "child-birth"
        with patch.object(control.os, "getpid", return_value=43210), \
                patch.object(control, "_process_start", side_effect=self.live.get), \
                patch.object(control.subprocess, "Popen", return_value=child):
            control._worker(self.root, self.dataset, "audit", result["job_id"])
        self.assertEqual(self.job()["error_code"], "WORKER_EXITED")
        child.kill.assert_not_called()

    def test_dead_overdue_job_is_durably_timed_out_by_status(self):
        self.launch()
        job = self.job()
        job["created_epoch"] -= 1801
        self.save_job(job)
        self.live.clear()
        self.assertEqual(self.status()["status"], "TIMED_OUT")
        self.assertEqual(self.job()["error_code"], "TIME_LIMIT")

    def test_missing_cached_audit_is_actionable_durable_failure(self):
        result = self.launch("validate")
        job = self.job()
        job["executor"] = control._identity(os.getpid())
        self.save_job(job)
        with patch.object(control.traceback, "print_exc"):
            code = control._execute(self.root, self.dataset, "validate", result["job_id"])
        self.assertEqual(code, 1)
        self.assertEqual(self.job()["error_code"], "AUDIT_REQUIRED")
        self.assertFalse((self.root / "runtime/dataset_release/audit.json").exists())

    def test_synthetic_enriched_audit_validation_preserves_existing_policy(self):
        self.bundle()
        before = self.snapshot()
        audit_result, phases = self.perform("audit")
        self.assertEqual(phases, ["AUDITING"])
        self.assertEqual(audit_result["audit"]["releasable_pairs"], 0)
        cache = json.loads((self.root / "runtime/dataset_release/audit.json").read_text())
        self.assertIn("generated_at", cache)
        self.assertIn("evidence_sources", cache)
        self.assertIn("frozen_patient_count", cache)
        from app.check_auto_apply import MIN_QUALITY_SCORE
        from app import dataset_validation as validator
        with patch("app.lge_execution.calibration", return_value=({"synthetic": True}, "synthetic-calibration")), \
                patch("app.quality_scoring.score_record", return_value={"score": MIN_QUALITY_SCORE}), \
                patch.object(validator, "validate_batch", wraps=validator.validate_batch) as validate:
            result, phases = self.perform("validate")
            self.assertEqual(validate.call_args.kwargs, {"limit": 100})
            self.assertEqual(set(validate.call_args.args[2]), set(control._AUDIT_CORE))
            self.assertEqual(validate.call_args.args[2]["audit_sha256"], cache["audit_sha256"])
            again, _ = self.perform("validate")
        self.assertEqual(phases, ["VALIDATING", "REFRESHING_AUDIT"])
        self.assertEqual(result["validation"]["counts"], {"accepted_pairs": 1, "accepted_targets": 3})
        self.assertEqual(result["audit"]["releasable_pairs"], 1)
        refreshed = json.loads((self.root / "runtime/dataset_release/audit.json").read_text())
        self.assertNotEqual(refreshed["audit_sha256"], cache["audit_sha256"])
        summary = json.loads((self.root / "runtime/dataset_release/summary.json").read_text())
        self.assertEqual(summary["summary"]["releasable_pairs"], 1)
        for key, count in result["audit"].items():
            self.assertEqual(summary["summary"][key], count)
        self.assertEqual(again["audit"], result["audit"])
        self.assertEqual(again["validation"]["attempted"], 0)
        self.assertEqual(again["validation"]["counts"]["unchanged_already_validated"], 1)
        self.assertEqual(before, self.snapshot())
        self.network.assert_not_called()

    def test_below_existing_threshold_never_accepts_or_freezes(self):
        self.bundle()
        self.perform("audit")
        from app.check_auto_apply import MIN_QUALITY_SCORE
        with patch("app.lge_execution.calibration", return_value=({"synthetic": True}, "synthetic-calibration")), \
                patch("app.quality_scoring.score_record", return_value={"score": MIN_QUALITY_SCORE - 1}):
            result, _ = self.perform("validate")
        self.assertEqual(result["validation"]["counts"]["needs_review_pairs"], 1)
        self.assertEqual(result["audit"]["releasable_pairs"], 0)
        ledger = self.root / "runtime/dataset_acceptance/receipts.jsonl"
        decisions = [json.loads(line) for line in ledger.read_text().splitlines()]
        self.assertEqual({r["status"] for r in decisions}, {"PENDING", "NEEDS_REVIEW"})
        with self.assertRaisesRegex(ValueError, "no_releasable_pairs"):
            self.perform("freeze")

    def test_refresh_phase_is_durable_and_refresh_failure_never_reports_success(self):
        self.bundle()
        self.perform("audit")
        launched = self.launch("validate")
        job = self.job()
        job["executor"] = control._identity(os.getpid())
        self.save_job(job)
        from app.check_auto_apply import MIN_QUALITY_SCORE
        def fail_refresh(root, dataset):
            self.assertEqual((root, dataset), (self.root, self.dataset))
            current = self.status()
            self.assertEqual(current["status"], "RUNNING")
            self.assertEqual(current["progress"]["phase"], "REFRESHING_AUDIT")
            raise OSError("SYNTHETIC_PRIVATE refresh failure")
        with patch("app.lge_execution.calibration", return_value=({"synthetic": True}, "synthetic-calibration")), \
                patch("app.quality_scoring.score_record", return_value={"score": MIN_QUALITY_SCORE}), \
                patch("scripts.dataset_release.run", side_effect=fail_refresh) as refresh, \
                patch.object(control.traceback, "print_exc"):
            code = control._execute(self.root, self.dataset, "validate", launched["job_id"])
        self.assertEqual(code, 1)
        refresh.assert_called_once_with(self.root, self.dataset)
        result = self.status()
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["error_code"], "OPERATION_FAILED")
        self.assertEqual(result["progress"]["phase"], "REFRESHING_AUDIT")
        self.assertNotIn("SYNTHETIC_PRIVATE", json.dumps(result))

    def test_stale_or_tampered_cached_audit_is_not_silently_refreshed_for_validation(self):
        _, paths = self.bundle()
        self.perform("audit")
        path = self.root / "runtime/dataset_release/audit.json"
        cached = path.read_bytes()
        report = json.loads(cached)
        report["summary"]["releasable_pairs"] = 100
        save_journal(path, report)
        with self.assertRaisesRegex(ValueError, "audit_changed"):
            self.perform("validate")
        path.write_bytes(cached)
        source = nib.load(paths["cmr_image"])
        nib.save(nib.Nifti1Image(np.asanyarray(source.dataobj) + 1, source.affine), paths["cmr_image"])
        with patch("app.lge_execution.calibration", return_value=({"synthetic": True}, "synthetic-calibration")):
            with self.assertRaisesRegex(ValueError, "stale_or_changed"):
                self.perform("validate")
        self.assertEqual(path.read_bytes(), cached)

    def test_freeze_uses_fresh_full_audit_and_never_overwrites(self):
        key, paths = self.bundle()
        self.perform("audit")  # cached non-releasable state must not control freeze
        self.evidence(key, paths)
        before = self.snapshot()
        export_id = "20261103T123456123456Z_0123456789abcdef"
        with patch("scripts.dataset_release.run", wraps=run) as release:
            result, phases = self.perform("freeze", export_id)
        self.assertEqual(phases, ["FREEZING"])
        self.assertEqual(result["release"]["pair_count"], 1)
        destination = self.root / "runtime/dataset_releases" / export_id
        self.assertEqual(release.call_args.args, (self.root, self.dataset, destination))
        self.assertFalse(destination.is_relative_to(self.dataset))
        manifest = destination / "manifest.json"
        self.assertEqual(hashlib.sha256(manifest.read_bytes()).hexdigest(), result["release"]["manifest_sha256"])
        self.assertFalse(manifest.stat().st_mode & stat.S_IWRITE)
        frozen = self.snapshot(destination)
        with self.assertRaises(FileExistsError):
            self.perform("freeze", export_id)
        self.assertTrue(result["release_created_by_current_audit"])
        self.assertEqual(result["release"]["release_id"], result["release"]["manifest_sha256"])
        self.assertIn("created_at", result["release"])
        later, _ = self.perform("audit")
        self.assertFalse(later["release_created_by_current_audit"])
        self.assertEqual(later["release"], result["release"])
        self.assertEqual(frozen, self.snapshot(destination))
        self.assertEqual(before, self.snapshot())
        self.assertNotIn("SYNTHETIC_PRIVATE", json.dumps(result))

    def test_real_detached_audit_survives_controller_recreation_and_preserves_pause(self):
        self.bundle()
        pause = self.root / "runtime/operator_control.json"
        pause.parent.mkdir(parents=True)
        pause.write_text('{"paused":true,"paid_calls":0}', encoding="utf-8")
        before_pause = pause.read_bytes(), pause.stat().st_mtime_ns
        before = self.snapshot()
        processes = []
        def spawn(*args, **kwargs):
            process = subprocess.Popen(*args, **kwargs)
            processes.append(process)
            return process
        start = time.monotonic()
        result = control.launch(self.root, self.dataset, "audit", popen=spawn)
        self.assertTrue(result["accepted"], result)
        self.assertLess(time.monotonic() - start, 5)
        until = time.monotonic() + 40
        while time.monotonic() < until:
            result = control.status(self.root)
            if not result["busy"]:
                break
            time.sleep(0.05)
        for process in processes:
            process.wait(timeout=5)  # Wait for Windows handle cleanup before deleting private logs.
        if result["status"] != "SUCCEEDED":
            self.fail(str(result) + "\n" + (control._job_path(self.folder, self.job()["job_id"]).parent / "worker.log").read_text())
        self.assertFalse(result["busy"])
        self.assertEqual(result["result"]["audit"]["keys"], 1)
        self.assertEqual(result["result"]["audit"]["releasable_pairs"], 0)
        self.assertEqual(result["progress"]["phase"], "COMPLETE")
        self.assertEqual(before, self.snapshot())
        self.assertEqual(before_pause, (pause.read_bytes(), pause.stat().st_mtime_ns))
        if os.name != "nt":
            self.assertEqual(stat.S_IMODE(self.folder.stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE(control._job_path(self.folder, result["job_id"]).stat().st_mode), 0o600)

    def test_process_identity_is_stable_for_actual_process_and_dead_pid_is_not_alive(self):
        first = control._identity(os.getpid())
        self.assertEqual(first, control._identity(os.getpid()))
        self.assertTrue(control._alive(first))
        self.assertFalse(control._alive({"pid": os.getpid(), "start": "old-birth"}))
        self.assertIsNone(control._process_start(2147483647))

    def test_python39_syntax(self):
        for path in (Path(control.__file__), Path(__file__)):
            ast.parse(path.read_text(encoding="utf-8"), feature_version=(3, 9))


if __name__ == "__main__":
    unittest.main()
