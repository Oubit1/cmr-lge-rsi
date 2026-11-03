"""Synthetic framework tests: no private vendor, torch, patients or provider calls."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from app import segmentation_training as training


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts" / "train_segmentation_candidate.py"

# A tiny vendor-shaped fixture, deliberately independent of actual training code.
# The checkpoint contains observations so the parent can check the adapter wiring.
FAKE_MAIN = r'''
from pathlib import Path
from types import SimpleNamespace
import json
import os
import subprocess
import sys
import time

Path(__file__).with_name("imported").write_text("yes")
events = []

def seed_everything(seed):
    events.append(["seed", seed])

def resolve_device(device):
    events.append(["device", device])
    return device

def load_image_hwt(path):
    assert Path(path).read_bytes().startswith(b"image-")
    # The third return is the original time AXIS, not number of frames.
    return SimpleNamespace(shape=(17, 19, 7)), None, 0

def load_image_hw(path):
    assert Path(path).read_bytes().startswith(b"image-")
    return SimpleNamespace(shape=(17, 19)), SimpleNamespace(shape=(17, 19, 1))

def build_model(config, device):
    events.append(["build_model", config["model"]])
    return {"built": "standard"}

def build_mi_model(config, device):
    events.append(["build_mi_model", config["model"]])
    return {"built": "mi"}

def load_checkpoint(path, model, device):
    assert model["built"] == "standard"
    events.append(["load_checkpoint", Path(path).read_text()])

def load_mi_checkpoint(path, model, device):
    assert model["built"] == "mi"
    events.append(["load_mi_checkpoint", Path(path).read_text()])

def load_frozen_myo_model(path, device):
    assert Path(path).read_text() == "frozen-myo"
    events.append(["frozen_myo", "frozen-myo"])
    return "frozen-myo", {}

def labeled_loader(records, config, device, shuffle, include_original):
    assert (shuffle, include_original) in [(True, False), (False, True)]
    for record in records:
        assert record.label_policy == "all"
        assert record.total_frames == 7
        assert record.labeled_frames == tuple(range(7))
    events.append(["labeled_loader", shuffle, include_original])
    return records

def train_loader(records, config, device):
    for record in records:
        assert record.native_shape == (17, 19, 1)
    events.append(["train_loader"])
    return records

def training_loader(records, config, device):
    events.append(["training_loader"])
    return train_loader(records, config, device)

def validation_loader(records, config, device):
    for record in records:
        assert record.native_shape == (17, 19, 1)
    events.append(["validation_loader"])
    return records

def train_model(model, *args):
    if model["built"] == "mi":
        myo, train, val, device, config, run_dir = args
        assert myo == "frozen-myo"
    else:
        train, val, device, config, run_dir = args
    assert {r.patient for r in train} == {"synthetic-train"}
    assert {r.patient for r in val} == {"synthetic-val"}
    assert "centers" not in config and "manifest_path" not in config
    assert "completion" not in config and "test" not in config
    assert config["training"]["epochs"] <= 3
    run_dir = Path(run_dir)
    mode = config["training"]["synthetic_mode"]
    (run_dir / "started").write_text("yes")
    if mode in (2, 5, 9):
        # A real descendant: heartbeat proves cleanup even when the worker exits.
        code = 'import pathlib,sys,time; p=pathlib.Path(sys.argv[1]);\nwhile True:\n p.write_text(str(time.monotonic())); time.sleep(.04)'
        child = subprocess.Popen([sys.executable, "-c", code, str(run_dir / "heartbeat")],
                                 creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        (run_dir / "child.pid").write_text(str(child.pid))
        deadline = time.monotonic() + 5
        while not (run_dir / "heartbeat").exists() and time.monotonic() < deadline:
            time.sleep(.02)
        if mode == 9:
            os._exit(19)
        if mode == 2:
            time.sleep(60)
    if mode == 1:
        raise RuntimeError("synthetic trainer failure")
    if mode == 3:
        os._exit(17)
    checkpoint = run_dir / "best.pt"
    if mode == 4:
        return run_dir / "missing.pt"
    if mode == 6:
        checkpoint = run_dir.parent / "outside.pt"
    if mode == 7:
        Path(train[0].label_path).write_text("mutated")
    payload = {"events": events, "config": config, "device": device,
               "images": [Path(r.image_path).read_text() for r in train + val],
               "labels": [Path(r.label_path).read_text() for r in train + val],
               "paths": [r.image_path for r in train + val]}
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")
    if mode == 8:
        checkpoint.write_bytes(b"")
    if train and hasattr(train[0], "total_frames"):
        return checkpoint, [{"synthetic_epoch": 1}]
    return checkpoint
'''

FAKE_CINE_RECORD = '''from dataclasses import dataclass
@dataclass
class SupervisedVolumeRecord:
    center: str
    patient: str
    relative_path: str
    image_path: str
    label_path: str
    label_policy: str
    labeled_frames: tuple
    total_frames: int
'''

FAKE_LGE_RECORD = '''from dataclasses import dataclass
@dataclass
class LGERecord:
    center: str
    patient: str
    relative_path: str
    image_path: str
    label_path: str
    native_shape: tuple
'''


def binding(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


class SegmentationTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="synthetic-training-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.dataset = self.root / "dataset"
        self.dataset.mkdir()
        self.records = []
        for split in ("train", "val", "heldout"):
            image, label = self.dataset / (split + ".image"), self.dataset / (split + ".label")
            image.write_text("image-" + split)
            label.write_text("label-" + split)
            self.records.append({"center": "synthetic-center", "patient": "synthetic-" + split,
                                 "relative_path": split + "/slice.nii.gz",
                                 "image": binding(image), "label": binding(label)})

    def make_job(self, target="lge_myo", mode=0):
        folder, entry, module = training.VENDORS[target]
        vendor = self.root / "vendor" / folder
        vendor.mkdir(parents=True, exist_ok=True)
        (vendor / entry).write_text(FAKE_MAIN, encoding="utf-8")
        package, record_file = module.split(".")
        (vendor / package).mkdir(exist_ok=True)
        (vendor / package / "__init__.py").write_text("")
        (vendor / package / (record_file + ".py")).write_text(
            FAKE_CINE_RECORD if target == "cmr_myo" else FAKE_LGE_RECORD)
        config = {"model": {"synthetic_architecture_value": 31},
                  "training": {"seed": 42, "epochs": 9, "synthetic_mode": mode,
                               "learning_rate": 0.123, "num_workers": 0},
                  "visualization": {"overlay_alpha": 0.38}, "roi": {"synthetic_roi": 1},
                  "loss": {"synthetic_loss": 2},
                  "centers": [{"test_reference": self.records[-1]["label"]["path"]}],
                  "manifest_path": str(self.dataset / "heldout-private-manifest"),
                  "test": self.records[-1], "completion": {"output_root": str(self.dataset)}}
        config_path = self.root / (target + "-config.json")
        config_path.write_text(json.dumps(config))
        job = {"schema_version": 1, "target": target, "dataset_root": str(self.dataset),
               "output_dir": str(self.root / ("output-" + target)), "config": binding(config_path),
               "vendor": {"root": str(vendor), "files": {
                   p.relative_to(vendor).as_posix(): binding(p)["sha256"] for p in vendor.rglob("*.py")}},
               "splits": {"train": ["synthetic-train"], "validation": ["synthetic-val"],
                          "test": ["synthetic-heldout"]}, "records": copy.deepcopy(self.records),
               "resources": {"max_seconds": 15, "max_epochs": 3, "device": "cpu"}}
        if target == "lge_mi":
            myo = self.root / "frozen-myo.pt"
            myo.write_text("frozen-myo")
            job["myo_weights"] = binding(myo)
        return job

    def cli(self, job, execute=False):
        path = self.root / "local-job.json"
        path.write_text(json.dumps(job))
        return subprocess.run([sys.executable, "-B", str(CLI), "--job", str(path)] +
                              (["--execute"] if execute else []), capture_output=True,
                              text=True, timeout=25)

    def test_preflight_pure_and_cli_requires_execute(self):
        job = self.make_job()
        before = copy.deepcopy(job)
        checked = training.validate_job(job)
        self.assertEqual(job, before)
        self.assertEqual(checked["counts"], {"train": 1, "validation": 1, "test": 1})
        self.assertEqual(checked["config"]["training"]["epochs"], 3)
        result = self.cli(job)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "PREFLIGHT")
        self.assertFalse(Path(job["output_dir"]).exists())
        self.assertFalse(list(Path(job["vendor"]["root"]).rglob("imported")))

    def test_all_three_adapters_and_parent_loaders_in_real_subprocess(self):
        for target in training.VENDORS:
            with self.subTest(target=target):
                job = self.make_job(target)
                parent = self.root / (target + "-parent.pt")
                parent.write_text("parent-weights")
                job["parent_weights"] = binding(parent)
                result = self.cli(job, execute=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                status = json.loads(result.stdout)
                self.assertEqual(status["status"], "TRAINED")
                self.assertGreater(status["elapsed_seconds"], 0)
                checkpoint = Path(status["checkpoint"])
                self.assertEqual(status["checkpoint_sha256"], binding(checkpoint)["sha256"])
                observation = json.loads(checkpoint.read_text())
                self.assertEqual(observation["images"], ["image-train", "image-val"])
                self.assertEqual(observation["labels"], ["label-train", "label-val"])
                loader = "load_mi_checkpoint" if target == "lge_mi" else "load_checkpoint"
                self.assertIn([loader, "parent-weights"], observation["events"])
                self.assertEqual(observation["config"]["training"]["learning_rate"], 0.123)
                self.assertEqual(observation["config"]["training"]["epochs"], 3)
                self.assertEqual(observation["config"]["model"], {"synthetic_architecture_value": 31})
                run_dir = checkpoint.parent.parent
                for name in ("worker_input.json", "effective_config.json"):
                    content = (run_dir / name).read_text()
                    self.assertNotIn("heldout", content)
                    self.assertNotIn(str(self.root / (target + "-config.json")), content)
                self.assertFalse((run_dir / "inputs" / "test").exists())
                self.assertTrue(all(str(run_dir) in path for path in observation["paths"]))
                self.assertEqual(training.read_json(run_dir / "status.json"), status)
                self.assertEqual(training.run_job(job, execute=True), status)

    def test_default_initialization_does_not_load_parent(self):
        status = training.run_job(self.make_job(), execute=True)
        self.assertEqual(status["status"], "TRAINED", status)
        events = json.loads(Path(status["checkpoint"]).read_text())["events"]
        self.assertTrue(any(e[0] == "build_model" for e in events))
        self.assertFalse(any(e[0].startswith("load_") for e in events))

    def test_split_validation_every_row_and_global_patient_ids(self):
        base = self.make_job()
        changes = [
            lambda j: j["splits"]["validation"].append("synthetic-train"),
            lambda j: j["splits"]["test"].append("synthetic-val"),
            lambda j: j["splits"].update(train=[]),
            lambda j: j["splits"].update(validation=[]),
            lambda j: j["splits"]["train"].append("missing-patient"),
            lambda j: j["records"][0].update(patient="unassigned"),
            lambda j: j["records"].append(copy.deepcopy(j["records"][0])),
            lambda j: j["records"][1].update(center="other-center", patient="synthetic-train"),
            lambda j: j["records"][0].pop("label"),
            lambda j: j["splits"].pop("test"),
            lambda j: j["splits"]["train"].append("synthetic-train"),
        ]
        for change in changes:
            job = copy.deepcopy(base)
            change(job)
            with self.subTest(change=change), self.assertRaises(training.JobError):
                training.validate_job(job)
        base["splits"]["test"] = []
        base["records"] = base["records"][:2]
        self.assertEqual(training.validate_job(base)["counts"]["test"], 0)

    def test_hash_bindings_include_test_config_vendor_and_dependencies(self):
        for name in ("train-image", "val-label", "test-label", "config", "vendor", "myo", "parent"):
            with self.subTest(name=name):
                job = self.make_job("lge_mi")
                parent = self.root / "parent.pt"
                parent.write_text("parent")
                job["parent_weights"] = binding(parent)
                paths = {"train-image": job["records"][0]["image"]["path"],
                         "val-label": job["records"][1]["label"]["path"],
                         "test-label": job["records"][2]["label"]["path"],
                         "config": job["config"]["path"], "parent": str(parent),
                         "myo": job["myo_weights"]["path"],
                         "vendor": str(Path(job["vendor"]["root"]) / "lge_mi_seg/data.py")}
                path = Path(paths[name])
                original = path.read_bytes()
                try:
                    path.write_bytes(original + b"changed")
                    with self.assertRaises(training.JobError):
                        training.validate_job(job)
                finally:
                    path.write_bytes(original)

    def test_schema_resources_and_output_constraints(self):
        base = self.make_job()
        changes = [lambda j: j.update(target="other"), lambda j: j.update(schema_version=True),
                   lambda j: j.update(output_dir=str(self.dataset / "run")),
                   lambda j: j.update(output_dir=str(self.root)),
                   lambda j: j.update(output_dir=j["vendor"]["root"]),
                   lambda j: j.update(output_dir="relative"),
                   lambda j: j.update(command="do-not-execute"),
                   lambda j: j["vendor"]["files"].pop("lge_seg/data.py"),
                   lambda j: j["records"][0].update(relative_path="../escape.nii.gz"),
                   lambda j: j["records"][0]["image"].update(sha256="bad")]
        for key, values in {"max_seconds": [0, -1, float("nan"), float("inf"), True, "3"],
                            "max_epochs": [0, -1, 1.5, True], "device": ["cpu; command", ""]}.items():
            for value in values:
                changes.append(lambda j, k=key, v=value: j["resources"].update({k: v}))
        for index, change in enumerate(changes):
            job = copy.deepcopy(base)
            change(job)
            with self.subTest(index=index), self.assertRaises(training.JobError):
                training.validate_job(job)

    def test_mi_requires_bound_dependency_and_non_mi_rejects_it(self):
        job = self.make_job("lge_mi")
        myo = job.pop("myo_weights")
        with self.assertRaises(training.JobError):
            training.validate_job(job)
        job = self.make_job()
        job["myo_weights"] = myo
        with self.assertRaises(training.JobError):
            training.validate_job(job)

    def test_numeric_config_projection_refuses_hidden_references_and_missing_epochs(self):
        job = self.make_job()
        path = Path(job["config"]["path"])
        original = json.loads(path.read_text())
        for change in (lambda c: c["training"].update(test_reference="secret-test-path"),
                       lambda c: c["training"].pop("epochs"),
                       lambda c: c.pop("model")):
            config = copy.deepcopy(original)
            change(config)
            path.write_text(json.dumps(config))
            job["config"] = binding(path)
            with self.assertRaises(training.JobError):
                training.validate_job(job)
        original["training"]["epochs"] = 1
        path.write_text(json.dumps(original))
        job["config"] = binding(path)
        self.assertEqual(training.validate_job(job)["config"]["training"]["epochs"], 1)

    def test_reject_aliased_files_and_copied_images_across_splits(self):
        job = self.make_job()
        job["records"][1]["image"] = job["records"][0]["image"]
        with self.assertRaises(training.JobError):
            training.validate_job(job)
        job = self.make_job()
        image = Path(job["records"][1]["image"]["path"])
        image.write_bytes(Path(job["records"][0]["image"]["path"]).read_bytes())
        job["records"][1]["image"] = binding(image)
        with self.assertRaises(training.JobError):
            training.validate_job(job)

    def test_worker_errors_bad_checkpoints_mutation_and_no_retries(self):
        for mode, reason in ((1, "synthetic trainer failure"), (3, "worker_exit_17"),
                             (4, "invalid checkpoint"), (6, "invalid checkpoint"),
                             (7, "SHA-256 mismatch"), (8, "invalid checkpoint")):
            with self.subTest(mode=mode):
                job = self.make_job(mode=mode)
                result = training.run_job(job, execute=True)
                self.assertEqual(result["status"], "FAILED", result)
                self.assertIn(reason, result["reason"])
                self.assertIsNone(result["checkpoint_sha256"])
                self.assertEqual(training.run_job(job, execute=True), result)
                self.assertEqual(Path(job["records"][0]["label"]["path"]).read_text(), "label-train")

    def assert_heartbeat_stopped(self, directory):
        heartbeat = directory / "heartbeat"
        self.assertTrue(heartbeat.exists(), "descendant must actually have started")
        time.sleep(0.15)
        before = heartbeat.read_bytes()
        time.sleep(0.25)
        self.assertEqual(heartbeat.read_bytes(), before, "descendant survived cleanup")

    def test_timeout_cleans_actual_descendant(self):
        job = self.make_job(mode=2)
        job["resources"]["max_seconds"] = 2.0
        start = time.monotonic()
        result = training.run_job(job, execute=True)
        self.assertLess(time.monotonic() - start, 8)
        self.assertEqual((result["status"], result["reason"]), ("FAILED", "deadline_exceeded"))
        self.assertIsNone(result["checkpoint_sha256"])
        self.assert_heartbeat_stopped(Path(training.validate_job(job)["run_dir"]) / "artifacts")

    def test_success_and_abrupt_exit_also_clean_descendants(self):
        for mode, expected in ((5, "TRAINED"), (9, "FAILED")):
            with self.subTest(mode=mode):
                job = self.make_job(mode=mode)
                result = training.run_job(job, execute=True)
                self.assertEqual(result["status"], expected, result)
                self.assert_heartbeat_stopped(Path(training.validate_job(job)["run_dir"]) / "artifacts")

    def test_preexisting_pause_and_cancel_do_not_launch(self):
        for kind in ("pause", "cancel"):
            with self.subTest(kind=kind):
                job = self.make_job()
                marker = self.root / kind.upper()
                marker.touch()
                job["controls"] = {kind + "_file": str(marker)}
                result = training.run_job(job, execute=True)
                self.assertEqual(result["status"], "CANCELLED")
                self.assertEqual(result["reason"], kind + "_marker")
                run_dir = Path(training.validate_job(job)["run_dir"])
                self.assertFalse((run_dir / "vendor").exists())
                marker.unlink()
                self.assertEqual(training.run_job(job, execute=True), result)

    def test_live_markers_persistent_training_and_cross_process_lock(self):
        for kind in ("pause", "cancel"):
            with self.subTest(kind=kind):
                job = self.make_job(mode=2)
                job["output_dir"] += "-" + kind
                run_dir = Path(training.validate_job(job)["run_dir"])
                job_path = self.root / (kind + "-job.json")
                job_path.write_text(json.dumps(job))
                process = subprocess.Popen([sys.executable, "-B", str(CLI), "--job", str(job_path), "--execute"],
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                try:
                    deadline = time.monotonic() + 10
                    while not (run_dir / "artifacts" / "heartbeat").exists():
                        if process.poll() is not None or time.monotonic() >= deadline:
                            self.fail("worker did not become ready")
                        time.sleep(0.05)
                    self.assertEqual(training.read_json(run_dir / "status.json")["status"], "TRAINING")
                    duplicate = self.cli(job, execute=True)
                    self.assertEqual(duplicate.returncode, 2, duplicate.stdout + duplicate.stderr)
                    self.assertIn("already running", duplicate.stdout)
                    (run_dir / kind.upper()).touch()
                    stdout, stderr = process.communicate(timeout=8)
                    self.assertEqual(process.returncode, 1, stdout + stderr)
                    result = json.loads(stdout)
                    self.assertEqual(result["status"], "CANCELLED")
                    self.assertEqual(result["reason"], kind + "_marker")
                    self.assert_heartbeat_stopped(run_dir / "artifacts")
                finally:
                    (run_dir / "CANCEL").touch()
                    if process.poll() is None:
                        process.communicate(timeout=8)
                    if process.stdout:
                        process.stdout.close()
                    if process.stderr:
                        process.stderr.close()

    def test_abandoned_training_receipt_is_failed_without_restart(self):
        job = self.make_job()
        checked = training.validate_job(job)
        run_dir = Path(checked["run_dir"])
        run_dir.mkdir(parents=True)
        (run_dir / "status.json").write_text(json.dumps({"status": "TRAINING", "elapsed_seconds": 2,
            "job_sha256": checked["job_sha256"], "checkpoint": None, "checkpoint_sha256": None}))
        result = training.run_job(job, execute=True)
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["reason"], "interrupted_controller_no_retry")
        self.assertFalse((run_dir / "vendor").exists())

    def test_changed_checkpoint_is_not_reused(self):
        job = self.make_job()
        result = training.run_job(job, execute=True)
        Path(result["checkpoint"]).write_text("tampered")
        with self.assertRaises(training.JobError):
            training.run_job(job, execute=True)

    def test_snapshot_hashes_exact_bytes_and_refuses_changed_source(self):
        source = self.dataset / "train.image"
        bound = binding(source)
        source.write_text("changed-after-preflight")
        with self.assertRaises(training.JobError):
            training._snapshot(bound, self.root / "snapshot")

    def test_orphan_run_contents_are_not_overwritten_or_retried(self):
        job = self.make_job()
        run_dir = Path(training.validate_job(job)["run_dir"])
        run_dir.mkdir(parents=True)
        orphan = run_dir / "worker.log"
        orphan.write_text("old attempt")
        result = training.run_job(job, execute=True)
        self.assertEqual(result["status"], "FAILED")
        self.assertIn("orphan run files", result["reason"])
        self.assertEqual(orphan.read_text(), "old attempt")
        self.assertFalse((run_dir / "vendor").exists())

    def test_strict_json_and_python39_syntax(self):
        path = self.root / "invalid.json"
        for content in ('{"target":1,"target":2}', '{"value":NaN}'):
            path.write_text(content)
            with self.assertRaises(training.JobError):
                training.read_json(path)
        for path in (Path(training.__file__), CLI, Path(__file__)):
            ast.parse(path.read_text(encoding="utf-8"), feature_version=(3, 9))


if __name__ == "__main__":
    unittest.main()
