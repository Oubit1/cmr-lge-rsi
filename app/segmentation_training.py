"""Bounded private-vendor training adapter (Python 3.9+), not an approval gate.

Job v1 (all paths absolute, SHA-256 lowercase hex)::

    {"schema_version": 1, "target": "cmr_myo|lge_myo|lge_mi",
     "dataset_root": "/private/frozen", "output_dir": "/private/candidates",
     "config": {"path": "/approved/config.json", "sha256": "..."},
     "vendor": {"root": "/private/vendor/automated_annotation_of_cine",
                "files": {"main_multicenter.py": "...", "cine_seg/data.py": "..."}},
     "splits": {"train": ["p1"], "validation": ["p2"], "test": ["p3"]},
     "records": [{"center": "synthetic", "patient": "p1", "relative_path": "p1/a.nii.gz",
                  "image": {"path": "/private/frozen/images/a.nii.gz", "sha256": "..."},
                  "label": {"path": "/private/frozen/labels/a.nii.gz", "sha256": "..."}}],
     "resources": {"max_seconds": 3600, "max_epochs": 2, "device": "cpu"},
     "parent_weights": {"path": "/approved/parent.pt", "sha256": "..."},
     "myo_weights": {"path": "/approved/myo.pt", "sha256": "..."},
     "controls": {"pause_file": "/operator/PAUSE", "cancel_file": "/operator/CANCEL"}}

parent_weights and controls are optional; myo_weights is required ONLY for MI.
vendor.files must enumerate EVERY .py under that target's vendor root, including
the fixed entrypoint. Patient IDs are global across centers. Every listed patient
must have rows, every row must be assigned, and train/validation must be nonempty.
Test may be empty. Dataset discovery, random splitting and clinical adjudication
are deliberately absent. The caller supplies an already approved JSON config;
hashes identify bytes, not reviewer authorization or a code sandbox.

Only model/training/visualization (plus roi/loss for MI) numeric config sections
are passed to the worker. Dataset/manifest/completion/test references are omitted.
Epochs are min(approved epochs, max_epochs); device is explicit. No hyperparameter
or architecture defaults are added. All train/validation bytes, weights and Python
vendor sources are hash-checked into per-run snapshots before vendor import/use.
Dependencies installed in the Python environment are outside this source binding.

run_job defaults to preflight. Execution uses this interpreter in a fresh process,
an OS file lock and output_dir/<job_sha256>. Existing receipts never auto-retry;
an abandoned TRAINING receipt becomes FAILED. PAUSE/CANCEL in that run directory
or optional external markers cancel the attempt, without resume. Status has
TRAINING/TRAINED/FAILED/CANCELLED, elapsed_seconds, reason, checkpoint and its SHA.
Partial checkpoints on failure/cancellation are not accepted. POSIX process groups
and Windows kill-on-close Job Objects clean up descendants. This is cooperative
local execution of approved code, not containment of hostile code. There is no
registry selection, model promotion, provider call, or label write here.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import json
import math
import os
import re
import signal
import stat
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path, PurePosixPath


VENDORS = {
    "cmr_myo": ("automated_annotation_of_cine", "main_multicenter.py", "cine_seg.multicenter"),
    "lge_myo": ("automated_annotation_of_lge", "main.py", "lge_seg.data"),
    "lge_mi": ("automated_annotation_of_lge_mi", "main_myo_prior.py", "lge_mi_seg.data"),
}
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_DEVICE = re.compile(r"(?:cpu|cuda(?::[0-9]+)?|mps|auto)\Z")
_CHUNK = 1024 * 1024


class JobError(ValueError):
    """Invalid job, changed input bytes, or invalid worker result."""


class JobBusy(RuntimeError):
    """Another process holds this identical job's lock."""


def _require(ok, message):
    if not ok:
        raise JobError(message)


def _keys(value, required, optional=()):
    _require(type(value) is dict and set(required) <= set(value)
             and set(value) <= set(required) | set(optional), "invalid object fields")


def _text(value):
    _require(type(value) is str and bool(value.strip()) and value == value.strip()
             and "\x00" not in value, "expected nonempty text")
    return value


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def _decode_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    def constant(value):
        raise JobError("nonfinite JSON constant: " + value)
    return json.loads(data, object_pairs_hook=pairs, parse_constant=constant)


def read_json(path):
    """Read strict JSON; duplicate keys and nonfinite constants are errors."""
    return _decode_json(Path(path).read_text(encoding="utf-8"))


def _absolute(value):
    path = Path(_text(value))
    _require(path.is_absolute(), "paths must be absolute")
    # Refuse links/reparse points before resolve can hide them.
    for part in (path,) + tuple(path.parents):
        if part.exists() or part.is_symlink():
            info = part.lstat()
            _require(not stat.S_ISLNK(info.st_mode)
                     and not (getattr(info, "st_file_attributes", 0) & 0x400),
                     "symlink/reparse point not allowed")
    return path.resolve()


def _relative(value):
    value = _text(value)
    path = PurePosixPath(value)
    _require(not path.is_absolute() and str(path) == value
             and not any(p in ("", ".", "..") for p in path.parts)
             and not any(c in value for c in "\\:\x00"), "invalid relative path")
    return value


def _sha(value):
    _require(type(value) is str and _SHA.fullmatch(value) is not None, "invalid SHA-256")
    return value


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        _require(stat.S_ISREG(os.fstat(stream.fileno()).st_mode), "expected regular file")
        for block in iter(lambda: stream.read(_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _file(value):
    _keys(value, ("path", "sha256"))
    path, expected = _absolute(value["path"]), _sha(value["sha256"])
    _require(path.is_file(), "input file missing")
    _require(sha256_file(path) == expected, "input SHA-256 mismatch")
    return {"path": str(path), "sha256": expected}


def _positive_int(value):
    _require(type(value) is int and value > 0, "expected positive integer")
    return value


def _numeric_config(value):
    # These existing trainers' required sections contain numeric/bool values only.
    # Refuse hidden path/string references instead of forwarding opaque metadata.
    if type(value) is dict:
        for key, child in value.items():
            _text(key)
            _numeric_config(child)
    elif type(value) is list:
        for child in value:
            _numeric_config(child)
    else:
        _require(type(value) in (int, float, bool), "training config must be numeric/data-free")
        if type(value) is float:
            _require(math.isfinite(value), "nonfinite config number")


def _vendor(value, target):
    _keys(value, ("root", "files"))
    root = _absolute(value["root"])
    folder, entry, _ = VENDORS[target]
    _require(root.is_dir() and root.name == folder, "wrong vendor root for target")
    files = value["files"]
    _require(type(files) is dict and entry in files, "missing vendor source manifest/entry")
    actual = set()
    for path in root.rglob("*"):
        _absolute(str(path))
        if path.is_file() and path.suffix == ".py":
            actual.add(path.relative_to(root).as_posix())
    _require(set(files) == actual, "vendor.files must cover all Python sources exactly")
    for relative, digest in files.items():
        _relative(relative)
        _require(sha256_file(root / relative) == _sha(digest), "vendor SHA-256 mismatch")
    return {"root": str(root), "files": dict(sorted(files.items()))}


def validate_job(job: dict) -> dict:
    """Pure preflight: validate every row/hash without importing private vendor code.

    Return {job, job_sha256, run_dir, counts, config}; input is not modified and
    no directories/status/locks are created. config is the data-free projection.
    """
    _keys(job, ("schema_version", "target", "dataset_root", "output_dir", "config",
                "vendor", "splits", "records", "resources"),
          ("parent_weights", "myo_weights", "controls"))
    job = copy.deepcopy(job)
    _require(type(job["schema_version"]) is int and job["schema_version"] == 1,
             "unsupported schema_version")
    _require(type(job["target"]) is str and job["target"] in VENDORS, "unsupported target")
    dataset, output = _absolute(job["dataset_root"]), _absolute(job["output_dir"])
    _require(dataset.is_dir(), "dataset_root missing")
    _require(not output.is_relative_to(dataset) and not dataset.is_relative_to(output),
             "output_dir and dataset_root must be disjoint")
    job.update(dataset_root=str(dataset), output_dir=str(output))
    resources = job["resources"]
    _keys(resources, ("max_seconds", "max_epochs", "device"))
    seconds = resources["max_seconds"]
    _require(type(seconds) in (int, float) and 0 < seconds < float("inf"),
             "max_seconds must be finite and positive")
    _positive_int(resources["max_epochs"])
    _require(type(resources["device"]) is str and _DEVICE.fullmatch(resources["device"]),
             "invalid device")
    job["config"] = _file(job["config"])
    config_bytes = Path(job["config"]["path"]).read_bytes()
    _require(hashlib.sha256(config_bytes).hexdigest() == job["config"]["sha256"],
             "config SHA-256 mismatch")
    source_config = _decode_json(config_bytes.decode("utf-8"))
    _require(type(source_config) is dict, "config must be a JSON object")
    sections = ["model", "training", "visualization"]
    if job["target"] == "lge_mi":
        sections += ["roi", "loss"]
    config = {}
    for key in sections:
        _require(type(source_config.get(key)) is dict, "missing approved config section: " + key)
        _numeric_config(source_config[key])
        config[key] = copy.deepcopy(source_config[key])
    epochs = _positive_int(config["training"].get("epochs"))
    _require(type(config["training"].get("seed")) is int, "missing integer approved seed")
    config["training"]["epochs"] = min(epochs, resources["max_epochs"])
    job["vendor"] = _vendor(job["vendor"], job["target"])
    for key in ("parent_weights", "myo_weights"):
        if key in job:
            job[key] = _file(job[key])
    _require(("myo_weights" in job) == (job["target"] == "lge_mi"),
             "hash-bound myo_weights required only for lge_mi")
    # Outputs cannot overwrite bound config, vendor sources or weights either.
    sources = [Path(job["config"]["path"]), Path(job["vendor"]["root"])]
    sources += [Path(job[k]["path"]) for k in ("parent_weights", "myo_weights") if k in job]
    _require(all(not p.is_relative_to(output) and not output.is_relative_to(p) for p in sources),
             "output overlaps a bound source")
    controls = job.get("controls", {})
    _keys(controls, (), ("pause_file", "cancel_file"))
    job["controls"] = {k: str(_absolute(v)) for k, v in controls.items()}
    _keys(job["splits"], ("train", "validation", "test"))
    assignment = {}
    for split, patients in job["splits"].items():
        _require(type(patients) is list, "patient assignments must be lists")
        for patient in patients:
            _text(patient)
            _require(patient not in assignment, "patient split overlap/duplicate")
            assignment[patient] = split
        job["splits"][split] = sorted(patients)
    _require(type(job["records"]) is list and job["records"], "records must be nonempty")
    counts = dict.fromkeys(("train", "validation", "test"), 0)
    patients_seen, rows_seen, files_seen, image_splits = set(), set(), set(), {}
    for row in job["records"]:
        _keys(row, ("center", "patient", "relative_path", "image", "label"))
        center, patient = _text(row["center"]), _text(row["patient"])
        relative = _relative(row["relative_path"])
        _require(patient in assignment, "unassigned patient row")
        split = assignment[patient]
        key = (center, relative)
        _require(key not in rows_seen, "duplicate record")
        rows_seen.add(key)
        patients_seen.add(patient)
        counts[split] += 1
        for kind in ("image", "label"):
            row[kind] = _file(row[kind])
            path = Path(row[kind]["path"])
            _require(path.is_relative_to(dataset), "record outside dataset_root")
            info = path.stat()
            identity = (info.st_dev, info.st_ino) if info.st_ino else str(path)
            _require(identity not in files_seen, "duplicate/aliased image or label file")
            files_seen.add(identity)
        digest = row["image"]["sha256"]
        _require(digest not in image_splits or image_splits[digest] == split,
                 "identical image bytes cross patient splits")
        image_splits[digest] = split
    _require(patients_seen == set(assignment), "assigned patient has no rows")
    _require(counts["train"] and counts["validation"], "empty train or validation")
    job["records"].sort(key=lambda row: (row["center"], row["patient"], row["relative_path"]))
    digest = hashlib.sha256(_canonical(job)).hexdigest()
    return {"job": job, "job_sha256": digest, "run_dir": str(_absolute(str(output / digest))),
            "counts": counts, "config": config}


def _atomic_json(path, value):
    path = Path(path)
    fd, name = tempfile.mkstemp(prefix=".pending-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(_canonical(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, str(path))
    finally:
        if os.path.exists(name):
            os.unlink(name)


@contextmanager
def _job_lock(path):
    """Cross-process file lock, without an optional third-party dependency."""
    with _absolute(str(path)).open("a+b") as stream:
        if stream.seek(0, os.SEEK_END) == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise JobBusy("identical job already running") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class _WindowsJob:
    """Attach worker before sending input, so it cannot spawn outside the job."""
    def __init__(self):
        import ctypes as c
        from ctypes import wintypes as w
        class Basic(c.Structure):
            _fields_ = [("PerProcessUserTimeLimit", c.c_int64), ("PerJobUserTimeLimit", c.c_int64),
                        ("LimitFlags", w.DWORD), ("MinimumWorkingSetSize", c.c_size_t),
                        ("MaximumWorkingSetSize", c.c_size_t), ("ActiveProcessLimit", w.DWORD),
                        ("Affinity", c.c_size_t), ("PriorityClass", w.DWORD), ("SchedulingClass", w.DWORD)]
        class IO(c.Structure):
            _fields_ = [(name, c.c_uint64) for name in ("ReadOperationCount", "WriteOperationCount",
                        "OtherOperationCount", "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]
        class Extended(c.Structure):
            _fields_ = [("BasicLimitInformation", Basic), ("IoInfo", IO),
                        ("ProcessMemoryLimit", c.c_size_t), ("JobMemoryLimit", c.c_size_t),
                        ("PeakProcessMemoryUsed", c.c_size_t), ("PeakJobMemoryUsed", c.c_size_t)]
        self.api = c.WinDLL("kernel32", use_last_error=True)
        for name, args, result in (
            ("CreateJobObjectW", [c.c_void_p, w.LPCWSTR], w.HANDLE),
            ("SetInformationJobObject", [w.HANDLE, c.c_int, c.c_void_p, w.DWORD], w.BOOL),
            ("AssignProcessToJobObject", [w.HANDLE, w.HANDLE], w.BOOL),
            ("TerminateJobObject", [w.HANDLE, w.UINT], w.BOOL),
            ("CloseHandle", [w.HANDLE], w.BOOL),
        ):
            fn = getattr(self.api, name)
            fn.argtypes, fn.restype = args, result
        self.handle = self.api.CreateJobObjectW(None, None)
        if not self.handle:
            raise c.WinError(c.get_last_error())
        info = Extended()
        info.BasicLimitInformation.LimitFlags = 0x2000  # KILL_ON_JOB_CLOSE
        if not self.api.SetInformationJobObject(self.handle, 9, c.byref(info), c.sizeof(info)):
            error = c.WinError(c.get_last_error())
            self.close()
            raise error

    def attach(self, process):
        if not self.api.AssignProcessToJobObject(self.handle, int(process._handle)):
            import ctypes
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self):
        if self.handle:
            self.api.TerminateJobObject(self.handle, 1)
            self.api.CloseHandle(self.handle)
            self.handle = None


def _stop_group(process, group):
    if group is not None:
        group.close()
    elif process is not None and os.name != "nt":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if process is not None:
        if process.poll() is None:
            process.kill()  # also covers Windows assignment failure
        process.wait(timeout=10)
        if process.stdin:
            process.stdin.close()


def _control_reason(job, run_dir):
    for kind in ("cancel", "pause"):
        external = job["controls"].get(kind + "_file")
        if (run_dir / kind.upper()).exists() or (external and Path(external).exists()):
            return kind + "_marker"
    return None


def _payload(checked):
    job = checked["job"]
    train, validation = set(job["splits"]["train"]), set(job["splits"]["validation"])
    # Never serialize the full job, splits, source config path, or dataset root.
    result = {"target": job["target"], "vendor": job["vendor"], "config": checked["config"],
              "config_sha256": job["config"]["sha256"], "device": job["resources"]["device"],
              "run_dir": checked["run_dir"], "job_sha256": checked["job_sha256"],
              "train": [r for r in job["records"] if r["patient"] in train],
              "validation": [r for r in job["records"] if r["patient"] in validation]}
    for key in ("parent_weights", "myo_weights"):
        if key in job:
            result[key] = job[key]
    return result


def run_job(job: dict, *, execute: bool = False) -> dict:
    """Preflight unless explicitly execute=True; never retry an existing attempt."""
    checked = validate_job(job)
    job, run_dir = checked["job"], Path(checked["run_dir"])
    if not execute:
        return {"status": "PREFLIGHT", "job_sha256": checked["job_sha256"],
                "run_dir": str(run_dir), "counts": checked["counts"],
                "epochs": checked["config"]["training"]["epochs"]}
    run_dir.mkdir(parents=True, exist_ok=True)
    with _job_lock(run_dir / ".lock"):
        status_file = _absolute(str(run_dir / "status.json"))
        if status_file.exists():
            previous = read_json(status_file)
            _require(previous.get("job_sha256") == checked["job_sha256"], "receipt/job mismatch")
            if previous["status"] == "TRAINING":
                previous.update(status="FAILED", reason="interrupted_controller_no_retry",
                                checkpoint=None, checkpoint_sha256=None)
                _atomic_json(status_file, previous)
            if previous["status"] == "TRAINED":
                checkpoint = _absolute(previous["checkpoint"])
                _require(checkpoint.is_relative_to(run_dir / "artifacts"), "stored checkpoint outside artifacts")
                _require(sha256_file(checkpoint) == previous["checkpoint_sha256"],
                         "stored checkpoint SHA-256 mismatch")
            _require(previous["status"] in ("TRAINED", "FAILED", "CANCELLED"), "invalid stored status")
            return previous
        start = time.monotonic()
        status = {"status": "TRAINING", "job_sha256": checked["job_sha256"],
                  "target": job["target"], "elapsed_seconds": 0.0, "reason": None,
                  "checkpoint": None, "checkpoint_sha256": None,
                  "config_sha256": job["config"]["sha256"],
                  "projected_config_sha256": hashlib.sha256(_canonical(checked["config"])).hexdigest(),
                  "vendor_sources_sha256": hashlib.sha256(_canonical(job["vendor"]["files"])).hexdigest(),
                  "parent_weights_sha256": job.get("parent_weights", {}).get("sha256"),
                  "myo_weights_sha256": job.get("myo_weights", {}).get("sha256")}
        process, group = None, None
        result_file = run_dir / "worker_result.json"
        try:
            _require(all(p.name in (".lock", "PAUSE", "CANCEL") for p in run_dir.iterdir()),
                     "orphan run files; use new output_dir")
            _atomic_json(status_file, status)
            reason = _control_reason(job, run_dir)
            if reason:
                status.update(status="CANCELLED", reason=reason)
            else:
                payload = _payload(checked)
                _atomic_json(run_dir / "worker_input.json", payload)
                command = [sys.executable, "-B", str(Path(__file__).resolve().parents[1] /
                           "scripts" / "train_segmentation_candidate.py"), "--worker", "--execute"]
                env = os.environ.copy()
                env["PYTHONDONTWRITEBYTECODE"] = "1"
                env["CMR_LGE_MPLCONFIGDIR"] = str(run_dir / "mpl-cache")
                if os.name == "nt":
                    group = _WindowsJob()
                with (run_dir / "worker.log").open("wb") as log:
                    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=log, stderr=log,
                                               cwd=str(run_dir), env=env, shell=False,
                                               start_new_session=os.name != "nt",
                                               creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                    if group is not None:
                        group.attach(process)
                    # Worker reads only this filename; it is already attached before it can import vendor.
                    process.stdin.write((str(run_dir / "worker_input.json") + "\n").encode("utf-8"))
                    process.stdin.flush()
                    last_update = start
                    while True:
                        now = time.monotonic()
                        reason = _control_reason(job, run_dir)
                        if reason:
                            status.update(status="CANCELLED", reason=reason)
                            break
                        if now - start >= job["resources"]["max_seconds"]:
                            status.update(status="FAILED", reason="deadline_exceeded")
                            break
                        if result_file.exists():
                            result = read_json(result_file)
                            if result.get("status") != "TRAINED":
                                status.update(status="FAILED", reason=result.get("reason", "worker_failed"))
                            else:
                                status.update(status="TRAINED", checkpoint=result["checkpoint"],
                                              checkpoint_sha256=result["checkpoint_sha256"])
                            break
                        if process.poll() is not None:
                            status.update(status="FAILED", reason="worker_exit_" + str(process.returncode))
                            break
                        if now - last_update >= 1:
                            status["elapsed_seconds"] = now - start
                            _atomic_json(status_file, status)
                            last_update = now
                        time.sleep(0.05)
        except KeyboardInterrupt:
            status.update(status="CANCELLED", reason="keyboard_interrupt")
        except Exception as exc:
            status.update(status="FAILED", reason=type(exc).__name__ + ": " + str(exc))
        finally:
            try:
                _stop_group(process, group)
                if status["status"] == "TRAINED":
                    checkpoint = _absolute(status["checkpoint"])
                    _require(checkpoint.is_relative_to(run_dir / "artifacts") and checkpoint.is_file()
                             and checkpoint.stat().st_size > 0, "invalid checkpoint location/file")
                    _require(sha256_file(checkpoint) == _sha(status["checkpoint_sha256"]),
                             "checkpoint SHA-256 mismatch")
            except Exception as exc:
                status.update(status="FAILED", reason="cleanup/result: " + str(exc))
            if status["status"] != "TRAINED":
                status.update(checkpoint=None, checkpoint_sha256=None)
            status["elapsed_seconds"] = time.monotonic() - start
            _atomic_json(status_file, status)
        return status


def _snapshot(binding, destination):
    """Read the exact bytes hashed, then vendor reads that private snapshot."""
    source = _absolute(binding["path"])
    expected = _sha(binding["sha256"])
    destination = _absolute(str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with source.open("rb") as src, destination.open("xb") as dst:
        _require(stat.S_ISREG(os.fstat(src.fileno()).st_mode), "expected regular source")
        for block in iter(lambda: src.read(_CHUNK), b""):
            digest.update(block)
            dst.write(block)
    _require(digest.hexdigest() == expected, "snapshot SHA-256 mismatch")
    return {"path": str(destination), "sha256": expected}


def _train_vendor(payload):
    """Internal worker only: import fixed existing trainer functions, never its CLI."""
    run_dir, target = Path(payload["run_dir"]), payload["target"]
    vendor = payload["vendor"]
    snapshots = []
    vendor_dir = run_dir / "vendor"
    for relative, digest in vendor["files"].items():
        snapshots.append(_snapshot({"path": str(Path(vendor["root"]) / _relative(relative)),
                                    "sha256": digest}, vendor_dir / relative))
    rows = {}
    for split in ("train", "validation"):
        rows[split] = copy.deepcopy(payload[split])
        for index, row in enumerate(rows[split]):
            for kind in ("image", "label"):
                row[kind] = _snapshot(row[kind], run_dir / "inputs" / split / str(index) /
                                      kind / Path(row[kind]["path"]).name)
                snapshots.append(row[kind])
    weights = {}
    for key in ("parent_weights", "myo_weights"):
        if key in payload:
            weights[key] = _snapshot(payload[key], run_dir / "inputs" / key / Path(payload[key]["path"]).name)
            snapshots.append(weights[key])
    sys.path.insert(0, str(vendor_dir))
    spec = importlib.util.spec_from_file_location("_segmentation_private_vendor", vendor_dir / VENDORS[target][1])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    record_module = importlib.import_module(VENDORS[target][2])
    config = copy.deepcopy(payload["config"])
    if target == "lge_mi":
        config["myocardium_model"] = {"weights": weights["myo_weights"]["path"],
                                      "sha256": weights["myo_weights"]["sha256"]}
    artifacts = run_dir / "artifacts"
    artifacts.mkdir()
    _atomic_json(run_dir / "effective_config.json", config)
    module.seed_everything(config["training"]["seed"])
    device = module.resolve_device(payload["device"])
    records = {}
    for split, items in rows.items():
        records[split] = []
        for row in items:
            args = {k: row[k] for k in ("center", "patient", "relative_path")}
            args.update(image_path=row["image"]["path"], label_path=row["label"]["path"])
            if target == "cmr_myo":
                image, _, _ = module.load_image_hwt(args["image_path"])
                frames = int(image.shape[2])  # HWT is guaranteed by the vendor loader.
                _positive_int(frames)
                args.update(label_policy="all", labeled_frames=tuple(range(frames)), total_frames=frames)
                record = record_module.SupervisedVolumeRecord(**args)
            else:
                _, nifti = module.load_image_hw(args["image_path"])
                args["native_shape"] = tuple(nifti.shape)
                record = record_module.LGERecord(**args)
            records[split].append(record)
    model = (module.build_mi_model if target == "lge_mi" else module.build_model)(config, device)
    if "parent_weights" in weights:
        loader = module.load_mi_checkpoint if target == "lge_mi" else module.load_checkpoint
        loader(weights["parent_weights"]["path"], model, device)
    if target == "cmr_myo":
        train = module.labeled_loader(records["train"], config, device, shuffle=True, include_original=False)
        val = module.labeled_loader(records["validation"], config, device, shuffle=False, include_original=True)
        checkpoint, _ = module.train_model(model, train, val, device, config, artifacts)
    elif target == "lge_myo":
        checkpoint = module.train_model(model, module.train_loader(records["train"], config, device),
                                        module.validation_loader(records["validation"], config, device),
                                        device, config, artifacts)
    else:
        myo_model, _ = module.load_frozen_myo_model(weights["myo_weights"]["path"], device)
        checkpoint = module.train_model(model, myo_model,
                                        module.training_loader(records["train"], config, device),
                                        module.validation_loader(records["validation"], config, device),
                                        device, config, artifacts)
    for binding in snapshots:
        _file(binding)
    checkpoint = _absolute(str(checkpoint))
    _require(checkpoint.is_relative_to(artifacts) and checkpoint.is_file()
             and checkpoint.stat().st_size > 0, "vendor returned invalid checkpoint")
    return {"status": "TRAINED", "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint)}


def worker_main():
    """Private bootstrap. Block until parent has attached process-tree ownership."""
    payload_path = sys.stdin.buffer.readline().decode("utf-8").rstrip("\r\n")
    payload = read_json(payload_path)
    try:
        result = _train_vendor(payload)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        result = {"status": "FAILED", "reason": type(exc).__name__ + ": " + str(exc)}
    _atomic_json(Path(payload["run_dir"]) / "worker_result.json", result)
    # Keep the group leader alive until the controller consumes the receipt and
    # kills the entire group, including any lingering DataLoader workers.
    sys.stdin.buffer.read(1)
