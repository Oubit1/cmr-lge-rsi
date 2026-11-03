"""Explicit, asynchronous read-only dataset jobs, independent of paid pause.

Board contract: ``launch(trusted_root, trusted_dataset, action)`` and
``status(trusted_root)``. Never populate either path from an HTTP payload.
Only launch accepts an action; status never retries or starts work. The private
current pointer, per-job journal and log live in runtime/dataset_control. A
detached supervisor enforces the deadline even if the board exits, using its
owned child handle (never killing a numeric PID obtained from a stale journal).
"""
from __future__ import annotations

import argparse
import errno
import json
import math
import os
import re
import stat
import subprocess
import sys
import threading
import time
import traceback
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from app.control_checkpoint import save_journal

ACTIONS = frozenset({"audit", "validate", "freeze"})
WALLTIME_SECONDS = 1800
VALIDATION_LIMIT = 100
_ID = re.compile(r"\d{8}T\d{12}Z_[a-f0-9]{16}\Z")
_STATES = {"STARTING", "RUNNING", "SUCCEEDED", "FAILED", "TIMED_OUT"}
_ACTIVE = {"STARTING", "RUNNING"}
_PHASES = {"QUEUED", "STARTING", "AUDITING", "VALIDATING", "REFRESHING_AUDIT", "FREEZING", "COMPLETE"}
_ERRORS = {"BUSY", "INVALID_ACTION", "INVALID_CONFIGURATION", "JOURNAL_UNREADABLE",
           "LAUNCH_FAILED", "WORKER_EXITED", "PROCESS_UNVERIFIABLE", "SUPERVISOR_EXITED",
           "AUDIT_REQUIRED", "OPERATION_FAILED", "TIME_LIMIT"}
_AUDIT_CORE = ("schema", "dataset", "pairing", "evidence", "excluded_patients",
               "issues", "rows", "summary", "audit_sha256")
_COUNTS = ("accepted_pairs", "needs_review_pairs", "accepted_targets",
           "existing_review_or_exclusion", "structural_or_frozen_exclusion",
           "unchanged_already_validated", "partial_volume_exclusion")


class _Busy(Exception):
    pass


class _AuditRequired(Exception):
    pass


def _stamp():
    return datetime.now(timezone.utc).isoformat()


def _safe_path(path):
    """Reject links/junctions before resolving, including nonexistent leaves."""
    path = Path(os.path.abspath(path))
    for part in reversed((path,) + tuple(path.parents)):
        if os.path.lexists(part):
            info = part.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                raise ValueError("unsafe_path")
    return path


def _root(root):
    root = _safe_path(root)
    if not root.is_dir():
        raise ValueError("missing_root")
    return root


def _folder(root):
    return _safe_path(root / "runtime/dataset_control")


def _configuration(root, dataset):
    root, dataset = _root(root), _root(dataset)
    for relative in ("dataset_control", "dataset_release", "dataset_releases", "dataset_acceptance"):
        output = _safe_path(root / "runtime" / relative)
        if output == dataset or dataset in output.parents or output in dataset.parents:
            raise ValueError("output_overlaps_dataset")
    return root, dataset


def _private_dir(path):
    _safe_path(path).mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name != "nt":
        path.chmod(0o700)


def _read(path):
    _safe_path(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("invalid_journal")
    return value


def _save(path, value):
    _safe_path(path)
    _safe_path(path.with_suffix(".pending"))
    save_journal(path, value)
    if os.name != "nt":
        path.chmod(0o600)


@contextmanager
def _lock(folder):
    # Nonblocking on HTTP paths. Workers retry only these short metadata locks.
    path = _safe_path(folder / "control.lock")
    with path.open("a+b") as handle:
        if path.stat().st_size == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                raise _Busy() from exc
            raise
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _process_start(pid):
    """Process birth identity, including boot identity on Linux; zombies are dead.

    Missing process -> None; permission/inspection failure -> exception. Unknown
    liveness must block a new launch rather than risk two writers.
    """
    if type(pid) is not int or pid <= 0:
        raise ValueError("invalid_pid")
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        api = ctypes.WinDLL("kernel32", use_last_error=True)
        api.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        api.OpenProcess.restype = wintypes.HANDLE
        api.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
        api.GetProcessTimes.restype = wintypes.BOOL
        api.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        api.GetExitCodeProcess.restype = wintypes.BOOL
        api.CloseHandle.argtypes = [wintypes.HANDLE]
        api.CloseHandle.restype = wintypes.BOOL
        handle = api.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
        if not handle:
            error = ctypes.get_last_error()
            if error == 87:  # ERROR_INVALID_PARAMETER: PID no longer exists
                return None
            raise ctypes.WinError(error)
        try:
            code = wintypes.DWORD()
            if not api.GetExitCodeProcess(handle, ctypes.byref(code)):
                raise ctypes.WinError(ctypes.get_last_error())
            if code.value != 259:  # STILL_ACTIVE
                return None
            created, exited, kernel, user = (wintypes.FILETIME() for _ in range(4))
            if not api.GetProcessTimes(handle, *map(ctypes.byref, (created, exited, kernel, user))):
                raise ctypes.WinError(ctypes.get_last_error())
            return "windows:" + str((created.dwHighDateTime << 32) | created.dwLowDateTime)
        finally:
            api.CloseHandle(handle)
    try:
        fields = (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
    except (FileNotFoundError, ProcessLookupError):
        return None
    if fields[0] in {"Z", "X"}:
        return None
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    return "linux:" + boot + ":" + fields[19]


def _identity(pid):
    start = _process_start(pid)
    if not start:
        raise ProcessLookupError("process_exited")
    return {"pid": pid, "start": start}


def _alive(identity):
    if identity is None:
        return False
    if not isinstance(identity, dict) or not isinstance(identity.get("start"), str):
        raise ValueError("invalid_process_identity")
    return _process_start(identity["pid"]) == identity["start"]


def _job_path(folder, job_id):
    if not isinstance(job_id, str) or not _ID.fullmatch(job_id):
        raise ValueError("invalid_job_id")
    return _safe_path(folder / "jobs" / job_id / "job.json")


def _load_job(folder, job_id):
    job = _read(_job_path(folder, job_id))
    if (job.get("schema") != 1 or job.get("job_id") != job_id
            or job.get("action") not in ACTIONS or job.get("status") not in _STATES
            or type(job.get("created_epoch")) not in (int, float)
            or not math.isfinite(job["created_epoch"])
            or type(job.get("walltime")) is not int or not 1 <= job["walltime"] <= WALLTIME_SECONDS
            or not isinstance(job.get("dataset"), str)):
        raise ValueError("invalid_job")
    return job


def _current(folder):
    pointer = folder / "current.json"
    if not pointer.exists():
        return None
    return _load_job(folder, _read(pointer)["job_id"])


def _finish(folder, job, status, error=None):
    job.update(status=status, error_code=error, finished_at=_stamp(), finished_epoch=time.time())
    if status == "SUCCEEDED":
        job["phase"] = "COMPLETE"
    _save(_job_path(folder, job["job_id"]), job)


def _reconcile(folder, job):
    supervisor, executor = _alive(job.get("supervisor")), _alive(job.get("executor"))
    if job["status"] in _ACTIVE and not supervisor:
        code = "SUPERVISOR_EXITED" if executor else "WORKER_EXITED"
        timed_out = time.time() >= job["created_epoch"] + job["walltime"]
        _finish(folder, job, "TIMED_OUT" if timed_out else "FAILED", "TIME_LIMIT" if timed_out else code)
    return supervisor or executor


def _integers(value, keys):
    if not isinstance(value, dict):
        return {}
    return {key: value[key] for key in keys if type(value.get(key)) is int and value[key] >= 0}


def _public_result(value):
    # An allowlist at the API boundary, even for apparently aggregate journals.
    if not isinstance(value, dict):
        return {}
    result = {}
    if isinstance(value.get("audit"), dict):
        result["audit"] = _integers(value["audit"], ("keys", "complete_files", "releasable_pairs", "blocked_pairs", "excluded_pairs"))
    if isinstance(value.get("validation"), dict):
        item = value["validation"]
        result["validation"] = {**_integers(item, ("attempted",)), "counts": _integers(item.get("counts"), _COUNTS)}
    release = value.get("release")
    if isinstance(release, dict) and (_ID.fullmatch(str(release.get("release_id", "")))
                                    or re.fullmatch(r"[a-f0-9]{64}", str(release.get("release_id", "")))):
        result["release"] = {"release_id": release["release_id"], **_integers(release, ("pair_count",))}
        digest = release.get("manifest_sha256")
        if isinstance(digest, str) and re.fullmatch(r"[a-f0-9]{64}", digest):
            result["release"]["manifest_sha256"] = digest
        try:
            created = datetime.fromisoformat(release["created_at"])
            if created.tzinfo is not None:
                result["release"]["created_at"] = created.isoformat()
        except (KeyError, ValueError, TypeError):
            pass
    if type(value.get("release_created_by_current_audit")) is bool:
        result["release_created_by_current_audit"] = value["release_created_by_current_audit"]
    return result


def _response(job=None, *, busy=False, error=None):
    result = {"status": "IDLE", "job_id": None, "action": None, "busy": busy,
              "progress": {"phase": "IDLE"}, "error_code": error, "result": {}}
    if job is None:
        if error:
            result["status"] = "BUSY" if error == "BUSY" else "FAILED"
        return result
    result.update(status=job["status"], job_id=job["job_id"], action=job["action"],
                  error_code=error or (job.get("error_code") if job.get("error_code") in _ERRORS else None),
                  progress={"phase": job.get("phase") if job.get("phase") in _PHASES else "STARTING"},
                  result=_public_result(job.get("result")))
    end = job.get("finished_epoch", time.time())
    if type(end) in (int, float) and math.isfinite(end):
        result["elapsed_seconds"] = max(0, round(end - job["created_epoch"], 1))
    return result


def status(root: Path) -> dict:
    """Small cached projection. No dataset scan, retry, provider or process spawn."""
    try:
        folder = _folder(_root(root))
        if not folder.exists():
            return _response()
        with _lock(folder):
            job = _current(folder)
            if job is None:
                return _response()
            try:
                busy = _reconcile(folder, job)
            except (OSError, ValueError, KeyError, IndexError):
                return _response(job, busy=True, error="PROCESS_UNVERIFIABLE")
            return _response(job, busy=busy)
    except _Busy:
        return _response(busy=True, error="BUSY")
    except (OSError, ValueError, KeyError, TypeError):
        return _response(busy=True, error="JOURNAL_UNREADABLE")


def _popen_options():
    options = {"stdin": subprocess.DEVNULL, "stderr": subprocess.STDOUT, "close_fds": True}
    if os.name == "nt":
        options["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        options["start_new_session"] = True
    return options


def _command(mode, root, dataset, action, job_id):
    return [sys.executable, "-u", "-m", "app.dataset_control", mode, str(root), str(dataset), action, job_id]


def _reap(process):
    # Reap detached children in a long-lived board without waiting on HTTP.
    threading.Thread(target=process.wait, daemon=True, name="dataset-reaper").start()


def launch(root: Path, dataset: Path, action: str, *, walltime: int = WALLTIME_SECONDS,
           popen=None) -> dict:
    """Launch one explicit job. Paths and optional limits are trusted configuration.

    accepted=False with BUSY means another job/metadata update owns the lock.
    Failures remain terminal until a subsequent explicit launch; no auto retry.
    """
    if not isinstance(action, str) or action not in ACTIONS:
        return {**_response(error="INVALID_ACTION"), "accepted": False}
    try:
        if type(walltime) is not int or not 1 <= walltime <= WALLTIME_SECONDS:
            raise ValueError("invalid_walltime")
        root, dataset = _configuration(root, dataset)
        folder = _folder(root)
        _private_dir(folder)
    except (OSError, ValueError, TypeError):
        return {**_response(error="INVALID_CONFIGURATION"), "accepted": False}
    try:
        with _lock(folder):
            previous = _current(folder)
            if previous:
                try:
                    busy = _reconcile(folder, previous)
                except (OSError, ValueError, KeyError, IndexError):
                    return {**_response(previous, busy=True, error="PROCESS_UNVERIFIABLE"), "accepted": False}
                if busy:
                    return {**_response(previous, busy=True, error="BUSY"), "accepted": False}
            job_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ_") + uuid.uuid4().hex[:16]
            path = _job_path(folder, job_id)
            path.parent.mkdir(parents=True, mode=0o700, exist_ok=False)
            job = {"schema": 1, "job_id": job_id, "action": action, "dataset": str(dataset),
                   "status": "STARTING", "phase": "QUEUED", "created_epoch": time.time(),
                   "created_at": _stamp(), "walltime": walltime, "supervisor": None,
                   "executor": None, "error_code": None, "result": {}}
            _save(path, job)
            _save(folder / "current.json", {"job_id": job_id})
            process = None
            try:
                with (path.parent / "worker.log").open("xb") as log:
                    if os.name != "nt":
                        os.chmod(log.name, 0o600)
                    process = (popen or subprocess.Popen)(
                        _command("--worker", root, dataset, action, job_id), cwd=Path(__file__).resolve().parents[1],
                        stdout=log, **_popen_options())
                job["supervisor"] = _identity(process.pid)
                _save(path, job)
            except Exception:
                # A child cannot pass its journal gate before this lock is released.
                if process is not None:
                    try:
                        process.kill()  # exact owned handle, not journal PID
                    except OSError:
                        pass
                    _reap(process)
                with (path.parent / "worker.log").open("a", encoding="utf-8") as log:
                    traceback.print_exc(file=log)
                _finish(folder, job, "FAILED", "LAUNCH_FAILED")
                return {**_response(job), "accepted": False}
            _reap(process)
            return {**_response(job, busy=True), "accepted": True}
    except _Busy:
        return {**_response(busy=True, error="BUSY"), "accepted": False}
    except (OSError, ValueError, KeyError, TypeError):
        return {**_response(busy=True, error="JOURNAL_UNREADABLE"), "accepted": False}


def _update(folder, job_id, change):
    # Only worker paths wait; locks never cover auditing/scoring/copying.
    until = time.monotonic() + 10
    while True:
        try:
            with _lock(folder):
                job = _load_job(folder, job_id)
                return change(job)
        except _Busy:
            if time.monotonic() >= until:
                raise
            time.sleep(0.02)


def _gate(folder, dataset, action, job_id, role):
    def check(job):
        if (job["dataset"] != str(dataset) or job["action"] != action
                or job["status"] not in _ACTIVE or job.get(role) != _identity(os.getpid())):
            raise ValueError("worker_identity_mismatch")
        return job
    return _update(folder, job_id, check)


def _phase(folder, job_id, phase):
    def change(job):
        if job["status"] not in _ACTIVE:
            raise RuntimeError("job_no_longer_active")
        job.update(status="RUNNING", phase=phase)
        _save(_job_path(folder, job_id), job)
    _update(folder, job_id, change)


def _perform(root, dataset, action, job_id, progress):
    """Domain policy stays entirely in the existing validator/release modules."""
    from scripts.dataset_release import run
    result = {}
    if action == "validate":
        from app.dataset_validation import validate_batch
        progress("VALIDATING")
        cached = root / "runtime/dataset_release/audit.json"
        if not cached.is_file():
            raise _AuditRequired()
        report = _read(cached)
        # Strip only run()'s known envelope, leaving unknown fields to fail the
        # validator's digest check. Never recompute or repair the cached digest.
        audit = dict(report)
        for key in ("generated_at", "evidence_sources", "frozen_patient_count"):
            audit.pop(key, None)
        validation = validate_batch(root, dataset, audit, limit=VALIDATION_LIMIT)
        result["validation"] = validation
        # Refresh dashboard totals from the new receipts before reporting success.
        # This full scan shares the worker's existing overall wall-time budget.
        progress("REFRESHING_AUDIT")
        summary = run(root, dataset)
    elif action == "freeze":
        progress("FREEZING")
        parent = root / "runtime/dataset_releases"
        _private_dir(parent)
        destination = _safe_path(parent / job_id)
        if os.path.lexists(destination):
            raise FileExistsError("release_exists")
        # run() always audits the entire dataset afresh before freeze_release's
        # independent full recheck; never freeze the cached audit.
        summary = run(root, dataset, destination)
    elif action == "audit":
        progress("AUDITING")
        summary = run(root, dataset)
    else:
        raise ValueError("invalid_action")
    result["audit"] = summary["summary"]
    result["release"] = summary.get("release")
    result["release_created_by_current_audit"] = summary.get("release_created_by_current_audit")
    return _public_result(result)


def _execute(root, dataset, action, job_id):
    root, dataset = _configuration(root, dataset)
    folder = _folder(root)
    job = _gate(folder, dataset, action, job_id, "executor")
    # Also bound an orphan executor if its supervisor is forcibly killed.
    remaining = max(0, min(job["walltime"], job["created_epoch"] + job["walltime"] - time.time()))
    watchdog = threading.Timer(remaining, lambda: os._exit(124))
    watchdog.daemon = True
    watchdog.start()
    try:
        for relative in ("dataset_release", "dataset_acceptance"):
            _private_dir(root / "runtime" / relative)
        result = _perform(root, dataset, action, job_id, lambda phase: _phase(folder, job_id, phase))
        def complete(current):
            if current["status"] in _ACTIVE:
                current["result"] = result
                _finish(folder, current, "SUCCEEDED")
        _update(folder, job_id, complete)
        return 0
    except Exception as exc:
        traceback.print_exc()  # redirected to the private log, never the API
        def failed(current):
            if current["status"] in _ACTIVE:
                _finish(folder, current, "FAILED", "AUDIT_REQUIRED" if isinstance(exc, _AuditRequired) else "OPERATION_FAILED")
        _update(folder, job_id, failed)
        return 1
    finally:
        watchdog.cancel()


def _worker(root, dataset, action, job_id):
    root, dataset = _configuration(root, dataset)
    folder = _folder(root)
    job = _gate(folder, dataset, action, job_id, "supervisor")
    process = None
    try:
        def start(current):
            nonlocal process
            if current["status"] not in _ACTIVE:
                raise RuntimeError("job_no_longer_active")
            with (_job_path(folder, job_id).parent / "worker.log").open("ab") as log:
                process = subprocess.Popen(_command("--execute", root, dataset, action, job_id),
                    cwd=Path(__file__).resolve().parents[1], stdout=log, **_popen_options())
            current.update(executor=_identity(process.pid), status="RUNNING", phase="STARTING")
            _save(_job_path(folder, job_id), current)
        _update(folder, job_id, start)
        remaining = max(0, min(job["walltime"], job["created_epoch"] + job["walltime"] - time.time()))
        timed_out = False
        try:
            code = process.wait(timeout=remaining)
            timed_out = code == 124
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            process.wait()
        def exited(current):
            if timed_out:
                _finish(folder, current, "TIMED_OUT", "TIME_LIMIT")
            elif current["status"] in _ACTIVE:
                _finish(folder, current, "FAILED", "WORKER_EXITED")
        _update(folder, job_id, exited)
        return 124 if timed_out else code
    except Exception:
        traceback.print_exc()
        if process is not None:
            try:
                process.kill()
            except OSError:
                pass
            process.wait()
        def failed(current):
            if current["status"] in _ACTIVE:
                _finish(folder, current, "FAILED", "LAUNCH_FAILED")
        _update(folder, job_id, failed)
        return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--worker", action="store_true")
    mode.add_argument("--execute", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("root", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("action", choices=sorted(ACTIONS))
    parser.add_argument("job_id")
    args = parser.parse_args()
    if os.name != "nt":
        os.umask(0o077)
    return (_worker if args.worker else _execute)(args.root, args.dataset, args.action, args.job_id)


if __name__ == "__main__":
    raise SystemExit(main())
