"""Controller-owned scoring-code installation verification; never writes labels.

An interrupted install leaves a durable consumer gate. Recovery verifies the
installed source, or restores only that release's source files. This is not a
clinical canary or a rollback of patient data.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import subprocess
import os
from contextlib import contextmanager
from pathlib import Path

from app.control_checkpoint import digest, restore_release, save_journal, source_path
from app.rsi_kernel import canonical_hash


@contextmanager
def installation_lock(root: Path, *, exclusive: bool):
    """Serialize source replacement against the entire consumer operation."""
    lock = root / "runtime/rsi/installation.lock"
    with file_lock(lock, exclusive=exclusive):
        yield


@contextmanager
def file_lock(lock: Path, *, exclusive: bool):
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open('a+b') as handle:
        if os.name == 'nt':
            import msvcrt
            handle.seek(0, 2)
            if handle.tell() == 0:
                handle.write(b'0')
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:
            yield
        finally:
            if os.name == 'nt':
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def gate_path(root: Path) -> Path:
    return root / "runtime/rsi/install_gate.json"


def consumer_gate(root: Path) -> dict:
    path = gate_path(root)
    if not path.exists():
        return {"ready": True, "status": "NOT_REQUIRED"}
    try:
        gate = json.loads(path.read_text(encoding="utf-8"))
        return {"ready": gate["status"] in {"VERIFIED", "ROLLED_BACK"},
                "status": gate["status"], "reason": gate.get("reason"),
                "action_id": gate.get("action_id")}
    except (OSError, ValueError, TypeError, KeyError):
        return {"ready": False, "status": "RECOVERY_REQUIRED", "reason": "invalid_install_gate"}


def begin_install(root: Path, workspace: Path, run_id: int, action_id: str) -> bool:
    selection_path = workspace / "rsi_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8")) if selection_path.exists() else {}
    if selection.get("scope") != "quality_scorer_all_targets" or selection.get("mode") != "enforce":
        return False
    if selection.get("decision") != "PROMOTE" or selection.get("blocks_promotion"):
        raise ValueError("unapproved_scoring_install")
    if not consumer_gate(root)["ready"]:
        raise ValueError("previous_install_not_verified")
    gate = {"status": "INSTALLING", "workspace": str(workspace.resolve()),
            "run_id": run_id, "action_id": action_id,
            "selection_sha256": canonical_hash(selection)}
    gate_path(root).parent.mkdir(parents=True, exist_ok=True)
    save_journal(gate_path(root), gate)
    return True


def verify_installed(root: Path, workspace: Path, selection: dict) -> dict:
    from app.rsi_runtime import freeze_parent, paired_evaluation, selection_summary
    from experiments.llm_maintenance import _copy_source
    # Pin the original parent, then evaluate the actually installed tree in a
    # fresh workspace. The Owner's accepted report cannot supply fresh scores.
    with tempfile.TemporaryDirectory(prefix="postinstall-", dir=str(root / "runtime/rsi")) as tmp:
        check = Path(tmp)
        shutil.copytree(workspace / "rsi_parent/orchestrator", check / "orchestrator",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        freeze_parent(check, root)
        installed = (check / "orchestrator").resolve()
        if not installed.is_relative_to((root / "runtime/rsi").resolve()):
            raise ValueError("postinstall_workspace_outside_runtime")
        shutil.rmtree(installed)
        _copy_source(root, check / "orchestrator")
        report = selection_summary(paired_evaluation(root, check, "lge_myo",
                                                     changed_files=["app/quality_scoring.py"]))
    if report.get("benchmark_hash") != selection.get("benchmark_hash"):
        return {"status": "RECOVERY_REQUIRED", "reason": "postinstall_benchmark_changed", "report": report}
    if report.get("baseline_metrics") != selection.get("baseline_metrics"):
        return {"status": "RECOVERY_REQUIRED", "reason": "postinstall_parent_environment_changed", "report": report}
    if report.get("decision") == "PROMOTE" and report.get("candidate_metrics") == selection.get("candidate_metrics"):
        return {"status": "VERIFIED", "report": report}
    return {"status": "REGRESSED", "reason": "installed_scores_differ_from_accepted", "report": report}


def rollback_source(root: Path, artifact: Path, manifest: dict) -> dict:
    """Reuse crash-replayable CAS rollback; no label/DB restoration."""
    checkpoint = artifact / "postinstall_checkpoint"
    files, installed = {}, {}
    for item in manifest["promoted"]:
        if item["owner"] != "orchestrator":
            raise ValueError("postinstall_mixed_owner_requires_recovery")
        relative = item["relative"]
        source_path(root, relative)
        files[relative], installed[relative] = item["old_hash"], item["new_hash"]
        if item["old_hash"]:
            backup = Path(item["backup"]).resolve()
            if not backup.is_relative_to((artifact / "backup").resolve()) or digest(backup) != item["old_hash"]:
                raise ValueError("postinstall_backup_changed")
            dest = source_path(checkpoint / "source", relative)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(backup, dest)
    checkpoint.mkdir(parents=True, exist_ok=True)
    save_journal(checkpoint / "manifest.json", {"root": str(root.resolve()), "files": files})
    return restore_release(root, checkpoint, installed, execute=True)


def finish_install(root: Path) -> dict:
    """Idempotent recovery before model calls or automatic consumers."""
    path = gate_path(root)
    if not path.exists() or consumer_gate(root)["ready"]:
        return consumer_gate(root)
    try:
        gate = json.loads(path.read_text(encoding="utf-8"))
        workspace = Path(gate["workspace"])
        selection = json.loads((workspace / "rsi_selection.json").read_text(encoding="utf-8"))
        if canonical_hash(selection) != gate["selection_sha256"]:
            raise ValueError("postinstall_selection_changed")
        journal = json.loads((workspace / "rsi_promotion_journal.json").read_text(encoding="utf-8"))
        if (journal.get("run_id"), journal.get("action_id")) != (gate["run_id"], gate["action_id"]):
            raise ValueError("postinstall_promotion_identity_changed")
        artifact = Path(journal["artifact"]).resolve()
        if not artifact.is_relative_to((root / "runtime/promotions").resolve()):
            raise ValueError("postinstall_artifact_outside_root")
        manifest_path = artifact / "promotion_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["status"] == "ROLLED_BACK":
            gate.update(status="ROLLED_BACK", reason=manifest.get("reason"))
        elif gate["status"] == "ROLLING_BACK":
            result = rollback_source(root, artifact, manifest)
            if result["status"] != "RESTORED":
                raise ValueError("postinstall_rollback_conflict")
            manifest.update(status="ROLLED_BACK", reason="postinstall_regression")
            save_journal(manifest_path, manifest)
            gate["status"] = "ROLLED_BACK"
        else:
            if manifest["status"] != "PROMOTED" or not manifest.get("promoted"):
                raise ValueError("postinstall_incomplete_promotion")
            for item in manifest["promoted"]:
                if item["owner"] != "orchestrator" or digest(source_path(root, item["relative"])) != item["new_hash"]:
                    raise ValueError("postinstall_concurrent_source_change")
            result = verify_installed(root, workspace, selection)
            gate.update(status=result["status"], reason=result.get("reason"), report=result.get("report"))
            if result["status"] == "REGRESSED":
                gate["status"] = "ROLLING_BACK"
                save_journal(path, gate)
                restored = rollback_source(root, artifact, manifest)
                if restored["status"] != "RESTORED":
                    raise ValueError("postinstall_rollback_conflict")
                manifest.update(status="ROLLED_BACK", reason="postinstall_regression")
                save_journal(manifest_path, manifest)
                gate["status"] = "ROLLED_BACK"
        save_journal(path, gate)
        save_journal(workspace / "postinstall_verification.json", {
            "status": gate["status"], "reason": gate.get("reason"), "report": gate.get("report")})
    except (OSError, ValueError, TypeError, KeyError, subprocess.TimeoutExpired) as exc:
        # Retain identity and ROLLING_BACK for journal replay on the next tick.
        if 'gate' in locals() and isinstance(gate, dict):
            gate['reason'] = str(exc)[:300]
            save_journal(path, gate)
        return {"ready": False, "status": "RECOVERY_REQUIRED", "reason": str(exc)[:300]}
    return consumer_gate(root)
