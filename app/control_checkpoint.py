"""Recoverable code checkpoints. Production labels are outside this module."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


SOURCE_DIRS = {"app", "agents", "contracts", "providers", "scripts", "tests",
               "experiments", "configs", "static", "deploy"}


def digest(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def save_journal(path: Path, value: dict) -> None:
    """Replace journals atomically and durably, preserving the prior record."""
    temporary = path.with_suffix(".pending")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(value, output, indent=2)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)
    if os.name != "nt":
        descriptor = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def source_path(root: Path, relative: str) -> Path:
    rel = PurePosixPath(relative)
    if (rel.is_absolute() or "\\" in relative or ":" in relative
            or any(part in {"", ".", ".."} for part in rel.parts)
            or (len(rel.parts) > 1 and rel.parts[0] not in SOURCE_DIRS)
            or (len(rel.parts) == 1 and rel.suffix not in {".md", ".py", ".toml", ".txt"})):
        raise ValueError("not a control-plane source path")
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink():
        raise ValueError("path leaves checkpoint root")
    if "__pycache__" in rel.parts or path.suffix in {".pyc", ".env"}:
        raise ValueError("not a source file")
    return path


def create_checkpoint(root: Path, destination: Path) -> dict:
    root, destination = root.resolve(), destination.resolve()
    if destination.is_relative_to(root):
        raise ValueError("checkpoint must be outside repository")
    destination.mkdir(parents=True, exist_ok=False, mode=0o700)
    copied = {}
    paths = [path for path in root.iterdir() if path.is_file()]
    for folder in sorted(SOURCE_DIRS):
        if (root / folder).is_dir():
            paths.extend((root / folder).rglob("*"))
    for path in sorted(paths):
        # Only selected control-plane folders; no runtime, images or labels.
        rel = path.relative_to(root).as_posix()
        if not path.is_file() or path.is_symlink():
            continue
        try:
            source_path(root, rel)
        except ValueError:
            continue
        if path.suffix == ".sqlite" or path.suffix == ".jsonl":
            continue
        target = destination / "source" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied[rel] = digest(target)
        if digest(path) != copied[rel]:
            raise RuntimeError("source changed during checkpoint: " + rel)
    db_path = root / "runtime/workflow.sqlite"
    database = None
    if db_path.is_file():
        database = destination / "workflow.sqlite"
        with closing(sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)) as source:
            with closing(sqlite3.connect(str(database))) as target:
                source.backup(target)
                if target.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                    raise RuntimeError("database checkpoint integrity failed")
    def git(*args):
        proc = subprocess.run(["git", *args], cwd=root, text=True, capture_output=True)
        if proc.returncode:
            raise RuntimeError(proc.stderr[-500:])
        return proc.stdout
    manifest = {"schema_version": "control-checkpoint-v1", "root": str(root),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "code_commit": git("rev-parse", "HEAD").strip(),
                "files": copied, "database_sha256": digest(database) if database else None,
                "database_restore": "manual disaster recovery only; normal code rollback keeps live DB"}
    (destination / "git-status.txt").write_text(git("status", "--short", "-uno"), encoding="utf-8")
    (destination / "tracked.patch").write_text(git("diff", "--binary", "HEAD"), encoding="utf-8")
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"checkpoint": str(destination), "files": len(copied),
            "database_saved": bool(database), "code_commit": manifest["code_commit"]}


def restoration_plan(root: Path, checkpoint: Path, installed: dict[str, str | None]) -> list[dict]:
    """Compare-and-swap restoration of only this release's touched files."""
    manifest = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
    if Path(manifest["root"]).resolve() != root.resolve():
        raise ValueError("checkpoint belongs to another repository")
    plan = []
    for relative, installed_hash in installed.items():
        target = source_path(root, relative)
        old_hash = manifest["files"].get(relative)
        backup = source_path(checkpoint / "source", relative)
        if old_hash and digest(backup) != old_hash:
            raise ValueError("checkpoint content corrupted: " + relative)
        current = digest(target)
        if current == old_hash:
            operation = "unchanged"
        elif current != installed_hash:
            operation = "conflict"
        else:
            operation = "restore" if old_hash else "remove_release_file"
        plan.append({"file": relative, "operation": operation, "before": current,
                     "after": old_hash, "installed": installed_hash})
    return plan


def restore_release(root: Path, checkpoint: Path, installed: dict[str, str | None], *,
                    execute: bool = False) -> dict:
    plan = restoration_plan(root, checkpoint, installed)
    if any(item["operation"] == "conflict" for item in plan):
        return {"status": "CONFLICT", "plan": plan, "executed": False}
    if execute:
        # Check the whole plan before writing, and retain displaced release
        # files so a failed/manual rollback can itself be recovered.
        recovery = checkpoint / ("rollback_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f"))
        for item in plan:
            if item["operation"] == "unchanged":
                continue
            target = source_path(root, item["file"])
            if digest(target) != item["before"]:
                raise RuntimeError("concurrent edit during rollback")
            displaced = recovery / item["file"]
            displaced.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                shutil.copy2(target, displaced)
        recovery.mkdir(parents=True, exist_ok=True)
        journal = {"status": "PREPARED", "root": str(root), "checkpoint": str(checkpoint),
                   "plan": plan, "completed": []}
        journal_path = recovery / "journal.json"
        save_journal(journal_path, journal)
        try:
            for item in plan:
                if item["operation"] == "unchanged":
                    continue
                target = source_path(root, item["file"])
                if digest(target) != item["before"]:
                    raise RuntimeError("concurrent edit during rollback")
                if item["operation"] == "remove_release_file":
                    target.unlink()
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temporary = target.with_suffix(target.suffix + ".rollback")
                    shutil.copy2(checkpoint / "source" / item["file"], temporary)
                    os.replace(temporary, target)
                journal["completed"].append(item["file"])
                save_journal(journal_path, journal)
            journal["status"] = "RESTORED"
        except Exception:
            failures = []
            for item in reversed(plan):
                if item["file"] not in journal["completed"]:
                    continue
                target = source_path(root, item["file"])
                try:
                    if digest(target) != item["after"]:
                        raise RuntimeError("concurrent edit during compensation")
                    displaced = recovery / item["file"]
                    if item["before"] is None:
                        target.unlink()
                    else:
                        shutil.copy2(displaced, target)
                except Exception as exc:
                    failures.append({"file": item["file"], "error": str(exc)})
            journal.update(status="RECOVERY_REQUIRED" if failures else "COMPENSATED", failures=failures)
            save_journal(journal_path, journal)
            raise
        save_journal(journal_path, journal)
    return {"status": "RESTORED" if execute else "PREVIEW", "plan": plan, "executed": execute}


def recover_rollback(root: Path, journal_path: Path, *, execute: bool = False) -> dict:
    """Replay an interrupted rollback from before/after hashes, not a counter."""
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    if Path(journal["root"]).resolve() != root.resolve():
        raise ValueError("rollback journal belongs to another repository")
    checkpoint = Path(journal["checkpoint"])
    installed = {item["file"]: item["installed"] for item in journal["plan"]}
    # restoration_plan accepts either the installed state or the original
    # checkpoint state, so it is safe even in the replace -> journal gap.
    result = restore_release(root, checkpoint, installed, execute=execute)
    if execute and result["status"] == "RESTORED":
        journal["status"] = "RECOVERED"
        save_journal(journal_path, journal)
    return result
