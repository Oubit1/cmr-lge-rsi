#!/usr/bin/env python3
"""Watch CMR check changes, refresh evidence, optionally apply eligible candidates.

The maintenance agents learn from the same state through autopilot. The
optional automatic apply path is bounded, audited, and skips the holdout.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agents.evidence import (approved_manual_edit_records,
                             approved_manual_review_records)  # noqa: E402
from app.check_validation import freeze_plan  # noqa: E402
from scripts.run_lge_feedback_consumer import run_once as run_lge_consumer_once  # noqa: E402

import sys as _path_sys
_path_sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import paths as _public_paths
DATASET = _public_paths.DATASET_ROOT
STATE = ROOT / "runtime/check_iterations/watch_state.json"

PIPELINE_FILES = (
    "scripts/run_check_iteration.py", "scripts/batch_screen.py", "scripts/rules_v2.py",
    "scripts/validate_batch_candidates.py", "scripts/evaluate_pending_check.py",
    "scripts/shadow_spatial_boundary.py", "app/manual_annotation.py",
    "app/quality_scoring.py", "app/check_auto_apply.py",
    "artifacts/image_label_quality_v1/calibration_profile.json",
)


def pipeline_fingerprint(root: Path) -> str:
    """Invalidate results after a change to code actually consumed by check.

Unconnected experimental code is deliberately absent: archiving an experiment
does not activate it as a production rule.
"""
    digest = hashlib.sha256()
    for relative in PIPELINE_FILES:
        path = root / relative
        digest.update(relative.encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest()
                      if path.is_file() else b"missing")
    return digest.hexdigest()


def fingerprint(dataset: Path) -> str:
    digest = hashlib.sha256()
    digest.update(pipeline_fingerprint(ROOT).encode())
    check = dataset / "labels/check"
    label_root = dataset / "labels/cmr_checked/cmr_Myo_mask"
    for state in ("待筛", "待处理", "checked", "垃圾桶"):
        for png in sorted((check / state / "cmr").glob("*/*.png")):
            digest.update(png.relative_to(check).as_posix().encode())
            if state == "待筛":
                label = label_root / png.parent.name / (png.stem + ".nii.gz")
                if label.exists():
                    stat = label.stat()
                    digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
    approved = approved_manual_edit_records(ROOT)
    digest.update(hashlib.sha256(json.dumps(
        approved, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).digest())
    reviews = approved_manual_review_records(dataset)
    digest.update(hashlib.sha256(json.dumps(
        reviews, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).digest())
    return digest.hexdigest()


def write_state(value: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(STATE)


def run_current_auto_apply(dataset: Path, runtime: Path, *, execute: bool,
                           limit: int = 10) -> dict:
    """Execute the currently promoted module, not the service-start snapshot."""
    importlib.reload(importlib.import_module("app.quality_scoring"))
    module = importlib.import_module("app.check_auto_apply")
    module = importlib.reload(module)
    return module.run(dataset, runtime, execute=execute, limit=limit)


def tick(dataset: Path, *, timeout: int = 900) -> bool:
    current = fingerprint(dataset)
    old = json.loads(STATE.read_text()) if STATE.exists() else {}
    if old.get("successful_fingerprint") == current:
        return False
    result = subprocess.run([sys.executable, str(ROOT / "scripts/run_check_iteration.py"),
                             "--dataset", str(dataset)], cwd=ROOT, text=True,
                            capture_output=True, timeout=timeout, check=False)
    stamp = datetime.now().astimezone().isoformat()
    if result.returncode:
        write_state({**old, "last_attempt": stamp, "last_error": result.stderr[-2000:]})
        raise RuntimeError(f"check iteration failed: {result.stderr[-500:]}")
    after = fingerprint(dataset)
    if after != current:
        write_state({**old, "last_attempt": stamp,
                     "last_error": "source state changed during iteration; rescan required"})
        return True
    write_state({"successful_fingerprint": current, "last_success": stamp,
                 "last_error": None})
    print(json.dumps({"event": "CHECK_ITERATION_COMPLETE", "at": stamp,
                      "fingerprint": current[:12]}, ensure_ascii=False), flush=True)
    return True


class ScorerReloadRequired(RuntimeError):
    pass


def run_consumers(dataset: Path, *, auto_apply: bool, scorer_boot_digest: str | None = None) -> None:
    from app.rsi_postpromotion import consumer_gate, installation_lock
    from app.autonomy_control import operator_paused
    if auto_apply and operator_paused(ROOT):
        return
    with installation_lock(ROOT, exclusive=False):
        install = consumer_gate(ROOT)
        if not install["ready"]:
            raise RuntimeError("SCORING_INSTALL_PENDING: " + str(install.get("reason") or install["status"]))
        if scorer_boot_digest is not None:
            from app.lge_execution import scorer_fingerprint
            if scorer_fingerprint(ROOT) != scorer_boot_digest:
                raise ScorerReloadRequired("scorer_or_dependency_version_changed")
        modern_lge = (ROOT / "configs/lge_execution.json").exists()
        if modern_lge:
            lge_result = run_lge_consumer_once(ROOT, dataset, execute=auto_apply, source_locked=True)
            if lge_result.get("status") == "PUBLISHED":
                print(json.dumps(lge_result, ensure_ascii=False), flush=True)
            from agents.evidence import dependent_workflow_status
            if dependent_workflow_status(dataset, ROOT)["stage"] != "cmr_paired":
                return
        tick(dataset)
        if auto_apply:
            freeze_plan(dataset, ROOT / "runtime")
            result = run_current_auto_apply(dataset, ROOT / "runtime", execute=True, limit=10)
            if result["applied"]:
                print(json.dumps({"event": "CHECK_AUTO_APPLIED", "items": result["applied"]},
                                 ensure_ascii=False), flush=True)
        if not modern_lge:
            lge_result = run_lge_consumer_once(ROOT, dataset, execute=auto_apply, source_locked=True)
            if lge_result.get("status") == "PUBLISHED":
                print(json.dumps(lge_result, ensure_ascii=False), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DATASET)
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--auto-apply", action="store_true",
                    help="Apply up to ten eligible non-holdout candidate volumes per tick")
    args = ap.parse_args()
    from app.lge_execution import scorer_fingerprint
    scorer_boot_digest = scorer_fingerprint(ROOT)
    from app.autonomy_control import heartbeat, operator_paused
    while True:
        try:
            if args.auto_apply:
                if operator_paused(ROOT):
                    heartbeat(ROOT, "PAUSED", execute=True)
                    if args.once:
                        break
                    time.sleep(args.interval)
                    continue
                heartbeat(ROOT, "RUNNING", execute=True)
            run_consumers(args.dataset, auto_apply=args.auto_apply, scorer_boot_digest=scorer_boot_digest)
            if args.auto_apply:
                heartbeat(ROOT, "READY", execute=True)
        except ScorerReloadRequired:
            # Re-exec the same service process only after the installation read
            # lock has closed. Helpers and package initializers cannot remain
            # cached from the previous accepted scorer generation.
            print(json.dumps({"event": "SCORER_GENERATION_RELOAD"}), flush=True)
            os.execv(sys.executable, [sys.executable, "-u", *sys.argv])
        except Exception as exc:
            if args.auto_apply:
                heartbeat(ROOT, "FAILED", execute=True, error=str(exc))
            print(json.dumps({"event": "CHECK_ITERATION_ERROR", "error": str(exc)},
                             ensure_ascii=False), flush=True)
            if args.once:
                raise
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
