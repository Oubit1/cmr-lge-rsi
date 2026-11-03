from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.artifact_registry import csv_rows, sha256_file
from app.database import Database
from app.state_machine import RunState


MANIFESTS = {
    "lge_issue_manifest": "labels/check/manifest_backup/lge_issue_manifest.csv",
    "cmr_middle_manifest": "labels/check/manifest_backup/cmr_middle_manifest.csv",
    "cmr_boundary_manifest": "labels/check/manifest_backup/cmr_boundary_manifest.csv",
    "lge_middle_manifest": "labels/check/manifest_backup/lge_middle_manifest.csv",
    "lge_boundary_manifest": "labels/check/manifest_backup/lge_boundary_manifest.csv",
    "trajectory_manifest": "labels/check/manifest_backup/trajectory_manifest.csv",
}


def snapshot_existing_qc(db: Database, project_root: Path, dataset_root: Path,
                         source_run_id: int | None = None,
                         iteration_history: list[dict[str, Any]] | None = None,
                         role_continuity: dict[str, Any] | None = None) -> dict[str, Any]:
    dataset_root = dataset_root.resolve()
    dataset_id = db.register_dataset(dataset_root.parent.name or dataset_root.name, dataset_root)
    config = {
        "dataset_root": str(dataset_root),
        "source_run_id": source_run_id,
        "input_mode": "existing_qc_snapshot",
        "production_writes": False,
        "iteration_history": iteration_history or [],
        "role_continuity": role_continuity or {},
    }
    run_id = db.create_run(dataset_id, "analysis", config)
    snapshot = project_root / "artifacts" / f"run_{run_id:06d}" / "evidence_snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)

    registered: list[dict[str, Any]] = []
    for artifact_type, relative in MANIFESTS.items():
        source = dataset_root / relative
        if not source.is_file():
            continue
        target = snapshot / f"{artifact_type}.csv"
        shutil.copy2(source, target)
        rows = csv_rows(target)
        metadata = {
            "rows": rows,
            "size": target.stat().st_size,
            "snapshot": True,
            "source_relative_path": relative,
        }
        db.register_artifact(run_id, None, artifact_type, target, metadata, sha256_file(target))
        registered.append({"artifact_type": artifact_type, "rows": rows, "path": str(target)})

    if not registered:
        raise RuntimeError(f"no issue manifests found under {dataset_root / 'labels/check'}")

    manifest = {
        "analysis_run_id": run_id,
        "source_run_id": source_run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "artifacts": registered,
    }
    (snapshot / "snapshot.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    db.transition(run_id, RunState.EVIDENCE_READY, current_step="existing_qc_snapshotted")
    return manifest
