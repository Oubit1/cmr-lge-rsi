from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from app.database import Database


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def csv_rows(path: Path) -> int:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return max(sum(1 for _ in csv.reader(handle)) - 1, 0)


def csv_issue_slices(path: Path) -> int:
    """Count unique review slices while preserving multi-issue rows in the CSV."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    identities = set()
    for row in rows:
        check = row.get("check_view") or row.get("check")
        if check:
            identities.add(("check", str(check)))
        else:
            identities.add((str(row.get("patient") or ""),
                            str(row.get("volume") or row.get("slice") or ""),
                            str(row.get("slice_index") or row.get("layer_index") or "")))
    return len(identities)


def collect_qc_artifacts(db: Database, run_id: int, task_id: int, dataset_root: Path) -> dict[str, Any]:
    patterns = {
        "full_pipeline_summary": "label_iteration/full_pipeline_runs/*/summary.json",
        "lge_qc_summary": "label_iteration/qc_runs/*/summary.json",
        "cmr_qc_summary": "label_iteration/cmr_qc_runs/*/summary.json",
        "lge_issue_manifest": "labels/check/manifest_backup/lge_issue_manifest.csv",
        "cmr_middle_manifest": "labels/check/manifest_backup/cmr_middle_manifest.csv",
        "cmr_boundary_manifest": "labels/check/manifest_backup/cmr_boundary_manifest.csv",
        "lge_middle_manifest": "labels/check/manifest_backup/lge_middle_manifest.csv",
        "lge_boundary_manifest": "labels/check/manifest_backup/lge_boundary_manifest.csv",
        "trajectory_manifest": "labels/check/manifest_backup/trajectory_manifest.csv",
    }
    issue_rows = 0
    registered = []
    for artifact_type, pattern in patterns.items():
        matches = sorted(dataset_root.glob(pattern))
        if not matches:
            continue
        path = matches[-1]
        metadata: dict[str, Any] = {"size": path.stat().st_size}
        if path.suffix == ".csv":
            metadata["rows"] = csv_rows(path)
            issue_rows += metadata["rows"]
        elif path.suffix == ".json":
            try:
                metadata["summary"] = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                metadata["summary"] = None
        checksum = sha256_file(path) if path.stat().st_size <= 50 * 1024 * 1024 else None
        db.register_artifact(run_id, task_id, artifact_type, path, metadata, checksum)
        registered.append(str(path))
    return {"registered": registered, "issue_rows": issue_rows}
