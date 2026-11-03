"""Small cached board projection; no filesystem-wide scans on HTTP requests."""
from __future__ import annotations
import json
from pathlib import Path


def status(root: Path) -> dict:
    summary_path = root / "runtime/dataset_release/summary.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        summary = {"status": "NOT_AUDITED", "summary": {}}
    # Deliberately expose only aggregate counters, never manifest rows/source paths.
    result = {"dataset_status": summary.get("status"),
              "generated_at": summary.get("generated_at"),
              "summary": summary.get("summary", {}),
              "evidence_sources": summary.get("evidence_sources", {}),
              "frozen_patient_count": summary.get("frozen_patient_count"),
              "release": summary.get("release"), "models": {},
              "training": {"status": "NOT_STARTED"}}
    # Keep dispatch readiness visible without implying that training is enabled.
    from app.segmentation_dispatch import readiness
    result["dispatch_readiness"] = readiness(root)
    from app.dataset_control import status as dataset_status
    result["dataset_job"] = dataset_status(root)
    registry = root / "runtime/segmentation_registry"
    for target in ("cmr_myo", "lge_myo", "lge_mi"):
        # No invented active version when an installation still uses legacy weights.
        result["models"][target] = {"status": "LEGACY_FIXED_WEIGHTS"}
        if registry.is_dir():
            from app.segmentation_registry import describe_active
            try:
                active = describe_active(registry, target)
                if active:
                    result["models"][target] = {"status": "SELECTED", "version": active["candidate_id"],
                        "kind": active["kind"], "dataset_sha256": active["dataset_sha256"]}
            except (OSError, ValueError, RuntimeError):
                result["models"][target] = {"status": "REGISTRY_ERROR"}
    latest = root / "runtime/segmentation_training/latest.json"
    if latest.exists():
        try:
            job = json.loads(latest.read_text(encoding="utf-8"))
            result["training"] = {k: job[k] for k in ("status", "target", "elapsed_seconds", "job_id", "error_code") if k in job}
        except (OSError, ValueError):
            result["training"] = {"status": "UNREADABLE"}
    return result
