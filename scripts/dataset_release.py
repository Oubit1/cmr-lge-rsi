#!/usr/bin/env python3
"""Audit paired acceptance and optionally freeze a private, immutable release."""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.control_checkpoint import save_journal
from app.paired_dataset import audit_pairs, freeze_release
from app.segmentation_evidence import collect_evidence, frozen_patients


def _read_object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _release_receipt(value: object) -> dict | None:
    """Only aggregate, path-free metadata; legacy timestamps remain unknown."""
    if not isinstance(value, dict):
        return None
    digest = value.get("manifest_sha256", value.get("manifest_sha"))
    count = value.get("pair_count")
    if (not isinstance(digest, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", digest)
            or type(count) is not int or count <= 0):
        return None
    created_at = value.get("created_at")
    try:
        if not isinstance(created_at, str) or datetime.fromisoformat(created_at).tzinfo is None:
            created_at = None
    except ValueError:
        created_at = None
    # Content identity does not depend on a potentially identifying folder name.
    return {"release_id": digest.lower(), "created_at": created_at,
            "manifest_sha256": digest.lower(), "pair_count": count}


def _latest_release(private: Path) -> dict | None:
    receipt_path = private / "latest_release.json"
    try:
        value = json.loads(receipt_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        pass
    except (OSError, ValueError) as exc:
        raise ValueError("latest_release_receipt_unreadable") from exc
    else:
        receipt = _release_receipt(value)
        if receipt is None:
            raise ValueError("invalid_latest_release_receipt")
        return receipt
    # Only a missing receipt permits legacy migration. An unreadable newer
    # receipt must never be overwritten by a potentially stale audit summary.
    receipt = _release_receipt(_read_object(private / "summary.json").get("release"))
    if receipt is not None:
        save_journal(receipt_path, receipt)
    return receipt


def run(root: Path, dataset: Path, destination: Path | None = None) -> dict:
    evidence, sources = collect_evidence(root, dataset)
    exclusions = frozen_patients(root, dataset, case_salt=os.environ.get("CMR_LGE_CASE_SALT"))
    audit = audit_pairs(dataset, evidence, excluded_patients=exclusions)
    stamp = datetime.now(timezone.utc).isoformat()
    report = {**audit, "generated_at": stamp, "evidence_sources": sources,
              "frozen_patient_count": len(exclusions)}
    private = root / "runtime/dataset_release"
    private.mkdir(parents=True, exist_ok=True, mode=0o700)
    save_journal(private / "audit.json", report)
    summary = {"generated_at": stamp, "summary": audit.get("summary", {}),
               "evidence_sources": sources, "frozen_patient_count": len(exclusions),
               "status": "AUDITED", "release": _latest_release(private),
               "release_created_by_current_audit": False}
    # A later audit (even a failed freeze attempt) must not imply that it created
    # or revalidated the prior export. Its counters describe the live dataset.
    save_journal(private / "summary.json", summary)
    if destination is not None:
        release = freeze_release(dataset, audit, destination)
        receipt = _release_receipt({"manifest_sha256": release["manifest_sha256"],
                                    "pair_count": release["releasable_pairs"],
                                    "created_at": datetime.now(timezone.utc).isoformat()})
        if receipt is None:
            raise ValueError("invalid_successful_release_receipt")
        # Commit export history first, so an interrupted summary refresh can be
        # recovered by the next audit without scanning private release folders.
        save_journal(private / "latest_release.json", receipt)
        summary["release"] = receipt
        summary["status"] = "RELEASED"
        summary["release_created_by_current_audit"] = True
        save_journal(private / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, help="New private output directory; never overwrites an existing release")
    args = parser.parse_args()
    print(json.dumps(run(args.project_root.resolve(), args.dataset.resolve(), args.freeze), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
