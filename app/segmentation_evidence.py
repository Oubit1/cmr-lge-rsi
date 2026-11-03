"""Read existing acceptance ledgers; never infer acceptance from folder names.

Legacy records without source-image binding remain visible as evidence debt.
All identifiers in this module's output are private runtime data.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("invalid_acceptance_record")
    return value


def _lines(path: Path):
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                yield json.loads(line)


def collect_evidence(root: Path, dataset: Path) -> tuple[list[dict], dict]:
    """Normalize only traceable successful transactions / whole-volume reviews.

    CMR old receipts bind labels only; they are not silently upgraded by hashing
    today's image. A current QC job must provide a new, image-bound acceptance.
    Myo's old MI hash may be refreshed only by the separately committed MI
    receipt binding that exact accepted Myo and image (the existing dependency).
    """
    evidence, debt, lge = [], Counter(), {}
    for path in sorted((root / "runtime/lge_execution/transactions").glob("*.json")):
        tx = _read(path)
        if tx.get("state") != "COMMITTED" or Path(tx.get("dataset", "")).resolve() != dataset.resolve():
            continue
        r = tx.get("receipt", {})
        stage, key = tx.get("stage"), tx.get("relative")
        if stage not in ("lge_myo", "lge_mi") or not key or r.get("status") != "CHECKED":
            continue
        if r.get("transaction_id") != tx.get("transaction_id") or r.get("after_hashes", {}).get("label") != tx.get("after_sha256"):
            debt["transaction_receipt_mismatch"] += 1
            continue
        current_path = root / "runtime/lge_execution/receipts" / stage / (hashlib.sha256(
            json.dumps(r["case_id"], sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest() + ".json")
        # The journal alone does not override a newer rejection or re-review.
        if not current_path.exists() or _read(current_path) != r:
            debt["superseded_lge_receipt"] += 1
            continue
        lge[(key, stage)] = {"key": key, "target": stage, "status": "ACCEPTED",
            "scope": "whole_volume", "hashes": dict(r["after_hashes"]),
            "provenance": {"kind": "automatic_qc", "record_id": tx["transaction_id"],
                "calibration_id": r.get("calibration_id"), "score": r.get("score_after"),
                "source": str(path.relative_to(root))}}
    for (key, target), item in lge.items():
        if target == "lge_myo":
            mi = lge.get((key, "lge_mi"))
            if mi and mi["hashes"].get("myo") == item["hashes"].get("label") and mi["hashes"].get("image") == item["hashes"].get("image"):
                item["hashes"]["mi"] = mi["hashes"]["label"]
                item["provenance"]["paired_mi_record_id"] = mi["provenance"]["record_id"]
        evidence.append(item)
    cmr = {}
    for row in _lines(root / "runtime/check_iterations/auto_apply.jsonl"):
        if row.get("event") == "applied" and row.get("key"):
            cmr[row["key"]] = row
    for key, row in cmr.items():
        if row.get("route") != "checked":
            continue
        if not row.get("source_image_sha256"):
            debt["cmr_legacy_image_hash_missing"] += 1
            continue
        evidence.append({"key": key, "target": "cmr_myo", "status": "ACCEPTED",
            "scope": "whole_volume", "hashes": {"image": row["source_image_sha256"], "label": row["source_sha_after"]},
            "provenance": {"kind": "automatic_qc", "record_id": row["timestamp"], "source": "runtime/check_iterations/auto_apply.jsonl"}})
    # Optional common acceptance ledger is written by configured validation jobs.
    # Latest means latest decision, including PENDING/ERROR/INELIGIBLE and
    # NEEDS_REVIEW. Filtering for ACCEPTED here would resurrect superseded success.
    # Saved drafts, single-frame APPLIED edits and PNG membership are insufficient.
    latest = {}
    for row in _lines(root / "runtime/dataset_acceptance/receipts.jsonl"):
        if Path(row.get("dataset", "")).resolve() == dataset.resolve():
            latest[(row.get("key"), row.get("target"))] = row
    overridden = set(latest)
    evidence = [e for e in evidence if (e["key"], e["target"]) not in overridden]
    evidence.extend(latest.values())
    return evidence, {"normalized_acceptances": sum(e.get("status") == "ACCEPTED" for e in evidence),
                      "normalized_decisions": len(evidence), "evidence_debt": dict(debt)}


def frozen_patients(root: Path, dataset: Path, *, case_salt: str | None = None) -> set[str]:
    """Union existing frozen exclusions at patient level, across all modalities."""
    excluded = set()
    cmr = root / "runtime/check_iterations/holdout_v1.json"
    if cmr.exists():
        excluded.update(r["key"].split("/", 1)[0] for r in _read(cmr)["holdout"])
    lge = root / "runtime/lge_execution/holdout.json"
    if lge.exists():
        value = _read(lge)
        if Path(value["dataset"]).resolve() != dataset.resolve():
            raise ValueError("frozen_dataset_mismatch")
        if not case_salt or hashlib.sha256(case_salt.encode()).hexdigest() != value["identity_salt_sha256"]:
            raise ValueError("frozen_patient_identity_unavailable")
        from experiments.runtime_evaluation import _case_id
        protected = set(value["holdout_patients"])
        for parent in (dataset / "images/cmr").iterdir():
            if parent.is_dir() and _case_id(case_salt, str(dataset.resolve()) + ":patient:" + parent.name) in protected:
                excluded.add(parent.name)
    return excluded
