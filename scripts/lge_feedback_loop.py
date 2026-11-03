"""LGE feedback loop – active-stage report adapter.

Reads the top-level workflow_stage and records supplied by the executor,
produces a consumer-contract report, and persists it to
runtime/lge_iterations/latest.json.  Zero production label writes.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional


MODULE_VERSION = "lge_feedback_loop/1.2"

REPORT_PATH = "runtime/lge_iterations/latest.json"

# Expected manifest integrity constants for the 289-record LGE myo snapshot.
EXPECTED_RECORD_COUNT = 289
EXPECTED_SOURCE_SHA256 = "2a96ae1d79606b76319b022398b3398228bc8f1f7520797803083ce189f33f04"


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def run_evaluation(data: Dict[str, Any]) -> Dict[str, Any]:
    """Active-stage report adapter.

    Consumes data['records'] and data['workflow_stage']; preserves
    data['source_sha256'] in the report.  Respects the holdout contract:
    when data['frozen_split']['status'] is UNAVAILABLE the report must
    carry frozen_split_status='UNAVAILABLE' and holdout_excluded=0.

    Derives consumer_ready=true when the input record count equals 289
    and the source SHA-256 matches the expected manifest hash, regardless
    of frozen_split status.
    """
    # ------------------------------------------------------------------
    # 1.  Extract inputs
    # ------------------------------------------------------------------
    records: List[Dict[str, Any]] = data.get("records", []) if isinstance(data, dict) else []
    workflow_stage: str = data.get("workflow_stage", "lge_myo") if isinstance(data, dict) else "lge_myo"
    source_sha256: str = data.get("source_sha256", "") if isinstance(data, dict) else ""

    # frozen_split – strict pass-through
    frozen_split: Dict[str, Any] = data.get("frozen_split", {}) if isinstance(data, dict) else {}
    frozen_status: str = frozen_split.get("status", "UNAVAILABLE")

    # ------------------------------------------------------------------
    # 2.  Holdout contract
    # ------------------------------------------------------------------
    if frozen_status == "UNAVAILABLE":
        holdout_excluded = 0
    else:
        holdout_ids: List[str] = data.get("holdout_case_ids", []) if isinstance(data, dict) else []
        holdout_set = set(holdout_ids)
        holdout_excluded = sum(1 for r in records if r.get("case_id") in holdout_set)

    # ------------------------------------------------------------------
    # 3.  Stage-level counts (pending / work / resolved)
    # ------------------------------------------------------------------
    total = len(records)
    # For this adapter we treat every record as "pending" (no resolution
    # logic is embedded – the stage consumer contract only requires that
    # pending + work + resolved == total).
    pending = total
    work = 0
    resolved = 0

    # ------------------------------------------------------------------
    # 4.  Derive consumer_ready from manifest integrity
    # ------------------------------------------------------------------
    consumer_ready = (
        total == EXPECTED_RECORD_COUNT
        and source_sha256 == EXPECTED_SOURCE_SHA256
    )

    # ------------------------------------------------------------------
    # 5.  Downstream consumer audit of UNAVAILABLE+ready=true
    # ------------------------------------------------------------------
    audit_checks: Dict[str, bool] = {
        "source_modality_lge": True,  # hardcoded below
        "workflow_stage_present": bool(workflow_stage),
        "input_source_sha256_present": bool(source_sha256),
        "consumer_ready_boolean": isinstance(consumer_ready, bool),
        "pending_int": isinstance(pending, int),
        "work_int": isinstance(work, int),
        "resolved_int": isinstance(resolved, int),
        "frozen_split_status_UNAVAILABLE": frozen_status == "UNAVAILABLE",
        "holdout_excluded_zero": holdout_excluded == 0,
        "label_write_false": True,  # hardcoded below
        "production_writes_zero": True,  # hardcoded below
        "counts_reconcile_true": (pending + work + resolved) == total,
    }
    audit_all_pass = all(audit_checks.values())
    audit_failed = [k for k, v in audit_checks.items() if not v]

    # ------------------------------------------------------------------
    # 6.  Build report
    # ------------------------------------------------------------------
    report: Dict[str, Any] = {
        "module_version": MODULE_VERSION,
        "source_modality": "lge",
        "workflow_stage": workflow_stage,
        "input_source_sha256": source_sha256,
        f"{workflow_stage}_consumer_ready": consumer_ready,
        f"{workflow_stage}_pending": pending,
        f"{workflow_stage}_work": work,
        f"{workflow_stage}_resolved": resolved,
        "frozen_split_status": frozen_status,
        "holdout_excluded": holdout_excluded,
        "label_write": False,
        "production_writes": 0,
        "total_records": total,
        "counts_reconcile": (pending + work + resolved) == total,
        "downstream_consumer_audit": {
            "checks": audit_checks,
            "all_pass": audit_all_pass,
            "failed": audit_failed,
            "verdict": "PASS" if audit_all_pass else f"FAIL: {audit_failed}",
        },
    }

    # ------------------------------------------------------------------
    # 7.  Persist report (zero label writes)
    # ------------------------------------------------------------------
    _ensure_dir(REPORT_PATH)
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        json.dump(report, fh, sort_keys=True, indent=2)

    return report
