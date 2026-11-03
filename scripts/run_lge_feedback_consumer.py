#!/usr/bin/env python3
"""Run the promoted LGE consumer and publish only a validated audit report."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.evidence import dependent_workflow_status  # noqa: E402
from experiments.runtime_evaluation import evaluation_input  # noqa: E402

import sys as _path_sys
_path_sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import paths as _public_paths
DATASET = _public_paths.DATASET_ROOT
INPUT_CONTRACT_VERSION = "lge-active-input-contract-v1"


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _audit(record: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def input_contract(data: dict[str, Any]) -> dict[str, Any]:
    """Publish only aggregate input-schema facts for owners and reviewers."""
    availability = data.get("field_availability")
    return {
        "version": INPUT_CONTRACT_VERSION,
        "schema": data.get("schema"),
        "workflow_stage": data.get("workflow_stage"),
        "record_count": len(data.get("records", [])),
        "field_availability": availability if isinstance(availability, dict) else {},
        "missing_values_must_not_be_imputed": True,
        "issue_codes_are_existing_qc_evidence_not_truth": True,
    }


def validate_report(data: dict[str, Any], report: Any, stage: str) -> dict[str, Any]:
    if not isinstance(report, dict):
        raise ValueError("run_evaluation must return an object")
    prefix = "lge_myo" if stage == "lge_myo" else "lge_mi"
    names = [f"{prefix}_pending", f"{prefix}_work", f"{prefix}_resolved"]
    counts = [report.get(name) for name in names]
    checks = {
        "source_modality": report.get("source_modality") == "lge",
        "workflow_stage": report.get("workflow_stage") == stage,
        "source_sha256": report.get("input_source_sha256") == data.get("source_sha256"),
        "consumer_ready": isinstance(report.get(f"{prefix}_consumer_ready"), bool),
        "counts": all(isinstance(value, int) and value >= 0 for value in counts),
        "reconciled": all(isinstance(value, int) for value in counts)
                      and sum(counts) == len(data.get("records", [])),
        "zero_write": report.get("label_write") is False
                      and report.get("production_writes") == 0,
    }
    if (data.get("frozen_split") or {}).get("status") == "UNAVAILABLE":
        checks["holdout_unavailable"] = (
            report.get("frozen_split_status") == "UNAVAILABLE"
            and report.get("holdout_excluded") == 0
        )
    if not all(checks.values()):
        raise ValueError("invalid LGE consumer report: " + ", ".join(
            name for name, passed in checks.items() if not passed
        ))
    return dict(report)


def run_once(root: Path = ROOT, dataset: Path = DATASET,
             output: Path | None = None, audit: Path | None = None, *, execute: bool = False,
             source_locked: bool = False) -> dict[str, Any]:
    output = output or root / "runtime/lge_iterations/latest.json"
    audit = audit or root / "runtime/lge_iterations/consumer_audit.jsonl"
    if (root / "configs/lge_execution.json").exists():
        from contextlib import nullcontext
        from app.rsi_postpromotion import installation_lock
        with (nullcontext() if source_locked else installation_lock(root, exclusive=False)):
            return run_execution(root, dataset, output, audit, execute=execute)
    stage = dependent_workflow_status(dataset, root).get("stage")
    if stage not in {"lge_myo", "lge_mi"}:
        return {"status": "SKIPPED", "reason": "stage_not_lge", "workflow_stage": stage}
    module_path = root / "scripts/lge_feedback_loop.py"
    if not module_path.is_file():
        return {"status": "SKIPPED", "reason": "consumer_not_promoted",
                "workflow_stage": stage}
    data = evaluation_input(root, workflow_stage=stage)
    if data.get("status") != "AVAILABLE" or not data.get("records"):
        raise RuntimeError("active-stage LGE evaluation input is unavailable or empty")
    module_sha256 = hashlib.sha256(module_path.read_bytes()).hexdigest()
    contract = input_contract(data)
    try:
        current = json.loads(output.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        current = {}
    if (isinstance(current, dict) and current.get("consumer_status") == "VALIDATED"
            and current.get("workflow_stage") == stage
            and current.get("input_source_sha256") == data.get("source_sha256")
            and current.get("consumer_module_sha256") == module_sha256
            and current.get("input_contract_version") == INPUT_CONTRACT_VERSION):
        return {"status": "UNCHANGED", "workflow_stage": stage}
    module = importlib.import_module("scripts.lge_feedback_loop")
    module = importlib.reload(module)
    entrypoint = getattr(module, "run_evaluation", None)
    if not callable(entrypoint):
        raise RuntimeError("promoted LGE consumer has no run_evaluation(data) entrypoint")
    report = validate_report(data, entrypoint(data), stage)
    previous = current if isinstance(current, dict) else {}
    published_at = datetime.now().astimezone().isoformat()
    published = {**previous, **report, "consumer_status": "VALIDATED",
                 "consumer_module_sha256": module_sha256,
                 "input_contract_version": INPUT_CONTRACT_VERSION,
                 "active_input_contract": contract,
                 "published_at": published_at}
    _atomic_json(output, published)
    event = {"event": "LGE_CONSUMER_PUBLISHED", "workflow_stage": stage,
             "input_source_sha256": data.get("source_sha256"),
             "input_contract_version": INPUT_CONTRACT_VERSION,
             "published_at": published_at}
    _audit(event, audit)
    return {"status": "PUBLISHED", **event}


def run_execution(root: Path, dataset: Path, output: Path, audit_path: Path, *, execute: bool) -> dict:
    from app.lge_execution import run_batch, SCHEMA
    previous = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {}
    if previous.get("schema_version") != SCHEMA:
        previous = {}  # Aggregate adapter claims cannot become apply receipts.
    statuses, inputs = {}, {}
    for stage in ("lge_myo", "lge_mi"):
        data = evaluation_input(root, workflow_stage=stage)
        inputs[stage] = data
        if data.get("status") != "AVAILABLE":
            raise ValueError("lge_execution_input_unavailable:" + stage)
        statuses[stage] = run_batch(root, dataset, stage, data, status_only=True)
    # Global Myo residuals cannot starve unrelated MI work. Each MI application
    # still independently validates its own paired Myo under the existing policy.
    ready=[name for name in ("lge_myo","lge_mi")
           if statuses[name].get(name+"_dispatchable",statuses[name].get(name+"_auto_pending",0))>0
           and statuses[name].get(name+"_consumer_ready")]
    previous_stage=(previous.get("last_execution") or {}).get("workflow_stage")
    stage=next((name for name in ready if name!=previous_stage),ready[0] if ready else None)
    outcomes = None
    if stage is not None:
        activity_path=root/"runtime/control/active_consumer_task.json"
        activity={"status":"RUNNING","workflow_stage":stage,"phase":"validated_candidate_batch",
                  "started_at":datetime.now().astimezone().isoformat(),"limit":10}
        if execute:_atomic_json(activity_path,activity)
        try:
            outcomes = run_batch(root, dataset, stage, inputs[stage], execute=execute)
        except Exception as exc:
            if execute:_atomic_json(activity_path,{**activity,"status":"FAILED","error_type":type(exc).__name__})
            raise
        if execute:
            _atomic_json(activity_path,{**activity,"status":"READY","finished_at":datetime.now().astimezone().isoformat(),
                "attempted":len(outcomes.get("outcomes",[])),
                "accepted":sum(r.get("status")=="CHECKED" for r in outcomes.get("outcomes",[])),
                "label_writes":outcomes.get("production_writes",0)})
        # Derive final counts from receipts, never preview plans or generator prose.
        for name in statuses:
            statuses[name] = run_batch(root, dataset, name, inputs[name], status_only=True)
    if stage is None:
        stage=next((name for name in ("lge_myo","lge_mi") if not statuses[name][name+"_auto_complete"]),None)
    published = {**previous, "schema_version": SCHEMA, "consumer_status": "VALIDATED",
                 "source_modality": "lge", "workflow_stage": stage or "paired_selection",
                 "audited_execution": True, "published_at": datetime.now().astimezone().isoformat(),
                 "execution_status": statuses, "last_execution": outcomes}
    published["production_writes"] = outcomes.get("production_writes", 0) if outcomes else 0
    published["label_write"] = published["production_writes"] > 0
    for name, report in statuses.items():
        for field in ("pending", "work", "resolved", "consumer_ready", "auto_pending", "auto_complete", "dispatchable"):
            published[name + "_" + field] = report.get(name + "_" + field,0)
    revision = hashlib.sha256(json.dumps({name: row["receipt_revision"] for name, row in statuses.items()},
                                        sort_keys=True).encode()).hexdigest()
    if previous.get("execution_revision") != revision:
        published["pairing_ready"] = False
        published["paired_slice_count"] = 0
    published["execution_revision"] = revision
    if execute and outcomes:
        from app.autonomy_feedback import record_execution
        published["experience"] = record_execution(root, outcomes)
    _atomic_json(output, published)
    _audit({"event": "LGE_EXECUTION_PUBLISHED", "workflow_stage": stage,
            "execute": execute, "execution_revision": revision,
            "production_writes": outcomes.get("production_writes", 0) if outcomes else 0}, audit_path)
    return {"status": "PUBLISHED", "workflow_stage": stage or "paired_selection",
            "execution_revision": revision,
            "production_writes": outcomes.get("production_writes", 0) if outcomes else 0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--execute", action="store_true", help="Service-only apply of independently validated candidates")
    args = parser.parse_args()
    print(json.dumps(run_once(dataset=args.dataset, execute=args.execute), ensure_ascii=False))


if __name__ == "__main__":
    main()
