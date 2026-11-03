#!/usr/bin/env python3
"""Persistent autonomous analyze -> implement -> validate -> iterate controller.

The daemon never sends source images or identifiers to external providers and
never mutates medical labels directly.  Reviewer-approved maintenance actions
run in disposable snapshots.  Tested code/rule/skill files are promoted with
backups; residual issue manifests are re-snapshotted for the next agent round.
It escalates only after repeated no-progress cycles or an explicit human-review
action whose required clinical evidence cannot be derived automatically.
"""
from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.governance import (agent_action_authorized, implementation_review_approved,
                               implementation_review_payload)  # noqa: E402
from agents.implementation_review import review_implementation  # noqa: E402
from agents.evidence import (action_builds_stage_consumer,
                             action_uses_stale_lge_shadow_contract,
                             approved_manual_edit_records,
                             approved_manual_review_records,
                             action_matches_workflow_stage,
                             dependent_workflow_status,
                             enrich_workflow_dependency,
                             safe_review_feedback,
                             semantic_action_workflow_stage)  # noqa: E402
from agents.shadow_analysis import run_shadow_analysis, stable_hash  # noqa: E402
from app.database import Database  # noqa: E402
from app.artifact_registry import csv_issue_slices  # noqa: E402
from app.existing_qc import MANIFESTS, snapshot_existing_qc  # noqa: E402
from app.agent_continuity import build_agent_continuity  # noqa: E402
from app.rsi_runtime import assess_action, global_promotion_preflight, reconcile_promotion  # noqa: E402
from app.state_machine import RunState  # noqa: E402
from experiments.llm_maintenance import run_llm_maintenance  # noqa: E402
from experiments.runtime_evaluation import evaluation_input, execute_evaluation  # noqa: E402


import sys as _path_sys
_path_sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import paths as _public_paths
PYTHON = _public_paths.PYTHON
SKILL = _public_paths.SKILL_ROOT
TERMINAL_ACTIONS = {"PROMOTED", "PROMOTED_SHADOW", "VALIDATED", "REJECTED", "HUMAN_REQUIRED"}
ISSUE_FINGERPRINT_SCHEMA = "v4-approved-edits-and-audited-lge-execution"


def shadow_only_files(changed_files: list[str]) -> bool:
    """Experiments and tests may be archived without activating a QC route."""
    if not changed_files:
        return False
    for item in changed_files:
        clean = item.removeprefix("orchestrator/")
        parts = clean.split("/")
        if (len(parts) < 2 or parts[0] not in {"experiments", "tests"}
                or any(part in {"", ".", ".."} for part in parts)
                or not clean.endswith(".py") or "\\" in clean):
            return False
    return True
MAINTENANCE_ACTION_CLASSES = {"isolated_code_experiment", "rule_draft", "skill_draft", "framework_repair"}
ORCHESTRATOR_ALLOW = ("agents/", "app/", "contracts/", "experiments/", "providers/",
                      "scripts/", "tests/", "configs/")
SKILL_ALLOW = ("tools/", "scripts/", "references/", "tests/")


def now() -> str:
    return datetime.now().astimezone().isoformat()


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def active_production_blocker(root: Path) -> dict[str, str] | None:
    """Return a measured control-plane blocker, never a medical verdict."""
    report = read_json(root / "runtime/check_iterations/latest.json", {})
    if (report.get("control_gate_status") == "PASS"
            and isinstance(report.get("pending_route_conflicts"), int)
            and report["pending_route_conflicts"] > 0):
        return {
            "target_path": "app/check_auto_apply.py",
            "owner_role": "qc_code_optimizer",
            "required_gate": (
                "route_conflicts_unresolved<pending_route_conflicts and "
                "holdout_selected==0 and preflight_errors==0"),
        }
    if (report.get("control_gate_status") == "FAIL"
            and isinstance(report.get("checked_false_routes"), int)
            and report["checked_false_routes"] > 0
            and isinstance(report.get("spatial_boundary_candidates"), int)
            and report["spatial_boundary_candidates"] > 0):
        return {
            "target_path": "scripts/shadow_spatial_boundary.py",
            "owner_role": "qc_code_optimizer",
            "required_gate": "checked_false_routes==0 and pending_spatial_boundary_candidates>0",
        }
    return None


def production_blocker_deferred(state: dict[str, Any],
                                blocker: dict[str, str] | None) -> bool:
    """Keep one exhausted blocker quarantined while other residual work proceeds."""
    deferred = state.get("deferred_production_blocker")
    return bool(
        blocker
        and isinstance(deferred, dict)
        and deferred.get("target_path") == blocker.get("target_path")
        and deferred.get("issue_fingerprint") == state.get("last_issue_fingerprint")
    )


def check_iteration_matches_queue(root: Path,
                                  issue_counts: dict[str, int]) -> tuple[bool, dict[str, int]]:
    """Prevent paid planning against a pre-routing check-iteration report."""
    report = read_json(root / "runtime/check_iterations/latest.json", {})
    observed = {
        "pending": int(report.get("pending", -1)),
        "work": int(report.get("work", -1)),
    }
    expected = {
        "pending": int(issue_counts.get("check_待筛", -2)),
        "work": int(issue_counts.get("check_待处理", -2)),
    }
    return observed == expected, {**observed,
                                  "expected_pending": expected["pending"],
                                  "expected_work": expected["work"]}


def action_addresses_blocker(payload: dict[str, Any], blocker: dict[str, str]) -> bool:
    """Only the named production-path action may consume tokens while blocked."""
    text = json.dumps(payload, ensure_ascii=False)
    return (payload.get("owner_role") == blocker["owner_role"]
            and payload.get("action_class") == "isolated_code_experiment"
            and blocker["target_path"] in text)


def operator_evidence_retry_pending(payload: dict[str, Any]) -> bool:
    """True only for the structured, unconsumed one-shot operator permit."""
    marker = payload.get("operator_evidence_retry")
    return bool(
        isinstance(marker, dict)
        and str(marker.get("edit_id") or "").strip()
        and str(marker.get("pool") or "").strip()
        and marker.get("consumed") is False
    )


def blocker_supersedes_action(payload: dict[str, Any],
                              blocker: dict[str, str] | None) -> bool:
    """Keep the production blocker, except for one exact operator example retry."""
    return bool(
        blocker
        and not action_addresses_blocker(payload, blocker)
        and not operator_evidence_retry_pending(payload)
    )


def changed_files_include_target(changed_files: list[str], target_path: str) -> bool:
    return target_path in {
        item.replace("\\", "/").removeprefix("orchestrator/").lstrip("./")
        for item in changed_files
    }


def owner_execution_requirements(payload: dict[str, Any],
                                 blocker: dict[str, str] | None) -> dict[str, str]:
    if payload.get("action_class") == "framework_repair":
        return {"required_target_path":"configs/lge_protocol_aliases.json",
                "required_gate":"lge_protocol_contract", "required_interface":"lge-protocol-v1 JSON aliases only",
                "required_edit_scope":"Only configs/lge_protocol_aliases.json and tests/test_lge_protocol_aliases.py; no medical decisions, labels, scorer/writer/evaluator/budget/permission changes."}
    stage = payload.get("workflow_stage") or semantic_action_workflow_stage(payload)
    action_text = json.dumps(payload, ensure_ascii=False).lower()
    if "scripts/lge_candidate_generator.py" in action_text:
        return {"required_target_path": "scripts/lge_candidate_generator.py",
                "required_gate": "lge_candidate_generator_contract + frozen_generator_doctor_reference_challenge",
                "required_interface": "generate_candidates(data: dict, output_dir: str) -> list[dict]",
                "required_edit_scope": "Generate repair/accept_existing/needs_review proposals for supplied case_id records. "
                    "Read only snapshot paths; write candidate binary NIfTI files only under output_dir. "
                    "Repair returns a candidate filename relative to output_dir and a reason. Never write source labels, "
                    "invent scores, hardcode case IDs, import dynamically, or call providers. The controller owns scoring, "
                    "holdout, apply and audit. Reuse existing domain tools and doctor-pair lessons for the algorithm.",
                "failure_if_omitted": "ACTION_MISSED_PRODUCTION_TARGET"}
    if "app/quality_scoring.py" in action_text:
        return {"required_target_path": "app/quality_scoring.py",
                "required_gate": "rsi_quality_all_targets",
                "required_interface": "score_record(record, profile) -> dict containing score",
                "required_edit_scope": "Preserve the fixed benchmark/profile contract and score schema; "
                                       "the configured Owner chooses implementation. All-target paired selection, "
                                       "existing quality gate and full regression are required. Keep calibration "
                                       "and interslice-context builders unchanged under this scoring contract.",
                "failure_if_omitted": "ACTION_MISSED_PRODUCTION_TARGET"}
    if (stage in {"lge_myo", "lge_mi"}
            and "scripts/lge_feedback_loop.py" in action_text):
        return {
            "required_target_path": "scripts/lge_feedback_loop.py",
            "required_gate": "lge_stage_consumer_contract",
            "required_interface": "run_evaluation(data: dict) -> dict",
            "required_edit_scope": (
                "Implement the active-stage report adapter in the named target. Read the "
                "top-level workflow_stage and records supplied by the executor; do not embed "
                "case IDs, invent holdout membership, write runtime state or touch labels."
            ),
            "failure_if_omitted": "ACTION_MISSED_PRODUCTION_TARGET",
        }
    if blocker and action_addresses_blocker(payload, blocker):
        if blocker["target_path"] == "app/check_auto_apply.py":
            return {
                "required_target_path": blocker["target_path"],
                "required_gate": blocker["required_gate"],
                "required_interface": (
                    "Preserve eligible_plan(dataset, runtime), run(..., execute=False) and the "
                    "default-abstaining resolve_route_conflict hook. Preview JSON must report "
                    "route_conflicts_detected/resolved/unresolved and planned records."
                ),
                "required_edit_scope": (
                    "Implement only the route-conflict discriminator in "
                    "resolve_route_conflict. Do not weaken holdout exclusion, candidate byte "
                    "validation, preflight, bounded execution, backups or audit logging."
                ),
                "failure_if_omitted": "ACTION_MISSED_PRODUCTION_TARGET",
            }
        return {
            "required_target_path": blocker["target_path"],
            "required_gate": blocker["required_gate"],
            "required_interface": (
                "Preserve the existing --dataset and --output CLI. --dataset is a DATASET ROOT "
                "DIRECTORY (never a JSON input file); the script must scan that root with its "
                "existing production loaders. A successful invocation must write JSON containing "
                "label_mutations, checked_false_routes, "
                "control_gate, pending_spatial_boundary_candidates and records."
            ),
            "required_edit_scope": (
                "Make a MINIMAL edit to the existing script. Preserve its nibabel/numpy/scipy "
                "filesystem scanner and the natkey, ring_fraction, trusted_core, evaluate and "
                "main functions. Do not replace evaluate(dataset: Path) with a JSON-record "
                "loader. Add only the discriminator needed inside the existing evaluation flow."
            ),
            "failure_if_omitted": "ACTION_MISSED_PRODUCTION_TARGET",
        }
    return {}


def owner_result_ready_for_validation(result: dict[str, Any],
                                      requirements: dict[str, str]) -> bool:
    """Let deterministic gates, not a cautious model boolean, validate output."""
    if result.get("tests_passed") is False:
        return False
    if result.get("ready_for_validation") is True:
        return True
    target = requirements.get("required_target_path")
    changed = result.get("changed_files")
    tests_run = result.get("tests_run")
    return bool(target and isinstance(changed, list)
                and changed_files_include_target(changed, target)
                and any(isinstance(item, str) and item.startswith("tests/")
                        and item.endswith(".py") for item in changed)
                and isinstance(tests_run, list) and tests_run)


def issue_fingerprint(dataset: Path, project_root: Path = ROOT) -> tuple[str, dict[str, int]]:
    digest = hashlib.sha256()
    counts: dict[str, int] = {}
    for name, relative in sorted(MANIFESTS.items()):
        path = dataset / relative
        if not path.is_file():
            continue
        payload = path.read_bytes()
        digest.update(name.encode())
        digest.update(hashlib.sha256(payload).digest())
        counts[name] = csv_issue_slices(path)
    check = dataset / "labels/check"
    # A reviewed work -> trash move changes both the membership and the
    # fingerprint. Valid audited discards count as workflow progress too.
    for state in ("待筛", "待处理", "垃圾桶", "checked"):
        base = check / state / "cmr"
        paths = sorted(base.rglob("*.png")) if base.is_dir() else []
        counts[f"check_{state}"] = len(paths)
        digest.update(state.encode("utf-8"))
        for path in paths:
            digest.update(path.relative_to(check).as_posix().encode("utf-8"))
    # Only an operator-applied edit is new knowledge.  The edit worker may
    # analyze a staged candidate before the operator accepts it, so hashing the
    # raw corpus would either resume too early or fail to notice the later
    # STAGED -> APPLIED transition.  Hash the approved view, which also includes
    # the applied provenance state, and let the normal paused-fingerprint path
    # resume the next bounded agent iteration.
    approved = approved_manual_edit_records(project_root)
    if approved:
        payload = json.dumps(
            approved, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        digest.update(b"approved_manual_edits")
        digest.update(hashlib.sha256(payload).digest())
    counts["analyzed_manual_edits"] = len(approved)
    reviews = approved_manual_review_records(dataset)
    if reviews:
        payload = json.dumps(
            reviews, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        digest.update(b"approved_manual_reviews")
        digest.update(hashlib.sha256(payload).digest())
    counts["confirmed_manual_reviews"] = len(reviews)
    lge = read_json(project_root / "runtime/lge_iterations/latest.json", {})
    if lge.get("schema_version") == "lge-execution-v1" and lge.get("audited_execution") is True:
        digest.update(str(lge.get("execution_revision", "")).encode())
        for stage in ("lge_myo", "lge_mi"):
            for suffix in ("pending", "work", "resolved"):
                name = stage + "_" + suffix
                value = lge.get(name)
                if type(value) is int and value >= 0:
                    counts[name] = value
    return digest.hexdigest(), counts


def rebase_issue_fingerprint_schema(state: dict[str, Any], current_fingerprint: str,
                                    current_counts: dict[str, int]) -> bool:
    """Migrate fingerprint semantics without treating the migration as new evidence."""
    if state.get("issue_fingerprint_schema") == ISSUE_FINGERPRINT_SCHEMA:
        return False
    had_history = bool(
        state.get("last_issue_fingerprint") or state.get("paused_issue_fingerprint")
    )
    was_paused = bool(state.get("paused_issue_fingerprint"))
    state["issue_fingerprint_schema"] = ISSUE_FINGERPRINT_SCHEMA
    if had_history:
        state["last_issue_fingerprint"] = current_fingerprint
        if was_paused:
            state["paused_issue_fingerprint"] = current_fingerprint
        state.setdefault("events", []).append({
            "timestamp": now(),
            "event": "ISSUE_FINGERPRINT_SCHEMA_REBASED",
            "schema": ISSUE_FINGERPRINT_SCHEMA,
            "paused_preserved": was_paused,
            "issue_counts": current_counts,
        })
        state["events"] = state["events"][-200:]
    return True


def historical_no_gain_cycles(events: list[dict[str, Any]],
                              issue_counts: dict[str, int], limit: int) -> int:
    """Recover no-gain state written by controllers predating the hard fuse."""
    matching = 0
    for event in reversed(events):
        if event.get("event") == "FRAMEWORK_FIX_RESUMED":
            break
        if event.get("event") != "NEXT_ITERATION":
            continue
        if event.get("issue_counts") != issue_counts:
            break
        matching += 1
        if matching >= limit:
            break
    return matching


def next_iteration_decision(*, auto_done: bool, stagnant_cycles: int,
                            max_stagnant: int, once: bool) -> str:
    """Choose the controller transition without hiding extra work in ``--once``.

    A one-shot invocation may analyze or process the current run, but it must
    never enqueue another model-analysis round in the same process.
    """
    if not auto_done:
        return "active"
    if stagnant_cycles >= max_stagnant:
        return "escalate"
    if once:
        return "one_shot_complete"
    return "next_iteration"


def production_failure_signature(validation: dict[str, Any]) -> str | None:
    """Identify a new, actionable production-gate defect without path noise.

    A changed failure earns another bounded repair window. Repeating the same
    defect does not, so a broken owner cannot create an infinite paid loop.
    """
    if validation.get("passed") is not False or validation.get("promotion_scope") != "production_path":
        return None
    normalized: list[dict[str, Any]] = []
    gate = validation.get("frozen_evaluation_gate")
    for result in gate.get("results", []) if isinstance(gate, dict) else []:
        if not isinstance(result, dict) or result.get("passed") is not False:
            continue
        command = result.get("command") if isinstance(result.get("command"), dict) else {}
        stderr = str(command.get("stderr") or "")
        exception = next((line.strip() for line in reversed(stderr.splitlines()) if line.strip()), "")
        normalized.append({
            "suite": result.get("suite"),
            "checks": result.get("checks"),
            "returncode": command.get("returncode"),
            "exception": exception,
        })
    if not normalized:
        normalized.append({"reason": validation.get("reason") or "production_validation_failed"})
    payload = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def named_production_contract_passed(validation: dict[str, Any]) -> bool:
    """True when the configured deterministic production acceptance contract passed."""
    if validation.get("passed") is not True or validation.get("promotion_scope") != "production_path":
        return False
    gate = validation.get("frozen_evaluation_gate")
    results = gate.get("results", []) if isinstance(gate, dict) else []
    spatial_passed = any(
        isinstance(result, dict)
        and result.get("suite") == "spatial_boundary_checked_controls"
        and result.get("passed") is True
        and all(result.get("checks", {}).get(name) is True for name in (
            "zero_label_mutations", "checked_false_routes_eq_0",
            "control_gate_pass", "nonempty_pending_coverage",
        ))
        for result in results
    )
    route_passed = any(
        isinstance(result, dict)
        and result.get("suite") == "route_conflict_dry_run"
        and result.get("passed") is True
        and all(result.get("checks", {}).get(name) is True for name in (
            "baseline_conflicts_positive", "conflicts_detected_ge_baseline",
            "at_least_one_conflict_resolved", "unresolved_conflicts_reduced",
            "resolved_route_in_bounded_plan", "preflight_errors_eq_0",
        ))
        for result in results
    )
    lge_consumer_passed = any(
        isinstance(result, dict)
        and result.get("suite") == "lge_stage_consumer_contract"
        and result.get("passed") is True
        and all(result.get("checks", {}).get(name) is True for name in (
            "lge_input", "stage_counts_present", "counts_reconcile",
            "consumer_ready_boolean", "report_modality_exact", "report_stage_exact",
            "report_source_hash_exact", "holdout_contract_exact",
            "no_embedded_case_ids", "zero_label_writes",
        ))
        for result in results
    )
    selection = validation.get("rsi_selection", {})
    quality_passed = (
        selection.get("scope") == "quality_scorer_all_targets"
        and selection.get("decision") == "PROMOTE"
        and bool(selection.get("constraints"))
        and all(value is True for value in selection["constraints"].values())
        and set(selection.get("suites", {})) == {"lge_myo", "lge_mi", "cmr_paired"}
        and validation.get("full_regression_required") is True
        and any(result.get("suite") == "image_label_quality" and result.get("passed") is True
                for result in results if isinstance(result, dict))
    )
    generator_passed = (validation.get("full_regression_required") is True and any(
        item.get("suite") in {"lge_candidate_generator_contract", "lge_protocol_contract"} and item.get("passed") is True
        and item.get("report", {}).get("production_writes") == 0 for item in results))
    return spatial_passed or route_passed or lge_consumer_passed or quality_passed or generator_passed


def compact_owner_report(report: Any, max_bytes: int = 8000) -> dict[str, Any]:
    """Keep aggregate evaluation evidence while excluding per-case payloads.

    Real shadow reports are often larger than the history prompt limit because
    they contain a ``cases`` mapping.  Dropping the whole report leaves the next
    role round blind and encourages it to repeat the same scanner.  Preserve
    top-level scalar metrics and small aggregate dictionaries instead; patient-
    level collections and operational prose remain outside the agent context.
    """
    if not isinstance(report, dict):
        return {"omitted": "owner_report_not_an_object"}
    excluded = {
        "cases", "suppression_candidates", "duplicate_case_ids",
        "no_transition_without_reason", "continuity_reconciliation_differences",
        "audit_diff", "rollback_command", "cohort_note", "no_write_clause",
    }
    compact: dict[str, Any] = {}
    for key, value in report.items():
        if key in excluded:
            continue
        if value is None or isinstance(value, (bool, int, float, str)):
            compact[key] = value[:300] if isinstance(value, str) else value
        elif isinstance(value, dict) and len(value) <= 30 and all(
                item is None or isinstance(item, (bool, int, float, str))
                for item in value.values()):
            compact[key] = value
        encoded = json.dumps(compact, ensure_ascii=False, sort_keys=True)
        if len(encoded.encode("utf-8")) > max_bytes:
            compact.pop(key, None)
            compact["truncated"] = True
            break
    compact["per_case_payload_omitted"] = True
    return compact


def paused_analysis_source(state: dict[str, Any], current_fingerprint: str) -> int:
    """Authorize one explicit analysis-only probe without clearing the fuse."""
    if state.get("status") != "ESCALATION_REQUIRED":
        raise ValueError("analysis-only resume requires ESCALATION_REQUIRED")
    if state.get("paused_issue_fingerprint") != current_fingerprint:
        raise ValueError("paused issue fingerprint changed; inspect current state first")
    source = state.get("escalated_for")
    if not isinstance(source, int) or source < 1:
        raise ValueError("missing escalated source run")
    return source


def maintenance_action_executable(action_class: str) -> bool:
    """Only snapshot-maintenance classes may enter the LLM maintenance executor."""
    return action_class in MAINTENANCE_ACTION_CLASSES


def action_attempts(run_id: int, action_id: str) -> int:
    root = ROOT / "runtime/maintenance" / f"run_{run_id:06d}"
    if not root.exists():
        return 0
    # A written request marks an attempt as started.  Counting interrupted
    # provider calls is deliberately conservative: otherwise a restart can
    # replay an expensive request whose response was lost during shutdown.
    count = 0
    for path in root.glob(f"{action_id}_*"):
        if not path.is_dir() or not (
            (path / "maintenance_request.json").is_file()
            or (path / "maintenance_manifest.json").is_file()
        ):
            continue
        count += 1
    return count


def command(command: list[str], cwd: Path, timeout: int = 900,
            extra_env: dict[str, str] | None = None) -> dict[str, Any]:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, env=env,
                            timeout=timeout, check=False)
    return {"command": command, "returncode": result.returncode,
            "stdout": result.stdout[-8000:], "stderr": result.stderr[-8000:]}


def prepare_full_regression_workspace(workspace: Path) -> bool:
    """Expose the isolated skill at the legacy path used by cross-repo tests.

    The link targets the copied snapshot, never the production skill tree.
    """
    skill = workspace / "skill"
    alias = workspace / "skills" / "cmr-lge-label-qc-workflow"
    if not skill.is_dir():
        return False
    if alias.exists() or alias.is_symlink():
        return True
    alias.parent.mkdir(parents=True, exist_ok=True)
    alias.symlink_to(skill.resolve(), target_is_directory=True)
    return True


def validate_workspace(workspace: Path, changed_files: list[str],
                       dataset: Path | None = None,
                       action: dict[str, Any] | None = None) -> dict[str, Any]:
    from app.lge_protocol import CONFIG as PROTOCOL_CONFIG, TEST as PROTOCOL_TEST
    clean_scope={name.removeprefix("orchestrator/") for name in changed_files}
    if ((action or {}).get("action_class") == "framework_repair" and not clean_scope.issubset({PROTOCOL_CONFIG,PROTOCOL_TEST})) or (PROTOCOL_CONFIG in clean_scope and (action or {}).get("owner_role") != "workflow_maintainer"):
        return {"passed":False,"checks":[],"targeted_tests":0,"full_regression_required":True,
                "frozen_evaluation_gate":{"required":True,"passed":False,"reason":"framework_repair_out_of_scope"}}
    checks = [command([str(PYTHON), "-m", "compileall", "-q", "agents", "app", "contracts",
                       "experiments", "providers", "scripts"], workspace / "orchestrator")]
    tests = []
    for item in changed_files:
        clean = item.removeprefix("orchestrator/")
        path = workspace / "orchestrator" / clean
        if clean.startswith("tests/") and path.suffix == ".py" and path.exists():
            tests.append(command(
                [str(PYTHON), str(path)], workspace / "orchestrator", 600,
                {"PYTHONPATH": str(workspace / "orchestrator")},
            ))
    checks.extend(tests)
    gate = frozen_evaluation_gate(workspace, changed_files, dataset, action)
    full_regression_required = bool(gate.get("required"))
    if full_regression_required:
        prepare_full_regression_workspace(workspace)
        checks.append(command(
            [str(PYTHON), "-m", "unittest", "discover", "-s", "tests"],
            workspace / "orchestrator", 900,
            {"PYTHONPATH": str(workspace / "orchestrator")},
        ))
    passed = all(item["returncode"] == 0 for item in checks) and gate["passed"]
    return {"passed": passed, "checks": checks, "targeted_tests": len(tests),
            "full_regression_required": full_regression_required,
            "frozen_evaluation_gate": gate}


def frozen_evaluation_gate(workspace: Path, changed_files: list[str],
                           dataset: Path | None = None,
                           action: dict[str, Any] | None = None) -> dict[str, Any]:
    """Require frozen-data evidence for production QC/scoring changes."""
    clean = [item.removeprefix("orchestrator/") for item in changed_files]
    needs_rules = any(name in {"scripts/rules_v2.py", "scripts/batch_screen.py"}
                      for name in clean)
    needs_quality = any(name in {"app/quality_scoring.py", "scripts/score_image_label_quality.py",
                                 "scripts/evaluate_quality_scoring.py"} for name in clean)
    needs_spatial = "scripts/shadow_spatial_boundary.py" in clean
    needs_generator = "scripts/lge_candidate_generator.py" in clean
    needs_protocol = "configs/lge_protocol_aliases.json" in clean
    needs_router = "app/check_auto_apply.py" in clean
    workflow_stage = (action or {}).get("workflow_stage")
    needs_lge_consumer = (
        "scripts/lge_feedback_loop.py" in clean
        and workflow_stage in {"lge_myo", "lge_mi"}
    )
    if (not needs_rules and not needs_quality and not needs_spatial and not needs_generator
            and not needs_router and not needs_lge_consumer and not needs_protocol):
        return {"passed": True, "required": False, "reason": "non-production-evaluator change"}
    candidate = workspace / "orchestrator"
    results: list[dict[str, Any]] = []
    passed = True
    if needs_generator or needs_protocol:
        if needs_protocol:
            from app.lge_protocol import load as validate_protocol
            try:
                validate_protocol(candidate)
            except (OSError, ValueError, TypeError) as exc:
                return {"required":True,"passed":False,"reason":"invalid_protocol_config:"+str(exc)}
        output = workspace / "lge_generator_contract.json"
        run = command([str(PYTHON), "scripts/validate_lge_generator.py", "--output", str(output)],
                      candidate, 600, {"PYTHONPATH": str(candidate)})
        report = read_json(output, {})
        ok = run["returncode"] == 0 and report.get("passed") is True and report.get("production_writes") == 0
        passed &= ok
        results.append({"suite": "lge_protocol_contract" if needs_protocol else "lge_candidate_generator_contract", "passed": ok, "report": report, "command": run})
    if needs_rules:
        eval_root = workspace / "frozen_eval"
        eval_root.mkdir(parents=True, exist_ok=True)
        for name in ("manifest.jsonl", "synthetic"):
            source = ROOT / "runtime/eval_set/v1" / name
            target = eval_root / name
            if not target.exists():
                target.symlink_to(source, target_is_directory=source.is_dir())
        run = command([str(PYTHON), "scripts/rules_v2.py", "--eval-dir", str(eval_root)],
                      candidate, 900, {"PYTHONPATH": str(candidate)})
        report = read_json(eval_root / "rules_v2_report.json", {})
        rates = report.get("detection_rate", {})
        checks = {
            "clean_fpr_le_0.10": float(report.get("clean_false_positive_rate", 1)) <= .10,
            "small_injury_le_0.05": float(report.get("true_small_false_repair", 1)) <= .05,
            "four_defect_recall_ge_0.60": len(rates) == 4 and all(float(x) >= .60 for x in rates.values()),
        }
        ok = run["returncode"] == 0 and all(checks.values())
        passed &= ok
        results.append({"suite": "controlled_defects", "passed": ok,
                        "checks": checks, "report": report, "command": run})
    if needs_quality:
        output = workspace / "quality_score_gate.json"
        run = command([str(PYTHON), "scripts/evaluate_quality_scoring.py",
                       "--eval-dir", str(ROOT / "runtime/eval_set/v1"),
                       "--output", str(output)], candidate, 1800,
                      {"PYTHONPATH": str(candidate)})
        report = read_json(output, {})
        classes = report.get("classes", {})
        checks = {
            "clean_below_70_le_0.10": float(report.get("clean_reference", {}).get("below_70", 1)) <= .10,
            "defect_drop_recall_ge_0.60": all(
                float(row.get("detected_drop_ge_5", 0)) >= .60
                for name, row in classes.items() if name != "true_small"),
            "true_small_injury_le_0.05": float(classes.get("true_small", {}).get(
                "detected_drop_ge_5", 1)) <= .05,
        }
        ok = run["returncode"] == 0 and bool(classes) and all(checks.values())
        passed &= ok
        results.append({"suite": "image_label_quality", "passed": ok,
                        "checks": checks, "report": report, "command": run})
    if needs_spatial:
        output = workspace / "spatial_boundary_production_gate.json"
        if dataset is None:
            run = {"returncode": 2, "stderr": "dataset is required for spatial gate"}
            report = {}
        else:
            run = command([str(PYTHON), "scripts/shadow_spatial_boundary.py",
                           "--dataset", str(dataset), "--output", str(output)],
                          candidate, 900, {"PYTHONPATH": str(candidate)})
            report = read_json(output, {})
        checks = {
            "zero_label_mutations": report.get("label_mutations") == 0,
            "checked_false_routes_eq_0": report.get("checked_false_routes") == 0,
            "control_gate_pass": report.get("control_gate") == "PASS",
            "nonempty_pending_coverage": int(
                report.get("pending_spatial_boundary_candidates") or 0) > 0,
        }
        ok = run["returncode"] == 0 and all(checks.values())
        passed &= ok
        results.append({"suite": "spatial_boundary_checked_controls", "passed": ok,
                        "checks": checks,
                        "report": {key: value for key, value in report.items()
                                   if key != "records"}, "command": run})
    if needs_router:
        baseline = read_json(ROOT / "runtime/check_iterations/latest.json", {})
        baseline_conflicts = int(baseline.get("pending_route_conflicts") or 0)
        probe = (
            "import json,sys; from pathlib import Path; "
            "from app.check_auto_apply import run; "
            "print(json.dumps(run(Path(sys.argv[1]),Path(sys.argv[2]),execute=False,limit=100)))"
        )
        run = command([str(PYTHON), "-c", probe, str(dataset), str(ROOT / "runtime")],
                      candidate, 900, {"PYTHONPATH": str(candidate)}) if dataset else {
                          "returncode": 2, "stdout": "", "stderr": "dataset is required"}
        try:
            router_report = json.loads(run.get("stdout") or "{}")
        except json.JSONDecodeError:
            router_report = {}
        planned = router_report.get("planned", [])
        preflight = router_report.get("preflight", [])
        checks = {
            "baseline_conflicts_positive": baseline_conflicts > 0,
            "conflicts_detected_ge_baseline": int(
                router_report.get("route_conflicts_detected") or 0) >= baseline_conflicts,
            "at_least_one_conflict_resolved": int(
                router_report.get("route_conflicts_resolved") or 0) > 0,
            "unresolved_conflicts_reduced": int(router_report.get(
                "route_conflicts_unresolved", baseline_conflicts)) < baseline_conflicts,
            "resolved_route_in_bounded_plan": any(
                isinstance(item, dict) and item.get("conflict_resolution") is True
                for item in planned),
            "preflight_errors_eq_0": bool(preflight) and all(
                isinstance(item, dict) and item.get("preflight") == "PASS"
                for item in preflight),
        }
        ok = run["returncode"] == 0 and all(checks.values())
        passed &= ok
        results.append({"suite": "route_conflict_dry_run", "passed": ok,
                        "checks": checks,
                        "report": {key: router_report.get(key) for key in (
                            "eligible", "holdout_excluded", "validation_excluded",
                            "route_conflicts_detected", "route_conflicts_resolved",
                            "route_conflicts_unresolved")}, "command": run})
    if needs_lge_consumer:
        data = evaluation_input(ROOT, workflow_stage=workflow_stage)
        evaluated = execute_evaluation(
            candidate, "scripts.lge_feedback_loop", data
        )
        owner_report = evaluated.get("report", {}) if isinstance(evaluated, dict) else {}
        prefix = "lge_myo" if workflow_stage == "lge_myo" else "lge_mi"
        count_names = [f"{prefix}_pending", f"{prefix}_work", f"{prefix}_resolved"]
        counts = [owner_report.get(name) for name in count_names]
        source_path = candidate / "scripts/lge_feedback_loop.py"
        source_text = source_path.read_text(encoding="utf-8", errors="replace") \
            if source_path.is_file() else ""
        frozen_status = (data.get("frozen_split") or {}).get("status")
        checks = {
            "lge_input": (data.get("status") == "AVAILABLE"
                          and data.get("modality") == "lge"
                          and data.get("workflow_stage") == workflow_stage
                          and len(data.get("records", [])) > 0),
            "stage_counts_present": all(
                isinstance(value, int) and value >= 0 for value in counts
            ),
            "counts_reconcile": (all(isinstance(value, int) for value in counts)
                                 and sum(counts) == len(data.get("records", []))),
            "consumer_ready_boolean": isinstance(
                owner_report.get(f"{prefix}_consumer_ready"), bool
            ),
            "report_modality_exact": owner_report.get("source_modality") == "lge",
            "report_stage_exact": owner_report.get("workflow_stage") == workflow_stage,
            "report_source_hash_exact": (
                owner_report.get("input_source_sha256") == data.get("source_sha256")
            ),
            "holdout_contract_exact": (
                frozen_status != "UNAVAILABLE"
                or (owner_report.get("frozen_split_status") == "UNAVAILABLE"
                    and owner_report.get("holdout_excluded") == 0)
            ),
            "no_embedded_case_ids": (
                frozen_status != "UNAVAILABLE"
                or re.search(r"case_[0-9a-f]{16}", source_text) is None
            ),
            "zero_label_writes": (owner_report.get("label_write") is False
                                  and owner_report.get("production_writes") == 0),
        }
        ok = evaluated.get("status") == "EXECUTED" and all(checks.values())
        passed &= ok
        results.append({
            "suite": "lge_stage_consumer_contract", "passed": ok,
            "checks": checks,
            "report": {
                "workflow_stage": workflow_stage,
                "input_records": len(data.get("records", [])),
                "input_source_sha256": data.get("source_sha256"),
                "consumer_ready": owner_report.get(f"{prefix}_consumer_ready"),
                "frozen_split_status": owner_report.get("frozen_split_status"),
                "holdout_excluded": owner_report.get("holdout_excluded"),
                **{name: owner_report.get(name) for name in count_names},
            },
            "command": {key: evaluated.get(key) for key in (
                "status", "reason", "error", "report_sha256"
            )},
        })
    report = {"passed": passed, "required": True, "results": results,
              "evaluated_at": now()}
    write_json(workspace / "frozen_evaluation_gate.json", report)
    return report


def resolve_candidate(workspace: Path, changed: str) -> tuple[Path, Path, str] | None:
    clean = changed.replace("\\", "/").lstrip("./")
    if clean.startswith("orchestrator/"):
        clean = clean[len("orchestrator/"):]
    if clean.startswith("skill/"):
        rel = clean[len("skill/"):]
        return workspace / "skill" / rel, SKILL / rel, "skill"
    orchestrator = workspace / "orchestrator" / clean
    skill = workspace / "skill" / clean
    if orchestrator.is_file():
        return orchestrator, ROOT / clean, "orchestrator"
    if skill.is_file():
        return skill, SKILL / clean, "skill"
    return None


def promote_workspace(run_id: int, action_id: str, workspace: Path,
                      changed_files: list[str]) -> dict[str, Any]:
    from app.rsi_postpromotion import installation_lock
    with installation_lock(ROOT, exclusive=True):
        return _promote_workspace_locked(run_id, action_id, workspace, changed_files)


def _promote_workspace_locked(run_id: int, action_id: str, workspace: Path,
                      changed_files: list[str]) -> dict[str, Any]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    promotion = ROOT / "runtime/promotions" / f"run_{run_id:06d}" / f"{action_id}_{stamp}"
    backup = promotion / "backup"
    promoted, skipped = [], []
    baseline_hashes = read_json(workspace / "baseline_hashes.json", {})
    conflicts = []
    resolved = []
    for changed in changed_files:
        item = resolve_candidate(workspace, changed)
        if item is None:
            skipped.append({"file": changed, "reason": "not_found"})
            continue
        source, destination, owner = item
        relative = destination.relative_to(ROOT if owner == "orchestrator" else SKILL).as_posix()
        allowed = ORCHESTRATOR_ALLOW if owner == "orchestrator" else SKILL_ALLOW
        if not relative.startswith(allowed):
            skipped.append({"file": changed, "reason": "not_allowlisted"})
            continue
        baseline_key = f"{owner}/{relative}"
        baseline_hash = baseline_hashes.get(baseline_key)
        current_hash = hashlib.sha256(destination.read_bytes()).hexdigest() if destination.exists() else None
        if baseline_hash is not None and current_hash != baseline_hash:
            conflicts.append({"file": baseline_key, "reason": "concurrent_change",
                              "baseline_hash": baseline_hash, "current_hash": current_hash})
            continue
        resolved.append((source, destination, owner, relative))
    if conflicts:
        return {"status": "CONFLICT", "promoted": [], "skipped": skipped,
                "conflicts": conflicts}
    if not resolved:
        return {"status": "NO_CHANGES", "promoted": [], "skipped": skipped,
                "conflicts": []}

    from app.rsi_postpromotion import begin_install, finish_install
    monitored = begin_install(ROOT, workspace, run_id, action_id)
    write_json(workspace / "rsi_promotion_journal.json", {
        "status": "PREPARED", "artifact": str(promotion),
        "run_id": run_id, "action_id": action_id,
    })

    for source, destination, owner, relative in resolved:
        old_hash = hashlib.sha256(destination.read_bytes()).hexdigest() if destination.exists() else None
        new_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        if old_hash == new_hash:
            skipped.append({"file": f"{owner}/{relative}", "reason": "already_integrated"})
            continue
        backup_path = backup / owner / relative
        if destination.exists():
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(destination, backup_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        promoted.append({"owner": owner, "relative": relative, "destination": str(destination),
                         "old_hash": old_hash, "new_hash": new_hash,
                         "backup": str(backup_path) if old_hash else None})

    post = command([str(PYTHON), "-m", "compileall", "-q", "agents", "app", "contracts",
                    "experiments", "providers", "scripts"], ROOT)
    if post["returncode"] != 0:
        for item in reversed(promoted):
            destination = Path(item["destination"])
            if item["backup"] and Path(item["backup"]).exists():
                shutil.copy2(item["backup"], destination)
            elif destination.exists():
                destination.unlink()
        result = {"status": "ROLLED_BACK", "reason": "post_promotion_compile_failed",
                  "promoted": promoted, "skipped": skipped, "post_check": post}
    else:
        result = {"status": "PROMOTED", "promoted": promoted,
                  "skipped": skipped, "conflicts": conflicts, "post_check": post}
    write_json(promotion / "promotion_manifest.json", result)
    if monitored:
        verified = finish_install(ROOT)
        result["postinstall_verification"] = verified
        if verified.get("status") == "ROLLED_BACK":
            result["status"] = "ROLLED_BACK"
            result["reason"] = "postinstall_regression"
        elif not verified.get("ready"):
            result["status"] = "RECOVERY_REQUIRED"
    result["artifact"] = str(promotion)
    return result


def history(db: Database, dataset: Path, limit: int = 50) -> list[dict[str, Any]]:
    result = []
    for run in db.list_runs(100):
        if Path(run["processed_path"]).resolve() != dataset.resolve():
            continue
        details = db.run_details(run["run_id"])
        actions = []
        for row in details.get("maintenance_actions", []):
            payload = json.loads(row["payload_json"])
            item = {"action_id": row["action_id"], "status": row["status"],
                    "owner_role": row["owner_role"],
                    "action_class": row["action_class"],
                    "objective": payload.get("objective", "")[:500],
                    "workflow_stage": payload.get("workflow_stage")}
            # Only recent feedback is needed for the next round; historical
            # objectives remain for the deterministic duplicate guard.
            if len(result) < 5:
                thread = f"run-{run['run_id']}:{row['action_id']}"
                reviews = [message for message in db.agent_inbox(
                    run["run_id"], row["owner_role"], thread_id=thread)
                    if message["sender_role"] == "independent_reviewer"
                    and message["message_type"] in {"action_review", "implementation_review"}]
                if reviews:
                    item["review_feedback"] = safe_review_feedback(reviews[-1]["payload"])
                # Return actual owner measurements to the next planning round,
                # not merely the status of an archived code experiment.
                artifact = row.get("artifact_path")
                if artifact:
                    workspace = Path(artifact).resolve()
                    base = (ROOT / "runtime/maintenance").resolve()
                    manifest_path = workspace / "maintenance_manifest.json"
                    if workspace.is_relative_to(base) and manifest_path.is_file():
                        try:
                            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                            measured = manifest.get("result", {}).get("real_evaluation", {})
                            if measured.get("status") == "EXECUTED":
                                item["measured_result"] = {
                                    key: measured.get(key) for key in (
                                        "status", "input_records", "source_sha256",
                                        "report_sha256", "input_modality", "input_cohort",
                                        "reference_labels_available")}
                                report = measured.get("report", {})
                                if len(json.dumps(report)) <= 12000:
                                    item["measured_result"]["owner_report"] = report
                                else:
                                    item["measured_result"]["owner_report_summary"] = (
                                        compact_owner_report(report)
                                    )
                        except (OSError, ValueError, AttributeError):
                            item["measurement_status"] = "UNREADABLE"
                    validation_path = workspace / "autopilot_validation.json"
                    if workspace.is_relative_to(base) and validation_path.is_file():
                        try:
                            validation = json.loads(validation_path.read_text(encoding="utf-8"))
                            frozen = validation.get("frozen_evaluation_gate") or {}
                            suites = []
                            for result_row in frozen.get("results") or []:
                                report = result_row.get("report") or {}
                                if len(json.dumps(report)) > 4000:
                                    report = {"omitted": "report_too_large"}
                                suites.append({
                                    "suite": result_row.get("suite"),
                                    "passed": result_row.get("passed"),
                                    "checks": result_row.get("checks") or {},
                                    "report": report,
                                    **({"immutable_contract": {
                                        "frozen_unavailable": (
                                            "When input frozen_split.status is UNAVAILABLE, output "
                                            "frozen_split_status must remain UNAVAILABLE and "
                                            "holdout_excluded must equal 0."
                                        ),
                                        "pure_adapter": (
                                            "run_evaluation returns the report only; the framework "
                                            "driver, not the adapter, persists latest.json."
                                        ),
                                    }} if result_row.get("suite") ==
                                           "lge_stage_consumer_contract" else {}),
                                })
                            item["validation_result"] = {
                                "passed": bool(validation.get("passed")),
                                "promotion_scope": validation.get("promotion_scope"),
                                "targeted_tests": validation.get("targeted_tests"),
                                "frozen_gate_passed": frozen.get("passed"),
                                "suites": suites,
                            }
                        except (OSError, ValueError, AttributeError):
                            item["validation_status"] = "UNREADABLE"
            actions.append(item)
        if actions:
            result.append({"run_id": run["run_id"], "status": run["status"], "actions": actions})
        if len(result) >= limit:
            break
    return result


class Autopilot:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.dataset = args.dataset.resolve()
        self.db = Database(args.db)
        self.db.initialize()
        self.state_path = args.state.resolve()
        self.state = read_json(self.state_path, {
            "dataset": str(self.dataset), "cycles": 0, "stagnant_cycles": 0,
            "last_issue_fingerprint": None, "last_code_fingerprint": None,
            "events": [], "status": "STARTING",
        })
        current_fingerprint, current_counts = issue_fingerprint(self.dataset)
        if rebase_issue_fingerprint_schema(
                self.state, current_fingerprint, current_counts):
            write_json(self.state_path, self.state)
        recovered = historical_no_gain_cycles(
            self.state.get("events", []), current_counts, self.args.max_stagnant
        )
        if (not self.state.get("paused_issue_fingerprint")
                and self.state.get("last_issue_fingerprint") == current_fingerprint
                and recovered >= self.args.max_stagnant):
            self.state["stagnant_cycles"] = recovered
            self.state["paused_issue_fingerprint"] = current_fingerprint
            self.state["status"] = "NO_PROGRESS_PAUSED"
            self.state.setdefault("events", []).append({
                "timestamp": now(), "event": "NO_PROGRESS_PAUSED",
                "reason": "historical_issue_manifests_unchanged",
                "stagnant_cycles": recovered, "issue_counts": current_counts,
            })
            self.state["events"] = self.state["events"][-200:]
            write_json(self.state_path, self.state)

    def event(self, event: str, **data: Any) -> None:
        record = {"timestamp": now(), "event": event, **data}
        self.state.setdefault("events", []).append(record)
        self.state["events"] = self.state["events"][-200:]
        self.state["status"] = event
        write_json(self.state_path, self.state)
        print(json.dumps(record, ensure_ascii=False), flush=True)

    def resume_after_fix(self, *, defer_blocker: bool = False) -> None:
        """One explicit, audited retry after a framework fix; preserve role gates."""
        fingerprint, counts = issue_fingerprint(self.dataset)
        status = self.state.get("status")
        paused_statuses = {"ESCALATION_REQUIRED", "NO_PROGRESS_PAUSED"}
        previous = status if status in paused_statuses else next((
            item.get("event") for item in reversed(self.state.get("events", []))
            if item.get("event") in paused_statuses
        ), None)
        if previous is None or not self.state.get("paused_issue_fingerprint"):
            raise ValueError("framework-fix resume requires a paused controller")
        if self.state.get("paused_issue_fingerprint") != fingerprint:
            raise ValueError("paused issue fingerprint changed; use normal resume")
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        if not commit or commit == self.state.get("last_code_fingerprint"):
            raise ValueError("no new framework commit for explicit retry")
        source_run_id = self.state.get("escalated_for")
        self.state.pop("paused_issue_fingerprint", None)
        self.state.pop("escalated_for", None)
        self.state["stagnant_cycles"] = 0
        self.state["last_code_fingerprint"] = commit
        blocker = active_production_blocker(ROOT) if defer_blocker else None
        if blocker:
            self.state["deferred_production_blocker"] = {
                "target_path": blocker["target_path"],
                "issue_fingerprint": fingerprint,
                "source_run_id": source_run_id,
                "reason": "bounded_attempts_exhausted",
            }
        self.event("FRAMEWORK_FIX_RESUMED", code_commit=commit,
                   previous_status=previous, issue_counts=counts,
                   deferred_blocker=(blocker or {}).get("target_path"))

    def latest_experiment_run(self) -> dict[str, Any] | None:
        """Newest run that still has an executable action waiting.

        A review-only round still needs no-progress accounting before another
        paid analysis. Prefer any executable action over that fallback.
        """
        review_only = None
        newest_executable = None
        for run in self.db.list_runs(100):
            if Path(run["processed_path"]).resolve() != self.dataset:
                continue
            if run["status"] not in (RunState.EXPERIMENT_PROPOSED.value,
                                     RunState.NEEDS_REVIEW.value):
                continue
            pending = []
            has_human_released_action = False
            for row in self.db.run_details(run["run_id"])["maintenance_actions"]:
                payload = json.loads(row["payload_json"])
                if payload.get("action_class") == "segmentation_training":
                    from app.segmentation_dispatch import proposal_context
                    if not proposal_context(ROOT, run["run_id"]).get("enabled"):
                        continue
                # RUNNING means the previous controller was interrupted; no
                # second controller can be active while this process owns the
                # global flock. SUCCEEDED may still need deterministic
                # validation/promotion. Both must resume before older runs.
                if (agent_action_authorized(self.db, run["run_id"], row, payload)
                        and payload.get("auto_execute", False)
                        and payload.get("action_class") != "human_review"):
                    pending.append(row)
                    has_human_released_action = (
                        has_human_released_action
                        or operator_evidence_retry_pending(payload)
                    )
            if pending:
                # An operator-applied example is an explicit answer to a
                # HUMAN_REQUIRED branch.  Scan past newer ordinary proposals
                # so that this answer cannot starve behind endlessly created
                # runs.  Among equal classes, retain newest-first ordering.
                if has_human_released_action:
                    return run
                if newest_executable is None:
                    newest_executable = run
            if review_only is None:
                review_only = run
        return newest_executable or review_only

    def analyze_new_snapshot(self, source_run_id: int | None) -> int:
        from app.workflow_budget import reserve
        reserve("analyses", ROOT)
        recovery = global_promotion_preflight(self.db, ROOT, SKILL)
        if recovery.get("status") == "RECOVERY_REQUIRED":
            raise RuntimeError("PROMOTION_RECOVERY_REQUIRED: " + str(recovery.get("reason")))
        iteration_history = history(self.db, self.dataset)
        continuity = build_agent_continuity(
            self.db, iteration_history=iteration_history
        )
        snap = snapshot_existing_qc(
            self.db, ROOT, self.dataset, source_run_id,
            iteration_history=iteration_history,
            role_continuity=continuity,
        )
        run_id = snap["analysis_run_id"]
        self.event("AGENT_ANALYSIS_STARTED", run_id=run_id)
        result = asyncio.run(run_shadow_analysis(
            self.db, run_id, self.args.providers, self.args.agents
        ))
        self.event("AGENT_ANALYSIS_FINISHED", run_id=run_id,
                   final_state=result["final_state"])
        return run_id

    def process_actions(self, run_id: int) -> dict[str, Any]:
        details = self.db.run_details(run_id)
        blocker = active_production_blocker(ROOT)
        dependency = enrich_workflow_dependency(
            dependent_workflow_status(self.dataset, ROOT), history(self.db, self.dataset)
        )
        if blocker and dependency["stage"] != "cmr_paired":
            self.event("CMR_BLOCKER_DEFERRED_FOR_LGE_FIRST", run_id=run_id,
                       active_stage=dependency["stage"],
                       target_path=blocker["target_path"])
            blocker = None
        if production_blocker_deferred(self.state, blocker):
            blocker = None
        promoted_any = False
        waiting_human = []
        results = []
        # Recover the whole run before executing any other action against its
        # source tree. A mixed install is a framework failure, not a domain task.
        for pending in details["maintenance_actions"]:
            if not pending.get("artifact_path") or pending["status"] in TERMINAL_ACTIONS:
                continue
            recovery = reconcile_promotion(
                self.db, ROOT, SKILL, Path(pending["artifact_path"]),
                run_id=run_id, action_id=pending["action_id"],
            )
            if recovery.get("status") == "RECOVERY_REQUIRED":
                self.event("PROMOTION_RECOVERY_REQUIRED", run_id=run_id,
                           action_id=pending["action_id"], reason=recovery.get("reason"))
                return {"promoted_any": False, "waiting_human": [pending["action_id"]],
                        "results": [], "recovery_required": recovery}
            if recovery.get("status") in {"PROMOTED", "PROMOTED_SHADOW"}:
                pending["status"] = recovery["status"]
        for row in details["maintenance_actions"]:
            from app.autonomy_control import operator_paused
            if operator_paused(ROOT) and not getattr(getattr(self, "args", None), "bounded_run", None):
                self.event("OPERATOR_PAUSED_BETWEEN_ACTIONS",run_id=run_id)
                break
            payload = json.loads(row["payload_json"])
            action_id = row["action_id"]
            if payload.get("action_class") == "segmentation_training":
                if row["status"] in TERMINAL_ACTIONS:
                    continue
                from app.segmentation_dispatch import dispatch
                outcome = dispatch(ROOT, self.db, run_id, action_id,
                                   providers_path=self.args.providers, agents_path=self.args.agents)
                results.append({"action_id": action_id, "segmentation": outcome})
                self.event("SEGMENTATION_ACTION", run_id=run_id, action_id=action_id, **outcome)
                # Dedicated typed-plan path: never run_llm_maintenance, code CAS,
                # clinical-label consumer integration or framework_repair.
                continue
            framework_repair = payload.get("action_class") == "framework_repair"
            if framework_repair:
                from app.lge_protocol import action_allowed
                if not action_allowed(payload):
                    self.db.update_maintenance_action(run_id, action_id, status="REJECTED")
                    self.event("FRAMEWORK_REPAIR_SCOPE_REJECTED",run_id=run_id,action_id=action_id)
                    continue
            integration_mismatch = (not framework_repair and dependency.get("integration_required")
                                    and not action_builds_stage_consumer(payload, require_generator=bool(dependency.get("lge_generator_required"))))
            stale_stage_denominator = (not framework_repair and dependency.get("integration_required")
                                       and action_uses_stale_lge_shadow_contract(payload))
            if (row["status"] not in TERMINAL_ACTIONS
                    and ((not framework_repair and not action_matches_workflow_stage(payload, dependency["stage"]))
                         or integration_mismatch or stale_stage_denominator)):
                self.db.update_maintenance_action(run_id, action_id, status="REJECTED")
                self.event(
                    ("ACTION_REJECTED_STALE_STAGE_DENOMINATOR" if stale_stage_denominator
                     else "ACTION_REJECTED_STAGE_CONSUMER_REQUIRED" if integration_mismatch
                     else "ACTION_REJECTED_WORKFLOW_STAGE_MISMATCH"), run_id=run_id,
                    action_id=action_id,
                    action_stage=semantic_action_workflow_stage(payload),
                    stamped_stage=payload.get("workflow_stage"),
                    active_stage=dependency["stage"],
                )
                continue
            requirements = owner_execution_requirements(payload, blocker)
            if (row["status"] not in TERMINAL_ACTIONS and not framework_repair
                    and blocker_supersedes_action(payload, blocker)):
                self.db.update_maintenance_action(run_id, action_id, status="REJECTED")
                self.event("ACTION_SUPERSEDED_BY_PRODUCTION_BLOCKER", run_id=run_id,
                           action_id=action_id, target_path=blocker["target_path"])
                continue
            if payload.get("action_class") == "human_review" or not payload.get("auto_execute", False):
                waiting_human.append(action_id)
                continue
            if not maintenance_action_executable(str(payload.get("action_class") or "")):
                self.db.update_maintenance_action(
                    run_id, action_id, status="HUMAN_REQUIRED"
                )
                waiting_human.append(action_id)
                self.event(
                    "ACTION_BLOCKED_NO_EXECUTOR", run_id=run_id, action_id=action_id,
                    action_class=payload.get("action_class"), owner_role=row["owner_role"],
                )
                continue
            attempts = action_attempts(run_id, action_id)
            max_attempts = int(payload.get("max_attempts", 1))
            if row["status"] in TERMINAL_ACTIONS:
                continue
            if not agent_action_authorized(self.db, run_id, row, payload):
                waiting_human.append(action_id)
                self.event("ACTION_BLOCKED_NO_AGENT_APPROVAL", run_id=run_id,
                           action_id=action_id, owner_role=row["owner_role"])
                continue
            reusable_manifest = (read_json(Path(row["artifact_path"]) / "maintenance_manifest.json", {})
                                 if row.get("artifact_path") else {})
            reusable_ready = owner_result_ready_for_validation(
                reusable_manifest.get("result", {}), requirements)
            if row.get("artifact_path") and (row["status"] == "SUCCEEDED" or reusable_ready):
                workspace = Path(row["artifact_path"])
                manifest = read_json(workspace / "maintenance_manifest.json", {})
                changed = manifest.get("result", {}).get("changed_files", [])
                if (blocker and action_addresses_blocker(payload, blocker)
                        and not changed_files_include_target(changed, blocker["target_path"])):
                    validation = {
                        "passed": False,
                        "reason": "production_blocker_target_not_changed",
                        "required_target": blocker["target_path"],
                        "changed_files": changed,
                    }
                    write_json(workspace / "autopilot_validation.json", validation)
                    self.db.update_maintenance_action(
                        run_id, action_id, status="REJECTED", artifact_path=workspace)
                    self.event("ACTION_MISSED_PRODUCTION_TARGET", run_id=run_id,
                               action_id=action_id, target_path=blocker["target_path"])
                    continue
                validation = validate_workspace(workspace, changed, self.dataset, payload)
                selection = assess_action(
                    self.db, ROOT, workspace, run_id=run_id,
                    action=payload, changed_files=changed, validation_passed=validation["passed"],
                )
                validation["rsi_selection"] = selection
                if selection.get("blocks_promotion"):
                    validation["passed"] = False
                shadow_only = shadow_only_files(changed)
                validation["promotion_scope"] = "shadow_only" if shadow_only else "production_path"
                write_json(workspace / "autopilot_validation.json", validation)
                if validation["passed"]:
                    review_input = dict(manifest.get("result", {}))
                    review_input["verified_workspace"] = str(workspace.resolve())
                    review_input["verified_source_hashes"] = {
                        name: hashlib.sha256((workspace / "orchestrator" / name.removeprefix("orchestrator/")).read_bytes()).hexdigest()
                        for name in changed if (workspace / "orchestrator" / name.removeprefix("orchestrator/")).is_file()}
                    prior_review = implementation_review_payload(
                        self.db, run_id, action_id, row["owner_role"]
                    )
                    if (prior_review and prior_review.get("implementation_sha256")
                            and prior_review.get("implementation_sha256") != stable_hash(review_input)):
                        prior_review = None
                    review_current = (not payload.get("runtime_failure_feedback")
                                      or bool(prior_review))
                    contract_passed = named_production_contract_passed(validation)
                    if (prior_review and prior_review.get("required_validation_metrics")
                            and not contract_passed
                            and not (shadow_only and prior_review.get("decision") == "accept_experiment")):
                        self.db.update_maintenance_action(
                            run_id, action_id, status="HUMAN_REQUIRED", artifact_path=workspace,
                            evolution_variant_id=selection.get("child_variant_id"),
                        )
                        self.event("IMPLEMENTATION_VALIDATION_DEBT", run_id=run_id,
                                   action_id=action_id,
                                   required_validation_metrics=prior_review[
                                       "required_validation_metrics"][:3])
                        continue
                    if (not review_current or not implementation_review_approved(
                        self.db, run_id, action_id, row["owner_role"]
                    ) and not (shadow_only and prior_review
                               and prior_review.get("decision") == "accept_experiment")):
                        try:
                            implementation_review = review_implementation(
                                db=self.db, run_id=run_id, action=payload,
                                implementation=review_input,
                                validation=validation, providers_path=self.args.providers,
                                agents_path=self.args.agents,
                            )
                        except Exception as exc:
                            self.db.update_maintenance_action(
                                run_id, action_id, status="HUMAN_REQUIRED", artifact_path=workspace,
                                evolution_variant_id=selection.get("child_variant_id"),
                            )
                            self.event("IMPLEMENTATION_REVIEW_ERROR", run_id=run_id,
                                       action_id=action_id, error=str(exc)[:1000])
                            continue
                        if implementation_review.decision != "accept_experiment":
                            status = ("REJECTED" if implementation_review.decision == "reject"
                                      else "HUMAN_REQUIRED")
                            self.db.update_maintenance_action(
                                run_id, action_id, status=status, artifact_path=workspace,
                                evolution_variant_id=selection.get("child_variant_id"),
                            )
                            self.event("IMPLEMENTATION_REVIEW_BLOCKED", run_id=run_id,
                                       action_id=action_id, decision=implementation_review.decision)
                            continue
                        if (implementation_review.required_validation_metrics and not shadow_only
                                and not contract_passed):
                            self.db.update_maintenance_action(
                                run_id, action_id, status="HUMAN_REQUIRED", artifact_path=workspace,
                                evolution_variant_id=selection.get("child_variant_id"),
                            )
                            self.event("IMPLEMENTATION_VALIDATION_DEBT", run_id=run_id,
                                       action_id=action_id,
                                       required_validation_metrics=(
                                           implementation_review.required_validation_metrics[:3]))
                            continue
                    promotion = promote_workspace(run_id, action_id, workspace, changed)
                    if promotion["status"] in {"PROMOTED", "NO_CHANGES"}:
                        status = "PROMOTED_SHADOW" if shadow_only else "PROMOTED"
                    elif promotion["status"] in {"CONFLICT", "ROLLED_BACK"}:
                        # A different agent changed the same file after this
                        # snapshot began. Reject the stale candidate and let the
                        # next autonomous iteration re-plan on the new baseline.
                        status = "REJECTED"
                    else:
                        status = "HUMAN_REQUIRED"
                    artifact = promotion.get("artifact")
                    self.db.update_maintenance_action(
                        run_id, action_id, status=status, artifact_path=workspace,
                        evolution_variant_id=selection.get("child_variant_id"),
                        rollback_manifest_path=(Path(artifact) / "promotion_manifest.json"
                                                if artifact else None),
                    )
                    if promotion.get("postinstall_verification"):
                        self.db.append_evolution_event("INSTALLATION_VERIFIED", promotion["postinstall_verification"],
                                                       variant_id=selection.get("child_variant_id"))
                    promoted_any |= status in {"PROMOTED", "PROMOTED_SHADOW"}
                    results.append({"action_id": action_id, "validation": validation,
                                    "promotion": promotion})
                    self.event("ACTION_VALIDATED", run_id=run_id, action_id=action_id,
                               status=status, promotion=promotion["status"])
                else:
                    failed_status = "APPROVED" if shadow_only else "REJECTED"
                    self.db.update_maintenance_action(
                        run_id, action_id, status=failed_status, artifact_path=workspace,
                        evolution_variant_id=selection.get("child_variant_id"))
                    self.event("ACTION_VALIDATION_FAILED", run_id=run_id,
                               action_id=action_id, status=failed_status,
                               promotion_scope=validation["promotion_scope"])
                    results.append({"action_id": action_id, "validation": validation,
                                    "promotion": None})
                continue
            if attempts >= max_attempts and row["status"] != "SUCCEEDED":
                self.db.update_maintenance_action(run_id, action_id, status="HUMAN_REQUIRED")
                waiting_human.append(action_id)
                self.event("ACTION_RETRIES_EXHAUSTED", run_id=run_id, action_id=action_id,
                           attempts=attempts)
                continue

            if operator_evidence_retry_pending(payload):
                marker = self.db.consume_operator_evidence_retry(run_id, action_id)
                payload["operator_evidence_retry"] = marker
                self.event(
                    "OPERATOR_EVIDENCE_RETRY_STARTED", run_id=run_id,
                    action_id=action_id, edit_id=marker["edit_id"], pool=marker["pool"],
                )
            self.db.update_maintenance_action(run_id, action_id, status="RUNNING")
            self.event("ACTION_STARTED", run_id=run_id, action_id=action_id,
                       attempt=attempts + 1)
            try:
                execution_payload = dict(payload)
                if requirements:
                    execution_payload["control_plane_requirements"] = requirements
                execution_payload["handoff_messages"] = self.db.agent_inbox(
                    run_id, row["owner_role"], thread_id=f"run-{run_id}:{action_id}"
                )
                result = run_llm_maintenance(
                    action=execution_payload,
                    output_root=ROOT / "runtime/maintenance" / f"run_{run_id:06d}",
                    orchestrator_source=ROOT, skill_source=SKILL,
                    providers_path=self.args.providers, agents_path=self.args.agents,
                    provider_role=payload.get("owner_role") or "qc_code_optimizer",
                    timeout_seconds=self.args.action_timeout,
                    db=self.db, run_id=run_id,
                )
                workspace = Path(result["workspace"])
                declared_ready = result["result"].get("ready_for_validation") is True
                ready = owner_result_ready_for_validation(result["result"], requirements)
                self.db.update_maintenance_action(run_id, action_id,
                                                  status="SUCCEEDED" if ready else "APPROVED",
                                                  artifact_path=workspace)
                self.event("ACTION_IMPLEMENTED", run_id=run_id, action_id=action_id,
                           ready_for_validation=ready,
                           owner_declared_ready=declared_ready,
                           deterministic_ready_override=ready and not declared_ready)
            except Exception as exc:
                self.db.update_maintenance_action(run_id, action_id, status="APPROVED")
                self.event("ACTION_ERROR", run_id=run_id, action_id=action_id,
                           error=str(exc)[:1000])
            # One expensive owner-agent task per tick keeps the daemon observable.
            break
        return {"promoted_any": promoted_any, "waiting_human": waiting_human,
                "results": results}

    def reopen_bounded_runtime_failure(self, record: dict) -> bool:
        """Return a genuine deployed producer crash to its original approved owner.

        No new analysis/assignment/review is synthesized. Original max_attempts
        and the twelve-step trial budget still apply; the changed implementation
        requires a fresh independent review bound to its result hash.
        """
        from app.lge_execution import governed_generator
        from app.autonomy_feedback import actual_execution
        result = actual_execution(ROOT)
        outcomes = result.get("outcomes") or []
        release = result.get("generator") or {}
        run_id = record.get("run_id")
        if (not outcomes or result.get("execution_mode") != "execute"
                or result.get("production_writes") != 0
                or release.get("run_id") != run_id
                or any(not str(r.get("reason", "")).startswith("generator_failure:") for r in outcomes)
                or release != governed_generator(ROOT)):
            return False
        action_id = release.get("action_id")
        rows = self.db.run_details(run_id)["maintenance_actions"]
        row = next((r for r in rows if r["action_id"] == action_id), None)
        if not row or row["status"] != "PROMOTED":
            return False
        feedback = {"module_sha256": release.get("module_sha256"),
                    "input_contract_version": release.get("input_contract_version"),
                    "failure_reasons": sorted({r["reason"] for r in outcomes}),
                    "failed_count": len(outcomes), "production_writes": 0}
        return self.claim_bounded_owner_repair(record, row, feedback)

    def reopen_bounded_validation_failure(self, record: dict) -> bool:
        """An uninstalled runtime-repair snapshot can use remaining approved attempts."""
        run_id = record.get("run_id")
        if not run_id:
            return False
        for row in self.db.run_details(run_id)["maintenance_actions"]:
            payload = json.loads(row["payload_json"])
            if row["status"] != "REJECTED" or not payload.get("runtime_failure_feedback") or not row.get("artifact_path"):
                continue
            workspace = Path(row["artifact_path"]).resolve()
            if not workspace.is_relative_to((ROOT / "runtime/maintenance" / f"run_{run_id:06d}").resolve()):
                continue
            validation = read_json(workspace / "autopilot_validation.json", {})
            if validation.get("passed") is not False:
                continue
            failures = [(c.get("stdout", "") + c.get("stderr", ""))[-2500:]
                        for c in validation.get("checks", []) if c.get("returncode") != 0]
            gate = validation.get("frozen_evaluation_gate") or {}
            if not gate.get("passed", True):
                failures.append(json.dumps(gate, ensure_ascii=False)[-3500:])
            if not failures:
                continue
            return self.claim_bounded_owner_repair(record, row, {
                "failure_type": "uninstalled_snapshot_validation", "failure_reasons": failures,
                "failed_count": len(failures), "production_writes": 0})
        return False

    def claim_bounded_owner_repair(self, record: dict, row: dict, feedback: dict) -> bool:
        run_id, action_id = record["run_id"], row["action_id"]
        payload = json.loads(row["payload_json"])
        if (action_attempts(run_id, action_id) >= int(payload.get("max_attempts", 1))
                or int(record.get("steps", 0)) >= 12):
            return False
        signature = stable_hash(feedback)
        if signature in record.get("runtime_repair_signatures", []):
            return False
        payload["runtime_failure_feedback"] = feedback
        with self.db.connect() as connection:
            cursor = connection.execute(
                "UPDATE maintenance_actions SET status='APPROVED',artifact_path=NULL,payload_json=? "
                "WHERE run_id=? AND action_id=? AND status=? AND artifact_path=?",
                (json.dumps(payload, ensure_ascii=False), run_id, action_id, row["status"], row["artifact_path"]))
            if cursor.rowcount != 1:
                raise RuntimeError("runtime repair action changed concurrently")
        self.db.post_agent_message(run_id, thread_id=f"run-{run_id}:{action_id}",
            sender_role="runtime_controller", recipient_role=row["owner_role"],
            message_type="runtime_execution_failure", payload=feedback)
        record.setdefault("runtime_repair_signatures", []).append(signature)
        record.setdefault("prior_artifacts", []).append(row["artifact_path"])
        record["status"] = "ACTIONS"
        record.pop("ended_at", None)
        self.event("BOUNDED_RUNTIME_REPAIR", run_id=run_id, action_id=action_id,
                   failed_count=feedback["failed_count"], remaining_attempts=int(payload.get("max_attempts", 1))-action_attempts(run_id, action_id))
        return True

    def extend_bounded_owner_budget(self, record_path: Path) -> None:
        """Explicit operator grant: one additional attempt of the same failed action.

        Never called by tick or the normal daemon. It neither buys an analysis
        nor accepts a test failure; normal owner/validation/review remain required.
        """
        record = read_json(record_path, {})
        if record.get("status") != "COMPLETE":
            raise ValueError("extension requires an ended unextended trial")
        rows = [r for r in self.db.run_details(record["run_id"])["maintenance_actions"]
                if r["status"] == "HUMAN_REQUIRED" and json.loads(r["payload_json"]).get("runtime_failure_feedback")]
        if len(rows) != 1:
            raise ValueError("extension requires one exhausted runtime-repair action")
        row = rows[0]
        workspace = Path(row["artifact_path"]).resolve()
        if not workspace.is_relative_to((ROOT / "runtime/maintenance" / f"run_{record['run_id']:06d}").resolve()):
            raise ValueError("untrusted repair snapshot")
        if str(workspace) in record.get("budget_extension_snapshots", []):
            raise ValueError("this failed snapshot already received an operator extension")
        result = read_json(workspace / "maintenance_manifest.json", {}).get("result", {})
        if result.get("tests_passed") is not False:
            raise ValueError("extension requires actual executor test failure")
        payload = json.loads(row["payload_json"])
        previous = int(payload.get("max_attempts", 1))
        if action_attempts(record["run_id"], row["action_id"]) != previous:
            raise ValueError("original attempt budget is not exactly exhausted")
        failure = (workspace / "test_output.log").read_text(encoding="utf-8")[-6000:]
        payload["max_attempts"] = previous + 1
        with self.db.connect() as connection:
            cursor = connection.execute("UPDATE maintenance_actions SET payload_json=? WHERE run_id=? AND action_id=? AND payload_json=?",
                (json.dumps(payload, ensure_ascii=False), record["run_id"], row["action_id"], row["payload_json"]))
            if cursor.rowcount != 1:
                raise RuntimeError("budget changed concurrently")
        row["payload_json"] = json.dumps(payload, ensure_ascii=False)
        record.setdefault("budget_extension_snapshots", []).append(str(workspace))
        record["owner_budget_extension"] = {"action_id": row["action_id"], "previous_max_attempts": previous,
                                              "new_max_attempts": previous + 1, "granted_at": now()}
        if not self.claim_bounded_owner_repair(record, row, {"failure_type": "operator_extended_failed_snapshot",
                "failure_reasons": [failure], "failed_count": 1, "production_writes": 0}):
            raise RuntimeError("extended repair could not be queued")
        write_json(record_path, record)
        self.event("OPERATOR_BOUNDED_OWNER_BUDGET_EXTENDED", run_id=record["run_id"],
                   action_id=row["action_id"], previous_max_attempts=previous, new_max_attempts=previous + 1)

    def bounded_run(self, record_path: Path, *, stop_requested=lambda: False) -> None:
        """One explicitly authorized analysis, then only that run's role actions.

        Durable STARTING prevents a crash from buying a second analysis. This
        controller never starts consumers or writes labels; an application trial
        is a separate single check-loop invocation with its own batch bound.
        """
        record = read_json(record_path, {})
        if record and record.get("dataset") != str(self.dataset):
            raise ValueError("bounded trial dataset mismatch")
        if record.get("status") == "COMPLETE" and (self.reopen_bounded_runtime_failure(record) or self.reopen_bounded_validation_failure(record)):
            write_json(record_path, record)
        if record.get("status") in {"COMPLETE", "FAILED", "STOPPED", "STEP_LIMIT"}:
            raise ValueError("bounded trial already ended; no new paid analysis")
        if record and not record.get("run_id"):
            raise ValueError("interrupted analysis requires reconciliation; refusing another analysis")
        if not record:
            recovery = global_promotion_preflight(self.db, ROOT, SKILL)
            if recovery.get("status") == "RECOVERY_REQUIRED":
                raise RuntimeError("promotion recovery required before bounded trial")
            record = {"dataset": str(self.dataset), "status": "STARTING",
                      "started_at": now(), "analysis_limit": 1, "application_limit": 10}
            write_json(record_path, record)
            self.event("BOUNDED_TRIAL_STARTED", record_path=str(record_path))
            try:
                record["run_id"] = self.analyze_new_snapshot(None)
            except Exception as exc:
                record.update(status="FAILED", error=f"{type(exc).__name__}: {exc}"[:1000])
                write_json(record_path, record)
                raise
            record["status"] = "ACTIONS"
            write_json(record_path, record)
        run_id = int(record["run_id"])
        for _ in range(12 - int(record.get("steps", 0))):
            if stop_requested():
                record["status"] = "STOPPED"
                break
            outcome = self.process_actions(run_id)
            record["steps"] = int(record.get("steps", 0)) + 1
            rows = self.db.run_details(run_id)["maintenance_actions"]
            active = [r for r in rows if r["status"] not in TERMINAL_ACTIONS
                      and json.loads(r["payload_json"]).get("auto_execute", False)
                      and json.loads(r["payload_json"]).get("action_class") != "human_review"]
            record["actions"] = [{"action_id": r["action_id"], "status": r["status"]} for r in rows]
            record["status"] = ("FAILED" if outcome.get("recovery_required")
                                else "ACTIONS" if active else "COMPLETE")
            write_json(record_path, record)
            if record["status"] != "ACTIONS":
                break
        if record["status"] == "ACTIONS":
            record["status"] = "STEP_LIMIT"
        record["ended_at"] = now()
        write_json(record_path, record)
        self.event("BOUNDED_TRIAL_ENDED", run_id=run_id, trial_status=record["status"],
                   actions=record.get("actions", []))

    def resume_validation_owner(self) -> bool:
        """Return measured generator rejection to its approved owner, within its budget."""
        from app.autonomy_control import policy
        cfg=policy(ROOT)
        if not cfg.get("enabled") or str(self.dataset.resolve()) != cfg.get("dataset"):
            return False
        for run in self.db.list_runs(10):
            if Path(run["processed_path"]).resolve() != self.dataset.resolve():continue
            run_id=run["run_id"]
            path=ROOT/"runtime/autonomy"/f"validation_repair_run_{run_id:06d}.json"
            record=read_json(path,{"run_id":run_id,"dataset":str(self.dataset),"steps":0,"status":"COMPLETE"})
            for row in self.db.run_details(run_id)["maintenance_actions"]:
                payload=json.loads(row["payload_json"])
                from agents.governance import original_action_approved
                if not original_action_approved(self.db,run_id,row,payload):continue
                if row["status"] == "REJECTED" and row.get("artifact_path"):
                    workspace=Path(row["artifact_path"]).resolve()
                    if not workspace.is_relative_to((ROOT/"runtime/maintenance"/f"run_{run_id:06d}").resolve()):continue
                    selection=read_json(workspace/"rsi_selection.json",{})
                    validation=read_json(workspace/"autopilot_validation.json",{})
                    if (selection.get("benchmark_id") not in {"reviewed-generator-v1","reviewed-generator-v2"}
                            or selection.get("reasons") != ["no_quality_improvement"]
                            or not validation.get("checks")
                            or any(c.get("returncode") != 0 for c in validation["checks"])):continue
                    feedback={"failure_type":"frozen_generator_quality_not_improved","failed_count":1,
                              "production_writes":0,"selection":{k:selection.get(k) for k in (
                                  "decision","reasons","benchmark_id","benchmark_hash","child_variant_id",
                                  "baseline_metrics","candidate_metrics","deltas")}}
                    if not self.claim_bounded_owner_repair(record,row,feedback):continue
                    write_json(path,record)
                elif not (record.get("status") == "ACTIONS" and row["status"] not in TERMINAL_ACTIONS
                          and payload.get("runtime_failure_feedback",{}).get("failure_type") == "frozen_generator_quality_not_improved"):
                    continue
                if int(record.get("steps",0)) >= 12:continue
                self.event("AUTONOMOUS_VALIDATION_REPAIR",run_id=run_id,action_id=row["action_id"])
                self.process_actions(run_id)
                record["steps"]=int(record.get("steps",0))+1
                write_json(path,record)
                return True
        return False

    def resume_runtime_owner(self) -> bool:
        """Normal-daemon recovery uses only an existing approval and remaining attempts."""
        from app.autonomy_control import policy
        from app.autonomy_feedback import actual_execution
        cfg = policy(ROOT)
        if not cfg.get("enabled") or str(self.dataset.resolve()) != cfg.get("dataset"):
            return False
        result = actual_execution(ROOT)
        release = result.get("generator") or {}
        run_id = release.get("run_id")
        if not run_id:
            return False
        path = ROOT / "runtime/autonomy" / f"repair_run_{int(run_id):06d}.json"
        record = read_json(path, {"run_id":run_id, "dataset":str(self.dataset), "steps":0, "status":"COMPLETE"})
        if record.get("status") == "COMPLETE":
            if not (self.reopen_bounded_runtime_failure(record) or self.reopen_bounded_validation_failure(record)):
                return False
            write_json(path,record)
        rows = self.db.run_details(run_id)["maintenance_actions"]
        active = [row for row in rows if row["status"] not in TERMINAL_ACTIONS
                  and json.loads(row["payload_json"]).get("runtime_failure_feedback")
                  and agent_action_authorized(self.db, run_id, row, json.loads(row["payload_json"]))]
        if not active or int(record.get("steps",0)) >= 12:
            record["status"] = "COMPLETE"
            write_json(path,record)
            return False
        self.event("AUTONOMOUS_OWNER_REPAIR", run_id=run_id, actions=[r["action_id"] for r in active])
        self.process_actions(run_id)
        record["steps"] = int(record.get("steps",0)) + 1
        write_json(path,record)
        return True

    def service_protocol_repair(self) -> bool:
        """One role-reviewed repair campaign per observed transport fault/version."""
        from app.autonomy_control import policy
        from app.lge_protocol import repair_status
        from app.autonomy_feedback import actual_execution
        cfg = policy(ROOT)
        if not cfg.get("enabled") or str(self.dataset.resolve()) != cfg.get("dataset"):
            return False
        fault = repair_status(ROOT)
        if fault.get("pending_verification"):
            self.event("WAITING_FOR_PROTOCOL_VERIFICATION")
            return True
        if not fault["required"]:
            return False
        for run in self.db.list_runs(50):
            if Path(run["processed_path"]).resolve() != self.dataset.resolve():
                continue
            for row in self.db.run_details(run["run_id"])["maintenance_actions"]:
                payload = json.loads(row["payload_json"])
                if payload.get("action_class") == "framework_repair" and agent_action_authorized(self.db,run["run_id"],row,payload):
                    self.process_actions(run["run_id"])
                    return True
        signature = stable_hash(fault)
        if self.state.get("protocol_repair_campaign") == signature:
            if self.state.get("status") != "PROTOCOL_REPAIR_EXHAUSTED":
                self.event("PROTOCOL_REPAIR_EXHAUSTED",reason="no_remaining_approved_action_for_same_fault")
            return True
        # Claim before analysis: interruption cannot repeatedly buy a campaign.
        self.state["protocol_repair_campaign"] = signature
        self.event("PROTOCOL_REPAIR_ANALYSIS",target=fault["target"])
        source_run = (actual_execution(ROOT).get("generator") or {}).get("run_id")
        self.analyze_new_snapshot(source_run)
        return True

    def tick(self) -> None:
        recovery = global_promotion_preflight(self.db, ROOT, SKILL)
        if recovery.get("status") == "RECOVERY_REQUIRED":
            self.event("PROMOTION_RECOVERY_REQUIRED", **{key: recovery[key] for key in (
                "run_id", "action_id", "reason") if key in recovery})
            return
        from app.autonomy_control import operator_paused, ensure_consumer
        if operator_paused(ROOT):
            if self.state.get("status") != "OPERATOR_PAUSED":
                self.event("OPERATOR_PAUSED")
            return
        consumer = ensure_consumer(ROOT, dataset=self.dataset)
        if not consumer["ready"]:
            from app.lge_protocol import repair_status
            if consumer["status"] == "REPAIR_REQUIRED" and repair_status(ROOT)["required"] and self.service_protocol_repair():
                return
            if self.state.get("consumer_recovery_status") != consumer["status"]:
                self.state["consumer_recovery_status"] = consumer["status"]
                self.event("WAITING_FOR_CONSUMER_RECOVERY", consumer=consumer)
            return
        if self.service_protocol_repair():
            return
        if self.resume_validation_owner():
            return
        if self.resume_runtime_owner():
            return
        current_fingerprint, current_counts = issue_fingerprint(self.dataset)
        deferred = self.state.get("deferred_production_blocker")
        if (isinstance(deferred, dict)
                and deferred.get("issue_fingerprint") != current_fingerprint):
            cleared_target = deferred.get("target_path")
            self.state.pop("deferred_production_blocker", None)
            self.event("PRODUCTION_BLOCKER_DEFERRAL_CLEARED",
                       target_path=cleared_target, issue_counts=current_counts)
        paused_fingerprint = self.state.get("paused_issue_fingerprint")
        if paused_fingerprint:
            if paused_fingerprint == current_fingerprint:
                return
            self.state.pop("paused_issue_fingerprint", None)
            self.state.pop("escalated_for", None)
            self.state["stagnant_cycles"] = 0
            self.event("ISSUE_CHANGE_RESUMED", issue_counts=current_counts)
        idle_until = self.state.get("escalated_until")
        if idle_until:
            if datetime.now().astimezone() < datetime.fromisoformat(idle_until):
                return
            self.state.pop("escalated_until", None)
        fingerprint, counts = current_fingerprint, current_counts
        if (ROOT / "configs/lge_execution.json").exists():
            dependency = dependent_workflow_status(self.dataset, ROOT)
            try:
                age = (datetime.now().astimezone() - datetime.fromisoformat(
                    dependency.get("lge_execution_published_at") or "")).total_seconds()
            except (TypeError, ValueError):
                age = float("inf")
            dispatchable=dependency.get("lge_dispatchable_pending",dependency.get("lge_auto_pending"))
            if (consumer.get("ready") and 0 <= age < 180
                    and isinstance(dispatchable,int) and dispatchable>0):
                if self.state.get("lge_consumer_wait_revision") != fingerprint:
                    self.state["lge_consumer_wait_revision"] = fingerprint
                    self.event("WAITING_FOR_LGE_CONSUMER", workflow_stage=dependency["stage"],
                               pending=dispatchable)
                return
        report_current, queue = check_iteration_matches_queue(ROOT, counts)
        if ((ROOT / "configs/lge_execution.json").exists()
                and dependent_workflow_status(self.dataset, ROOT)["stage"] != "cmr_paired"):
            report_current = True  # A CMR refresh cannot block the prerequisite LGE stage.
        if not report_current:
            wait_key = json.dumps(queue, sort_keys=True)
            if self.state.get("check_refresh_wait_key") != wait_key:
                self.state["check_refresh_wait_key"] = wait_key
                self.event("WAITING_FOR_CHECK_REFRESH", queue=queue)
            return
        if self.state.pop("check_refresh_wait_key", None) is not None:
            self.event("CHECK_REFRESHED", queue=queue)
        self.event("TICK", issue_fingerprint=fingerprint, issue_counts=counts)
        run = self.latest_experiment_run()
        if run is None:
            run_id = self.analyze_new_snapshot(None)
            run = self.db.get_run(run_id)
        outcome = self.process_actions(run["run_id"])
        if outcome.get("recovery_required"):
            return

        refreshed = self.db.run_details(run["run_id"])
        auto_rows = []
        for row in refreshed["maintenance_actions"]:
            payload = json.loads(row["payload_json"])
            if payload.get("auto_execute", False) and payload.get("action_class") != "human_review":
                auto_rows.append(row)
        # A review-only round with every proposed action removed by the
        # deterministic duplicate guard is also complete; do not analyze it
        # again forever.
        auto_done = ((bool(auto_rows) and all(row["status"] in TERMINAL_ACTIONS
                                              for row in auto_rows))
                     or (not auto_rows and refreshed["status"] in (
                         RunState.NEEDS_REVIEW.value, RunState.EXPERIMENT_PROPOSED.value)))
        new_fingerprint, new_counts = issue_fingerprint(self.dataset)
        issue_progress = new_fingerprint != self.state.get("last_issue_fingerprint")
        failure_signatures = [
            signature for result in outcome["results"]
            if (signature := production_failure_signature(result.get("validation", {})))
        ]
        new_production_failure = bool(
            failure_signatures
            and failure_signatures[-1] != self.state.get("last_production_failure_signature")
        )
        if failure_signatures:
            self.state["last_production_failure_signature"] = failure_signatures[-1]
        if issue_progress:
            self.state["stagnant_cycles"] = 0
        elif new_production_failure:
            self.state["stagnant_cycles"] = 0
            self.event("PRODUCTION_REPAIR_WINDOW_OPENED", source_run_id=run["run_id"],
                       failure_signature=failure_signatures[-1], issue_counts=new_counts)
        elif auto_done:
            done_key = f"run{run['run_id']}"
            if self.state.get("stagnant_key") != done_key:
                self.state["stagnant_key"] = done_key
                self.state["stagnant_cycles"] = int(self.state.get("stagnant_cycles", 0)) + 1
        self.state["last_issue_fingerprint"] = new_fingerprint

        transition = next_iteration_decision(
            auto_done=auto_done,
            stagnant_cycles=self.state["stagnant_cycles"],
            max_stagnant=self.args.max_stagnant,
            once=self.args.once,
        )
        if transition == "next_iteration":
            self.state["cycles"] = int(self.state.get("cycles", 0)) + 1
            self.event("NEXT_ITERATION", source_run_id=run["run_id"],
                       stagnant_cycles=self.state["stagnant_cycles"], issue_counts=new_counts)
            self.analyze_new_snapshot(run["run_id"])
        elif transition == "escalate":
            self.state["paused_issue_fingerprint"] = new_fingerprint
            if self.state.get("escalated_for") == run["run_id"]:
                return
            self.state["escalated_for"] = run["run_id"]
            self.event("ESCALATION_REQUIRED", source_run_id=run["run_id"],
                       reason="no_issue_progress_after_bounded_iterations",
                       waiting_human=outcome["waiting_human"], issue_counts=new_counts)
        elif transition == "one_shot_complete":
            self.event("ONE_SHOT_COMPLETE", source_run_id=run["run_id"],
                       stagnant_cycles=self.state["stagnant_cycles"], issue_counts=new_counts)
        else:
            self.event("ITERATION_ACTIVE", run_id=run["run_id"],
                       waiting_human=outcome["waiting_human"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--db", type=Path, default=ROOT / "runtime/workflow.sqlite")
    parser.add_argument("--providers", type=Path, default=ROOT / "configs/providers.yaml")
    parser.add_argument("--agents", type=Path, default=ROOT / "configs/agents.aliyun-roles.yaml")
    parser.add_argument("--state", type=Path, default=ROOT / "runtime/autopilot_state.json")
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--max-stagnant", type=int, default=3)
    parser.add_argument("--action-timeout", type=int, default=240)
    parser.add_argument("--extend-bounded-owner-budget", action="store_true", help="Explicit operator grant of one extra same-action attempt; requires bounded-run")
    parser.add_argument("--bounded-run", type=Path, help="Explicit one-analysis trial record; never starts label consumers")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--resume-paused-analysis-once", action="store_true",
                        help="Explicitly run one agent analysis chain; keep the pause fuse and skip owner execution")
    parser.add_argument("--resume-after-fix", action="store_true",
                        help="Record one audited retry after a committed framework fix; do not run agents")
    parser.add_argument("--defer-active-blocker", action="store_true",
                        help="With --resume-after-fix, quarantine the exhausted blocker for this unchanged queue")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    lock_path = ROOT / "runtime/autopilot.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("autopilot already running")
        pilot = Autopilot(args)
        if getattr(args, "bounded_run", None):
            if args.resume_after_fix or args.resume_paused_analysis_once or args.defer_active_blocker:
                raise ValueError("bounded-run cannot be combined with resume flags")
            if getattr(args, "extend_bounded_owner_budget", False):
                pilot.extend_bounded_owner_budget(args.bounded_run)
            pilot.bounded_run(args.bounded_run, stop_requested=lambda: stop_requested)
            return
        if getattr(args, "extend_bounded_owner_budget", False):
            raise ValueError("budget extension requires --bounded-run")
        if getattr(args, "resume_after_fix", False):
            pilot.resume_after_fix(
                defer_blocker=bool(getattr(args, "defer_active_blocker", False)))
            return
        if getattr(args, "defer_active_blocker", False):
            raise ValueError("--defer-active-blocker requires --resume-after-fix")
        if args.resume_paused_analysis_once:
            fingerprint, _ = issue_fingerprint(pilot.dataset)
            source_run_id = paused_analysis_source(pilot.state, fingerprint)
            try:
                pilot.event("RESUME_ANALYSIS_STARTED", source_run_id=source_run_id)
                run_id = pilot.analyze_new_snapshot(source_run_id)
                pilot.event("RESUME_ANALYSIS_FINISHED", run_id=run_id,
                            source_run_id=source_run_id)
            except Exception as exc:
                pilot.event("RESUME_ANALYSIS_FAILED", source_run_id=source_run_id,
                            error=f"{type(exc).__name__}: {str(exc) or repr(exc)}"[:1000])
                raise
            finally:
                # Never arm the daemon from this diagnostic one-shot. A later
                # operator decision must address the resulting action separately.
                pilot.state["paused_issue_fingerprint"] = fingerprint
                pilot.state["status"] = "ESCALATION_REQUIRED"
                write_json(pilot.state_path, pilot.state)
            return
        while not stop_requested:
            try:
                pilot.tick()
            except Exception as exc:
                pilot.event(
                    "AUTOPILOT_ERROR",
                    error=f"{type(exc).__name__}: {str(exc) or repr(exc)}"[:2000],
                )
            if args.once or stop_requested:
                break
            deadline = time.monotonic() + max(args.interval, 5.0)
            while not stop_requested and time.monotonic() < deadline:
                time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
        if stop_requested:
            pilot.event("CONTROLLER_STOPPED_AFTER_TICK")


if __name__ == "__main__":
    main()
