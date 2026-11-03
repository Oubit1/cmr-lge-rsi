from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

from app.rsi_quality_benchmark import benchmark_status
from app.lge_execution import status as lge_execution_status
from app.autonomy_feedback import feedback_summary
from app.rsi_generator_benchmark import status as generator_benchmark_status
from app.lge_protocol import repair_status as protocol_repair_status


PATIENT_COLUMNS = ("patient", "patient_id", "case", "subject")
ISSUE_COLUMNS = ("issues", "issue", "issue_type", "reason", "decision")
MODALITY_COLUMNS = ("modality",)
WORKFLOW_STAGES = ("lge_myo", "lge_mi", "paired_selection", "cmr_paired")


def semantic_action_workflow_stage(payload: dict[str, Any]) -> str | None:
    """Classify an action's declared target without interpreting medical data."""
    text = " ".join(
        str(payload.get(key) or "")
        for key in ("action_id", "objective", "hypothesis", "target_path")
    ).lower()
    if not text.strip():
        return None
    if "scripts/lge_candidate_generator.py" in text and payload.get("workflow_stage") in {"lge_myo", "lge_mi"}:
        # The same producer serves both LGE targets. Mentioning transferable
        # CMR priors in its hypothesis does not make this a CMR execution job.
        return payload["workflow_stage"]
    if "cmr" in text:
        return "cmr_paired"
    if any(token in text for token in ("paired_slice", "pairing", "pair selection")):
        return "paired_selection"
    if "lge" in text and any(token in text for token in ("_mi", " mi ", "infarct")):
        return "lge_mi"
    if "lge" in text and any(token in text for token in ("myo", "myocard")):
        return "lge_myo"
    return None


def action_matches_workflow_stage(payload: dict[str, Any], current_stage: str) -> bool:
    """Require both the persisted stage stamp and recognizable target to agree."""
    explicit = payload.get("workflow_stage")
    if explicit is not None and explicit != current_stage:
        return False
    semantic = semantic_action_workflow_stage(payload)
    return semantic is None or semantic == current_stage


def action_builds_stage_consumer(payload: dict[str, Any], *, require_generator: bool = False) -> bool:
    """Recognize a control-plane integration action, not its medical rule content."""
    text = json.dumps(payload, ensure_ascii=False).lower()
    generator = ("scripts/lge_candidate_generator.py" in text and any(marker in text for marker in (
        "generate_candidates", "lge_candidate_generator_contract")))
    if require_generator:
        return generator
    output_contract_named = any(marker in text for marker in (
        "runtime/lge_iterations/latest.json",
        "latest.json",
        "lge_stage_consumer_contract",
    ))
    return generator or ("scripts/lge_feedback_loop.py" in text and output_contract_named)


def action_uses_stale_lge_shadow_contract(payload: dict[str, Any]) -> bool:
    """Reject old bounded-scan denominators from a production consumer action."""
    # ``addresses_metric`` must quote an old reviewer metric verbatim for an
    # auditable continuation.  Judge only the proposed work contract, not that
    # historical citation.
    text = json.dumps({
        key: payload.get(key) for key in (
            "objective", "acceptance_metrics", "continuation_delta"
        )
    }, ensure_ascii=False).lower()
    patterns = (
        r"(?:==|=)\s*9\b",
        r"\b9\s*(?:个|条)?\s*(?:non[- ]?holdout|非留出|lge)",
        r"holdout_excluded_count\s*(?:==|=)\s*48\b",
        r"\b48\s*(?:个|条)?\s*(?:holdout|留出)",
    )
    return any(re.search(pattern, text) for pattern in patterns)


def enrich_workflow_dependency(dependency: dict[str, Any],
                               iteration_history: list[dict[str, Any]]) -> dict[str, Any]:
    """Add bounded progress requirements shared by planning and execution."""
    result = dict(dependency)
    stage_shadow_count = sum(
        1 for run in iteration_history for action in run.get("actions", [])
        if action.get("status") == "PROMOTED_SHADOW"
        and semantic_action_workflow_stage(action) == result["stage"]
    )
    consumer_missing = (
        result["stage"] == "lge_myo" and not result["lge_myo_consumer_ready"]
    ) or (
        result["stage"] == "lge_mi" and not result["lge_mi_consumer_ready"]
    )
    result["shadow_evidence_count"] = stage_shadow_count
    result["integration_required"] = consumer_missing and stage_shadow_count >= 2
    if result.get("lge_generator_required"):
        result["integration_required"] = True
        result["required_output"] = (
            "A tested scripts/lge_candidate_generator.py with generate_candidates(data, output_dir). "
            "It returns case_id, decision repair/accept_existing/needs_review, reason, and for repairs a "
            "relative candidate NIfTI filename. It reads snapshot image/label/paired-mask paths and writes "
            "only candidate output files. The independent controller owns frozen holdout, calibrated "
            "grading, geometry validation, label apply and audit; do not create another aggregate report adapter.")
        return result
    result["required_output"] = (
        "A tested scripts/lge_feedback_loop.py production consumer that continuously emits "
        "runtime/lge_iterations/latest.json. It must consume the active-stage LGE input, use "
        "the exact lge_myo_pending/lge_myo_work/lge_myo_resolved (or corresponding lge_mi) "
        "fields, reconcile them to the supplied stage queue, and retain zero-write audit and "
        "rollback evidence. It must not reuse a CMR denominator or hard-code a shadow-sample "
        "count. An experiments-only module does not satisfy this output."
        if result["integration_required"] else None
    )
    return result


def dependent_workflow_status(dataset_root: Path, project_root: Path) -> dict[str, Any]:
    """Return the deterministic LGE-first processing stage.

    Stage promotion is driven only by a persisted production-consumer report;
    model prose cannot advance the dependency chain. Missing or partial reports
    stay at the earliest unmet stage.
    """
    path = project_root / "runtime/lge_iterations/latest.json"
    try:
        report = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, json.JSONDecodeError):
        report = {}
    if not isinstance(report, dict):
        report = {}

    def nonnegative(name: str) -> int | None:
        value = report.get(name)
        return value if isinstance(value, int) and value >= 0 else None

    myo_pending = nonnegative("lge_myo_pending")
    myo_work = nonnegative("lge_myo_work")
    mi_pending = nonnegative("lge_mi_pending")
    mi_work = nonnegative("lge_mi_work")
    paired = nonnegative("paired_slice_count")
    myo_ready = report.get("lge_myo_consumer_ready") is True
    mi_ready = report.get("lge_mi_consumer_ready") is True
    audited_execution = report.get("schema_version") == "lge-execution-v1" and report.get("audited_execution") is True
    execution_policy = project_root / "configs/lge_execution.json"
    execution_required = False
    if execution_policy.is_file():
        try:
            declared = json.loads(execution_policy.read_text(encoding="utf-8"))["dataset"]
            execution_required = Path(declared).resolve() == dataset_root.resolve()
        except (OSError, ValueError, KeyError, TypeError):
            execution_required = True  # Invalid active policy cannot revive an old report's pass.
    if execution_required and not audited_execution:
        myo_ready = mi_ready = False
    pairing_ready = report.get("pairing_ready") is True and isinstance(paired, int) and paired > 0

    myo_done = report.get("lge_myo_auto_complete") is True if audited_execution else (
        myo_pending is not None and myo_work is not None and myo_pending + myo_work == 0)
    mi_done = report.get("lge_mi_auto_complete") is True if audited_execution else (
        mi_pending is not None and mi_work is not None and mi_pending + mi_work == 0)
    if not myo_ready or not myo_done:
        stage = "lge_myo"
    elif not mi_ready or not mi_done:
        stage = "lge_mi"
    elif not pairing_ready:
        stage = "paired_selection"
    else:
        stage = "cmr_paired"
    stage_report = (report.get("execution_status") or {}).get(stage, {})
    generator_improvement_required = bool(audited_execution and stage in {"lge_myo", "lge_mi"}
        and stage_report.get(stage + "_consumer_ready")
        and (stage_report.get(stage + "_execution_failed", 0) > 0
             or (stage_report.get(stage + "_auto_pending") == 0 and not stage_report.get(stage + "_auto_complete"))))
    return {
        "stage": stage,
        "protocol_repair_required": protocol_repair_status(project_root)["required"],
        "order": ["lge_myo", "lge_mi", "paired_selection", "cmr_paired"],
        "report_present": bool(report),
        "audited_execution": audited_execution,
        "lge_generator_required": (execution_required and not myo_ready) or generator_improvement_required,
        "lge_dispatchable_pending": sum(int((report.get("execution_status") or {}).get(name,{}).get(name+"_dispatchable",0))
            for name in ("lge_myo","lge_mi")
            if (report.get("execution_status") or {}).get(name,{}).get(name+"_consumer_ready")),
        "lge_generator_improvement_required": generator_improvement_required,
        "lge_auto_pending": stage_report.get(stage + "_auto_pending"),
        "lge_execution_published_at": report.get("published_at"),
        "lge_last_execution_mode": (report.get("last_execution") or {}).get("execution_mode"),
        "lge_myo_consumer_ready": myo_ready,
        "lge_myo_unresolved": None if myo_pending is None or myo_work is None else myo_pending + myo_work,
        "lge_mi_consumer_ready": mi_ready,
        "lge_mi_unresolved": None if mi_pending is None or mi_work is None else mi_pending + mi_work,
        "pairing_ready": pairing_ready,
        "paired_slice_count": paired,
        "active_input_contract": (
            report.get("active_input_contract")
            if isinstance(report.get("active_input_contract"), dict) else {}
        ),
        "transfer_prior": (
            "LGE myocardium may test already-validated CMR myocardium layer/topology priors. "
            "Do not transfer CMR temporal-frame adjacency constraints; measure LGE-specific "
            "validity before rule promotion."
        ),
        "contract": (
            "LGE myocardium precedes LGE infarct; stages advance from audited receipts, with frozen review samples and a bounded "
            "manual residual retained visibly. Pending automatic candidates cannot be skipped; only "
            "the persisted paired slice set may enter CMR myocardium repair."
        ),
    }


def approved_manual_edit_records(project_root: Path) -> list[dict[str, Any]]:
    """Use only analyzed edits that an operator actually applied and signed off."""
    manual = project_root / "runtime/manual_annotations"
    corpus = manual / "analyzed_training_corpus.jsonl"
    if not corpus.is_file():
        return []
    approved: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line in corpus.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        edit_id = record.get("edit_id")
        if not isinstance(edit_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", edit_id):
            continue
        if edit_id in seen:
            continue
        provenance = manual / "edits" / edit_id / "provenance.json"
        try:
            state = json.loads(provenance.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (not isinstance(state, dict) or state.get("edit_id") != edit_id
                or state.get("status") != "APPLIED" or not state.get("applied_by")
                or not state.get("applied_hash")):
            continue
        seen.add(edit_id)
        approved.append(record)
    return approved


def approved_manual_review_records(dataset_root: Path) -> list[dict[str, Any]]:
    """Return de-identified latest same-state human confirmations.

    A same-state decision means the operator inspected an existing checked or
    trash item and confirmed that disposition without drawing a mask delta.
    Only hashes, bounded issue codes and the disposition leave this function.
    """
    history = dataset_root / "labels/check/boundary_review_history.jsonl"
    manifest = dataset_root / "labels/check/manifest_backup/cmr_boundary_manifest.csv"
    if not history.is_file():
        return []
    issue_by_key: dict[str, list[str]] = {}
    if manifest.is_file():
        with manifest.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                patient = str(row.get("patient") or row.get("patient_id") or "").strip()
                volume = str(row.get("volume") or row.get("file") or "").strip()
                if not patient or not volume:
                    continue
                raw = str(row.get("issues") or row.get("issue") or "")
                issue_by_key[f"{patient}/{volume}"] = [
                    value for value in re.split(r"[;,]", raw)
                    if re.fullmatch(r"[a-z][a-z0-9_]{0,63}", value.strip())
                    for value in [value.strip()]
                ][:8]
    latest: dict[str, dict[str, Any]] = {}
    for line in history.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        key = event.get("relative_key")
        if isinstance(key, str) and 1 <= len(key) <= 500:
            latest[key] = event
    approved: list[dict[str, Any]] = []
    for key, event in latest.items():
        status = str(event.get("status") or "")
        if status not in {"trash", "checked"} or event.get("previous_status") != status:
            continue
        approved.append({
            "review_id": hashlib.sha256(key.encode("utf-8")).hexdigest()[:20],
            "modality": "cmr",
            "pool": "cmr_boundary_manifest",
            "decision": "confirmed_trash" if status == "trash" else "confirmed_checked",
            "issues": issue_by_key.get(key, []),
            "timestamp": str(event.get("timestamp") or "")[:40],
        })
    return sorted(approved, key=lambda item: item["review_id"])


def check_feedback_summary(dataset_root: Path, project_root: Path) -> dict[str, Any]:
    """Aggregate the live review queue and reviewed edit lessons without identifiers."""
    check = dataset_root / "labels/check"
    states = {
        state: sum(1 for _ in (check / state / "cmr").rglob("*.png"))
        for state in ("待筛", "待处理", "垃圾桶", "checked")
    }
    patterns: Counter[str] = Counter()
    operations: Counter[str] = Counter()
    lessons: dict[str, dict[str, Any]] = {}
    for record in approved_manual_edit_records(project_root):
        if record.get("modality") != "cmr" or record.get("target") != "myo":
            continue
        pattern = str(record.get("suggested_pattern") or "unknown")
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", pattern):
            pattern = "unknown"
        patterns[pattern] += 1
        if record.get("operation") in {"add", "remove", "mixed"}:
            operations[record["operation"]] += 1
        analysis = record.get("analysis") or {}
        if not isinstance(analysis, dict) or pattern in lessons:
            continue
        # These are model-written, de-identified rule descriptions from
        # the edit worker. Never forward raw edit records, pixels or IDs.
        fields = {}
        for key in ("likely_edit_intent", "reusable_casebook_pattern",
                    "qc_rule_candidate"):
            value = analysis.get(key)
            if not isinstance(value, str) or len(value) > 800:
                continue
            if re.search(r"(?:/data\d*/|/home/|/Users/|[A-Za-z]:\\)|[A-Z]+_[A-Z]+_[A-Z]+_\d{2,}", value):
                continue
            fields[key] = value
        if fields:
            lessons[pattern] = fields
    review_decisions: Counter[str] = Counter()
    review_issues: Counter[str] = Counter()
    for record in approved_manual_review_records(dataset_root):
        review_decisions[str(record["decision"])] += 1
        for issue in record.get("issues", []):
            review_issues[str(issue)] += 1
    return {
        "unit": "CMR spatial volume",
        "states": states,
        "reviewed_edit_patterns": dict(patterns.most_common()),
        "confirmed_edit_operations": dict(operations),
        "deidentified_edit_lessons": lessons,
        "confirmed_no_edit_reviews": dict(review_decisions.most_common()),
        "confirmed_review_issue_types": dict(review_issues.most_common()),
        "instruction": (
            "For the pending queue, propose a reusable casebook/tool match and a held-out "
            "validation experiment before escalating whole classes to human review. "
            "Only aggregate evidence is supplied; do not infer patient-specific edits."
        ),
    }


def check_iteration_evidence(project_root: Path) -> dict[str, Any]:
    """Expose measured outcomes to planning agents without sample IDs or paths."""
    path = project_root / "runtime/check_iterations/latest.json"
    if not path.is_file():
        return {}
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    allowed = ("pending", "work", "checked", "trash", "scanned_frames",
               "candidate_frames", "gate_passed_frames", "volumes_with_candidates",
               "counterfactual_salvage", "residual_pending", "residual_clusters",
               "control_gate_status", "spatial_boundary_candidates",
               "checked_false_routes", "pending_holdout", "pending_non_holdout",
               "pending_route_conflicts")
    result = {key: report[key] for key in allowed if key in report}
    result["gate_scope"] = {
        "control_gate_status": "Existing spatial-boundary rule for automatic trash routing only.",
        "not_global_veto": "FAIL does not prohibit isolated experiments or independently eligible checked candidates.",
        "checked_policy": "Changed candidates require validator plus SALVAGE or calibrated improvement; unchanged non-spatial labels may pass at calibrated score>=70. Frozen validation cases stay excluded.",
        "policy_source": "app/check_auto_apply.py:eligible_plan",
        "holdout_role": "20-case human adjudication is monitoring/feedback, not a per-candidate manual approval requirement under current operator authorization.",
    }
    false_routes = result.get("checked_false_routes")
    pending_candidates = result.get("spatial_boundary_candidates")
    route_conflicts = result.get("pending_route_conflicts")
    if (result.get("control_gate_status") == "PASS"
            and isinstance(route_conflicts, int) and route_conflicts > 0):
        result["production_blocker"] = {
            "priority": "RESOLVE_BEFORE_NEW_SHADOWS",
            "target_path": "app/check_auto_apply.py",
            "measured_predecessor": "runtime/check_iterations/latest.json",
            "required_gate": (
                "route_conflicts_unresolved<pending_route_conflicts and "
                "holdout_selected==0 and preflight_errors==0"),
            "owner": "qc_code_optimizer",
            "instruction": (
                "Assign one isolated production-path snapshot patch to the existing conflict "
                "resolver hook. The owner must choose the discriminator from aggregate scorer, "
                "candidate-validation and spatial-control evidence; the control plane must not "
                "choose a medical route. Preserve frozen holdout exclusion and dry-run preflight."
            ),
        }
    elif (result.get("control_gate_status") == "FAIL"
            and isinstance(false_routes, int) and false_routes > 0
            and isinstance(pending_candidates, int) and pending_candidates > 0):
        result["production_blocker"] = {
            "priority": "RESOLVE_BEFORE_NEW_SHADOWS",
            "target_path": "scripts/shadow_spatial_boundary.py",
            "measured_predecessor": "runtime/check_iterations/*/spatial_boundary_shadow.json",
            "required_gate": "checked_false_routes==0 and pending_spatial_boundary_candidates>0",
            "owner": "qc_code_optimizer",
            "instruction": (
                "Assign one isolated production-path snapshot patch. The owner must derive the "
                "discriminator from supplied anonymous spatial_control records; do not prescribe "
                "thresholds in the control plane and do not start another measurement-only shadow."
            ),
        }
    blocker = result.get("production_blocker")
    if isinstance(blocker, dict):
        state = {}
        try:
            state = json.loads((project_root / "runtime/autopilot_state.json").read_text(
                encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
        deferred = state.get("deferred_production_blocker")
        if (isinstance(deferred, dict)
                and deferred.get("target_path") == blocker.get("target_path")
                and deferred.get("issue_fingerprint") == state.get("last_issue_fingerprint")):
            result.pop("production_blocker", None)
            result["deferred_production_blocker"] = {
                "target_path": blocker["target_path"],
                "reason": "bounded_attempts_exhausted",
                "instruction": (
                    "Do not retry this target for the unchanged queue. Treat its conflicts as "
                    "abstained residuals and work another independently measurable issue class."
                ),
            }
    quality = report.get("candidate_quality")
    if isinstance(quality, dict):
        fields = ("scored_volumes", "improved", "degraded", "unchanged",
                  "unscorable", "median_delta", "min_delta")
        safe = {key: quality[key] for key in fields
                if isinstance(quality.get(key), (int, float))
                and not isinstance(quality[key], bool)}
        if quality.get("status") in {"SCORED", "NOT_RUN"}:
            safe["status"] = quality["status"]
        result["candidate_quality"] = safe
    return result


def check_validation_evidence(project_root: Path) -> dict[str, Any]:
    """Only aggregate operator judgments; never pass identities to providers."""
    path = project_root / "runtime/check_validation/report.json"
    if not path.is_file():
        return {"status": "NO_COMPLETED_HOLDOUT_ADJUDICATION"}
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"status": "INVALID_HOLDOUT_ADJUDICATION"}
    if not isinstance(report, dict) or not report.get("ready_for_independent_review"):
        return {"status": "INCOMPLETE_HOLDOUT_ADJUDICATION"}
    fields = ("selected", "reviewed", "suggested", "total_suggestions",
              "false_routes", "false_route_rate", "false_route_wilson_upper_95",
              "suggestion_coverage", "strata", "shadow_control_gate")
    return {"status": "COMPLETE", **{key: report[key] for key in fields if key in report}}


def safe_review_feedback(payload: dict[str, Any]) -> dict[str, Any]:
    """Return bounded reviewer feedback suitable for another provider call."""
    if not isinstance(payload, dict):
        return {}
    decision = payload.get("decision")
    if decision not in {"accept_experiment", "needs_human_review", "reject"}:
        return {}
    forbidden = re.compile(
        r"(?:/data\d*/|/home/|/Users/|/mnt/|/tmp/|[A-Za-z]:[\\/]"
        r"|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|[A-Z]{2,}_[A-Z0-9_]*\d{2,})"
    )

    def public_notes(key: str) -> list[str]:
        notes = payload.get(key)
        if not isinstance(notes, list):
            return []
        return [note for note in notes[:3]
                if isinstance(note, str) and 0 < len(note) <= 240
                and not forbidden.search(note) and "\n" not in note]

    return {
        "decision": decision,
        "reasons": public_notes("reasons"),
        "required_validation_metrics": public_notes("required_validation_metrics"),
    }


def validation_backlog(iteration_history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prioritize unmet independent-review evidence, never infer label success."""
    backlog = []
    for run in iteration_history:
        for action in run.get("actions", []):
            feedback = action.get("review_feedback") or {}
            metrics = feedback.get("required_validation_metrics") or []
            if not metrics or action.get("status") not in {"PROMOTED", "HUMAN_REQUIRED"}:
                continue
            backlog.append({
                "run_id": run.get("run_id"), "action_id": action.get("action_id"),
                "owner_role": action.get("owner_role"), "status": action.get("status"),
                "review_decision": feedback.get("decision"),
                "required_validation_metrics": metrics,
            })
            if len(backlog) >= 6:
                return backlog
    return backlog


def progress_ledger(iteration_history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compact, de-identified work state shared by all roles, not a new verdict."""
    ledger = []
    for run in iteration_history:
        for action in run.get("actions", []):
            feedback = action.get("review_feedback") or {}
            ledger.append({
                "run_id": run.get("run_id"),
                "action_id": action.get("action_id"),
                "action_class": action.get("action_class"),
                "status": action.get("status"),
                "objective": action.get("objective", ""),
                "review_decision": feedback.get("decision"),
                "review_reasons": feedback.get("reasons") or [],
                "unmet_metrics": feedback.get("required_validation_metrics") or [],
                "measured_result": action.get("measured_result"),
            })
            if len(ledger) >= 8:
                return ledger
    return ledger


def anonymous_case_id(value: str, salt: str) -> str:
    return "case_" + hashlib.sha256((salt + "\0" + value).encode("utf-8")).hexdigest()[:12]


def _first(row: dict[str, str], names: tuple[str, ...], default: str) -> str:
    for name in names:
        if row.get(name):
            return row[name]
    return default


def build_evidence_package(run_details: dict[str, Any], *, max_cases: int = 80) -> dict[str, Any]:
    salt = os.getenv("CMR_LGE_CASE_SALT")
    if not salt:
        raise RuntimeError("CMR_LGE_CASE_SALT is required to create stable anonymous case IDs")
    issue_counts: Counter[str] = Counter()
    issue_counts_by_modality: dict[str, Counter[str]] = {
        "cmr": Counter(), "lge": Counter(),
    }
    modalities: set[str] = set()
    representative: list[dict[str, str]] = []
    manifests = [item for item in run_details.get("artifacts", []) if "manifest" in item["artifact_type"]]
    for artifact in manifests:
        path = Path(artifact["path"])
        if not path.exists():
            continue
        fallback_modality = "cmr" if "cmr" in artifact["artifact_type"] else "lge"
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                patient = _first(row, PATIENT_COLUMNS, "unknown")
                issue = _first(row, ISSUE_COLUMNS, artifact["artifact_type"])
                modality = _first(row, MODALITY_COLUMNS, fallback_modality).lower()
                for tag in (part.strip() for part in issue.replace(";", "|").split("|")):
                    if tag:
                        issue_counts[tag] += 1
                        if modality in issue_counts_by_modality:
                            issue_counts_by_modality[modality][tag] += 1
                modalities.add(modality)
                if len(representative) < max_cases:
                    representative.append({
                        "case_id": anonymous_case_id(patient, salt),
                        "modality": modality,
                        "issue_type": issue,
                    })
    dataset = Path(str(run_details.get("config", {}).get("dataset_root") or ""))
    project_root = Path(__file__).resolve().parents[1]
    iteration_history = run_details.get("config", {}).get("iteration_history", [])
    dependency = enrich_workflow_dependency(
        dependent_workflow_status(dataset, project_root), iteration_history
    )
    check_iteration = check_iteration_evidence(project_root)
    if dependency["stage"] != "cmr_paired" and check_iteration.get("production_blocker"):
        check_iteration["deferred_cmr_production_blocker"] = check_iteration.pop(
            "production_blocker")
    workstreams = [
        {
            "key": "cmr_myo", "modality": "cmr",
            "specialist_role": "cmr_specialist",
            "issue_counts": dict(issue_counts_by_modality["cmr"].most_common()),
            "execution_status": "production_check_loop_available",
            "eligible_now": dependency["stage"] == "cmr_paired",
            "input_scope": "persisted paired_slice_count only",
        },
        {
            "key": "lge_myo", "modality": "lge",
            "specialist_role": "lge_myo_specialist",
            "issue_counts": dict(issue_counts_by_modality["lge"].most_common()),
            "execution_status": "audited_candidate_executor" if dependency.get("audited_execution") else "production_check_loop_missing",
            "eligible_now": dependency["stage"] == "lge_myo",
        },
        {
            "key": "lge_mi", "modality": "lge",
            "specialist_role": "lge_mi_specialist",
            "issue_counts": dict(issue_counts_by_modality["lge"].most_common()),
            "execution_status": "audited_candidate_executor" if dependency.get("audited_execution") else "production_check_loop_missing",
            "eligible_now": dependency["stage"] == "lge_mi",
        },
    ]
    workstreams = [item for item in workstreams if item["modality"] in modalities]
    return {
        "run_id": run_details["run_id"],
        "dataset_id": run_details["dataset_id"],
        "issue_counts": dict(issue_counts.most_common()),
        "modalities": sorted(modalities),
        "workstreams": workstreams,
        "workflow_dependency": dependency,
        "segmentation_lifecycle": __import__("app.segmentation_status", fromlist=["status"]).status(project_root),
        "segmentation_dispatch": __import__("app.segmentation_dispatch", fromlist=["proposal_context"]).proposal_context(project_root, run_details["run_id"]),
        "representative_cases": representative,
        "check_feedback": check_feedback_summary(dataset, project_root),
        "check_iteration": check_iteration,
        "check_validation": check_validation_evidence(project_root),
        "autonomous_execution_feedback": feedback_summary(project_root),
        "iteration_history": iteration_history,
        "role_continuity": run_details.get("config", {}).get("role_continuity", {}),
        "validation_backlog": validation_backlog(
            run_details.get("config", {}).get("iteration_history", [])),
        "progress_ledger": progress_ledger(
            run_details.get("config", {}).get("iteration_history", [])),
        "protocol_repair": protocol_repair_status(project_root),
        "execution_capabilities": {
            "lge_candidates": lge_execution_status(project_root),
            "owner": "qc_code_optimizer",
            "rsi_quality_benchmarks": benchmark_status(project_root),
            "rsi_generator_benchmark": generator_benchmark_status(project_root),
            "real_evaluation": "Owner may implement experiments.<module>.run_evaluation(data); executor runs it after tests and persists the actual report.",
            "available_input": "The active stage is selected by workflow_dependency. LGE has snapshotted issue manifests, checked image-label pairs, scoring tools and role-specific casebook/tool evidence. CMR production processing is permitted only after a persisted paired-slice set exists. Absent fields remain absent.",
            "unavailable_input": "No raw images, masks, patient identifiers or per-record independent reference verdicts are sent to the provider.",
            "purpose": "Prefer feasible measurement/report continuations using these inputs. Do not request precision or CI that requires unavailable truth; report those metrics unavailable.",
        },
        "active_work_scope": {
            "queue": "Dependency order: LGE myocardium -> LGE MI -> paired slice selection -> paired CMR myocardium only.",
            "success_measure": "Per-workstream audited movement from pending/work to checked/trash, with unresolved reasons preserved.",
            "issue_counts_scope": "Global issue_counts are historical manifests; use workstreams for modality ownership and never treat duplicate manifest rows as unique unresolved labels.",
            "scheduling": "Only the active dependency stage is analyzed. Parallelism is allowed across samples inside that stage, never across dependent stages.",
            "transfer_prior": dependency["transfer_prior"],
        },
        "iteration_instruction": (
            "Do not repeat completed or failed experiments without new evidence. Analyze only "
            "workflow_dependency.stage; later stages are ineligible. Parallelize samples only "
            "inside that stage. For LGE myocardium, CMR layer/topology priors are a testable "
            "starting point, but CMR temporal-frame adjacency must not be transferred. Select "
            "the highest-volume unresolved issue class in the active stage that is not already "
            "covered by iteration_history. "
            "Use execution_capabilities.rsi_quality_benchmarks for existing independent calibration "
            "and validation evidence. When it reports a measured weakness for the active stage, "
            "a named-component scoring experiment may address that metric; the configured domain "
            "Owner chooses the implementation. Reuse existing reviewed-pair evidence instead of "
            "requesting new per-case ground truth by default. A quality scoring action must name "
            "app/quality_scoring.py to receive its current source and training-only profile. "
            "Controlled-defect metrics are not clinical acceptance of pending labels. "
            "When workflow_dependency.integration_required is true, do not propose another "
            "measurement-only shadow scan. Build and validate the named stage consumer and its "
            "persisted audit report; domain thresholds remain owned by the specialist. The "
            "production denominator is the supplied active-stage input length. Ignore historical "
            "validation metrics that hard-code a prior 9-case shadow sample or 48-case CMR holdout. "
            "Address independent-review validation_backlog before proposing another similar rule. "
            "If check_iteration.production_blocker is present in the active CMR stage, it is the next control-plane "
            "priority: assign its named owner an isolated patch to its target_path before any "
            "new measurement-only shadow action. The domain owner chooses the rule from the "
            "anonymous spatial_control evidence and the deterministic gate decides promotion. "
            "If check_iteration reports pending volumes with zero counterfactual salvage, "
            "assign a bounded volume-level check experiment; code promotion alone is not label acceptance. "
            "When a measured shadow result identifies a concrete production consumer and the remaining blocker, the next action may request an isolated snapshot patch to that named path. It must preserve zero label writes and pass post-implementation independent review plus deterministic production gates."
        ),
        "privacy": "Identifiers and absolute paths removed; no source image included.",
    }
