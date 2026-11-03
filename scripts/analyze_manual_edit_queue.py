#!/usr/bin/env python3
"""Consume de-identified manual mask deltas and turn them into reusable guidance.

The worker never receives patient names, original paths, images, masks, or free
text notes.  It reads only the geometric/topological delta saved in
``agent_request.json`` and writes a structured analysis beside the staged edit.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contracts.compat import model_dump  # noqa: E402
from app.manual_annotation import execute_prompt_repair  # noqa: E402
from app.role_routing import (  # noqa: E402
    assert_role_owns,
    responsibility_for_label_task,
    role_for_label_task,
    validate_agent_config,
)
from contracts.models import ManualEditAnalysis, PromptRepairPlan  # noqa: E402
from providers.registry import ProviderRegistry  # noqa: E402


SYSTEM_PROMPT = """You analyze de-identified CMR/LGE manual segmentation deltas.
Input contains only modality, target, issue types, before/after topology, and
added/removed pixel geometry relative to the myocardial centroid. It contains no
image, mask, patient name, or source path. Describe what the human changed,
where it lies in image-coordinate myocardial polar coordinates, and abstract a
reusable but conservative casebook/prompt lesson. When prompt_supervision is
present, compare the prior agent decision with the human delta and propose a
more precise future prompt without copying identifiers or reviewer notes. Never invent AHA segments,
clinical diagnoses, intensity findings, or adjacent-slice evidence absent from
the input. A manual edit is evidence for review, not automatic authorization to
broaden production rules. Explicitly list contraindications and whether another
human review is needed. Return only the requested structured schema."""

PROMPT_REPAIR_SYSTEM = """You are the modality specialist responsible for one
de-identified CMR/LGE mask correction. Convert the operator instruction into the
smallest deterministic CURRENT-frame operation plan.

The request's qc_context is binding domain knowledge, not optional advice. Obey
its hard_constraints, evidence_priority, allowed_operations and forbidden_operations.
Use the operator instruction first, then bilateral neighbor evidence, current
topology/geometry, matching reviewed demonstrations and iteration history. Never
invent image intensity, diagnosis, AHA segments or evidence absent from input.
When mode is refine, keep the original operator instruction unchanged. The only
new feedback is that the previous candidate was unsatisfactory. Reflect on its
plan and supplied candidate geometry; choose a safer/different supported plan
if possible, or return manual_required. Do not invent a specific clinical error.

Decision order:
1. no_change only when supplied geometry already satisfies the instruction;
2. apply_operations only when every operation is allowlisted and supported;
3. manual_required when evidence conflicts or a hard constraint prevents a safe plan.

Use at most three operations. Prefer polygon/box/circle primitives for a local
boundary. For a local gap explicitly supported by previous and next frame/layer,
use fill_from_adjacent_consensus; never use global dilation. CMR may legitimately
have two dominant LV/RV components, so never collapse it to one component. Myo
blood-pool holes are physiological and must not be filled as defects. MI must stay
inside Myo; separated MI territories and a high MI fraction are not errors by
themselves.

Return compact JSON only. rationale and expected_effect must be Simplified Chinese,
each <=60 characters. evidence_used must name the supplied evidence actually used;
constraints_checked must list the QC constraints checked. This creates a staging
candidate only and never authorizes a production-label write."""

PROMPT_OPERATIONS_V2 = [
    "keep_largest_component", "remove_components_below_area",
    "fill_internal_holes_below_area", "binary_close", "binary_dilate",
    "binary_erode", "clip_mi_to_myo", "remove_components_in_region",
    "keep_components_in_region", "erase_region_pixels", "erase_box",
    "fill_box", "erase_circle", "fill_circle", "erase_polygon",
    "fill_polygon", "fill_from_adjacent_consensus",
]


def _prompt_timeout_seconds() -> int:
    return max(15, min(int(os.getenv("CMR_LGE_EDIT_PROMPT_TIMEOUT", "35")), 60))


def enforce_prompt_policy(request: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    """Reject unsafe/ungrounded plans locally without another slow model call."""
    plan = dict(plan)
    policy = request.get("qc_context") or {}
    allowed = set(request.get("allowed_operations") or PROMPT_OPERATIONS_V2)
    forbidden = set(policy.get("forbidden_operations") or [])
    operations = list(plan.get("operations") or [])[:3]
    violations: list[str] = []
    if plan.get("target") != request.get("target"):
        violations.append("target_mismatch")
    for operation in operations:
        name = str(operation.get("operation") or "")
        if name not in allowed or name in forbidden:
            violations.append(f"operation_not_allowed:{name}")
        if name == "fill_from_adjacent_consensus" and len(request.get("neighbors") or []) < 2:
            violations.append("bilateral_neighbor_support_missing")
        if (request.get("modality") == "cmr"
                and name == "remove_components_below_area"
                and int((request.get("current") or {}).get("components") or 0) < 3):
            violations.append("cmr_top_two_components_must_be_preserved")
    if plan.get("decision") == "apply_operations" and not operations:
        violations.append("empty_operation_plan")
    if violations:
        return {
            **plan,
            "decision": "manual_required",
            "operations": [],
            "rationale": "计划违反当前QC约束，已阻止自动候选。",
            "expected_effect": "保留原Mask，等待更精确指令或人工编辑。",
            "risk_flags": sorted(set(list(plan.get("risk_flags") or []) + violations)),
            "constraints_checked": list(policy.get("hard_constraints") or [])[:6],
            "confidence": min(float(plan.get("confidence") or 0.0), 0.4),
        }
    plan["operations"] = operations
    return plan


def deterministic_prompt_plan(request: dict[str, Any]) -> dict[str, Any] | None:
    """Resolve narrow, high-confidence intents without a slow provider call."""
    instruction = str(request.get("operator_instruction") or "").lower()
    target = str(request.get("target") or "")
    adjacent_words = ("相邻", "前后", "上下", "neighbor", "adjacent")
    fill_words = ("补", "填", "修复", "fill", "repair")
    gap_words = ("缺口", "缺损", "断裂", "gap", "notch")
    region_map = {
        "左上": "left_upper", "右上": "right_upper",
        "左下": "left_lower", "右下": "right_lower",
    }
    region = next((value for key, value in region_map.items() if key in instruction), None)
    if (target == "myo" and len(request.get("neighbors") or []) == 2
            and any(word in instruction for word in adjacent_words)
            and any(word in instruction for word in fill_words)
            and any(word in instruction for word in gap_words)
            and region is not None):
        return {
            "decision": "apply_operations", "target": target,
            "operations": [{"operation": "fill_from_adjacent_consensus",
                            "radius": 3, "region": region}],
            "rationale": "双侧相邻证据支持指定区域的局部补全。",
            "expected_effect": "仅补指定象限内、靠近当前轮廓的共识像素。",
            "risk_flags": ["visual_review_required"],
            "evidence_used": ["operator_instruction", "both_adjacent_masks",
                              "current_mask_geometry"],
            "constraints_checked": ["local_delta_only", "preserve_dominant_components",
                                    "preserve_blood_pool_holes"],
            "confidence": 0.9,
        }
    if target == "mi" and any(word in instruction for word in (
            "心肌外", "超出心肌", "裁剪到心肌", "outside myo", "clip to myo")):
        return {
            "decision": "apply_operations", "target": target,
            "operations": [{"operation": "clip_mi_to_myo"}],
            "rationale": "MI超出心肌的像素可按包含约束确定性裁剪。",
            "expected_effect": "仅删除Myo外MI像素。",
            "risk_flags": ["visual_review_required"],
            "evidence_used": ["operator_instruction", "myocardium_constraint"],
            "constraints_checked": ["mi_inside_myo"],
            "confidence": 0.98,
        }
    return None


def _update_local_status(edit_dir: Path, status: str, **extra: Any) -> None:
    path = edit_dir / "local_request.json"
    if not path.exists():
        return
    local = json.loads(path.read_text(encoding="utf-8"))
    local.update({"status": status, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **extra})
    from app.control_checkpoint import save_journal
    save_journal(path,local)


def expand_env(value: Any) -> Any:
    if isinstance(value, str):
        match = re.fullmatch(
            r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?:\|([A-Za-z_][A-Za-z0-9_]*))?\}", value)
        if match:
            primary, fallback = match.groups()
            resolved = os.getenv(primary) or (os.getenv(fallback) if fallback else None)
            if not resolved:
                raise RuntimeError(f"missing model environment variable: {primary}"
                                   + (f" (and fallback {fallback})" if fallback else ""))
            return resolved
    if isinstance(value, dict):
        return {key: expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_env(item) for item in value]
    return value


async def analyze(request: dict[str, Any], providers: Path, agents: Path) -> dict[str, Any]:
    configs = expand_env(yaml.safe_load(agents.read_text(encoding="utf-8")))["agents"]
    role = role_for_label_task(request)
    validate_agent_config(configs, required_roles={role})
    assert_role_owns(role, responsibility_for_label_task(request, prompt=False))
    config = configs[role]
    provider = ProviderRegistry(providers).get(config["provider"])
    result = await provider.generate_structured(
        system_prompt=SYSTEM_PROMPT,
        input_data={
            "role": role,
            "manual_edit_delta": request,
            "privacy_contract": {
                "identifiers": False,
                "paths": False,
                "images_or_masks": False,
                "free_text_note": False,
            },
        },
        output_schema=ManualEditAnalysis,
        model=config["model"],
    )
    return {"role": role, "provider": config["provider"], "model": config["model"],
            "analysis": model_dump(result)}


async def plan_prompt_repair(request: dict[str, Any], providers: Path,
                             agents: Path, request_budget=None) -> dict[str, Any]:
    # Migrate queued v1 jobs in memory so a service restart can use the latest
    # deterministic tools without exposing or rewriting local identifiers.
    request = dict(request)
    request["schema_version"] = "prompt-repair-v2"
    request.setdefault("allowed_operations", PROMPT_OPERATIONS_V2)
    configs = expand_env(yaml.safe_load(agents.read_text(encoding="utf-8")))["agents"]
    role = role_for_label_task(request)
    validate_agent_config(configs, required_roles={role})
    assert_role_owns(role, responsibility_for_label_task(request, prompt=True))
    # A refinement must consider the failed candidate; replaying the same
    # deterministic first-pass plan would defeat the operator's feedback.
    deterministic = None if request.get("mode") == "refine" else deterministic_prompt_plan(request)
    if deterministic is not None:
        return {"role": role, "provider": "local", "model": "deterministic-qc-router-v1",
                "plan": enforce_prompt_policy(request, deterministic)}
    config = configs[role]
    provider = ProviderRegistry(providers).get(config["provider"])
    result = await provider.generate_structured(
        system_prompt=PROMPT_REPAIR_SYSTEM,
        input_data={
            "role": role,
            "prompt_repair": request,
            "privacy_contract": {
                "identifiers": False, "paths": False,
                "image_pixels": False, "mask_geometry_grid": True,
            },
            "_max_output_tokens": 1200,
            "_fallback_on_error": False,
            "_task_kind": "prompt_repair",
            "_request_budget_callback": request_budget,
        },
        output_schema=PromptRepairPlan,
        model=config["model"],
        timeout_seconds=_prompt_timeout_seconds(),
    )
    plan = enforce_prompt_policy(request, model_dump(result))
    return {"role": role, "provider": config["provider"], "model": config["model"],
            "plan": plan}


def claim_one(work_root: Path, job_id: str | None=None) -> tuple[str, Path, Path] | None:
    sources = [
        ("prompt", work_root / "prompt_jobs"),
        ("manual_edit", work_root / "edits"),
    ]
    if job_id:
        from app.prompt_dispatch import job_folder
        folder=job_folder(work_root,job_id)
        sources=[("prompt",folder.parent)]
    for kind, source in sources:
        paths=[folder/"agent_request.json"] if job_id else sorted(source.glob("*/agent_request.json"))
        for request in paths:
            from app.prompt_dispatch import job_lock
            # Recovery must not see the rename before the live claim is durable.
            with job_lock(request.parent) if kind == "prompt" else nullcontext():
                if kind == "prompt":
                    local_path=request.parent/"local_request.json"
                    local=json.loads(local_path.read_text(encoding="utf-8")) if local_path.exists() else {}
                    if local.get("status") in {"READY","NO_CHANGE","MANUAL_REQUIRED","ERROR"}:continue
                    if not job_id and (local.get("execution_scope")=="interactive_prompt" or (request.parent/"interactive_budget.json").exists()):
                        continue
                if (request.parent / "agent_analysis.json").exists():
                    continue
                if kind == "prompt" and (request.parent / "candidate.png").exists():
                    continue
                claimed = request.with_suffix(".running")
                try:
                    request.rename(claimed)
                except FileNotFoundError:
                    continue
                if kind == "prompt":
                    from app.prompt_dispatch import register_claim
                    register_claim(request.parent)
                return kind, request, claimed
    return None


def recover_stale_claims(work_root: Path, stale_seconds: int | None = None) -> int:
    """Recover abandoned claims once at worker startup.

    A live worker owns its ``.running`` file. Recovery shares the prompt claim
    lock with explicit workers, including the rename-to-publication interval.
    """
    if stale_seconds is None:
        # This function runs before the new worker enters its polling loop;
        # every extant claim therefore belongs to the previous dead process.
        stale_seconds = 0
    recovered = 0
    now = time.time()
    for source in (work_root / "prompt_jobs", work_root / "edits"):
        for running in sorted(source.glob("*/agent_request.running")):
            edit_dir = running.parent
            from app.prompt_dispatch import job_lock, worker_alive
            with job_lock(edit_dir) if source.name == "prompt_jobs" else nullcontext():
                if not running.exists():
                    continue
                if source.name == "prompt_jobs" and worker_alive(edit_dir):
                    continue
                local_path=edit_dir/"local_request.json"
                local=json.loads(local_path.read_text(encoding="utf-8")) if local_path.exists() else {}
                terminal=any((edit_dir/name).exists() for name in ("candidate.png","agent_analysis.json")) or (
                    source.name=="prompt_jobs" and local.get("status") in {"READY","NO_CHANGE","MANUAL_REQUIRED","ERROR"})
                if terminal:
                    queued=edit_dir/"agent_request.json"
                    if source.name=="prompt_jobs" and not queued.exists():running.rename(queued)
                    else:running.unlink(missing_ok=True)
                    continue
                if now - running.stat().st_mtime < stale_seconds:
                    continue
                queued = edit_dir / "agent_request.json"
                if not queued.exists():
                    running.rename(queued)
                    if source.name == "prompt_jobs":
                        _update_local_status(edit_dir, "QUEUED", recovery="stale_claim")
                    recovered += 1
    return recovered


def process_one(work_root: Path, providers: Path, agents: Path, job_id: str | None=None) -> bool:
    claimed = claim_one(work_root,job_id) if job_id else claim_one(work_root)
    if claimed is None:
        return False
    kind, request_path, running_path = claimed
    edit_dir = running_path.parent
    request: dict[str, Any] = {}
    try:
        request = json.loads(running_path.read_text(encoding="utf-8"))
        if kind == "prompt":
            from app.manual_annotation import validate_prompt_source
            validate_prompt_source(work_root,edit_dir.name)
            _update_local_status(edit_dir, "PROCESSING", started_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
            plan_path = edit_dir / "agent_plan.json"
            if plan_path.exists():
                output = json.loads(plan_path.read_text(encoding="utf-8"))
                result = {key: output[key] for key in ("role", "provider", "model", "plan")}
            else:
                planning_started = time.perf_counter()
                result = asyncio.run(asyncio.wait_for(
                    plan_prompt_repair(request, providers, agents, request_budget=(
                        (lambda: __import__('app.prompt_dispatch',fromlist=['reserve']).reserve(work_root,edit_dir.name))
                        if (edit_dir/'interactive_budget.json').exists() else None)),
                    timeout=_prompt_timeout_seconds() + 5,
                ))
                output = {
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "planning_latency_ms": round((time.perf_counter() - planning_started) * 1000, 1),
                    **result,
                }
                from app.control_checkpoint import save_journal
                save_journal(plan_path,output)
            outcome = execute_prompt_repair(work_root, edit_dir.name, result["plan"])
            (edit_dir / "agent_error.json").unlink(missing_ok=True)
            running_path.rename(request_path)  # Keep the immutable prompt for refinement and edit learning.
            print(json.dumps({"job_id": edit_dir.name, "status": outcome["status"], "role": result["role"]},
                             ensure_ascii=False), flush=True)
            return True
        result = asyncio.run(analyze(request, providers, agents))
        output = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **result}
        (edit_dir / "agent_analysis.json").write_text(
            json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with (work_root / "analyzed_training_corpus.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({**request, **output}, ensure_ascii=False) + "\n")
        running_path.unlink()
        print(json.dumps({"edit_id": edit_dir.name, "status": "ANALYZED", "role": result["role"]},
                         ensure_ascii=False), flush=True)
    except Exception as exc:
        error = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "error": str(exc)}
        (edit_dir / "agent_error.json").write_text(
            json.dumps(error, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        request = request if isinstance(request, dict) else {}
        retries = int(request.get("retry_count") or 0) + 1
        stale_source = kind == "prompt" and "当前标签已被其他流程更新" in str(exc)
        terminal_safety = kind == "prompt" and "manual review required" in str(exc).lower()
        max_retries = 2 if kind == "prompt" else 3
        if terminal_safety or stale_source or retries >= max_retries:
            if kind == "prompt":
                _update_local_status(
                    edit_dir, "MANUAL_REQUIRED" if terminal_safety else "ERROR",
                    retry_count=retries, terminal_error=str(exc)[:1000],
                )
            if kind == "prompt":running_path.rename(request_path)
            else:running_path.unlink(missing_ok=True)
        else:
            request["retry_count"] = retries
            running_path.write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
            running_path.rename(request_path)
            if kind == "prompt":
                _update_local_status(edit_dir, "RETRYING", retry_count=retries,
                                     last_error=str(exc)[:1000])
        print(json.dumps({"job_id" if kind == "prompt" else "edit_id": edit_dir.name,
                          "status": "MANUAL_REQUIRED" if terminal_safety else "ERROR",
                          "retry_count": retries, "error": str(exc)},
                         ensure_ascii=False), flush=True)
        if not terminal_safety:
            time.sleep(2)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, default=ROOT / "runtime/manual_annotations")
    parser.add_argument("--providers", type=Path, default=ROOT / "configs/providers.yaml")
    parser.add_argument("--agents", type=Path, default=ROOT / "configs/agents.production.yaml")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--job-id", help="Process only this explicit editor prompt; no backlog learning")
    args = parser.parse_args()
    startup_configs = expand_env(yaml.safe_load(args.agents.read_text(encoding="utf-8")))["agents"]
    validate_agent_config(startup_configs)
    args.work_root.mkdir(parents=True, exist_ok=True)
    recovered = 0 if args.job_id else recover_stale_claims(args.work_root)
    if recovered:
        print(json.dumps({"status": "RECOVERED_STALE_CLAIMS", "count": recovered}), flush=True)
    if args.job_id:
        for _ in range(2):
            if not process_one(args.work_root,args.providers,args.agents,job_id=args.job_id):break
        return
    while True:
        processed = process_one(args.work_root, args.providers, args.agents)
        if not args.watch:
            break
        if not processed:
            time.sleep(max(1.0, args.poll_seconds))


if __name__ == "__main__":
    main()
