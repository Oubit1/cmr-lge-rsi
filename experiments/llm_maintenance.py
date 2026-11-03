"""Maintenance executor backed by the configured LLM provider.

The orchestrator originally delegated code changes to the Codex CLI. That is an
external agent the project never onboarded, so this module replaces it: the same
isolated-snapshot contract, but the implementation is produced by whichever
provider/model the agent profile assigns (Aliyun DashScope by default).

Guarantees kept identical to the old path:
  * work happens in a disposable copy, never in the production tree
  * only allow-listed directories may be written
  * the snapshot's own unit tests must pass before anything is marked ready
"""
from __future__ import annotations

import asyncio
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, List

import yaml
from pydantic import BaseModel

from agents.shadow_analysis import expand_env
from agents.invocation import invoke_traced
from app.database import Database
from app.rsi_quality_benchmark import owner_context
from app.autonomy_feedback import feedback_summary
from providers.registry import ProviderRegistry
from experiments.runtime_evaluation import evaluation_input, execute_evaluation

ALLOWED_PREFIXES = ("experiments/", "tests/", "scripts/", "configs/", "agents/", "app/")
MAX_FILES = 2
MAX_CHARS = 40_000
ROLE_ALIASES = {"lge_specialist": "lge_myo_specialist"}

LGE_CONSUMER_TARGET = "scripts/lge_feedback_loop.py"
LGE_GENERATOR_TARGET = "scripts/lge_candidate_generator.py"
PROTOCOL_CONFIG = "configs/lge_protocol_aliases.json"
PROTOCOL_TEST = "tests/test_lge_protocol_aliases.py"
LGE_CONSUMER_CONTRACT = {
    "entrypoint": "run_evaluation(data: dict) -> dict",
    "input_contract": [
        "Read workflow_stage from data['workflow_stage']; records do not contain workflow_stage.",
        "Consume data['records'] directly and preserve data['source_sha256'] in the report.",
        "Use only data['holdout_case_ids']; never embed case IDs or a holdout count.",
    ],
    "holdout_contract": (
        "When data['frozen_split']['status'] is UNAVAILABLE, report "
        "frozen_split_status='UNAVAILABLE' and holdout_excluded=0."
    ),
    "required_report_fields": [
        "source_modality='lge'", "workflow_stage", "input_source_sha256",
        "<stage>_consumer_ready", "<stage>_pending", "<stage>_work",
        "<stage>_resolved", "frozen_split_status", "holdout_excluded",
        "label_write=false", "production_writes=0",
    ],
    "count_contract": "pending + work + resolved must equal len(data['records']).",
    "write_boundary": (
        "run_evaluation returns a report only and must not write labels or runtime state; "
        "the control-plane driver persists a validated report atomically."
    ),
    "test_contract": [
        "Tests must call run_evaluation(data) and assert only the returned report; "
        "they must not expect run_evaluation to persist a file or runtime state.",
        "For frozen_split.status='UNAVAILABLE', tests must require "
        "frozen_split_status='UNAVAILABLE' and holdout_excluded=0.",
        "<stage>_consumer_ready must be a boolean derived from the supplied records; "
        "tests must not force it to true or use it to bypass unresolved records.",
        "pending, work and resolved are mutually exclusive and must reconcile exactly once "
        "to len(data['records']).",
    ],
}


def _with_lge_consumer_contract(action: dict[str, Any]) -> dict[str, Any]:
    """Attach the immutable production adapter contract to the assigned owner."""
    normalized = dict(action)
    requirements = dict(action.get("control_plane_requirements") or {})
    if requirements.get("required_target_path") == PROTOCOL_CONFIG:
        from app.lge_protocol import DEFAULT
        requirements["protocol_contract"] = {"schema":DEFAULT,"scope":[PROTOCOL_CONFIG,PROTOCOL_TEST],
            "invariants":"Canonical identities/decisions/reasons, source paths, clinical thresholds, writer, evaluator and budgets are immutable. Only add field aliases to existing image/label/myo/mi paths and candidate filename. Full regression and independent review required."}
        normalized["control_plane_requirements"] = requirements
        return normalized
    if requirements.get("required_target_path") == LGE_GENERATOR_TARGET:
        requirements["required_interface"] = "generate_candidates(data: dict, output_dir: str) -> list[dict]"
        requirements["lge_candidate_contract"] = {
            "input": "workflow_stage, calibration_profile, records with case_id, issue_codes, measurements and paths",
            "record_paths": {"paths": {"image": "snapshot image NIfTI", "label": "snapshot target NIfTI",
                                        "myo": "paired Myo for MI stage", "mi": "paired MI for Myo stage"},
                             "aliases": {"snapshot_image_path": "paths.image", "snapshot_mask_path": "paths.label",
                                         "snapshot_myo_path": "paths.myo when supplied", "snapshot_mi_path": "paths.mi when supplied"}},
            "test_runtime": "Use unittest (pytest is not installed); test supplied snapshot paths, not invented field names. Snapshot geometry is unchanged: the interface probe supplies 2D, 3D singleton and 3D multislice NIfTI arrays for both stages. Unsupported inputs must return an explicit needs_review decision without crashing the batch.",
            "decisions": ["repair", "accept_existing", "needs_review"],
            "repair_output": "case_id, decision, reason, candidate filename relative to output_dir (candidate_filename is a supported alias)",
            "other_output": "case_id, decision, reason; return exactly one decision per supplied record",
            "boundary": "No provider calls or source-label writes. Controller alone grades, excludes holdout, applies and audits.",
            "validation": "Binary NIfTI, original shape/affine; independent grading and paired-mask checks occur after generation. Promotion additionally compares the candidate against the frozen parent on doctor-reference copies and existing controlled defects for both LGE targets; preserve clean labels and all measured defect-class gains. Reference answers are never supplied to the producer or provider.",
        }
        normalized["control_plane_requirements"] = requirements
        return normalized
    if requirements.get("required_target_path") != LGE_CONSUMER_TARGET:
        return normalized
    requirements["required_interface"] = "run_evaluation(data: dict) -> dict"
    requirements["lge_consumer_contract"] = LGE_CONSUMER_CONTRACT
    normalized["control_plane_requirements"] = requirements
    return normalized


def _lge_candidate_tools(root: Path) -> list[dict]:
    """Share actual existing tool signatures, never patient data or invented rules."""
    path = root / "experiments/lge_shadow.py"
    if not path.is_file():
        return []
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {"extract_slice_features", "extract_case_features", "simulate_candidates", "ShadowParameters"}
    return [{"module": "experiments.lge_shadow", "name": node.name,
             "signature": ast.unparse(node.args)[:700] if isinstance(node, ast.FunctionDef) else "dataclass parameters; inspect defaults in existing source",
             "description": (ast.get_docstring(node) or "")[:250]}
            for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]


def resolve_provider_role(agents: dict[str, Any], requested: str) -> str:
    """Never silently execute a supervisor action as a different owner."""
    resolved = ROLE_ALIASES.get(requested, requested)
    if resolved not in agents:
        raise ValueError(f"unconfigured maintenance owner role: {requested}")
    return resolved

SYSTEM_PROMPT = """You are a bounded maintenance engineer for a CMR/LGE cardiac-MRI QC
orchestrator. You receive one approved maintenance action plus its acceptance metrics.

Hard rules:
- Return COMPLETE file contents, never diffs.
- Only create or modify files under experiments/ or tests/ (scripts/, configs/,
  agents/, app/ only when the objective explicitly says so).
- Never propose touching datasets, patient images, production labels, credentials or
  runtime databases. Never weaken a QC safety check.
- Prefer one small, self-contained module plus a unittest file that can run offline
  without downloading anything.
- Return at most two files and keep each file below 250 lines. Reuse existing helpers;
  do not generate a second harness for a problem already covered in the request.
- If action.control_plane_requirements.required_target_path is present, the FIRST
  FileEdit.path must equal that path exactly and contain its complete updated source.
  An experiments-only report, validator, holdout, rule draft or test does not satisfy
  this action. Use the optional second file only for a focused test. Omitting the
  required target is a failed action, even if measurements or tests are useful.
- Preserve every interface named by action.control_plane_requirements.required_interface.
  In particular, do not replace an existing command-line entrypoint with only a library
  function or test helper; the production replay gate invokes the command directly and
  requires its JSON output contract.
- Obey action.control_plane_requirements.required_edit_scope literally. The target's
  existing loaders and entrypoint are part of the production contract; do not rewrite
  them into a standalone experiment even if that would be shorter.
- required_target_source contains the current complete source for the mandatory target.
  Use that exact source as the base and make the smallest necessary edit; do not recreate
  the file from memory or from experiment modules.
- If action.control_plane_requirements.lge_consumer_contract is present, it is an
  immutable machine-checked interface and overrides older handoff prose. Implement
  every named field exactly. Never invent a holdout split, repeat workflow_stage on
  records, embed pseudonymous case IDs, or write runtime state from run_evaluation.
  The optional test file is subordinate to this contract: test the pure returned
  report, never expect persistence, never force consumer_ready true, and preserve
  frozen_split.status='UNAVAILABLE' exactly as UNAVAILABLE with holdout_excluded=0.
- Set ready_for_validation true only if you added at least one test that passes
  without external data.
- If real_evaluation_input.status is AVAILABLE, implement run_evaluation(data) in
  an experiments module, or in the explicitly requested scripts module for a read-only
  shadow scanner, and set evaluation_module to its dotted name. This
  function receives that exact JSON and returns a JSON object; the executor runs
  it on the real supplied measurements after tests. Every pending CMR volume may
  include a pseudonymous case_id; holdout_case_ids and frozen_split identify the
  frozen evaluation cohort without exposing patient identity. Exclude those IDs
  when the objective explicitly targets non-holdout pending work.
  include the existing score, flags and numeric subscores; changed candidates may
  also include before/after area and cavity values. quality_benchmark contains
  aggregate gold-derived score/defect validation, not per-record truth. State
  spatial_control contains anonymous checked-control and pending-candidate rank/core
  records from the existing zero-write spatial replay. Use it to measure a proposed
  discriminator; never infer identity. State
  unsupported metrics as unavailable. Do not invent images, masks, reference
  labels or clinical accuracy. Reuse supplied source interfaces.
"""


class FileEdit(BaseModel):
    path: str
    content: str


class MaintenancePatch(BaseModel):
    summary: str
    files: List[FileEdit]
    tests_run: List[str] = []
    remaining_risks: List[str] = []
    ready_for_validation: bool = False
    evaluation_module: str | None = None


COPY_IGNORE = shutil.ignore_patterns(
    ".git", "__pycache__", "*.pyc", "runtime", "artifacts", "*.sqlite", "*.sqlite-*",
    "models", "vendor", "archive", "label_iteration", "*.nii", "*.nii.gz", "*.dcm", "*.png",
)


def _copy_source(src: Path, dst: Path, *, include: tuple[str, ...] | None = None) -> None:
    src = src.resolve()
    if include is None:
        shutil.copytree(src, dst, ignore=COPY_IGNORE)
        return
    dst.mkdir(parents=True)
    for name in include:
        item = src / name
        if not item.exists():
            continue
        target = dst / name
        if item.is_dir():
            shutil.copytree(item, target, ignore=COPY_IGNORE)
        else:
            shutil.copy2(item, target)


def _baseline_hashes(root: Path, owner: str) -> dict[str, str]:
    return {
        f"{owner}/{path.relative_to(root).as_posix()}": hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*") if path.is_file()
    }


def _run_tests(workspace: Path, changed: list[str], timeout: int = 900) -> tuple[bool, str]:
    """Run only the tests this patch added or touched.

    A full discover in the snapshot also runs pre-existing tests that depend on
    the production skill path, which would fail for unrelated reasons and make
    every valid patch look broken.
    """
    modules = []
    for rel in changed:
        if rel.startswith("tests/") and rel.endswith(".py"):
            modules.append(".".join(Path(rel).with_suffix("").parts))
    command = [sys.executable, "-m", "unittest", *modules] if modules else [
        sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", "."
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(workspace) + os.pathsep + env.get("PYTHONPATH", "")
    try:
        result = subprocess.run(
            command,
            cwd=workspace, env=env, text=True, capture_output=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, "test run timed out"
    output = (result.stdout or "") + (result.stderr or "")
    return result.returncode == 0, output[-4000:]


def _run_lge_consumer_contract_probe(workspace: Path,
                                     evaluation_data: dict[str, Any] | None = None) -> tuple[bool, str]:
    """Exercise the immutable LGE adapter contract before paid review/promotion."""
    module_path = workspace / LGE_CONSUMER_TARGET
    if not module_path.is_file():
        return False, f"LGE contract probe: missing {LGE_CONSUMER_TARGET}"
    import importlib.util

    spec = importlib.util.spec_from_file_location("_lge_contract_probe", module_path)
    if spec is None or spec.loader is None:
        return False, "LGE contract probe: module could not be loaded"
    probe_data = evaluation_data or {
            "status": "AVAILABLE",
            "modality": "lge",
            "workflow_stage": "lge_myo",
            "source_sha256": "contract-probe-sha256",
            "records": [{"case_id": "probe-a"}, {"case_id": "probe-b"}],
            "holdout_case_ids": [],
            "frozen_split": {"status": "UNAVAILABLE"},
        }
    records = probe_data.get("records") or []
    workflow_stage = probe_data.get("workflow_stage") or "lge_myo"
    source_sha256 = probe_data.get("source_sha256")
    try:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        run_evaluation = getattr(module, "run_evaluation")
        # Candidate code runs with a disposable cwd. A pure adapter must leave it empty;
        # a relative runtime write is caught without contaminating production state.
        previous_cwd = Path.cwd()
        with tempfile.TemporaryDirectory(prefix="lge-contract-probe-") as tmp:
            os.chdir(tmp)
            try:
                report = run_evaluation(probe_data)
                written = [p.relative_to(tmp).as_posix() for p in Path(tmp).rglob("*")
                           if p.is_file() and "__pycache__" not in p.parts]
            finally:
                os.chdir(previous_cwd)
        prefix = workflow_stage
        counts = [report.get(f"{prefix}_{name}") for name in ("pending", "work", "resolved")]
        checks = {
            "report_dict": isinstance(report, dict),
            "source_modality": report.get("source_modality") == "lge",
            "workflow_stage": report.get("workflow_stage") == workflow_stage,
            "source_hash": report.get("input_source_sha256") == source_sha256,
            "consumer_ready_bool": isinstance(report.get(f"{prefix}_consumer_ready"), bool),
            "counts": all(isinstance(value, int) and value >= 0 for value in counts)
                      and sum(counts) == len(records),
            "frozen_unavailable": report.get("frozen_split_status") == "UNAVAILABLE",
            "holdout_excluded_zero": report.get("holdout_excluded") == 0,
            "label_write_false": report.get("label_write") is False,
            "production_writes_zero": report.get("production_writes") == 0,
            "pure_no_files_written": not written,
        }
    except Exception as exc:  # noqa: BLE001 - surfaced to the owner repair round
        return False, f"LGE contract probe raised {type(exc).__name__}: {exc}"
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        return False, "LGE contract probe failed: " + ", ".join(failed)
    return True, "LGE contract probe passed"


def _run_owner_validation(workspace: Path, changed: list[str],
                          action: dict[str, Any],
                          evaluation_data: dict[str, Any] | None = None) -> tuple[bool, str]:
    tests_ok, output = _run_tests(workspace, changed)
    requirements = action.get("control_plane_requirements") or {}
    if tests_ok and requirements.get("required_target_path") == LGE_CONSUMER_TARGET:
        probe_ok, probe_output = _run_lge_consumer_contract_probe(
            workspace, evaluation_data=evaluation_data)
        tests_ok = tests_ok and probe_ok
        output = (output + "\n" + probe_output).strip()
    if requirements.get("required_target_path") in {LGE_GENERATOR_TARGET, PROTOCOL_CONFIG}:
        with tempfile.TemporaryDirectory(prefix="generator-owner-probe-") as tmp:
            result = subprocess.run([sys.executable, "scripts/validate_lge_generator.py", "--output", str(Path(tmp) / "probe.json")],
                                    cwd=workspace, capture_output=True, text=True, timeout=180)
        tests_ok = tests_ok and result.returncode == 0
        output = (output + "\n" + (result.stdout + result.stderr)[-3000:]).strip()
    return tests_ok, output


def _required_target_sources(action: dict[str, Any], root: Path) -> dict[str, str]:
    """Expose only the explicitly required project source, never runtime or data."""
    requirements = action.get("control_plane_requirements")
    relative = requirements.get("required_target_path") if isinstance(requirements, dict) else None
    if not isinstance(relative, str) or not (relative.endswith(".py") or relative == PROTOCOL_CONFIG) or "\\" in relative:
        return {}
    parts = Path(relative).parts
    if not parts or (parts[0] not in {"agents", "app", "scripts", "experiments"} and relative != PROTOCOL_CONFIG) \
            or any(part in {"", ".", ".."} for part in parts):
        return {}
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        return {}
    if not path.is_file() or path.stat().st_size > MAX_CHARS:
        return {}
    return {relative: path.read_text(encoding="utf-8")}


def _owner_model_config(config: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    selected = dict(config)
    alternates = config.get("runtime_repair_models") or []
    if action.get("runtime_failure_feedback") and alternates:
        selected["model"] = alternates[0]
    return selected


def _test_framework_diagnostics(failure: str) -> str:
    if "array_equal" in failure and "AssertionError" in failure:
        return ("NumPy array_equal compares both shape and values; comparing a matrix with a scalar "
                "returns false even if every element equals that scalar. Inspect the generated "
                "test's intended elementwise assertion. Keep immutable framework tests unchanged.")
    return ""


def _previous_owner_attempt(output_root: Path, action: dict[str, Any], current: Path) -> dict[str, Any]:
    """Read code-only failed context from this action, never install it for the owner."""
    if not action.get("runtime_failure_feedback"):
        return {}
    root = output_root.resolve()
    candidates = [p for p in output_root.iterdir() if p.is_dir()
                  and p.name.startswith(str(action["action_id"]) + "_")
                  and p.resolve() != current.resolve() and p.resolve().is_relative_to(root)]
    for previous in sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            manifest = json.loads((previous / "maintenance_manifest.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        result = manifest.get("result") or {}
        selection_path=previous / "rsi_selection.json"
        measured_rejection=(json.loads(selection_path.read_text(encoding="utf-8")).get("decision") == "RETAIN_PARENT") if selection_path.exists() else False
        if result.get("tests_passed") is not False and result.get("ready_for_validation") is not False and not measured_rejection:
            continue
        files = []
        for name in result.get("changed_files", [])[:MAX_FILES]:
            if not isinstance(name, str) or not name.startswith(ALLOWED_PREFIXES):
                continue
            path = (previous / "orchestrator" / name).resolve()
            if (not path.is_relative_to((previous / "orchestrator").resolve())
                    or not path.is_file() or path.stat().st_size > MAX_CHARS):
                continue
            content = path.read_text(encoding="utf-8")
            files.append({"path": name, "content": content,
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        if files:
            return {"snapshot": previous.name, "files": files,
                    "instruction": "Continue the same approved action using the prior failed implementation as context. Preserve working fixes; correct the actual reported failures. This snapshot is unapproved and is not installed. Do not weaken framework tests or source-label boundaries."}
    return {}


def run_llm_maintenance(*, action: dict[str, Any], output_root: Path, orchestrator_source: Path,
                        skill_source: Path, providers_path: Path, agents_path: Path,
                        provider_role: str = "qc_code_optimizer",
                        timeout_seconds: int = 900, db: Database | None = None,
                        run_id: int | None = None) -> dict[str, Any]:
    action = _with_lge_consumer_contract(action)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    workspace = output_root / f"{action['action_id']}_{stamp}"
    orchestrator_copy = workspace / "orchestrator"
    skill_copy = workspace / "skill"
    workspace.mkdir(parents=True, exist_ok=False)
    _copy_source(orchestrator_source, orchestrator_copy)
    _copy_source(
        skill_source, skill_copy,
        include=("SKILL.md", "README.md", "tools", "scripts", "references", "tests", "agents"),
    )
    baseline = {
        **_baseline_hashes(orchestrator_copy, "orchestrator"),
        **_baseline_hashes(skill_copy, "skill"),
    }
    (workspace / "baseline_hashes.json").write_text(
        json.dumps(baseline, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    from app.rsi_runtime import freeze_parent
    freeze_parent(workspace, orchestrator_source, db=db)

    (workspace / "maintenance_request.json").write_text(
        json.dumps(action, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    agents = expand_env(yaml.safe_load(agents_path.read_text(encoding="utf-8")))["agents"]
    resolved_role = resolve_provider_role(agents, provider_role)
    role_cfg = _owner_model_config(agents[resolved_role], action)
    if not role_cfg.get("can_write_snapshot", False):
        raise PermissionError(
            f"agent role {resolved_role} is not allowed to implement snapshot changes"
        )
    registry = ProviderRegistry(providers_path)
    provider = registry.get(role_cfg["provider"])

    existing = sorted(p.relative_to(orchestrator_copy).as_posix()
                      for p in (orchestrator_copy / "experiments").glob("*.py"))
    real_input = evaluation_input(
        orchestrator_source, workflow_stage=action.get("workflow_stage")
    )
    # Names alone do not let an owner reuse an existing implementation. Provide
    # a bounded recent source catalog; all files are project code, not patient data.
    recent = sorted((orchestrator_source / "experiments").glob("*.py"),
                    key=lambda p: p.stat().st_mtime, reverse=True)
    sources = {}
    for path in recent:
        if path.name in {"llm_maintenance.py", "runtime_evaluation.py"}:
            continue
        content = path.read_text(encoding="utf-8")
        if len(content) <= 18000:
            sources["experiments/" + path.name] = content
        if len(sources) >= 2:
            break
    required_sources = _required_target_sources(action, orchestrator_source)
    layout = ["experiments/<name>.py", "tests/test_<name>.py",
              "skill/ is a read-only copy of the QC skill"]
    layout.extend(required_sources)
    payload = {
        "role": resolved_role,
        "requested_role": provider_role,
        # Maintenance actions are expensive and run unattended.  One bounded
        # structured attempt is safer than multiplying the per-request timeout
        # across schema retries and a semantically different fallback model.
        "_max_output_tokens": 10000,
        "_schema_retries": 0,
        "_fallback_on_error": False,
        "objective": action.get("objective", ""),
        "action_class": action.get("action_class", ""),
        "owner_role": action.get("owner_role", ""),
        "acceptance_metrics": action.get("acceptance_metrics", []),
        "agent_handoffs": action.get("handoff_messages", []),
        "actual_execution_feedback": feedback_summary(orchestrator_source),
        "generator_fitness_contract": (__import__('app.rsi_generator_benchmark',fromlist=['status']).status(orchestrator_source)
            if action.get("control_plane_requirements",{}).get("required_target_path")==LGE_GENERATOR_TARGET else {}),
        "independent_validation_feedback": action.get("runtime_failure_feedback",{}),
        "previous_owner_attempt": _previous_owner_attempt(output_root, action, workspace),
        "test_framework_diagnostics": _test_framework_diagnostics(str(action.get("runtime_failure_feedback", {}))),
        "control_plane_requirements": action.get("control_plane_requirements", {}),
        "required_target_source": required_sources,
        "protocol_peer_source": (_required_target_sources({"control_plane_requirements":{"required_target_path":LGE_GENERATOR_TARGET}},orchestrator_source) if PROTOCOL_CONFIG in required_sources else {}),
        "existing_experiment_modules": existing,
        "existing_module_sources": sources,
        "real_evaluation_input": real_input,
        "rsi_quality_context": (
            owner_context(orchestrator_source, action.get("workflow_stage", ""))
            if ("app/quality_scoring.py" in required_sources or action.get("control_plane_requirements", {}).get("required_target_path") == LGE_GENERATOR_TARGET)
            else {"status": "NOT_REQUESTED"}
        ),
        "lge_existing_tools": (_lge_candidate_tools(orchestrator_source)
            if action.get("control_plane_requirements", {}).get("required_target_path") == LGE_GENERATOR_TARGET else []),
        "layout": layout,
    }

    async def invoke(input_data: dict[str, Any]) -> MaintenancePatch:
        if db is not None and run_id is not None:
            return await invoke_traced(
                db=db, run_id=run_id,
                thread_id=f"run-{run_id}:{action['action_id']}",
                role=resolved_role, provider_name=role_cfg["provider"], provider=provider,
                system_prompt=SYSTEM_PROMPT, input_data=input_data,
                output_schema=MaintenancePatch, model=role_cfg["model"],
                timeout_seconds=timeout_seconds,
            )
        return await provider.generate_structured(
            system_prompt=SYSTEM_PROMPT, input_data=input_data,
            output_schema=MaintenancePatch, model=role_cfg["model"],
            timeout_seconds=timeout_seconds,
        )

    try:
        patch: MaintenancePatch = asyncio.run(invoke(payload))
    except Exception as exc:
        failed = {"workspace": str(workspace), "exit_code": 2, "result": {
            "summary": "provider call failed before a patch was produced",
            "changed_files": [], "tests_run": [], "remaining_risks": [str(exc)[:1000]],
            "ready_for_validation": False,
            "executor": f"llm:{role_cfg['provider']}/{role_cfg['model']}",
        }}
        (workspace / "maintenance_manifest.json").write_text(
            json.dumps(failed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise

    written: list[str] = []
    skipped: list[str] = []
    for edit in patch.files[:MAX_FILES]:
        rel = edit.path.strip().lstrip("/")
        if action.get("action_class") == "framework_repair" and rel not in {PROTOCOL_CONFIG,PROTOCOL_TEST}:
            skipped.append(rel)
            continue
        if rel.startswith("..") or not rel.startswith(ALLOWED_PREFIXES):
            skipped.append(rel)
            continue
        target = (orchestrator_copy / rel).resolve()
        if not str(target).startswith(str(orchestrator_copy.resolve())):
            skipped.append(rel)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        content = edit.content[:MAX_CHARS]
        target.write_text(content, encoding="utf-8")
        written.append(rel)

    tests_ok, test_output = (False, "no files written")
    if written:
        tests_ok, test_output = _run_owner_validation(
            orchestrator_copy, written, action, evaluation_data=real_input)

    repair_rounds = 0
    # One self-repair round: feed the failing test output back to the model.
    # Generated code regularly misses an import or a fixture; without this the
    # loop would just mark a fixable patch as failed and move on.
    if written and not tests_ok:
        repair_input = dict(payload)
        repair_input["previous_attempt_files"] = [
            {"path": rel, "content": (orchestrator_copy / rel).read_text(encoding="utf-8", errors="replace")[:20000]}
            for rel in written
        ]
        repair_input["test_failure"] = test_output[-6000:]
        repair_input["test_framework_diagnostics"] = _test_framework_diagnostics(test_output)
        repair_input["instruction"] = (
            "The validation above failed. Return the complete corrected files (same paths), "
            "fixing the actual failure. Mandatory framework tests and the immutable "
            "control_plane_requirements must not be weakened. If a generated test contradicts "
            "that immutable contract, correct the generated test so it asserts the contract: "
            "run_evaluation is pure and does not persist files; consumer_ready is not forced "
            "true; frozen_split UNAVAILABLE remains UNAVAILABLE with holdout_excluded=0."
        )
        try:
            repaired = asyncio.run(invoke(repair_input))
            for edit in repaired.files[:MAX_FILES]:
                rel = edit.path.strip().lstrip("/")
                if action.get("action_class") == "framework_repair" and rel not in {PROTOCOL_CONFIG,PROTOCOL_TEST}:
                    continue
                if rel.startswith("..") or not rel.startswith(ALLOWED_PREFIXES):
                    continue
                target = (orchestrator_copy / rel).resolve()
                if not str(target).startswith(str(orchestrator_copy.resolve())):
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(edit.content[:MAX_CHARS], encoding="utf-8")
            tests_ok, test_output = _run_owner_validation(
                orchestrator_copy, written, action, evaluation_data=real_input)
            repair_rounds = 1
            patch.summary = repaired.summary or patch.summary
            patch.evaluation_module = repaired.evaluation_module or patch.evaluation_module
        except Exception as exc:  # noqa: BLE001 - repair is best effort
            test_output = test_output + " | repair round failed: " + str(exc)
    (workspace / "test_output.log").write_text(test_output, encoding="utf-8")

    ready = bool(written) and tests_ok and patch.ready_for_validation
    real_result = {"status": "NOT_REQUESTED"}
    if ready and patch.evaluation_module:
        module_path = patch.evaluation_module.replace(".", "/") + ".py"
        if module_path not in written:
            real_result = {"status": "FAILED", "reason": "entrypoint_not_in_owner_patch"}
        else:
            real_result = execute_evaluation(orchestrator_copy, patch.evaluation_module, real_input)
        ready = ready and real_result["status"] == "EXECUTED"
    result = {
        "summary": patch.summary,
        "changed_files": written,
        "skipped_files": skipped,
        "tests_run": patch.tests_run + [f"targeted unittest: {'ok' if tests_ok else 'FAILED'}"],
        "remaining_risks": patch.remaining_risks,
        "ready_for_validation": ready,
        "tests_passed": tests_ok,
        "executor": f"llm:{role_cfg['provider']}/{role_cfg['model']}",
        "repair_rounds": repair_rounds,
        "real_evaluation": real_result,
    }
    manifest = {"workspace": str(workspace), "exit_code": 0 if ready else 1, "result": result}
    (workspace / "maintenance_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
