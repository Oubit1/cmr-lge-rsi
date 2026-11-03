"""Version archive, paired evaluation and role inheritance for maintenance.

The trusted benchmark contract lives outside the Owner snapshot. Observe mode
records the exact selection that enforce mode would make. Neither mode applies
labels. This is orchestration machinery, not a medical evaluator.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import hashlib
import ast
from pathlib import Path
from typing import Any

from app.database import Database
from app.rsi_kernel import MetricSpec, canonical_hash, compare_fitness, variant_id


MODES = {"off", "observe", "enforce"}
PROTECTED = {"app/rsi_kernel.py", "app/rsi_runtime.py", "app/control_checkpoint.py",
             "app/rsi_quality_benchmark.py", "app/rsi_postpromotion.py", "app/lge_execution.py",
             "configs/lge_execution.json", "configs/rsi.json", "scripts/rsi_control.py"}
PROTECTED.update({"scripts/run_lge_feedback_consumer.py", "scripts/validate_lge_generator.py",
                  "providers/openai_compatible.py", "agents/invocation.py", "app/prompt_dispatch.py", "app/workflow_budget.py", "scripts/verify_autonomous_run.py", "app/autonomy_control.py", "app/autonomy_feedback.py", "configs/autonomy.json", "app/lge_protocol.py", "scripts/autopilot.py", "scripts/board_server.py", "app/rsi_generator_benchmark.py", "configs/rsi_generator_benchmark.json"})


def read_policy(root: Path) -> dict:
    path = root / "configs/rsi.json"
    policy = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"mode": "off"}
    if policy.get("mode") not in MODES:
        raise ValueError("invalid RSI mode")
    return policy


def benchmark_contracts(root: Path) -> dict:
    """Load one atomically activated generation (legacy files remain readable)."""
    folder = root / "configs/rsi_benchmarks"
    active = folder / "active.json"
    if active.is_file():
        registry = json.loads(active.read_text(encoding="utf-8"))
        contracts = registry["contracts"]
        if not isinstance(contracts, dict) or canonical_hash(contracts) != registry["generation_sha256"]:
            raise ValueError("benchmark_generation_hash_mismatch")
        return contracts
    return {path.stem: json.loads(path.read_text(encoding="utf-8"))
            for path in folder.glob("*.json") if path.stem != "active"}


def write_record(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def parent_record_path(workspace: Path, root: Path) -> Path:
    return root / "runtime/rsi/parents" / (canonical_hash(str(workspace.resolve())) + ".json")


def freeze_parent(workspace: Path, root: Path, *, db: Database | None = None) -> None:
    """Pin the actual starting tree, including WIP, before any Owner edits."""
    if read_policy(root)["mode"] == "off":
        return
    parent = workspace / "rsi_parent"
    if parent.exists():
        raise FileExistsError("RSI parent already frozen")
    for owner in ("orchestrator", "skill"):
        source = workspace / owner
        if source.is_dir():
            shutil.copytree(source, parent / owner,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    parent_digest = tree_digest(parent)
    parent_id = "base_" + parent_digest[:20]
    if db is not None:
        matching = [row for row in db.list_evolution_variants(1000)
                    if row["metadata"].get("tree_digest") == parent_digest
                    and row["status"] in {"PROMOTED", "PROMOTED_SHADOW", "CANARY", "ACTIVE"}]
        if matching:
            parent_id = matching[0]["variant_id"]
    pinned = {"schema_version": "rsi-parent-v1", "workspace": str(workspace.resolve()),
              "parent_digest": parent_digest, "parent_variant_id": parent_id}
    path = parent_record_path(workspace, root)
    if path.exists():
        raise FileExistsError("parent lineage is already pinned")
    write_record(path, pinned)


def frozen_parent_record(workspace: Path, root: Path) -> dict:
    path = parent_record_path(workspace, root)
    if not path.is_file():
        raise ValueError("trusted_parent_record_missing")
    pinned = json.loads(path.read_text(encoding="utf-8"))
    if pinned.get("workspace") != str(workspace.resolve()):
        raise ValueError("parent_workspace_mismatch")
    if tree_digest(workspace / "rsi_parent") != pinned["parent_digest"]:
        raise ValueError("frozen_parent_modified")
    return pinned


def tree_digest(workspace: Path) -> str:
    import hashlib
    entries = {}
    for owner in ("orchestrator", "skill"):
        folder = workspace / owner
        if folder.is_dir():
            for path in sorted(folder.rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                    if path.is_symlink():
                        raise ValueError("RSI snapshots must contain files, not symlinks")
                    entries[path.relative_to(workspace).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return canonical_hash(entries)


def _metric_value(report: dict, dotted: str) -> Any:
    value: Any = report
    for field in dotted.split("."):
        if not isinstance(value, dict) or field not in value:
            return None
        value = value[field]
    return value


def score_predictions(data: dict, report: dict) -> dict:
    """Score outputs against frozen, externally supplied reference answers.

    The solver returns predictions only. It cannot award its own accuracy or
    drop hard cases from the denominator. Reference construction belongs to the
    existing independent evaluator/domain roles.
    """
    references, predictions = data.get("references"), report.get("predictions")
    if not isinstance(references, dict) or not references or not isinstance(predictions, dict):
        raise ValueError("frozen_references_or_predictions_missing")
    if any(key not in references for key in predictions):
        raise ValueError("prediction_outside_frozen_cohort")
    correct = wrong = abstained = 0
    for key, expected in references.items():
        if key not in predictions or predictions[key] is None:
            abstained += 1
        elif canonical_hash(predictions[key]) == canonical_hash(expected):
            correct += 1
        else:
            wrong += 1
    return {"correct": correct, "wrong": wrong, "abstained": abstained,
            "total": len(references), "accuracy": correct / len(references)}


def _measure(code: Path, contract: dict, input_bytes: str) -> dict:
    # Use a fresh interpreter per version; no shared import cache or provider
    # credentials. This is process separation, not an OS security sandbox.
    runner = (
        "import importlib,json,sys; sys.path.insert(0,sys.argv[1]); "
        "data=json.load(sys.stdin); "
        "report=importlib.import_module(sys.argv[2]).run_evaluation(data); "
        "print(json.dumps(report,allow_nan=False))"
    )
    if contract.get("scorer") == "reviewed_quality_scores_v1":
        runner = """import copy,importlib,json,math,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
data=json.load(sys.stdin)
# Restrict normal Python file/network/process access before loading candidate code.
# This is an evaluation integrity guard, not a same-UID OS security sandbox.
allowed=[Path(sys.argv[1]).resolve(),Path(sys.base_prefix).resolve(),Path(sys.prefix).resolve()]
if data.get('local_asset_root'):
    allowed.append(Path(data['local_asset_root']).resolve())
def audit(event,args):
    if event=='open' and not isinstance(args[0],int):
        path=Path(args[0]).resolve()
        if not any(path.is_relative_to(base) for base in allowed):
            raise PermissionError('evaluation read outside supplied inputs')
        mode=args[1]
        if (isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(args[2],int) and args[2] & 3):
            raise PermissionError('evaluation inputs are read only')
    if event.startswith(('socket.','subprocess.')) or event in ('os.system','os.exec','os.spawn','ctypes.dlopen'):
        raise PermissionError('evaluation external access disabled')
# Scientific libraries need native extensions while importing. Load them before
# enabling the candidate boundary; provider secrets are absent from this process.
try:
    import numpy,scipy.ndimage,nibabel
except ImportError:
    pass
sys.addaudithook(audit)
module=importlib.import_module(sys.argv[2])
report={'scores':{},'errors':0}
for row in data['records']:
    try:
        if 'raw_pair' in row:
            paths={k:Path(data['local_asset_root'])/v for k,v in row['raw_pair'].items()}
            spec=module.PairSpec(dataset='frozen_reference',tier='gold',target=row['target'],
                                 image=paths['image'],label=paths['label'],myo=paths.get('myo'),
                                 relative_id='anonymous',case_id='anonymous',slice_position=0,reviewed=True)
            extracted=module.extract_pair(spec)
            if not extracted.get('valid'):
                raise ValueError('invalid extraction')
            features={k:float(extracted.get(k,math.nan)) for k in data['feature_schema']}
            features.update(valid=True,target=row['target'],sparse_phase=bool(extracted.get('sparse_phase',False)),
                            myo_exists=bool(extracted.get('myo_exists',False)),dataset='frozen_reference',
                            case_id='anonymous',relative_id='anonymous',tier='gold',reviewed=True,failure='')
        else:
            features={k:(math.nan if v is None else v) for k,v in row['features'].items()}
        value=module.score_record(features,copy.deepcopy(data['profile']))['score']
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
            raise ValueError('invalid score')
        report['scores'][row['id']]=float(value)
    except Exception:
        report['errors']+=1
print(json.dumps(report,allow_nan=False))
"""
    environment = {k: v for k, v in os.environ.items()
                   if k.upper() in {"PATH", "SYSTEMROOT", "LANG", "LC_ALL", "TMP", "TEMP"}}
    proc = subprocess.run(
        [sys.executable, "-I", "-c", runner, str(code.resolve()), contract["module"]],
        cwd=code, env=environment, input=input_bytes,
        capture_output=True, text=True, timeout=int(contract.get("timeout_seconds", 90)),
    )
    if proc.returncode:
        raise ValueError("evaluator_execution_failed:" + proc.stderr[-800:])
    if len(proc.stdout) > 2_000_000:
        raise ValueError("evaluator_report_too_large")
    report = json.loads(proc.stdout)
    if not isinstance(report, dict):
        raise ValueError("evaluator_report_not_object")
    return report


def _paired_evaluation_single(root: Path, workspace: Path, stage: str, *,
                              contracts: dict | None = None) -> dict:
    """Controller-owned execution; never accept an Owner-authored score file."""
    if not re.fullmatch(r"[a-z_]+", stage):
        raise ValueError("invalid workflow stage")
    pinned = frozen_parent_record(workspace, root)
    contracts = benchmark_contracts(root) if contracts is None else contracts
    if stage not in contracts:
        return {"decision": "NEEDS_EVIDENCE", "reasons": ["benchmark_contract_missing"],
                "baseline_metrics": {}, "candidate_metrics": {}, "workflow_stage": stage}
    contract = contracts[stage]
    scorer_fingerprint = canonical_hash({
        "contract": contract,
        "controller_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "kernel_sha256": hashlib.sha256(Path(__file__).with_name("rsi_kernel.py").read_bytes()).hexdigest(),
        "quality_adapter_sha256": (hashlib.sha256(Path(__file__).with_name("rsi_quality_benchmark.py").read_bytes()).hexdigest()
                                   if contract.get("scorer") == "reviewed_quality_scores_v1" else None),
    })
    if (contract.get("schema_version") != "rsi-benchmark-v1"
            or not re.fullmatch(r"(?:scripts|experiments|app)\.[A-Za-z_][A-Za-z0-9_]*", contract.get("module", ""))):
        raise ValueError("invalid benchmark contract")
    if contract.get("scorer") not in {"reference_predictions_v1", "reviewed_quality_scores_v1"}:
        raise ValueError("independent_scorer_missing")
    if contract["scorer"] == "reviewed_quality_scores_v1":
        unmeasured = {"build_profile", "add_interslice_context", "load_specs", "write_reports", "PROFILE_METRICS"}
        def definitions(code_root):
            path = code_root / "app/quality_scoring.py"
            tree = ast.parse(path.read_text(encoding="utf-8"))
            result = {}
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name in unmeasured:
                    result[node.name] = ast.dump(node)
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id in unmeasured:
                            result[target.id] = ast.dump(node)
            return result
        before = definitions(workspace / "rsi_parent/orchestrator")
        after = definitions(workspace / "orchestrator")
        unsupported = [name for name in unmeasured if before.get(name) != after.get(name)]
        if unsupported:
            raise ValueError("mutation_outside_frozen_scoring_contract:" + ",".join(sorted(unsupported)))
    input_rel = Path(contract["input_path"])
    source = (root / input_rel).resolve()
    if input_rel.is_absolute() or not source.is_relative_to((root / "runtime/rsi/benchmarks").resolve()):
        raise ValueError("benchmark input must be frozen under runtime/rsi/benchmarks")
    data = json.loads(source.read_text(encoding="utf-8"))
    if canonical_hash(data) != contract["input_sha256"]:
        raise ValueError("benchmark_input_changed")
    def assets_valid():
        referenced = {value for row in data.get("records", [])
                      for value in row.get("raw_pair", {}).values()}
        if referenced - set(data.get("asset_hashes", {})):
            raise ValueError("reference_asset_hash_missing")
        for relative, expected in data.get("asset_hashes", {}).items():
            asset = (source.parent / relative).resolve()
            if not asset.is_relative_to((source.parent / "assets").resolve()):
                raise ValueError("reference_asset_outside_frozen_directory")
            if hashlib.sha256(asset.read_bytes()).hexdigest() != expected:
                raise ValueError("frozen_reference_asset_changed")
    assets_valid()
    parent_code = workspace / "rsi_parent/orchestrator"
    if not parent_code.is_dir():
        raise ValueError("frozen_parent_missing")
    frozen_input = workspace / "rsi_input.json"
    parent_digest, child_digest = tree_digest(workspace / "rsi_parent"), tree_digest(workspace)
    # Keep the assessment schema in the controller-owned contract; the Owner
    # changes the solver entrypoint, not the metric mapping or benchmark set.
    # Solver input never includes the private reference answers.
    solver_input = {k: v for k, v in data.items() if k not in {"references", "asset_hashes"}}
    asset_copies = {}
    if data.get("asset_hashes"):
        from app.rsi_quality_benchmark import SCORER_NUMERIC_FIELDS
        solver_input.update(feature_schema=sorted(SCORER_NUMERIC_FIELDS))
        for side in ("baseline", "candidate"):
            folder = workspace / "rsi_input_assets" / side
            asset_copies[side] = folder
            for relative, expected in data["asset_hashes"].items():
                destination = folder / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source.parent / relative, destination)
                if hashlib.sha256(destination.read_bytes()).hexdigest() != expected:
                    raise ValueError("input_asset_copy_changed")
    write_record(frozen_input, solver_input)
    def run_side(side, code):
        payload = dict(solver_input)
        if side in asset_copies:
            payload["local_asset_root"] = str(asset_copies[side].resolve())
        report = _measure(code, contract, json.dumps(payload, allow_nan=False))
        for relative, expected in data.get("asset_hashes", {}).items():
            if hashlib.sha256((asset_copies[side] / relative).read_bytes()).hexdigest() != expected:
                raise ValueError("evaluation_asset_mutation")
        return report
    baseline_report = run_side("baseline", parent_code)
    candidate_report = run_side("candidate", workspace / "orchestrator")
    assets_valid()
    if contract["scorer"] == "reviewed_quality_scores_v1":
        from app.rsi_quality_benchmark import score_quality_outputs
        baseline_score = score_quality_outputs(data, baseline_report, contract)
        candidate_score = score_quality_outputs(data, candidate_report, contract)
    else:
        baseline_score = score_predictions(data, baseline_report)
        candidate_score = score_predictions(data, candidate_report)
    specs, baseline, candidate = {}, {}, {}
    for name, metric in contract["metrics"].items():
        specs[name] = MetricSpec(**{k: v for k, v in metric.items() if k != "report_field"})
        for destination, report in ((baseline, baseline_score), (candidate, candidate_score)):
            value = _metric_value(report, metric["report_field"])
            if value is not None:
                destination[name] = value
    constraints = {"snapshot_unchanged": parent_digest == tree_digest(workspace / "rsi_parent")
                   and child_digest == tree_digest(workspace)}
    if contract.get("require_discrimination", contract["scorer"] == "reviewed_quality_scores_v1"):
        constraints["candidate_discriminates_controls"] = candidate_score.get("discrimination_ready") is True
    for name, condition in contract.get("constraints", {}).items():
        constraints[name] = all(_metric_value(report, condition["report_field"]) == condition["equals"]
                                for report in (baseline_report, candidate_report))
    result = compare_fitness(
        baseline=baseline, candidate=candidate,
        baseline_benchmark_id=contract["benchmark_id"], candidate_benchmark_id=contract["benchmark_id"],
        baseline_benchmark_hash=scorer_fingerprint, candidate_benchmark_hash=scorer_fingerprint,
        metric_specs=specs, constraints=constraints,
    ).as_dict()
    return {**result, "benchmark_id": contract["benchmark_id"], "benchmark_hash": scorer_fingerprint,
            "baseline_metrics": baseline, "candidate_metrics": candidate,
            "constraints": constraints, "workflow_stage": stage,
            "parent_digest": parent_digest, "child_digest": child_digest,
            "scope": contract.get("scope", "unspecified"),
            "scorer": contract["scorer"], "target": contract.get("target"),
            "metric_specs": {key: {k: v for k, v in metric.items() if k != "report_field"}
                             for key, metric in contract["metrics"].items()},
            "baseline_report": baseline_report, "candidate_report": candidate_report,
            "baseline_summary": baseline_score, "candidate_summary": candidate_score}


def scorer_dependency_paths(code: Path, module: str) -> tuple[set[str], bool]:
    """Static local import closure; unresolved dynamic imports disable exemption."""
    pending, paths, visited, dynamic = [module], set(), set(), False
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        visited.add(name)
        components = name.split(".")
        for index in range(1, len(components)):
            package = "/".join(components[:index]) + "/__init__.py"
            paths.add(package)
            package_name = ".".join(components[:index])
            if (code / package).is_file():
                pending.append(package_name)
        relative = name.replace(".", "/") + ".py"
        path = code / relative
        if not path.is_file():
            relative = name.replace(".", "/") + "/__init__.py"
            path = code / relative
        if not path.is_file():
            continue
        paths.add(relative)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            dynamic = True
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                pending.extend(alias.name for alias in node.names)
                dynamic |= any(alias.name == "builtins" or alias.name == "importlib"
                               or alias.name.startswith("importlib.") for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                prefix = components if relative.endswith("/__init__.py") else components[:-1]
                if node.level:
                    prefix = prefix[:len(prefix) - node.level + 1]
                    imported = ".".join([*prefix, *([node.module] if node.module else [])])
                else:
                    imported = node.module or ""
                if imported:
                    pending.append(imported)
                    pending.extend(imported + "." + alias.name for alias in node.names if alias.name != "*")
                dynamic |= imported == "builtins" or imported.startswith("importlib")
            elif isinstance(node, ast.Call):
                function = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
                dynamic |= function in {"__import__", "import_module", "exec", "eval"}
    return paths, dynamic


def paired_evaluation(root: Path, workspace: Path, stage: str, *,
                      changed_files: list[str] | None = None) -> dict:
    if not re.fullmatch(r"[a-z_]+", stage):
        raise ValueError("invalid workflow stage")
    requested_stage = stage
    normalized = {name.removeprefix("orchestrator/") for name in (changed_files or [])}
    if "scripts/lge_candidate_generator.py" in normalized:
        from app.rsi_generator_benchmark import paired as generator_paired
        try:
            return generator_paired(root,workspace)
        except (OSError,ValueError,TypeError,KeyError,subprocess.TimeoutExpired) as exc:
            return {"decision":"NEEDS_EVIDENCE","reasons":["generator_benchmark:"+str(exc)[-700:]],
                    "baseline_metrics":{},"candidate_metrics":{},"workflow_stage":requested_stage,
                    "scope":"lge_generator_doctor_reference_challenge"}
    contracts = benchmark_contracts(root)
    if stage not in contracts and changed_files is not None:
        normalized = {name.removeprefix("orchestrator/") for name in changed_files}
        registered = list(contracts)
        for registered_stage in registered:
            candidate_contract = contracts[registered_stage]
            applies = candidate_contract.get("applies_to")
            if not applies:
                continue
            before, dynamic_before = scorer_dependency_paths(workspace / "rsi_parent/orchestrator", candidate_contract["module"])
            after, dynamic_after = scorer_dependency_paths(workspace / "orchestrator", candidate_contract["module"])
            if normalized.intersection(set(applies) | before | after) or dynamic_before or dynamic_after:
                stage = registered_stage
                break
        else:
            if registered:
                return {"decision": "NEEDS_EVIDENCE", "reasons": ["benchmark_scope_not_applicable"],
                        "baseline_metrics": {}, "candidate_metrics": {}, "workflow_stage": requested_stage,
                        "scope_not_applicable": True}
    if changed_files is not None and stage in contracts:
        contract = contracts[stage]
        applies = contract.get("applies_to")
        normalized = [name.removeprefix("orchestrator/") for name in changed_files]
        parent_paths, parent_dynamic = scorer_dependency_paths(workspace / "rsi_parent/orchestrator", contract["module"])
        child_paths, child_dynamic = scorer_dependency_paths(workspace / "orchestrator", contract["module"])
        executable_dependencies = parent_paths | child_paths | {"requirements.txt", "pyproject.toml"}
        if (applies and not set(applies).union(executable_dependencies).intersection(normalized)
                and not parent_dynamic and not child_dynamic):
            return {"decision": "NEEDS_EVIDENCE", "reasons": ["benchmark_scope_not_applicable"],
                    "baseline_metrics": {}, "candidate_metrics": {}, "workflow_stage": stage,
                    "scope_not_applicable": True}
    first = _paired_evaluation_single(root, workspace, stage, contracts=contracts)
    if first.get("scorer") != "reviewed_quality_scores_v1":
        return first
    suites = {stage: first}
    for other in ("lge_myo", "lge_mi", "cmr_paired"):
        if other != stage:
            report = _paired_evaluation_single(root, workspace, other, contracts=contracts)
            if report.get("scorer") != "reviewed_quality_scores_v1":
                raise ValueError("quality_cross_target_contract_missing:" + other)
            suites[other] = report
    baseline, candidate, specs = {}, {}, {}
    for report in suites.values():
        for name, specification in report["metric_specs"].items():
            key = report["target"] + "." + name
            specs[key] = MetricSpec(**specification)
            if name in report["baseline_metrics"]:
                baseline[key] = report["baseline_metrics"][name]
            if name in report["candidate_metrics"]:
                candidate[key] = report["candidate_metrics"][name]
    fingerprint = canonical_hash({key: value["benchmark_hash"] for key, value in suites.items()})
    constraints = {key + "." + name: passed for key, report in suites.items()
                   for name, passed in report["constraints"].items()}
    selection = compare_fitness(
        baseline=baseline, candidate=candidate,
        baseline_benchmark_id="reviewed-quality-all-targets", candidate_benchmark_id="reviewed-quality-all-targets",
        baseline_benchmark_hash=fingerprint, candidate_benchmark_hash=fingerprint,
        metric_specs=specs, constraints=constraints).as_dict()
    return {**selection, "benchmark_id": "reviewed-quality-all-targets", "benchmark_hash": fingerprint,
            "baseline_metrics": baseline, "candidate_metrics": candidate, "constraints": constraints,
            "workflow_stage": requested_stage, "scope": "quality_scorer_all_targets", "suites": suites}


def assess_action(db: Database, root: Path, workspace: Path, *, run_id: int,
                  action: dict, changed_files: list[str], validation_passed: bool = True) -> dict:
    policy = read_policy(root)
    if policy["mode"] == "off":
        return {"mode": "off", "blocks_promotion": False}
    stage = str(action.get("workflow_stage") or "unknown")
    try:
        result = paired_evaluation(root, workspace, stage, changed_files=changed_files)
    except (OSError, ValueError, TypeError, KeyError, subprocess.TimeoutExpired) as exc:
        result = {"decision": "NEEDS_EVIDENCE", "reasons": [str(exc)[:1000]],
                  "baseline_metrics": {}, "candidate_metrics": {}, "workflow_stage": stage}
    clean = [name.removeprefix("orchestrator/") for name in changed_files]
    protected_change = any(name in PROTECTED or name.startswith("configs/rsi_benchmarks/") for name in clean)
    if protected_change:
        result.update(decision="RETAIN_PARENT", reasons=["trusted_rsi_kernel_changed"])
    if validation_passed is not True:
        result.update(decision="RETAIN_PARENT", reasons=["regression_suite_failed", *result["reasons"]])
    pin_path = parent_record_path(workspace, root)
    pinned = json.loads(pin_path.read_text(encoding="utf-8")) if pin_path.exists() else {}
    parent_digest = pinned.get("parent_digest")
    child_digest = tree_digest(workspace)
    parent_id = pinned.get("parent_variant_id") or "base_" + canonical_hash({"legacy_workspace": str(workspace)})[:20]
    child_id = variant_id(parent_variant_id=parent_id, source_run_id=run_id,
                          action_id=action["action_id"], code_commit=None,
                          changed_files=clean, artifact_digest=child_digest)
    if parent_id.startswith("base_"):
        db.register_evolution_variant(
            variant_id=parent_id, parent_variant_id=None, source_run_id=None, action_id=None,
            workflow_stage=None, code_commit=None, continuity_revision=None,
            artifact_path=None, status="BASELINE", metadata={"tree_digest": parent_digest})
    db.register_evolution_variant(
        variant_id=child_id, parent_variant_id=parent_id, source_run_id=run_id,
        action_id=action["action_id"], workflow_stage=stage, code_commit=None,
        continuity_revision=None, artifact_path=workspace, metadata={
            "tree_digest": child_digest, "owner_role": action.get("owner_role"),
            "changed_files": clean, "hypothesis": str(action.get("objective", ""))[:500]})
    db.record_evolution_evaluation(
        variant_id=child_id, parent_variant_id=parent_id,
        benchmark_id=result.get("benchmark_id", "UNAVAILABLE"),
        benchmark_hash=result.get("benchmark_hash", "UNAVAILABLE"),
        baseline_metrics=result["baseline_metrics"], candidate_metrics=result["candidate_metrics"],
        constraints=result.get("constraints", {}), decision=result["decision"], reasons=result["reasons"])
    result.update(mode=policy["mode"], parent_variant_id=parent_id, child_variant_id=child_id,
                  blocks_promotion=protected_change or (policy["mode"] == "enforce"
                  and result["decision"] != "PROMOTE" and not result.get("scope_not_applicable")))
    write_record(workspace / "rsi_selection.json", result)
    db.append_evolution_event("SELECTION_RECORDED", {
        "decision": result["decision"], "mode": policy["mode"],
        "reasons": result["reasons"], "blocks_promotion": result["blocks_promotion"]}, variant_id=child_id)
    return selection_summary(result)


def selection_summary(result: dict) -> dict:
    """Provider handoffs receive aggregate evidence, not thousands of outputs."""
    summary = {key: value for key, value in result.items()
               if key not in {"baseline_report", "candidate_report", "suites"}}
    if "suites" in result:
        summary["suites"] = {
            stage: {key: report.get(key) for key in (
                "target", "benchmark_id", "benchmark_hash", "baseline_metrics", "candidate_metrics",
                "constraints", "scope", "baseline_summary", "candidate_summary")}
            for stage, report in result["suites"].items()}
    return summary


def inheritance(db: Database, *, stage: str | None = None, role: str | None = None) -> dict:
    """Retrieve tested successes and failed hypotheses for the next role model."""
    try:
        rows = db.list_evolution_variants(1000)
    except Exception as exc:
        import sqlite3
        if isinstance(exc, sqlite3.OperationalError) and "no such table" in str(exc):
            return {"schema_version": "rsi-inheritance-v1", "variants": [], "ready": False}
        raise
    selected = []
    for row in rows:
        if row["parent_variant_id"] is None or (stage and row["workflow_stage"] != stage):
            continue
        if role and row["metadata"].get("owner_role") != role:
            continue
        evaluations = db.evolution_evaluations(row["variant_id"])
        latest = evaluations[-1] if evaluations else {}
        verification = {}
        artifact = row.get("artifact_path")
        if artifact and (Path(artifact) / "postinstall_verification.json").is_file():
            try:
                installed = json.loads((Path(artifact) / "postinstall_verification.json").read_text(encoding="utf-8"))
                verification = {key: installed.get(key) for key in ("status", "reason")}
            except (OSError, ValueError, TypeError):
                verification = {"status": "UNAVAILABLE"}
        selected.append({"variant_id": row["variant_id"], "parent_variant_id": row["parent_variant_id"],
                         "workflow_stage": row["workflow_stage"], "status": row["status"],
                         "owner_role": row["metadata"].get("owner_role"),
                         "changed_files": row["metadata"].get("changed_files", []),
                         "hypothesis": row["metadata"].get("hypothesis"),
                         "installation_verification": verification,
                         "decision": latest.get("decision"), "reasons": latest.get("reasons", []),
                         "baseline_metrics": latest.get("baseline_metrics", {}),
                         "candidate_metrics": latest.get("candidate_metrics", {}),
                         "benchmark_id": latest.get("benchmark_id")})
        if len(selected) == 8:
            break
    return {"schema_version": "rsi-inheritance-v1", "variants": selected, "ready": bool(selected),
            "contract": "Reuse measured improvements; avoid repeating failed hypotheses without new evidence. "
                        "An observed selection does not establish production or clinical acceptance."}


def _reconcile_promotion(db: Database, root: Path, skill: Path, workspace: Path, *,
                        run_id: int, action_id: str) -> dict:
    """Recover DB bookkeeping after installed files but before a DB commit.

    An incomplete filesystem promotion requires recovery; restarting an Owner
    or sending another paid request cannot repair it by implication.
    """
    import hashlib
    path = workspace / "rsi_promotion_journal.json"
    selection_path = workspace / "rsi_selection.json"
    if not path.exists():
        return {"status": "NOT_APPLICABLE"}
    journal = json.loads(path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8")) if selection_path.exists() else {}
    if journal.get("run_id") != run_id or journal.get("action_id") != action_id:
        return {"status": "RECOVERY_REQUIRED", "reason": "promotion_identity_mismatch"}
    artifact = Path(journal["artifact"]).resolve()
    if not artifact.is_relative_to((root / "runtime/promotions").resolve()):
        return {"status": "RECOVERY_REQUIRED", "reason": "promotion_artifact_outside_root"}
    manifest_path = artifact / "promotion_manifest.json"
    if not manifest_path.is_file():
        return {"status": "RECOVERY_REQUIRED", "reason": "filesystem_promotion_incomplete"}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") == "ROLLED_BACK":
        db.update_maintenance_action(run_id, action_id, status="REJECTED", artifact_path=workspace,
                                     evolution_variant_id=selection.get("child_variant_id"))
        return {"status": "REJECTED", "reason": manifest.get("reason", "promotion_rolled_back")}
    if manifest.get("status") != "PROMOTED":
        return {"status": "NOT_APPLICABLE"}
    for item in manifest.get("promoted", []):
        base = root if item.get("owner") == "orchestrator" else skill
        target = (base / item["relative"]).resolve()
        if not target.is_relative_to(base.resolve()):
            return {"status": "RECOVERY_REQUIRED", "reason": "promotion_destination_outside_root"}
        if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != item["new_hash"]:
            return {"status": "RECOVERY_REQUIRED", "reason": "installed_promotion_changed"}
    validation = json.loads((workspace / "autopilot_validation.json").read_text(encoding="utf-8"))
    if validation.get("passed") is not True:
        return {"status": "RECOVERY_REQUIRED", "reason": "promotion_validation_not_passed"}
    status = "PROMOTED_SHADOW" if validation.get("promotion_scope") == "shadow_only" else "PROMOTED"
    row = db.get_maintenance_action(run_id, action_id)
    if row["status"] not in {"PROMOTED", "PROMOTED_SHADOW"}:
        db.update_maintenance_action(
            run_id, action_id, status=status, artifact_path=workspace,
            evolution_variant_id=selection.get("child_variant_id"),
            rollback_manifest_path=manifest_path)
        db.append_evolution_event("PROMOTION_RECONCILED", {"run_id": run_id, "action_id": action_id},
                                  variant_id=selection.get("child_variant_id"))
    return {"status": status}


def reconcile_promotion(db: Database, root: Path, skill: Path, workspace: Path, *,
                        run_id: int, action_id: str) -> dict:
    try:
        return _reconcile_promotion(db, root, skill, workspace, run_id=run_id, action_id=action_id)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return {"status": "RECOVERY_REQUIRED", "reason": str(exc)[:1000]}


def global_promotion_preflight(db: Database, root: Path, skill: Path) -> dict:
    """An interrupted older run blocks newer runs using the same source tree."""
    from app.rsi_postpromotion import finish_install, installation_lock
    with installation_lock(root, exclusive=True):
        gate = finish_install(root)
    if not gate.get("ready"):
        return {"status": "RECOVERY_REQUIRED", "reason": gate.get("reason", "postinstall_verification_pending")}
    with db.connect() as connection:
        rows = connection.execute(
            "SELECT run_id,action_id,artifact_path FROM maintenance_actions "
            "WHERE artifact_path IS NOT NULL AND status NOT IN "
            "('PROMOTED','PROMOTED_SHADOW','REJECTED','VALIDATED') ORDER BY run_id"
        ).fetchall()
    for row in rows:
        workspace = Path(row["artifact_path"])
        if not (workspace / "rsi_promotion_journal.json").exists():
            continue
        result = reconcile_promotion(db, root, skill, workspace,
                                     run_id=row["run_id"], action_id=row["action_id"])
        if result.get("status") == "RECOVERY_REQUIRED":
            return {**result, "run_id": row["run_id"], "action_id": row["action_id"]}
    return {"status": "READY"}
