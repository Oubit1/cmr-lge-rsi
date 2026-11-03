from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


COPY_IGNORE = shutil.ignore_patterns(
    ".git", "__pycache__", "*.pyc", "runtime", "artifacts", "models", "vendor",
    "label_iteration", "*.nii", "*.nii.gz", "*.dcm", "*.png",
)


RESULT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "changed_files": {"type": "array", "items": {"type": "string"}},
        "tests_run": {"type": "array", "items": {"type": "string"}},
        "remaining_risks": {"type": "array", "items": {"type": "string"}},
        "ready_for_validation": {"type": "boolean"},
    },
    "required": ["summary", "changed_files", "tests_run", "remaining_risks", "ready_for_validation"],
    "additionalProperties": False,
}


def _copy_source(source: Path, target: Path, *, include: tuple[str, ...] | None = None) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    if include is None:
        shutil.copytree(source, target, ignore=COPY_IGNORE)
        return
    target.mkdir(parents=True)
    for name in include:
        item = source / name
        if not item.exists():
            continue
        destination = target / name
        if item.is_dir():
            shutil.copytree(item, destination, ignore=COPY_IGNORE)
        else:
            shutil.copy2(item, destination)


def _baseline_hashes(root: Path, owner: str) -> dict[str, str]:
    import hashlib

    hashes: dict[str, str] = {}
    for path in root.rglob("*"):
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            hashes[f"{owner}/{relative}"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def run_codex_maintenance(*, action: dict[str, Any], output_root: Path,
                          orchestrator_source: Path, skill_source: Path,
                          codex_cli: Path, timeout_seconds: int = 3600) -> dict[str, Any]:
    """Run a coding task only inside a disposable snapshot; never production."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    workspace = output_root / f"{action['action_id']}_{stamp}"
    orchestrator_copy = workspace / "orchestrator"
    skill_copy = workspace / "skill"
    workspace.mkdir(parents=True, exist_ok=False)
    _copy_source(orchestrator_source.resolve(), orchestrator_copy)
    # The coding agent needs rule/tool sources, not 267 MB of checkpoints,
    # vendor inference code or archived dataset scripts.  An explicit allowlist
    # also prevents new large/sensitive skill directories entering snapshots.
    _copy_source(
        skill_source.resolve(), skill_copy,
        include=("SKILL.md", "README.md", "tools", "scripts", "references", "tests", "agents"),
    )
    baseline = {
        **_baseline_hashes(orchestrator_copy, "orchestrator"),
        **_baseline_hashes(skill_copy, "skill"),
    }
    (workspace / "baseline_hashes.json").write_text(
        json.dumps(baseline, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    request = {
        "objective": action["objective"],
        "owner_role": action["owner_role"],
        "action_class": action["action_class"],
        "acceptance_metrics": action["acceptance_metrics"],
        "constraints": [
            "Work only in this isolated snapshot.",
            "Do not access datasets, patient images, production labels, credentials or network data.",
            "Update tests for behavioral changes.",
            "Keep medical repair rules conservative and measurable.",
            "Do not declare promotion; report readiness for deterministic validation only.",
        ],
    }
    request_path = workspace / "maintenance_request.json"
    schema_path = workspace / "result_schema.json"
    result_path = workspace / "codex_result.json"
    events_path = workspace / "codex_events.jsonl"
    request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    schema_path.write_text(json.dumps(RESULT_SCHEMA, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    prompt = (
        "Implement the bounded CMR/LGE QC maintenance request in maintenance_request.json. "
        "The orchestrator copy is the current directory and the skill copy is available as an "
        "additional writable directory. Inspect existing architecture first, make the narrowest "
        "change, run relevant tests, and return the required structured result."
    )
    command = [
        str(codex_cli), "exec", "--ephemeral", "--skip-git-repo-check",
        "--sandbox", "workspace-write", "--json",
        "--output-schema", str(schema_path), "--output-last-message", str(result_path),
        "-C", str(orchestrator_copy), "--add-dir", str(skill_copy), prompt,
    ]
    env = os.environ.copy()
    for key in list(env):
        if key.endswith("_API_KEY") or key in {"DASHSCOPE_BASE_URL", "CMR_LGE_CASE_SALT"}:
            env.pop(key, None)
    with events_path.open("w", encoding="utf-8") as events, (workspace / "codex_stderr.log").open("w", encoding="utf-8") as errors:
        result = subprocess.run(command, cwd=orchestrator_copy, env=env, stdout=events,
                                stderr=errors, stdin=subprocess.DEVNULL, text=True,
                                timeout=timeout_seconds, check=False,
                                start_new_session=True)
    payload = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {
        "summary": "Codex did not produce a structured result.",
        "changed_files": [], "tests_run": [],
        "remaining_risks": [f"codex_exit_{result.returncode}"],
        "ready_for_validation": False,
    }
    manifest = {"workspace": str(workspace), "exit_code": result.returncode, "result": payload}
    (workspace / "maintenance_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
