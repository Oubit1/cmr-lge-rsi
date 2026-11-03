"""Control-plane adapter for owner-written experiments on existing QC measurements.

No image loading, label decisions or production writes occur here.
"""
from __future__ import annotations

import hashlib
import csv
import json
import math
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

NUMERIC_FIELDS = ("reference_area", "before_area", "before_cavity",
                  "after_area", "after_cavity", "quality_before", "quality_after",
                  "quality_delta")
STRING_FIELDS = ("decision", "quality_status")
LIST_FIELDS = ("quality_flags_before", "quality_flags_after")
DICT_FIELDS = ("quality_subscores_before", "quality_subscores_after")
LGE_STAGES = {"lge_myo": "myo_", "lge_mi": "mi_"}
LGE_INPUT_SCHEMA = "existing-lge-qc-measurements-v2"


def _case_id(salt: str, key: str) -> str:
    return "case_" + hashlib.sha256((salt + "\0" + key).encode()).hexdigest()[:16]


def _frozen_holdout_input(root: Path, salt: str, current_keys: list[str]) -> dict:
    """Expose only pseudonymous membership for the frozen evaluation split."""
    path = root / "runtime/check_iterations/holdout_v1.json"
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
        rows = payload["holdout"]
        canonical = json.dumps(sorted(rows, key=lambda row: row["key"]),
                               ensure_ascii=False, sort_keys=True,
                               separators=(",", ":")).encode()
        if payload.get("holdout_sha256") != hashlib.sha256(canonical).hexdigest():
            raise ValueError("frozen holdout hash mismatch")
        frozen = {row["key"] for row in rows if isinstance(row.get("key"), str)}
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return {"status": "INVALID", "reason": str(exc)[:200], "case_ids": []}
    current = set(current_keys)
    selected = sorted(current & frozen)
    return {
        "status": "AVAILABLE",
        "case_ids": [_case_id(salt, key) for key in selected],
        "frozen_holdout_in_queue": len(selected),
        "non_holdout_in_queue": len(current - frozen),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _quality_benchmark(root: Path) -> dict:
    """Return only aggregate, already-published scoring validation evidence."""
    base = root / "artifacts/image_label_quality_v1"
    try:
        summary = json.loads((base / "summary.json").read_text(encoding="utf-8"))
        defect = json.loads((base / "defect_validation.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"status": "UNAVAILABLE"}
    groups = summary.get("groups", {})
    cmr = {name: value for name, value in groups.items()
           if isinstance(name, str) and name.endswith(":cmr_myo")
           and isinstance(value, dict)}
    return {
        "status": "AVAILABLE",
        "cmr_reference_groups": cmr,
        "clean_reference": defect.get("clean_reference", {}),
        "defect_classes": defect.get("classes", {}),
    }


def _spatial_control_input(report_path: Path, salt: str) -> dict:
    path = report_path.parent / "spatial_boundary_shadow.json"
    try:
        raw = path.read_bytes()
        report = json.loads(raw)
    except (OSError, json.JSONDecodeError):
        return {"status": "UNAVAILABLE", "records": []}
    records = []
    for row in report.get("records", []):
        key = row.get("key")
        if not isinstance(key, str):
            continue
        item = {"case_id": _case_id(salt, key)}
        for field in ("state", "decision", "reason"):
            value = row.get(field)
            if isinstance(value, str) and len(value) <= 80:
                item[field] = value
        rank = row.get("rank")
        if type(rank) is int:
            item["rank"] = rank
        core = row.get("core")
        if (isinstance(core, list) and len(core) == 2
                and all(type(value) is int for value in core)):
            item["core"] = core
        records.append(item)
    return {
        "status": "AVAILABLE",
        "rule": report.get("rule"),
        "control_gate": report.get("control_gate"),
        "checked_false_routes": report.get("checked_false_routes"),
        "pending_candidates": report.get("pending_spatial_boundary_candidates"),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "records": records,
    }


def _latest_lge_manifest(root: Path) -> Path | None:
    candidates = []
    for path in (root / "artifacts").glob("run_*/evidence_snapshot/lge_issue_manifest.csv"):
        try:
            run_id = int(path.parents[1].name.removeprefix("run_"))
        except ValueError:
            continue
        if path.is_file():
            candidates.append((run_id, path))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def _numeric_evidence(value: str) -> dict[str, int | float]:
    """Preserve existing QC measurements without interpreting their meaning."""
    result: dict[str, int | float] = {}
    for token in value.split(";")[:20]:
        name, separator, raw = token.partition("=")
        if not separator or not name or len(name) > 80:
            continue
        try:
            number = float(raw)
        except ValueError:
            continue
        if not math.isfinite(number):
            continue
        result[name] = int(number) if number.is_integer() else number
    return result


def _lge_evaluation_input(root: Path, salt: str, workflow_stage: str) -> dict:
    path = _latest_lge_manifest(root)
    if path is None:
        return {"status": "UNAVAILABLE", "modality": "lge", "records": [],
                "reason": "lge_issue_manifest_missing"}
    raw = path.read_bytes()
    prefix = LGE_STAGES[workflow_stage]
    records = []
    seen = set()
    try:
        rows = csv.DictReader(raw.decode("utf-8-sig").splitlines())
        for row in rows:
            issues = [item.strip() for item in str(row.get("issues") or "").split(";")
                      if item.strip().startswith(prefix) and len(item.strip()) <= 80]
            if not issues:
                continue
            patient = str(row.get("patient") or "")
            slice_name = str(row.get("slice") or "")
            layer = str(row.get("layer_index") or "")
            if not patient or not slice_name:
                continue
            key = f"{patient}/{slice_name}#{layer}"
            case_id = _case_id(salt, key)
            if case_id in seen:
                continue
            seen.add(case_id)
            # Supply the same already-published QC evidence in both its
            # canonical grouped form and a compatibility form.  This is a
            # schema adapter only: it does not infer missing measurements or
            # turn an issue code into a label verdict.
            item: dict[str, object] = {
                "case_id": case_id,
                "issue_codes": issues,
                "issues": list(issues),
            }
            try:
                item["layer_index"] = int(layer)
            except ValueError:
                pass
            measurements = _numeric_evidence(str(row.get("evidence") or ""))
            if measurements:
                item["measurements"] = measurements
                item.update(measurements)
            records.append(item)
    except (csv.Error, UnicodeDecodeError) as exc:
        return {"status": "UNAVAILABLE", "modality": "lge", "records": [],
                "reason": str(exc)[:200]}
    issue_counts = Counter(code for item in records for code in item["issue_codes"])
    measurement_counts = Counter(
        name for item in records for name in item.get("measurements", {})
    )
    return {
        "status": "AVAILABLE", "schema": LGE_INPUT_SCHEMA,
        "workflow_stage": workflow_stage, "modality": "lge", "unit": "spatial_slice",
        "cohort": "current_lge_issue_manifest", "source_run": path.parents[1].name,
        "source_sha256": hashlib.sha256(raw).hexdigest(), "records": records,
        "full_stage_queue": len(records), "holdout_case_ids": [],
        "frozen_split": {"status": "UNAVAILABLE", "reason": "no_lge_holdout_membership"},
        "reference_labels_available": False,
        "field_availability": {
            "record_count": len(records),
            "issue_code_counts": dict(sorted(issue_counts.items())),
            "measurement_field_counts": dict(sorted(measurement_counts.items())),
            "issue_codes_field": "issue_codes",
            "measurement_container": "measurements",
            "compatibility_aliases": ["issues", "top-level numeric measurements"],
            "missing_values_imputed": False,
        },
        "limitations": [
            "Existing LGE QC issue codes and measurements are not independent truth.",
            "Patient and slice identifiers and source paths are excluded.",
            "No source images or masks are supplied; missing measurements must not be imputed.",
            "Holdout membership is unavailable for this LGE manifest.",
        ],
    }


def evaluation_input(root: Path, workflow_stage: str | None = None) -> dict:
    salt = os.environ.get("CMR_LGE_CASE_SALT")
    if not salt:
        return {"status": "UNAVAILABLE", "records": []}
    if workflow_stage in LGE_STAGES:
        return _lge_evaluation_input(root, salt, workflow_stage)
    latest = root / "runtime/check_iterations/latest.json"
    if not latest.is_file():
        return {"status": "UNAVAILABLE", "records": []}
    summary = json.loads(latest.read_text(encoding="utf-8"))
    path = Path(summary.get("report_path", "")).resolve()
    base = (root / "runtime/check_iterations").resolve()
    if not path.is_relative_to(base) or not path.is_file():
        return {"status": "UNAVAILABLE", "records": []}
    raw = path.read_bytes()
    report = json.loads(raw)
    records = []
    current_keys = []
    for row in report.get("records", []):
        key = row.get("key")
        if not isinstance(key, str):
            continue
        current_keys.append(key)
        item = {"case_id": _case_id(salt, key)}
        for field in NUMERIC_FIELDS:
            value = row.get(field)
            if type(value) in (int, float) and math.isfinite(value):
                item[field] = value
        for field in STRING_FIELDS:
            value = row.get(field)
            if isinstance(value, str) and len(value) <= 80:
                item[field] = value
        for field in LIST_FIELDS:
            value = row.get(field)
            if isinstance(value, list):
                item[field] = [entry for entry in value[:12]
                               if isinstance(entry, str) and len(entry) <= 80]
        for field in DICT_FIELDS:
            value = row.get(field)
            if isinstance(value, dict):
                item[field] = {
                    str(key): number for key, number in list(value.items())[:20]
                    if isinstance(key, str) and len(key) <= 80
                    and type(number) in (int, float) and math.isfinite(number)
                }
        records.append(item)
    split = _frozen_holdout_input(root, salt, current_keys)
    return {"status": "AVAILABLE", "schema": "existing-qc-measurements-v1",
            "modality": "cmr", "unit": "spatial_volume",
            "cohort": "current_pending_queue_with_frozen_split",
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "created_at": summary.get("created_at"), "records": records,
            "full_pending_queue": len(records),
            "holdout_case_ids": split.get("case_ids", []),
            "frozen_split": {key: value for key, value in split.items()
                             if key != "case_ids"},
            "spatial_control": _spatial_control_input(path, salt),
            "quality_benchmark": _quality_benchmark(root),
            "reference_labels_available": False,
            "limitations": ["Measurements from existing QC evaluation, not independent truth.",
                            "No source images, masks or per-record reference verdicts supplied.",
                            "Frozen split membership is pseudonymous and is not a clinical verdict.",
                            "Gold-derived quality benchmarks are aggregate evidence, not per-record truth.",
                            "Missing measurements must not be imputed as zero."]}


def execute_evaluation(snapshot: Path, module: str, data: dict) -> dict:
    """Run an explicit owner entrypoint; retain its actual report for review."""
    import re
    snapshot = snapshot.resolve()
    if not re.fullmatch(r"(?:experiments|scripts)\.[a-zA-Z_][a-zA-Z0-9_]*", module):
        return {"status": "FAILED", "reason": "invalid_entrypoint"}
    if data.get("status") != "AVAILABLE":
        return {"status": "UNAVAILABLE", "reason": "no_real_measurements"}
    folder = snapshot.parent / "real_evaluation"
    folder.mkdir(exist_ok=True)
    source = folder / "input.json"
    source.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    runner = (
        "import importlib,json,sys; "
        "data=json.load(open(sys.argv[2],encoding='utf-8')); "
        "result=importlib.import_module(sys.argv[1]).run_evaluation(data); "
        "assert isinstance(result,dict), 'report must be an object'; "
        "print(json.dumps(result,ensure_ascii=False,allow_nan=False))"
    )
    env = {k: v for k, v in os.environ.items()
           if k in {"PATH", "LANG", "LC_ALL", "TMPDIR", "SYSTEMROOT"}}
    env["PYTHONPATH"] = str(snapshot)
    try:
        proc = subprocess.run([sys.executable, "-c", runner, module, str(source)],
                              cwd=snapshot, env=env, capture_output=True,
                              text=True, timeout=90)
        if proc.returncode:
            return {"status": "FAILED", "error": proc.stderr[-2000:]}
        if len(proc.stdout) > 2_000_000:
            return {"status": "FAILED", "reason": "report_too_large"}
        report = json.loads(proc.stdout)
        output = folder / "report.json"
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"status": "EXECUTED", "input_records": len(data["records"]),
                "input_modality": data.get("modality"),
                "input_cohort": data.get("cohort"),
                "source_sha256": data["source_sha256"],
                "report_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "report": report, "reference_labels_available": False,
                "note": "Owner-produced measurement report; independent review still required."}
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        return {"status": "FAILED", "error": str(exc)[:1000]}
