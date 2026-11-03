"""LGE candidate execution control plane. Domain algorithms live in governed generators.

This module resolves source identities, freezes a patient holdout, independently
scores staged masks, and applies validated bytes with recoverable receipts.
It never invents a repair, interprets a QC issue as a verdict, or trusts model
reported scores/counts. Calling preview never writes production labels.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np

from app.control_checkpoint import save_journal
from app.database import Database, utcnow
from app.manual_annotation import TARGET_DIRS
from app.rsi_kernel import canonical_hash
from app.lge_protocol import input_aliases as protocol_input_aliases, output_name as protocol_output_name, load as protocol_load

STAGES = {"lge_myo": "myo", "lge_mi": "mi"}
GENERATOR = "scripts/lge_candidate_generator.py"
SCHEMA = "lge-execution-v1"
INPUT_CONTRACT_VERSION = "lge-snapshot-paths-v4"
_SCORER_DIGEST = None


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root: Path, event: dict) -> None:
    path = root / "runtime/lge_execution/apply_audit.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"timestamp": utcnow(), **event}, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def contained(base: Path, relative: str) -> Path:
    rel = Path(relative)
    if rel.is_absolute() or "\\" in relative or ":" in relative or ".." in rel.parts:
        raise ValueError("invalid_relative_path")
    path = base / rel
    if not path.resolve().is_relative_to(base.resolve()) or path.is_symlink():
        raise ValueError("source_path_escape")
    return path


def policy(root: Path, dataset: Path) -> dict:
    config = json.loads((root / "configs/lge_execution.json").read_text(encoding="utf-8"))
    if Path(config["dataset"]).resolve() != dataset.resolve():
        raise ValueError("lge_dataset_not_authorized")
    if config.get("schema_version") != SCHEMA:
        raise ValueError("invalid_lge_execution_policy")
    return config


def source_queue(root: Path, dataset: Path, stage: str, data: dict) -> tuple[list[dict], str]:
    from experiments.runtime_evaluation import _latest_lge_manifest, _case_id
    salt = os.environ.get("CMR_LGE_CASE_SALT")
    if not salt:
        raise ValueError("missing_case_salt")
    manifest = _latest_lge_manifest(root)
    if manifest is None or sha(manifest) != data["source_sha256"]:
        raise ValueError("lge_manifest_changed")
    raw_rows = list(csv.DictReader(manifest.read_text(encoding="utf-8-sig").splitlines()))
    wanted = {r["case_id"] for r in data["records"]}
    evidence_by_id = {r["case_id"]: {key: r[key] for key in ("issue_codes", "measurements", "layer_index") if key in r}
                      for r in data["records"]}
    upstream_file = root / "runtime/lge_execution/upstream_myo.json"
    if stage == "lge_myo" and upstream_file.exists():
        upstream = json.loads(upstream_file.read_text(encoding="utf-8"))
        if upstream.get("dataset") != str(dataset.resolve()) or upstream.get("sha256") != canonical_hash(
                {k: v for k, v in upstream.items() if k != "sha256"}):
            raise ValueError("invalid_upstream_myo_request")
        present = {_case_id(salt, f"{r.get('patient','')}/{r.get('slice','')}#{r.get('layer_index','')}") for r in raw_rows}
        for case_id in set(upstream["case_ids"]) & present:
            wanted.add(case_id)
            evidence_by_id.setdefault(case_id, {"issue_codes": ["paired_myo_not_ready"]})
    found, rows, all_patients = set(), [], set()
    editor_manifest = dataset / "labels/check/manifest_backup/lge_issue_manifest.csv"
    editor_rows = list(csv.DictReader(editor_manifest.read_text(encoding="utf-8-sig").splitlines())) if editor_manifest.exists() else []
    editor_indices = {(r.get("patient"), r.get("slice") or r.get("volume"), r.get("layer_index", "")): i
                      for i, r in enumerate(editor_rows)}
    for row in raw_rows:
        patient, volume, layer = row.get("patient", ""), row.get("slice", ""), row.get("layer_index", "")
        if not patient or not volume:
            continue
        patient_id = _case_id(salt, str(dataset.resolve()) + ":patient:" + patient)
        all_patients.add(patient_id)
        case_id = _case_id(salt, f"{patient}/{volume}#{layer}")
        if case_id not in wanted or case_id in found:
            continue
        found.add(case_id)
        relative = (Path(patient) / volume).as_posix()
        image = contained(dataset / "images/lge", relative)
        labels = {name: contained(dataset / TARGET_DIRS[("lge", name)], relative) for name in ("myo", "mi")}
        target = labels[STAGES[stage]]
        sources = {"image": image, "label": target}
        # Both targets bind the paired mask. A changed Myo invalidates old MI
        # receipts; a Myo repair also checks the existing MI score for regression.
        other = "mi" if stage == "lge_myo" else "myo"
        if labels[other].is_file():
            sources[other] = labels[other]
        if any(not path.resolve().is_relative_to(dataset.resolve()) for path in sources.values()):
            raise ValueError("lge_source_outside_authorized_dataset")
        rows.append({"case_id": case_id, "patient_id": patient_id, "sources": sources,
                     "evidence": evidence_by_id[case_id],
                     "editor_index": editor_indices.get((patient, volume, layer)), "target": STAGES[stage]})
    if wanted != found:
        raise ValueError("lge_case_mapping_incomplete")
    # Never apply two independently generated candidates to one file in a batch.
    files = [str(r["sources"]["label"].resolve()) for r in rows]
    if len(files) != len(set(files)):
        raise ValueError("lge_duplicate_label_identity")
    frozen_path = root / "runtime/lge_execution/holdout.json"
    if frozen_path.exists():
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        expected = {k: v for k, v in frozen.items() if k != "sha256"}
        if frozen.get("sha256") != canonical_hash(expected) or frozen.get("dataset") != str(dataset.resolve()):
            raise ValueError("lge_holdout_changed")
        if frozen.get("identity_salt_sha256") != hashlib.sha256(salt.encode()).hexdigest():
            raise ValueError("lge_identity_salt_changed")
    else:
        patients = sorted(all_patients, key=lambda p: canonical_hash(p))
        count = min(2, max(1, len(patients) // 20)) if len(patients) > 1 else len(patients)
        frozen = {"dataset": str(dataset.resolve()), "patients": patients,
                  "identity_salt_sha256": hashlib.sha256(salt.encode()).hexdigest(),
                  "holdout_patients": patients[:count], "sampling": "patient_fixed_first_snapshot_v1"}
        frozen["sha256"] = canonical_hash(frozen)
        frozen_path.parent.mkdir(parents=True, exist_ok=True)
        save_journal(frozen_path, frozen)
    known, excluded = set(frozen["patients"]), set(frozen["holdout_patients"])
    for row in rows:
        row["excluded"] = row["patient_id"] in excluded or row["patient_id"] not in known
        row["exclusion_reason"] = "冻结抽样复核" if row["patient_id"] in excluded else "新病例尚未纳入冻结划分"
        row["hashes"] = {name: sha(path) for name, path in row["sources"].items() if path.is_file()}
        row["available"] = all(path.is_file() for path in row["sources"].values())
        if stage == "lge_mi" and "myo" not in row["sources"]:
            row["available"] = False
    return rows, frozen["sha256"]


def governed_generator(root: Path) -> dict:
    from app.rsi_runtime import scorer_dependency_paths
    path = root / GENERATOR
    if not path.is_file():
        return {"ready": False, "reason": "generator_not_promoted", "required_target": GENERATOR}
    module_hash = sha(path)
    db = Database(root / "runtime/workflow.sqlite")
    try:
        with db.connect() as connection:
            actions = connection.execute("SELECT run_id,action_id,artifact_path,payload_json FROM maintenance_actions "
                                         "WHERE status='PROMOTED' AND instr(payload_json,?)>0 "
                                         "ORDER BY run_id DESC LIMIT 200", (GENERATOR,)).fetchall()
        for action in actions:
            if GENERATOR not in (action["payload_json"] or ""):
                continue
            if not action["artifact_path"]:
                continue
            workspace = Path(action["artifact_path"])
            journal_path = workspace / "rsi_promotion_journal.json"
            if not journal_path.is_file():
                continue
            journal = json.loads(journal_path.read_text(encoding="utf-8"))
            if (journal.get("run_id"), journal.get("action_id")) != (action["run_id"], action["action_id"]):
                continue
            artifact = Path(journal["artifact"]).resolve()
            if not artifact.is_relative_to((root / "runtime/promotions").resolve()):
                continue
            manifest = json.loads((artifact / "promotion_manifest.json").read_text(encoding="utf-8"))
            validation = json.loads((workspace / "autopilot_validation.json").read_text(encoding="utf-8"))
            if validation.get("passed") is not True or validation.get("full_regression_required") is not True:
                continue
            if manifest.get("status") == "PROMOTED" and any(
                item.get("owner") == "orchestrator" and item.get("relative") == GENERATOR
                and item.get("new_hash") == module_hash for item in manifest.get("promoted", [])):
                dependencies, dynamic = scorer_dependency_paths(root, "scripts.lge_candidate_generator")
                dependencies = {dep for dep in dependencies if (root / dep).is_file()}
                if dynamic or any(not (workspace / "orchestrator" / dep).is_file()
                                  or sha(root / dep) != sha(workspace / "orchestrator" / dep) for dep in dependencies):
                    continue
                return {"ready": True, "module_sha256": module_hash,
                        "dependency_sha256": canonical_hash({dep: sha(root / dep) for dep in sorted(dependencies)}),
                        "input_contract_version": INPUT_CONTRACT_VERSION,
                        "protocol_sha256": canonical_hash(protocol_load(root)),
                        "run_id": action["run_id"], "action_id": action["action_id"]}
    except Exception as exc:
        return {"ready": False, "reason": "generator_release_unverifiable:" + type(exc).__name__}
    return {"ready": False, "reason": "generator_has_no_approved_release", "required_target": GENERATOR}


def calibration(root: Path, stage: str) -> tuple[dict, str]:
    from app.rsi_runtime import benchmark_contracts
    contract = benchmark_contracts(root)[stage]
    source = contained(root, contract["input_path"])
    data = json.loads(source.read_text(encoding="utf-8"))
    if canonical_hash(data) != contract["input_sha256"]:
        raise ValueError("lge_calibration_benchmark_changed")
    return data["profile"], canonical_hash({"profile": data["profile"], "scorer": scorer_fingerprint(root)})


def scorer_fingerprint(root: Path) -> str:
    from app.rsi_runtime import scorer_dependency_paths
    paths, dynamic = scorer_dependency_paths(root, "app.quality_scoring")
    if dynamic:
        paths.update(path.relative_to(root).as_posix() for folder in ("app", "scripts", "experiments")
                     for path in (root / folder).rglob("*.py"))
    return canonical_hash({name: sha(root / name) for name in sorted(paths) if (root / name).is_file()})


def measure(row: dict, label: Path, profile: dict) -> dict:
    import importlib
    global _SCORER_DIGEST
    module = importlib.import_module("app.quality_scoring")
    source = Path(module.__file__).resolve()
    current_digest = sha(source)
    if _SCORER_DIGEST != current_digest:
        cached = Path(module.__cached__).resolve() if getattr(module, "__cached__", None) else None
        if cached and cached.parent == source.parent / "__pycache__":
            cached.unlink(missing_ok=True)
        module = importlib.reload(module)
        _SCORER_DIGEST = current_digest
    PairSpec, extract_pair, score_record = module.PairSpec, module.extract_pair, module.score_record
    sources = row["sources"]
    common = dict(dataset="active_lge", tier="unreviewed", image=sources["image"],
                  relative_id="anonymous", case_id="anonymous", slice_position=0, reviewed=False)
    spec = PairSpec(label=label, target="lge_" + row["target"],
                    myo=sources.get("myo") if row["target"] == "mi" else None, **common)
    feature = extract_pair(spec)
    if not feature.get("valid"):
        raise ValueError("invalid_lge_pair")
    score = score_record(feature, profile)["score"]
    if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 100:
        raise ValueError("invalid_lge_score")
    result = {"score": float(score)}
    if row["target"] == "myo" and "mi" in sources:
        other = extract_pair(PairSpec(label=sources["mi"], myo=label, target="lge_mi", **common))
        if not other.get("valid"):
            raise ValueError("invalid_paired_mi")
        result["paired_mi_score"] = float(score_record(other, profile)["score"])
        if not math.isfinite(result["paired_mi_score"]):
            raise ValueError("invalid_paired_mi_score")
    return result


def validate_mask(source: Path, candidate: Path, staged: Path) -> None:
    original, proposed = nib.load(str(source)), nib.load(str(candidate))
    if original.shape != proposed.shape or not np.allclose(original.affine, proposed.affine, atol=1e-6):
        raise ValueError("candidate_geometry_mismatch")
    values = np.asanyarray(proposed.dataobj)
    if not np.isfinite(values).all() or not np.isin(values, (0, 1)).all():
        raise ValueError("candidate_not_binary")
    # Retain the original geometry/header and dtype when serializing new bytes.
    nib.save(original.__class__(values.astype(original.get_data_dtype()), original.affine, original.header.copy()), str(staged))


def producer(root: Path, rows: list[dict], work: Path, stage: str, module_hash: str,
             training_profile: dict | None = None) -> list[dict]:
    inputs, output = work / "inputs", work / "outputs"
    inputs.mkdir(); output.mkdir()
    records = []
    for row in rows:
        copies = {}
        for name, source in row["sources"].items():
            ext = ".nii.gz" if source.name.endswith(".nii.gz") else ".nii"
            copied = inputs / (row["case_id"] + "_" + name + ext)
            shutil.copy2(source, copied)
            if sha(copied) != row["hashes"][name]:
                raise ValueError("source_changed_during_snapshot")
            copies[name] = str(copied)
        records.append({**row.get("evidence", {}), "case_id": row["case_id"], "paths": copies,
                        "source_sha256": row["hashes"]["label"],
                        **protocol_input_aliases(root, copies)})
    runner = '''import sys,json,importlib,os
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import numpy,scipy.ndimage,nibabel
data=json.load(sys.stdin)
code=Path(sys.argv[1]).resolve();work=Path(sys.argv[2]).resolve();out=work/'outputs'
allowed=[code/'app',code/'scripts',code/'experiments',work,Path(sys.prefix).resolve(),Path(sys.base_prefix).resolve()]
def guard(event,args):
 if event=='open' and not isinstance(args[0],int):
  p=Path(args[0]).resolve();mode=args[1]
  if not any(p.is_relative_to(base) for base in allowed):raise PermissionError('unsupplied input')
  writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(args[2],int) and args[2]&3)
  if writing and not p.is_relative_to(out):raise PermissionError('write outside candidate output')
 if event in ('os.remove','os.rmdir','os.mkdir','os.chmod','os.chown','os.utime','os.truncate','os.rename','os.symlink','os.link'):
  paths=args[:2] if event in ('os.rename','os.symlink','os.link') else args[:1]
  if any(not isinstance(p,(str,bytes,os.PathLike)) or not Path(os.fsdecode(p)).resolve().is_relative_to(out) for p in paths):raise PermissionError('mutation outside candidate output')
 if event.startswith(('socket.','subprocess.')) or event in ('os.system','ctypes.dlopen','os.exec','os.spawn','os.posix_spawn','os.fork'):raise PermissionError('external execution')
sys.addaudithook(guard)
module=importlib.import_module('scripts.lge_candidate_generator')
report=module.generate_candidates(data,str(out))
print(json.dumps(report,allow_nan=False))
'''
    if sha(root / GENERATOR) != module_hash:
        raise ValueError("generator_changed")
    environment = {k: v for k, v in os.environ.items() if k.upper() in {"PATH", "LANG", "LC_ALL", "SYSTEMROOT", "TMP", "TEMP"}}
    run = subprocess.run([sys.executable, "-I", "-c", runner, str(root.resolve()), str(work.resolve())],
                         input=json.dumps({"schema_version": SCHEMA, "input_contract_version": INPUT_CONTRACT_VERSION, "workflow_stage": stage,
                                           "calibration_profile": training_profile or {}, "records": records}),
                         text=True, capture_output=True, env=environment, cwd=work, timeout=180)
    if run.returncode or len(run.stdout) > 2_000_000:
        raise ValueError("lge_generator_failed:" + run.stderr[-300:])
    result = json.loads(run.stdout)
    if not isinstance(result, list) or len(result) > len(rows):
        raise ValueError("invalid_candidate_batch")
    ids = [r.get("case_id") for r in result if isinstance(r, dict)]
    if len(ids) != len(result) or len(ids) != len(set(ids)) or set(ids) != {r["case_id"] for r in rows}:
        raise ValueError("candidate_identity_mismatch")
    if sha(root / GENERATOR) != module_hash:
        raise ValueError("generator_changed")
    for item in result:
        if item.get("decision") == "repair":
            item["candidate"] = protocol_output_name(root, item)
            if not isinstance(item.get("candidate"), str):
                raise ValueError("generator_protocol_missing_candidate_filename")
            candidate = contained(output, item["candidate"])
            if not candidate.is_file():
                raise ValueError("generator_protocol_candidate_missing")
    for row in rows:
        for name in row["sources"]:
            if sha(Path(next(r for r in records if r["case_id"] == row["case_id"])["paths"][name])) != row["hashes"][name]:
                raise ValueError("generator_modified_input")
    return result


def receipt_path(root: Path, stage: str, case_id: str) -> Path:
    path = root / "runtime/lge_execution/receipts" / stage / (canonical_hash(case_id) + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def receipt_current(receipt: dict, row: dict, calibration_id: str) -> bool:
    actual = row["hashes"]
    approved = receipt.get("after_hashes", {})
    if row["target"] == "myo":
        # Independently validated MI updates do not change Myo itself. In the
        # other direction, MI always binds the exact Myo bytes it was graded on.
        actual = {k: v for k, v in actual.items() if k != "mi"}
        approved = {k: v for k, v in approved.items() if k != "mi"}
    return (receipt.get("status") == "CHECKED" and receipt.get("case_id") == row["case_id"]
            and receipt.get("calibration_id") == calibration_id
            and approved == actual)


def human_acceptances(root: Path) -> dict:
    """Read actual signed editor applies, never model descriptions of a review."""
    result = {}
    for path in (root / "runtime/manual_annotations/edits").glob("*/provenance.json"):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
            if (item.get("status") == "APPLIED" and item.get("applied_by") and item.get("applied_hash")
                    and item.get("modality") == "lge" and item.get("applied_context")):
                key = (str(Path(item["source_label"]).resolve()), item["applied_hash"])
                result[key] = item
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return result


def recover(root: Path, dataset: Path) -> None:
    """Recover apply -> receipt/audit gaps without reapplying candidate bytes."""
    folder = root / "runtime/lge_execution/transactions"
    for path in folder.glob("*.json"):
        item = json.loads(path.read_text(encoding="utf-8"))
        if item.get("state") == "COMMITTED":
            continue
        if item.get("dataset") != str(dataset.resolve()):
            raise ValueError("lge_recovery_dataset_mismatch")
        target = contained(dataset / TARGET_DIRS[("lge", item["target"])], item["relative"])
        current = sha(target)
        if current == item["after_sha256"]:
            for dependency, expected in item["dependencies"].items():
                if sha(contained(dataset, dependency)) != expected:
                    raise ValueError("lge_recovery_dependency_conflict")
            audit(root, {"event": "LGE_APPLY_RECOVERED", "transaction_id": item["transaction_id"],
                         "case_id": item["case_id"], "target": item["target"], "label_write": item["label_write"]})
            save_journal(receipt_path(root, item["stage"], item["case_id"]), item["receipt"])
            item["state"] = "COMMITTED"
            save_journal(path, item)
        elif current == item["before_sha256"]:
            item["state"] = "COMMITTED"
            item["result"] = "NOT_APPLIED"
            save_journal(path, item)
        else:
            raise ValueError("lge_recovery_source_conflict")


def apply_one(root: Path, dataset: Path, row: dict, proposal: dict, work: Path,
              profile: dict, calibration_id: str, stage: str, release: dict, execute: bool) -> dict:
    from app.rsi_postpromotion import file_lock
    with file_lock(dataset / "label_iteration/control/label_writer.lock", exclusive=True):
        return _apply_one_locked(root, dataset, row, proposal, work, profile, calibration_id, stage, release, execute)


def _apply_one_locked(root: Path, dataset: Path, row: dict, proposal: dict, work: Path,
                      profile: dict, calibration_id: str, stage: str, release: dict, execute: bool) -> dict:
    decision = proposal.get("decision")
    reason = str(proposal.get("reason") or "").strip()[:500]
    if not reason or decision not in {"repair", "accept_existing", "needs_review"}:
        raise ValueError("invalid_candidate_decision")
    if decision == "needs_review":
        return {"case_id": row["case_id"], "status": "NEEDS_REVIEW", "reason": reason}
    source = row["sources"]["label"]
    before = measure(row, source, profile)
    staged = work / ("validated_" + uuid.uuid4().hex + (".nii.gz" if source.name.endswith(".nii.gz") else ".nii"))
    if decision == "repair":
        candidate = contained(work / "outputs", str(proposal["candidate"]))
        validate_mask(source, candidate, staged)
        after = measure(row, staged, profile)
        if before["score"] >= after["score"] or sha(staged) == row["hashes"]["label"]:
            raise ValueError("candidate_has_no_scored_improvement")
        if "paired_mi_score" in before and after["paired_mi_score"] < before["paired_mi_score"]:
            raise ValueError("candidate_degrades_paired_mi")
    else:
        after = before
    if after["score"] < 70:
        raise ValueError("candidate_below_existing_grade_C")
    if row["target"] == "mi":
        myo_row = {**row, "target": "myo", "sources": {"image": row["sources"]["image"]}}
        if measure(myo_row, row["sources"]["myo"], profile)["score"] < 70:
            raise ValueError("paired_myo_not_ready")
    actual = {name: sha(path) for name, path in row["sources"].items()}
    if actual != row["hashes"]:
        raise ValueError("stale_lge_source")
    result = {"case_id": row["case_id"], "status": "VALIDATED", "reason": reason,
              "score_before": before["score"], "score_after": after["score"],
              "label_write": decision == "repair"}
    if not execute:
        return result
    row["_write_started"] = True
    transaction_id = uuid.uuid4().hex
    folder = root / "runtime/lge_execution/transactions"
    folder.mkdir(parents=True, exist_ok=True)
    after_hash = sha(staged) if decision == "repair" else actual["label"]
    receipt = {**result, "status": "CHECKED", "case_id": row["case_id"], "calibration_id": calibration_id,
               "after_hashes": {**actual, "label": after_hash}, "release": release,
               "transaction_id": transaction_id, "timestamp": utcnow()}
    backup = folder / (transaction_id + (".backup.nii.gz" if source.name.endswith(".nii.gz") else ".backup.nii"))
    if decision == "repair":
        shutil.copy2(source, backup)
        backup.chmod(0o600)
        if sha(backup) != actual["label"]:
            raise ValueError("lge_backup_source_changed")
    item = {"transaction_id": transaction_id, "state": "PREPARED", "stage": stage,
            "dataset": str(dataset.resolve()),
            "case_id": row["case_id"], "target": row["target"],
            "relative": source.relative_to(dataset / TARGET_DIRS[("lge", row["target"])]).as_posix(),
            "before_sha256": actual["label"], "after_sha256": after_hash,
            "label_write": decision == "repair", "receipt": receipt,
            "dependencies": {str(path.relative_to(dataset)): actual[name] for name, path in row["sources"].items() if name != "label"},
            "backup": str(backup) if decision == "repair" else None}
    journal = folder / (transaction_id + ".json")
    save_journal(journal, item)
    audit(root, {"event": "LGE_APPLY_PREPARED", **result, "transaction_id": transaction_id,
                 "sha_before": actual["label"], "sha_after": after_hash, "backup": item["backup"]})
    if {name: sha(path) for name, path in row["sources"].items()} != actual:
        raise ValueError("stale_lge_source_before_write")
    if decision == "repair":
        temporary = source.with_name("." + transaction_id + source.name)
        shutil.copy2(staged, temporary)
        temporary.chmod(source.stat().st_mode & 0o777)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, source)
        if os.name != "nt":
            descriptor = os.open(str(source.parent), os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    audit(root, {"event": "LGE_APPLIED", **result, "transaction_id": transaction_id,
                 "sha_before": actual["label"], "sha_after": after_hash})
    save_journal(receipt_path(root, stage, row["case_id"]), receipt)
    item["state"] = "COMMITTED"
    save_journal(journal, item)
    return dict(receipt)


def run_batch(root: Path, dataset: Path, stage: str, data: dict, *, execute: bool = False,
              limit: int = 10, status_only: bool = False) -> dict:
    from app.rsi_postpromotion import installation_lock, file_lock
    config = policy(root, dataset)
    if stage not in STAGES or data.get("workflow_stage") != stage:
        raise ValueError("lge_execution_stage_mismatch")
    if execute and config.get("enabled") is not True:
        raise ValueError("lge_execution_disabled")
    operator = root / "runtime/control/operator_state.json"
    if execute and operator.exists() and json.loads(operator.read_text(encoding="utf-8")).get("paused"):
        raise ValueError("operator_paused")
    # Separate ledger lock; outer Check Loop holds the source-installation read lock.
    with installation_lock(root / "runtime/lge_execution", exclusive=True):
        if execute:
            with file_lock(dataset / "label_iteration/control/label_writer.lock", exclusive=True):
                recover(root, dataset)
        rows, holdout = source_queue(root, dataset, stage, data)
        profile, calibration_id = calibration(root, stage)
        release = governed_generator(root)
        if not rows:
            release = {"ready": True, "reason": "empty_stage"}
        humans = human_acceptances(root)
        resolved, queued, heldout, parked = [], [], [], []
        for row in rows:
            receipt = receipt_path(root, stage, row["case_id"])
            old = json.loads(receipt.read_text(encoding="utf-8")) if receipt.exists() else {}
            human = humans.get((str(row["sources"]["label"].resolve()), row["hashes"].get("label")))
            required = ("image", "myo") if stage == "lge_mi" else ("image",)
            human_current = bool(human and human.get("target") == row["target"] and row["available"]
                and all(human["applied_context"].get(name) == row["hashes"].get(name) for name in required))
            if human_current:
                resolved.append(row)
            elif row["excluded"]:
                heldout.append(row)
            elif row["available"] and receipt_current(old, row, calibration_id):
                resolved.append(row)
            elif not row["available"] or (old.get("status") == "NEEDS_REVIEW" and old.get("after_hashes") == row["hashes"]
                 and old.get("calibration_id") == calibration_id and old.get("release") == release):
                row["parked_reason"] = old.get("reason", "missing_source_pair")
                row["execution_error"] = str(row["parked_reason"]).startswith("generator_failure:")
                parked.append(row)
            else:
                queued.append(row)
        outcomes = []
        if release["ready"] and not status_only:
            selected = [r for r in queued if r["available"]][:max(0, min(int(limit), 10))]
            if selected:
                temp_parent = root / "runtime/lge_execution/candidates"
                temp_parent.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(dir=str(temp_parent), prefix="batch-") as tmp:
                    work = Path(tmp)
                    try:
                        proposals = producer(root, selected, work, stage, release["module_sha256"], profile)
                    except (OSError, ValueError, TypeError, KeyError, subprocess.TimeoutExpired) as exc:
                        proposals = [{"case_id": r["case_id"], "decision": "needs_review",
                                      "reason": "generator_failure:" + str(exc)[-500:]} for r in selected]
                    by_id = {r["case_id"]: r for r in selected}
                    for proposal in proposals:
                        row = by_id[proposal["case_id"]]
                        try:
                            outcome = apply_one(root, dataset, row, proposal, work, profile, calibration_id, stage, release, execute)
                        except (OSError, ValueError, TypeError, KeyError, nib.filebasedimages.ImageFileError) as exc:
                            if isinstance(exc, OSError) and row.get("_write_started"):
                                raise  # An apply/audit failure must be recovered, not parked as a bad mask.
                            outcome = {"case_id": row["case_id"], "status": "NEEDS_REVIEW", "reason": str(exc)[:300]}
                            if execute and stage == "lge_mi" and type(exc) is ValueError and str(exc) == "paired_myo_not_ready":
                                path = root / "runtime/lge_execution/upstream_myo.json"
                                previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"case_ids": []}
                                request = {"dataset": str(dataset.resolve()), "case_ids": sorted(set(previous["case_ids"]) | {row["case_id"]})}
                                request["sha256"] = canonical_hash(request)
                                save_journal(path, request)
                                outcome["upstream_dependency"] = "lge_myo"
                        if str(outcome.get("reason", "")).startswith("generator_failure:"):
                            outcome["execution_error"] = True
                        outcomes.append(outcome)
                        if execute:
                            audit(root, {"event": "LGE_CANDIDATE_RESULT", **outcome, "stage": stage})
                            if outcome["status"] == "NEEDS_REVIEW":
                                save_journal(receipt_path(root, stage, row["case_id"]), {
                                    **outcome, "after_hashes": row["hashes"], "calibration_id": calibration_id, "release": release})
        checked = {r["case_id"] for r in resolved} | {r["case_id"] for r in outcomes if r["status"] == "CHECKED"}
        manual = {r["case_id"] for r in heldout + parked if not r.get("execution_error")} | {r["case_id"] for r in outcomes if r["status"] == "NEEDS_REVIEW" and not r.get("execution_error")}
        execution_failed = {r["case_id"] for r in parked + outcomes if r.get("execution_error")}
        prefix = stage
        report = {"schema_version": SCHEMA, "execution_mode": "execute" if execute else "preview",
                  "workflow_stage": stage, "source_modality": "lge", "input_source_sha256": data["source_sha256"],
                  prefix + "_consumer_ready": release["ready"], prefix + "_resolved": len(checked),
                  prefix + "_work": len(manual - checked), prefix + "_pending": len(rows) - len(checked | manual),
                  "holdout_sha256": holdout, "holdout_excluded": len(heldout),
                  "label_write": any(r.get("label_write") and r["status"] == "CHECKED" for r in outcomes),
                  "production_writes": sum(bool(r.get("label_write")) and r["status"] == "CHECKED" for r in outcomes),
                  "generator": release, "calibration_id": calibration_id, "outcomes": outcomes,
                  prefix + "_execution_failed": len(execution_failed),
                  "audited_resolved": len(checked), "published_at": utcnow()}
        report[prefix + "_auto_pending"] = report[prefix + "_pending"]
        attempted_ids = {r["case_id"] for r in outcomes} if execute else set()
        report[prefix + "_dispatchable"] = sum(bool(r["available"]) and r["case_id"] not in attempted_ids for r in queued)
        review_limit = min(20, max(2, (len(rows) - len(heldout)) // 20))
        report[prefix + "_auto_complete"] = bool(release["ready"] and (checked or not rows)
            and report[prefix + "_pending"] == 0 and len(manual) - len(heldout) <= review_limit)
        report["receipt_revision"] = canonical_hash({"stage": stage, "checked": sorted(checked),
            "sources": {r["case_id"]: r["hashes"] for r in rows}, "calibration": calibration_id})
        failures = Counter(r.get("parked_reason", "needs_review") for r in parked)
        failures.update(r["reason"] for r in outcomes if r["status"] == "NEEDS_REVIEW")
        report["failure_reason_counts"] = dict(failures.most_common(20))
        report["manual_review_items"] = [{"case_id": r["case_id"], "editor_index": r["editor_index"],
            "href": ("/editor?pool=lge_issue_manifest&mode=manual&index=" + str(r["editor_index"])) if r["editor_index"] is not None else "/editor?pool=lge_issue_manifest&mode=manual",
            "reason": r["exclusion_reason"] if r["excluded"] else "需要人工示例或新的领域修法"}
            for r in rows if r["case_id"] in manual]
        if execute:
            out = root / "runtime/lge_execution" / (stage + ".json")
            save_journal(out, report)
        return report


def status(root: Path) -> dict:
    """Small read-only capsule for roles and UI; no candidate generation."""
    configured = (root / "configs/lge_execution.json").is_file()
    if not configured:
        return {"configured": False}
    release = governed_generator(root)
    path = root / "runtime/lge_iterations/latest.json"
    try:
        report = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        if not isinstance(report, dict):
            raise ValueError("invalid_execution_report")
        stages = report.get("execution_status", {}) if report.get("schema_version") == SCHEMA else {}
        result = {"configured": True, "generator": release, "schema_version": SCHEMA,
                  "audited_execution": report.get("audited_execution") is True,
                  "execution_revision": report.get("execution_revision"), "stages": {}}
        for stage, values in stages.items():
            result["stages"][stage] = {key: values.get(key) for key in (
                stage + "_pending", stage + "_work", stage + "_resolved", stage + "_auto_complete",
                "holdout_excluded", "calibration_id", stage + "_execution_failed")}
            result["stages"][stage]["failure_reason_counts"] = values.get("failure_reason_counts", {})
            items = values.get("manual_review_items", [])
            result["stages"][stage]["manual_review_count"] = len(items)
            result["stages"][stage]["manual_review_items"] = items[:20]
        return result
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return {"configured": True, "generator": release, "status": "UNAVAILABLE", "reason": str(exc)[:200]}
