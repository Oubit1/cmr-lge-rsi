"""Content-addressed segmentation lineage and a mechanical selection gate.

This module never trains, loads a model, computes a metric, or chooses a medical
threshold. It uses only the standard library (Python 3.9+).

Trust boundary: a trusted controller owns the registry directory. Candidate code
must have read-only access, and must NOT be allowed to call ``approve_policy``,
``record_assessment``, selection, or rollback through an unauthenticated API.
The controller authenticates policy approvers and evaluator receipts *outside*
this module. Actor strings, hashes, and patient-disjoint attestations are not
signatures or proof of scientific validity. Owner-supplied approval flags are
never accepted. ``assess_candidate`` is a pure preview, not an authorization.

Training receipts require status=SUCCEEDED, training_source=training/fine_tuning,
owner_role, attempt_id, finite numeric metrics (possibly empty), and
artifact_hashes for weights/config/architecture/dataset_manifest. Dataset
manifests are opaque byte snapshots; their producers must enumerate immutable
data/split/content hashes. The registry does not copy the underlying dataset.

``approve_policy(root, definition, approver=...)`` archives an already approved
policy received by the controller. A definition has target, dataset_sha256,
benchmark_sha256, and metrics: {name: {kind: segmentation/qc, direction: min/max,
min_delta: nonnegative number, max_regression: nonnegative number}}. There must
be a segmentation metric. All regression limits apply; at least one *strictly*
positive segmentation delta must meet its min_delta. No thresholds are defaulted.

A comparison has candidate_id, parent_id, target, candidate_artifact_hashes,
parent_artifact_hashes, dataset_sha256, benchmark_sha256, two literal-True flags
validation_test_patient_disjoint/training_evaluation_patient_disjoint, evaluator
{actor, role: independent_evaluator, status: SUCCEEDED}, measurement {source:
predictions, self_reported: false, candidate_predictions_sha256,
parent_predictions_sha256, reference_sha256}, and metrics {target: {name:
{candidate, parent, candidate_count, parent_count}}}. Counts are equal positive
integers for a paired benchmark. QC cannot supply the required improvement.

``record_assessment`` ingests independently obtained evidence against a stored
policy and returns a handle. ``activate`` accepts that handle, reloads immutable
evidence/policy/lineage, and recomputes the gate under a process-safe lock. A
baseline can only initialize an empty selector via ``select_baseline``; it is
explicitly unmeasured. Import and assessment never select anything themselves.

Stored files are append-only and read-only, with hashes checked on use. This is
tamper detection, not protection against an administrator who can rewrite the
entire registry. Local filesystems with atomic rename and OS file locking are
required. Mutations fail fast with ConcurrentUpdate when busy (caller retries).
History precedes the atomic pointer swap, so a failed swap leaves only an orphan
history record. Rollback is an explicit CAS selection of the previous version.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path


TARGETS = frozenset({"cmr_myo", "lge_myo", "lge_mi"})
ARTIFACTS = ("weights", "config", "architecture", "dataset_manifest")
CHUNK_SIZE = 1024 * 1024
_SHA = re.compile(r"[0-9a-f]{64}\Z")


class RegistryError(ValueError):
    """Invalid, missing, or corrupted registry input/evidence."""


class ConcurrentUpdate(RuntimeError):
    """The selector changed or another registry writer holds the lock."""


def _require(condition, message):
    if not condition:
        raise RegistryError(message)


def _keys(value, expected, label):
    _require(type(value) is dict and set(value) == set(expected), label + ": keys")


def _text(value, label):
    _require(type(value) is str and bool(value.strip()) and value == value.strip(), label)
    return value


def _identifier(value):
    _require(type(value) is str and _SHA.fullmatch(value) is not None, "invalid id/hash")
    return value


def _target(value):
    _require(type(value) is str and value in TARGETS, "invalid target")
    return value


def _number(value):
    _require(type(value) in (int, float), "metric/threshold must be numeric, not boolean")
    try:
        valid = math.isfinite(value)
    except OverflowError:
        valid = False
    _require(valid, "non-finite metric/threshold")
    return value


def _canonical(value):
    def validate(item):
        if type(item) is dict:
            _require(all(type(key) is str for key in item), "JSON keys must be strings")
            for child in item.values():
                validate(child)
        elif type(item) is list:
            for child in item:
                validate(child)
        else:
            _require(item is None or type(item) in (str, bool, int, float), "not JSON")
            if type(item) in (int, float):
                _number(item)
    validate(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def _hash(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _record(payload):
    return dict(payload, id=_hash(payload))


def _check_record(record):
    _require(type(record) is dict and "id" in record, "missing record id")
    _identifier(record["id"])
    _require(_hash({k: v for k, v in record.items() if k != "id"}) == record["id"],
             "record hash mismatch")


def _root(root):
    return Path(root).resolve()


def _no_link(path):
    if path.exists() or path.is_symlink():
        info = path.lstat()
        _require(not stat.S_ISLNK(info.st_mode) and
                 not (getattr(info, "st_file_attributes", 0) & 0x400),
                 "symlink/reparse point forbidden")


def _path(root, *parts):
    path = root
    for part in parts:
        _require(type(part) is str and part not in ("", ".", "..") and
                 not any(c in part for c in "/\\:"), "unsafe path")
        path = path / part
        _no_link(path)
    _require(path.resolve().is_relative_to(root), "path escapes registry")
    return path


def _sha_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        _require(stat.S_ISREG(os.fstat(handle.fileno()).st_mode), "not a regular file")
        for block in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    try:
        result = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs)
        _canonical(result)
        return result
    except (OSError, ValueError) as exc:
        raise RegistryError("cannot read valid registry JSON: " + str(path)) from exc


def _sync_dir(path):
    if os.name != "nt":
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _atomic_json(path, value, *, immutable):
    data = _canonical(value)
    if immutable and path.exists():
        _require(_canonical(_load(path)) == data, "immutable record collision")
        return
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=str(path.parent))
    pending = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(pending), str(path))
        if immutable:
            path.chmod(stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
        _sync_dir(path.parent)
    finally:
        if pending.exists():
            pending.unlink()


@contextmanager
def _locked(root):
    root.mkdir(parents=True, exist_ok=True)
    lock_path = _path(root, ".registry.lock")
    with lock_path.open("a+b") as handle:
        if lock_path.stat().st_size == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ConcurrentUpdate("registry busy; retry") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _store(root, collection, record):
    folder = _path(root, collection)
    folder.mkdir(exist_ok=True)
    _atomic_json(_path(root, collection, _identifier(record["id"]) + ".json"),
                 record, immutable=True)


def _read_record(root, collection, identifier):
    record = _load(_path(root, collection, _identifier(identifier) + ".json"))
    _check_record(record)
    _require(record["id"] == identifier, "record address mismatch")
    return record


def _artifact_hashes(candidate):
    return {key: candidate["artifacts"][key]["sha256"] for key in ARTIFACTS}


def _hashes(value):
    _keys(value, ARTIFACTS, "artifact hashes")
    for sha in value.values():
        _identifier(sha)


def _training(record, hashes):
    _require(type(record) is dict, "training record required")
    _require(record.get("status") == "SUCCEEDED", "training did not succeed")
    _require(record.get("training_source") in ("training", "fine_tuning"),
             "training source must describe actual training")
    _require(record.get("evaluation_only", False) is False, "evaluation-only receipt is not training")
    _text(record.get("attempt_id"), "training attempt_id")
    role = _text(record.get("owner_role"), "training owner_role")
    _require(role.casefold() not in ("independent_evaluator", "policy_approver"), "owner role conflict")
    _hashes(record.get("artifact_hashes"))
    _require(record["artifact_hashes"] == hashes, "training artifact hash mismatch")
    _require(type(record.get("metrics")) is dict, "attempt metrics required")
    for name, value in record["metrics"].items():
        _text(name, "attempt metric name")
        _number(value)
    _canonical(record)  # Preserve additional JSON provenance, including provider replacement.


def _candidate_shape(candidate):
    _check_record(candidate)
    _keys(candidate, {"id", "schema", "kind", "target", "parent_id", "owner",
                      "parent_artifact_hashes", "dataset_sha256", "artifacts",
                      "training_record"}, "candidate")
    _require(candidate["schema"] == "segmentation-candidate-v1", "candidate schema")
    _target(candidate["target"])
    _text(candidate["owner"], "owner")
    _keys(candidate["artifacts"], ARTIFACTS, "artifacts")
    for name, artifact in candidate["artifacts"].items():
        _keys(artifact, {"path", "sha256", "size"}, "artifact")
        _require(artifact["path"] == name, "artifact path mismatch")
        _identifier(artifact["sha256"])
        _require(type(artifact["size"]) is int and artifact["size"] > 0, "empty artifact")
    hashes = _artifact_hashes(candidate)
    _require(candidate["dataset_sha256"] == hashes["dataset_manifest"], "dataset digest mismatch")
    if candidate["parent_id"] is None:
        _require(candidate["parent_artifact_hashes"] is None, "unexpected parent hashes")
    else:
        _identifier(candidate["parent_id"])
        _hashes(candidate["parent_artifact_hashes"])
    if candidate["kind"] == "trained":
        _training(candidate["training_record"], hashes)
    else:
        _require(candidate["kind"] == "baseline" and candidate["parent_id"] is None and
                 candidate["training_record"] == {"status": "EXISTING_UNMEASURED"},
                 "invalid baseline")


def _read_candidate(root, candidate_id):
    _identifier(candidate_id)
    candidate = _load(_path(root, "candidates", candidate_id, "candidate.json"))
    _candidate_shape(candidate)
    _require(candidate["id"] == candidate_id, "candidate address mismatch")
    for name, artifact in candidate["artifacts"].items():
        path = _path(root, "candidates", candidate_id, name)
        _require(path.is_file() and path.stat().st_size == artifact["size"] and
                 _sha_file(path) == artifact["sha256"], "artifact hash/size mismatch: " + name)
    return candidate


def _copy(source, destination):
    source = Path(source).absolute()
    for part in (source,) + tuple(source.parents):
        _no_link(part)
    def stamp(info):
        # Windows stat/fstat may expose different ctime semantics (creation vs
        # metadata change). Identity, size, mtime and the second full hash still
        # detect changed/replaced sources there; POSIX also checks ctime.
        return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns,
                info.st_ctime_ns if os.name != "nt" else None)
    digest = hashlib.sha256()
    with source.open("rb") as incoming, destination.open("xb") as outgoing:
        before = os.fstat(incoming.fileno())
        _require(stat.S_ISREG(before.st_mode) and before.st_size > 0, "empty/non-file source")
        for block in iter(lambda: incoming.read(CHUNK_SIZE), b""):
            digest.update(block)
            outgoing.write(block)
        outgoing.flush()
        os.fsync(outgoing.fileno())
        _require(stamp(before) == stamp(os.fstat(incoming.fileno())), "source changed during copy")
    sha = digest.hexdigest()
    _require(stamp(before) == stamp(source.stat()) and _sha_file(source) == sha and
             stamp(before) == stamp(source.stat()),
             "source changed during copy")
    return {"path": destination.name, "sha256": sha, "size": before.st_size}


def _register(root, *, target, weights, config, architecture, dataset_manifest,
              parent_id, owner, training_record, kind):
    root = _root(root)
    _target(target)
    _text(owner, "owner")
    # Detach mutable caller-owned receipt before copying anything.
    training_record = json.loads(_canonical(training_record))
    with _locked(root):
        parent = None if parent_id is None else _read_candidate(root, parent_id)
        if parent is not None:
            _require(parent["target"] == target, "parent target mismatch")
        folder = _path(root, "candidates")
        folder.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".candidate-", dir=str(root)) as temporary:
            stage = Path(temporary) / "bundle"
            stage.mkdir()
            sources = dict(weights=weights, config=config, architecture=architecture,
                           dataset_manifest=dataset_manifest)
            artifacts = {name: _copy(source, stage / name) for name, source in sources.items()}
            candidate = _record({"schema": "segmentation-candidate-v1", "kind": kind,
                                 "target": target, "parent_id": parent_id, "owner": owner,
                                 "parent_artifact_hashes": _artifact_hashes(parent) if parent else None,
                                 "dataset_sha256": artifacts["dataset_manifest"]["sha256"],
                                 "artifacts": artifacts, "training_record": training_record})
            _candidate_shape(candidate)
            destination = _path(root, "candidates", candidate["id"])
            if destination.exists():
                return _read_candidate(root, candidate["id"])
            _atomic_json(stage / "candidate.json", candidate, immutable=True)
            for path in stage.iterdir():
                path.chmod(stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
            os.rename(str(stage), str(destination))
            _sync_dir(folder)
            return candidate


def register_candidate(registry_root: Path, *, target: str, weights: Path,
                       config: Path, architecture: Path, dataset_manifest: Path,
                       parent_id: str | None, owner: str, training_record: dict) -> dict:
    """Snapshot a completed training attempt; never change the active selector."""
    return _register(registry_root, target=target, weights=weights, config=config,
                     architecture=architecture, dataset_manifest=dataset_manifest,
                     parent_id=parent_id, owner=owner, training_record=training_record, kind="trained")


def register_baseline(registry_root: Path, *, target: str, weights: Path,
                      config: Path, architecture: Path, dataset_manifest: Path, owner: str) -> dict:
    """Import existing unmeasured artifacts without claiming training or improvement."""
    return _register(registry_root, target=target, weights=weights, config=config,
                     architecture=architecture, dataset_manifest=dataset_manifest,
                     parent_id=None, owner=owner,
                     training_record={"status": "EXISTING_UNMEASURED"}, kind="baseline")


def _policy_definition(policy):
    _keys(policy, {"target", "dataset_sha256", "benchmark_sha256", "metrics"}, "policy")
    _target(policy["target"])
    _identifier(policy["dataset_sha256"])
    _identifier(policy["benchmark_sha256"])
    _require(type(policy["metrics"]) is dict and policy["metrics"], "empty policy metrics")
    segmentation = False
    for name, rule in policy["metrics"].items():
        _text(name, "metric name")
        _keys(rule, {"kind", "direction", "min_delta", "max_regression"}, "metric rule")
        _require(rule["kind"] in ("segmentation", "qc"), "metric kind")
        _require(rule["direction"] in ("min", "max"), "metric direction")
        normalized = re.sub(r"[^a-z0-9]", "", name.lower())
        if normalized in {"qc", "qcscore", "qualitycontrolscore"}:
            _require(rule["kind"] == "qc", "QC is not a segmentation metric")
        for field in ("min_delta", "max_regression"):
            _require(_number(rule[field]) >= 0, "negative policy tolerance")
        segmentation |= rule["kind"] == "segmentation"
    _require(segmentation, "independent segmentation metrics required")


def _policy(record):
    _check_record(record)
    _keys(record, {"id", "schema", "approver", "definition"}, "approved policy")
    _require(record["schema"] == "segmentation-policy-v1", "policy schema")
    _keys(record["approver"], {"actor", "role"}, "approver")
    _text(record["approver"]["actor"], "approver actor")
    _require(record["approver"]["role"] == "policy_approver", "approver role")
    _policy_definition(record["definition"])


def approve_policy(registry_root: Path, definition: dict, *, approver: str) -> dict:
    """CONTROL ONLY: archive externally approved policy, after authenticating approver."""
    definition = json.loads(_canonical(definition))
    _policy_definition(definition)
    _text(approver, "approver")
    policy = _record({"schema": "segmentation-policy-v1", "definition": definition,
                      "approver": {"actor": approver, "role": "policy_approver"}})
    root = _root(registry_root)
    with _locked(root):
        _store(root, "policies", policy)
    return policy


def _assess(candidate, comparison, policy):
    _candidate_shape(candidate)
    _require(candidate["kind"] == "trained" and candidate["parent_id"] is not None,
             "trained candidate and measured parent comparison required")
    _policy(policy)
    rules = policy["definition"]
    _require(policy["approver"]["actor"].casefold() != candidate["owner"].casefold(),
             "owner cannot approve policy")
    _keys(comparison, {"candidate_id", "parent_id", "target", "candidate_artifact_hashes",
                       "parent_artifact_hashes", "dataset_sha256", "benchmark_sha256",
                       "validation_test_patient_disjoint", "training_evaluation_patient_disjoint",
                       "evaluator", "measurement", "metrics"}, "comparison")
    for field in ("candidate_id", "parent_id", "target", "dataset_sha256"):
        expected = candidate["id"] if field == "candidate_id" else candidate[field]
        _require(comparison[field] == expected, "comparison binding mismatch: " + field)
    _require(rules["target"] == candidate["target"] and
             rules["dataset_sha256"] == candidate["dataset_sha256"] and
             rules["benchmark_sha256"] == comparison["benchmark_sha256"], "policy binding mismatch")
    for field, expected in (("candidate_artifact_hashes", _artifact_hashes(candidate)),
                             ("parent_artifact_hashes", candidate["parent_artifact_hashes"])):
        _hashes(comparison[field])
        _require(comparison[field] == expected, "comparison binding mismatch: " + field)
    for field in ("validation_test_patient_disjoint", "training_evaluation_patient_disjoint"):
        _require(comparison[field] is True, "patient disjointness not established: " + field)
    evaluator = comparison["evaluator"]
    _keys(evaluator, {"actor", "role", "status"}, "evaluator")
    actor = _text(evaluator["actor"], "evaluator actor")
    _require(actor.casefold() != candidate["owner"].casefold() and
             evaluator["role"] == "independent_evaluator" and
             evaluator["role"] != candidate["training_record"]["owner_role"], "evaluator not independent")
    _require(evaluator["status"] == "SUCCEEDED", "evaluation did not succeed")
    measurement = comparison["measurement"]
    _keys(measurement, {"source", "self_reported", "candidate_predictions_sha256",
                        "parent_predictions_sha256", "reference_sha256"}, "measurement")
    _require(measurement["source"] == "predictions" and measurement["self_reported"] is False,
             "independent prediction measurement required")
    for field in ("candidate_predictions_sha256", "parent_predictions_sha256", "reference_sha256"):
        _identifier(measurement[field])
    _keys(comparison["metrics"], {candidate["target"]}, "metric targets")
    measured = comparison["metrics"][candidate["target"]]
    _keys(measured, rules["metrics"], "metric names")
    deltas, improved, reasons = {}, [], []
    for name, rule in rules["metrics"].items():
        metric = measured[name]
        _keys(metric, {"candidate", "parent", "candidate_count", "parent_count"}, "measurement values")
        for field in ("candidate_count", "parent_count"):
            _require(type(metric[field]) is int and metric[field] > 0, "empty/invalid denominator")
        _require(metric["candidate_count"] == metric["parent_count"], "unpaired denominators")
        delta = _number(metric["candidate"]) - _number(metric["parent"])
        if rule["direction"] == "min":
            delta = -delta
        _number(delta)
        deltas[name] = delta
        if delta < -rule["max_regression"]:
            reasons.append("regression:" + name)
        if rule["kind"] == "segmentation" and delta > 0 and delta >= rule["min_delta"]:
            improved.append(name)
    if not improved:
        reasons.append("no_positive_segmentation_improvement")
    return {"passed": not reasons, "reasons": reasons, "deltas": deltas, "improved_metrics": improved,
            "candidate_id": candidate["id"], "parent_id": candidate["parent_id"], "policy_id": policy["id"]}


def assess_candidate(candidate: dict, comparison: dict, policy: dict) -> dict:
    """Pure fail-closed preview. Policy is the envelope returned by approve_policy."""
    try:
        _canonical(comparison)
        return _assess(candidate, comparison, policy)
    except (RegistryError, KeyError, TypeError, OverflowError) as exc:
        return {"passed": False, "reasons": [str(exc)], "deltas": {}, "improved_metrics": []}


def record_assessment(registry_root: Path, candidate_id: str, comparison: dict, *, policy_id: str) -> dict:
    """CONTROL ONLY: ingest a receipt obtained from the authenticated evaluator.

    Never expose this as an owner-submit-and-approve endpoint. Failed assessments
    are also retained. The returned id is the only authorization handle activate
    uses; all mutable verdict fields supplied by its caller are ignored.
    """
    root = _root(registry_root)
    comparison = json.loads(_canonical(comparison))
    with _locked(root):
        candidate = _read_candidate(root, candidate_id)
        policy = _read_record(root, "policies", policy_id)
        result = assess_candidate(candidate, comparison, policy)
        evidence = _record({"schema": "segmentation-assessment-v1", "candidate_id": candidate_id,
                            "policy_id": policy_id, "comparison": comparison, "result": result})
        _store(root, "assessments", evidence)
        return dict(result, assessment_id=evidence["id"])


def _pointer(root, target):
    path = _path(root, "active", _target(target) + ".json")
    if not path.exists():
        return None
    pointer = _load(path)
    _keys(pointer, {"event_id"}, "pointer")
    event = _read_record(root, "history", pointer["event_id"])
    _require(event["target"] == target, "pointer target mismatch")
    _identifier(event["candidate_id"])
    return event


def _select(root, candidate, previous, action, assessment_id=None):
    event = _record({"schema": "segmentation-selection-v1", "target": candidate["target"],
                     "candidate_id": candidate["id"], "previous_event_id": previous["id"] if previous else None,
                     "previous_candidate_id": previous["candidate_id"] if previous else None,
                     "action": action, "assessment_id": assessment_id, "nonce": uuid.uuid4().hex})
    _store(root, "history", event)
    folder = _path(root, "active")
    folder.mkdir(exist_ok=True)
    _atomic_json(_path(root, "active", candidate["target"] + ".json"),
                 {"event_id": event["id"]}, immutable=False)
    return event


def select_baseline(registry_root: Path, candidate_id: str) -> dict:
    """CONTROL ONLY: explicitly initialize an empty target with an unmeasured baseline."""
    root = _root(registry_root)
    with _locked(root):
        candidate = _read_candidate(root, candidate_id)
        _require(candidate["kind"] == "baseline", "only existing unmeasured baseline allowed")
        if _pointer(root, candidate["target"]) is not None:
            raise ConcurrentUpdate("baseline initialization requires an empty selector")
        return _select(root, candidate, None, "BASELINE_UNMEASURED")


def activate(registry_root: Path, candidate_id: str, assessment: dict,
             expected_parent_id: str | None) -> dict:
    """CONTROL ONLY: explicit atomic CAS; revalidate stored evidence, never a passed flag."""
    root = _root(registry_root)
    _identifier(candidate_id)
    if expected_parent_id is not None:
        _identifier(expected_parent_id)
    _require(type(assessment) is dict, "stored assessment handle required")
    assessment_id = _identifier(assessment.get("assessment_id"))
    with _locked(root):
        candidate = _read_candidate(root, candidate_id)
        previous = _pointer(root, candidate["target"])
        current_id = previous["candidate_id"] if previous else None
        if current_id != expected_parent_id or candidate["parent_id"] != expected_parent_id:
            raise ConcurrentUpdate("active parent changed or candidate has another parent")
        _require(current_id is not None, "select an existing baseline before promotion")
        parent = _read_candidate(root, current_id)
        _require(parent["target"] == candidate["target"] and
                 _artifact_hashes(parent) == candidate["parent_artifact_hashes"], "parent lineage mismatch")
        evidence = _read_record(root, "assessments", assessment_id)
        _keys(evidence, {"id", "schema", "candidate_id", "policy_id", "comparison", "result"}, "evidence")
        _require(evidence["schema"] == "segmentation-assessment-v1" and
                 evidence["candidate_id"] == candidate_id, "assessment candidate mismatch")
        policy = _read_record(root, "policies", evidence["policy_id"])
        result = assess_candidate(candidate, evidence["comparison"], policy)
        _require(result == evidence["result"] and result["passed"] is True, "assessment failed revalidation")
        return _select(root, candidate, previous, "ACTIVATE", assessment_id)


def rollback(registry_root: Path, target: str, *, expected_active_id: str) -> dict:
    """CONTROL ONLY: CAS restore the preceding selection, retaining the complete history."""
    root = _root(registry_root)
    _identifier(expected_active_id)
    with _locked(root):
        current = _pointer(root, target)
        if current is None or current["candidate_id"] != expected_active_id:
            raise ConcurrentUpdate("active candidate changed")
        _require(current["previous_event_id"] is not None, "no previous selection")
        previous = _read_record(root, "history", current["previous_event_id"])
        _require(previous["target"] == target and
                 previous["candidate_id"] == current["previous_candidate_id"], "rollback history mismatch")
        candidate = _read_candidate(root, previous["candidate_id"])
        _require(candidate["target"] == target, "rollback target mismatch")
        return _select(root, candidate, current, "ROLLBACK")


def resolve_candidate(registry_root: Path, candidate_id: str) -> dict:
    """Read verified immutable artifacts without changing the active model."""
    root = _root(registry_root)
    candidate = _read_candidate(root, candidate_id)
    return {"candidate_id": candidate["id"], "target": candidate["target"],
            "kind": candidate["kind"], "parent_id": candidate["parent_id"],
            "dataset_sha256": candidate["dataset_sha256"],
            "training_record": candidate["training_record"],
            "artifacts": {
                name: dict(artifact, path=str(_path(root, "candidates", candidate["id"], name)))
                for name, artifact in candidate["artifacts"].items()}}


def describe_active(registry_root: Path, target: str) -> dict | None:
    """Small display projection; model loading still uses resolve_active.

    Verify pointer/metadata identities without hashing large checkpoints on every
    dashboard refresh. This describes selection, not an artifact health check.
    """
    root = _root(registry_root)
    event = _pointer(root, target)
    if event is None:
        return None
    candidate = _load(_path(root, "candidates", event["candidate_id"], "candidate.json"))
    _candidate_shape(candidate)
    _require(candidate["id"] == event["candidate_id"] and candidate["target"] == target,
             "selected metadata mismatch")
    return {"candidate_id": candidate["id"], "target": target, "kind": candidate["kind"],
            "dataset_sha256": candidate["dataset_sha256"]}


def resolve_active(registry_root: Path, target: str) -> dict | None:
    """Return verified immutable artifact references for one atomic pointer snapshot.

    None means no selection. Corruption raises RegistryError; it never silently
    falls back. Consumers must retain/read these immutable paths, not reread the
    selector between loading config and weights. Selection may change afterwards.
    """
    root = _root(registry_root)
    event = _pointer(root, target)
    if event is None:
        return None
    candidate = resolve_candidate(root, event["candidate_id"])
    _require(candidate["target"] == target, "selected candidate target mismatch")
    return dict(candidate, selection=event)
