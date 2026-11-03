"""Offline framework tests. All artifacts and metric values are synthetic."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import multiprocessing
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app import segmentation_registry as registry


def sha(data):
    return hashlib.sha256(data).hexdigest()


def activation_worker(root, candidate_id, assessment, parent_id, ready, start, results):
    """Separate interpreters exercise OS locks, not just an in-process mutex."""
    ready.put(True)
    if not start.wait(20):
        results.put("timeout")
        return
    try:
        registry.activate(Path(root), candidate_id, assessment, parent_id)
        results.put("activated")
    except registry.ConcurrentUpdate:
        results.put("conflict")
    except Exception as exc:
        results.put(type(exc).__name__ + ": " + str(exc))


def lock_worker(root, ready):
    with registry._locked(Path(root)):
        ready.put(True)
        # The parent terminates this process to verify crash lock release.
        multiprocessing.Event().wait(30)


class SegmentationRegistryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base / "registry"
        self.sources = self.base / "sources"
        self.sources.mkdir()
        self.files = {}
        for name in registry.ARTIFACTS:
            path = self.sources / (name + ".synthetic")
            path.write_bytes(("synthetic-" + name).encode())
            self.files[name] = path
        self.target = "cmr_myo"
        self.baseline = registry.register_baseline(
            self.root, target=self.target, owner="existing-custodian", **self.files)
        self.candidate = self.make_candidate()
        self.definition = {
            "target": self.target,
            "dataset_sha256": self.candidate["dataset_sha256"],
            "benchmark_sha256": sha(b"synthetic-frozen-benchmark"),
            "metrics": {
                "synthetic_overlap": {"kind": "segmentation", "direction": "max",
                                      "min_delta": 0.125, "max_regression": 0.0},
                "synthetic_error": {"kind": "segmentation", "direction": "min",
                                    "min_delta": 0.125, "max_regression": 0.125},
                "qc_score": {"kind": "qc", "direction": "max",
                             "min_delta": 0.0, "max_regression": 0.0},
            },
        }
        self.policy = registry.approve_policy(self.root, self.definition, approver="policy-board")
        self.comparison = self.make_comparison(self.candidate)

    def receipt(self, attempt="attempt-1"):
        return {"status": "SUCCEEDED", "training_source": "training",
                "owner_role": "candidate_owner", "attempt_id": attempt,
                "metrics": {"synthetic_attempt_measure": 2.0},
                "artifact_hashes": {name: sha(path.read_bytes()) for name, path in self.files.items()},
                "provider": "replaceable-provider-v1"}

    def make_candidate(self, attempt="attempt-1", **overrides):
        receipt = overrides.pop("training_record") if "training_record" in overrides else self.receipt(attempt)
        args = dict(target=self.target, owner="model-owner", parent_id=self.baseline["id"],
                    training_record=receipt, **self.files)
        args.update(overrides)
        return registry.register_candidate(self.root, **args)

    def test_dashboard_description_does_not_rehash_checkpoint_bytes(self):
        registry.select_baseline(self.root, self.baseline["id"])
        with mock.patch.object(registry, "_sha_file", side_effect=AssertionError("large file read")):
            result = registry.describe_active(self.root, self.target)
        self.assertEqual(result["candidate_id"], self.baseline["id"])
        self.assertEqual(result["kind"], "baseline")

    def make_comparison(self, candidate):
        return {"candidate_id": candidate["id"], "parent_id": candidate["parent_id"],
                "target": candidate["target"],
                "candidate_artifact_hashes": registry._artifact_hashes(candidate),
                "parent_artifact_hashes": candidate["parent_artifact_hashes"],
                "dataset_sha256": candidate["dataset_sha256"],
                "benchmark_sha256": self.definition["benchmark_sha256"],
                "validation_test_patient_disjoint": True,
                "training_evaluation_patient_disjoint": True,
                "evaluator": {"actor": "evaluation-service", "role": "independent_evaluator",
                              "status": "SUCCEEDED"},
                "measurement": {"source": "predictions", "self_reported": False,
                                "candidate_predictions_sha256": sha(b"synthetic-candidate-predictions"),
                                "parent_predictions_sha256": sha(b"synthetic-parent-predictions"),
                                "reference_sha256": sha(b"synthetic-reference")},
                "metrics": {candidate["target"]: {
                    "synthetic_overlap": {"candidate": 0.75, "parent": 0.5,
                                          "candidate_count": 4, "parent_count": 4},
                    "synthetic_error": {"candidate": 0.25, "parent": 0.5,
                                        "candidate_count": 4, "parent_count": 4},
                    "qc_score": {"candidate": 1.0, "parent": 1.0,
                                 "candidate_count": 4, "parent_count": 4},
                }}}

    def store(self, comparison=None, candidate=None, policy=None):
        return registry.record_assessment(self.root, (candidate or self.candidate)["id"],
                                          comparison or self.comparison,
                                          policy_id=(policy or self.policy)["id"])

    def select_and_activate(self):
        registry.select_baseline(self.root, self.baseline["id"])
        return registry.activate(self.root, self.candidate["id"], self.store(), self.baseline["id"])

    def assert_rejected(self, comparison=None, candidate=None, policy=None):
        result = registry.assess_candidate(candidate or self.candidate,
                                          comparison or self.comparison, policy or self.policy)
        self.assertIs(result["passed"], False)
        self.assertTrue(result["reasons"])

    def corrupt(self, path, data):
        path.chmod(stat.S_IREAD | stat.S_IWRITE)
        path.write_bytes(data)

    def test_registration_is_content_addressed_detached_and_not_selected(self):
        self.assertEqual(self.make_candidate(), self.candidate)
        expected_id = sha(json.dumps({k: v for k, v in self.candidate.items() if k != "id"},
                                    sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=True, allow_nan=False).encode())
        self.assertEqual(expected_id, self.candidate["id"])
        self.assertIsNone(registry.resolve_active(self.root, self.target))
        record = self.receipt("detached")
        candidate = self.make_candidate(training_record=record)
        record["metrics"]["synthetic_attempt_measure"] = 100
        self.assertEqual(candidate["training_record"]["metrics"]["synthetic_attempt_measure"], 2)
        self.assertEqual(registry._read_candidate(self.root, candidate["id"]), candidate)

    def test_source_and_provider_replacement_do_not_rewrite_history(self):
        original = registry._read_candidate(self.root, self.candidate["id"])
        self.files["weights"].write_bytes(b"replacement-synthetic-weights")
        receipt = self.receipt("replacement-attempt")
        receipt["provider"] = "replacement-provider"
        child = self.make_candidate(parent_id=self.candidate["id"], training_record=receipt)
        self.assertNotEqual(child["id"], original["id"])
        self.assertEqual(child["parent_id"], original["id"])
        self.assertEqual(child["parent_artifact_hashes"], registry._artifact_hashes(original))
        self.assertEqual(registry._read_candidate(self.root, original["id"]), original)

    def test_dataset_owner_attempt_and_parent_are_part_of_identity(self):
        variants = [self.make_candidate(owner="another-owner"), self.make_candidate("attempt-2"),
                    self.make_candidate(parent_id=self.candidate["id"])]
        self.files["dataset_manifest"].write_bytes(b"different-synthetic-dataset-digest")
        variants.append(self.make_candidate())
        self.assertEqual(len({self.candidate["id"]} | {c["id"] for c in variants}), 5)

    def test_large_and_small_files_stream_without_read_bytes(self):
        self.files["weights"].write_bytes(b"x" * (registry.CHUNK_SIZE * 3 + 17))
        receipt = self.receipt()
        reads = []
        real_open = Path.open

        class Reader:
            def __init__(self, handle):
                self.handle = handle

            def __enter__(self):
                self.handle.__enter__()
                return self

            def __exit__(self, *args):
                return self.handle.__exit__(*args)

            def fileno(self):
                return self.handle.fileno()

            def read(self, size=-1):
                reads.append(size)
                if not 0 < size <= registry.CHUNK_SIZE:
                    raise AssertionError("unbounded file read")
                return self.handle.read(size)

        def open_file(path, mode="r", *args, **kwargs):
            handle = real_open(path, mode, *args, **kwargs)
            return Reader(handle) if mode == "rb" else handle

        with mock.patch.object(Path, "read_bytes", side_effect=AssertionError("read_bytes forbidden")), \
                mock.patch.object(Path, "open", open_file):
            candidate = self.make_candidate(training_record=receipt)
            registry._read_candidate(self.root, candidate["id"])
        self.assertGreater(len(reads), 10)
        self.assertEqual(candidate["artifacts"]["weights"]["sha256"], receipt["artifact_hashes"]["weights"])

    def test_changed_source_is_rejected_and_stage_removed(self):
        real_copy = registry._copy

        def changing_copy(source, destination):
            if destination.name == "weights":
                with mock.patch.object(registry, "_sha_file", return_value="0" * 64):
                    return real_copy(source, destination)
            return real_copy(source, destination)

        with mock.patch.object(registry, "_copy", side_effect=changing_copy):
            with self.assertRaisesRegex(registry.RegistryError, "source changed"):
                self.make_candidate("unstable")
        self.assertFalse(list(self.root.glob(".candidate-*")))

    def test_training_receipt_must_be_completed_real_training(self):
        changes = [("status", None), ("status", False), ("status", "RUNNING"),
                   ("training_source", "evaluation_only"), ("training_source", "evaluation-only"),
                   ("training_source", ""), ("attempt_id", ""),
                   ("evaluation_only", True), ("evaluation_only", 0),
                   ("owner_role", "independent_evaluator"), ("metrics", {"x": False}),
                   ("metrics", {"x": float("nan")}), ("metrics", {"x": float("inf")})]
        for field, value in changes:
            with self.subTest(field=field, value=value):
                receipt = self.receipt()
                receipt[field] = value
                with self.assertRaises(registry.RegistryError):
                    self.make_candidate(training_record=receipt)
        for field in ("status", "training_source", "artifact_hashes", "metrics"):
            receipt = self.receipt()
            del receipt[field]
            with self.subTest(missing=field), self.assertRaises(registry.RegistryError):
                self.make_candidate(training_record=receipt)

    def test_all_training_artifacts_must_match_receipt(self):
        for name in registry.ARTIFACTS:
            receipt = self.receipt()
            receipt["artifact_hashes"][name] = "0" * 64
            with self.subTest(name=name), self.assertRaisesRegex(registry.RegistryError, "hash mismatch"):
                self.make_candidate(training_record=receipt)

    def test_targets_parent_target_and_parent_integrity(self):
        for target in registry.TARGETS:
            baseline = registry.register_baseline(self.root, target=target, owner="custodian", **self.files)
            candidate = self.make_candidate(target=target, parent_id=baseline["id"])
            self.assertEqual(candidate["target"], target)
        with self.assertRaises(registry.RegistryError):
            self.make_candidate(target="all")
        with self.assertRaisesRegex(registry.RegistryError, "parent target"):
            self.make_candidate(target="lge_mi")
        path = self.root / "candidates" / self.baseline["id"] / "architecture"
        self.corrupt(path, b"corrupt")
        with self.assertRaises(registry.RegistryError):
            self.make_candidate()

    def test_invalid_ids_and_paths_are_rejected(self):
        for value in ("../escape", "x/y", "C:\\escape", "A" * 64, "0" * 63, "0" * 65,
                      "0" * 64 + "\n", False, [], ""):
            with self.subTest(value=value):
                with self.assertRaises(registry.RegistryError):
                    self.make_candidate(parent_id=value)
                with self.assertRaises(registry.RegistryError):
                    registry.activate(self.root, value, {"passed": True}, None)
        with self.assertRaises(registry.RegistryError):
            registry.resolve_active(self.root, "../cmr_myo")
        with self.assertRaises(registry.RegistryError):
            registry._path(self.root, "../outside")

    def test_symlink_source_and_registry_escape_rejected(self):
        link = self.sources / "linked-weights"
        try:
            link.symlink_to(self.files["weights"])
        except (OSError, NotImplementedError) as exc:
            self.skipTest("OS disallows symlinks: " + str(exc))
        with self.assertRaises(registry.RegistryError):
            self.make_candidate(weights=link)
        outside = self.base / "outside"
        outside.mkdir()
        (self.root / "active").symlink_to(outside, target_is_directory=True)
        with self.assertRaises(registry.RegistryError):
            registry.resolve_active(self.root, self.target)
        self.assertFalse(list(outside.iterdir()))

    def test_baseline_is_unmeasured_and_requires_explicit_empty_selection(self):
        self.assertEqual(self.baseline["training_record"], {"status": "EXISTING_UNMEASURED"})
        self.assertIsNone(registry.resolve_active(self.root, self.target))
        registry.select_baseline(self.root, self.baseline["id"])
        resolved = registry.resolve_active(self.root, self.target)
        self.assertEqual(resolved["kind"], "baseline")
        self.assertEqual(resolved["selection"]["action"], "BASELINE_UNMEASURED")
        self.assert_rejected(candidate=self.baseline)
        with self.assertRaises(registry.ConcurrentUpdate):
            registry.select_baseline(self.root, self.baseline["id"])
        with self.assertRaises(registry.RegistryError):
            registry.select_baseline(self.root, self.candidate["id"])

    def test_unparented_training_can_be_registered_but_not_claim_improvement(self):
        candidate = self.make_candidate(parent_id=None)
        self.assert_rejected(candidate=candidate)
        with self.assertRaises(registry.RegistryError):
            registry.select_baseline(self.root, candidate["id"])

    def test_positive_direction_min_and_max(self):
        result = registry.assess_candidate(self.candidate, self.comparison, self.policy)
        self.assertIs(result["passed"], True)
        self.assertEqual(result["deltas"]["synthetic_overlap"], 0.25)
        self.assertEqual(result["deltas"]["synthetic_error"], 0.25)
        self.assertEqual(set(result["improved_metrics"]), {"synthetic_overlap", "synthetic_error"})

    def test_every_artifact_and_lineage_binding_is_checked(self):
        for field in ("candidate_id", "parent_id", "dataset_sha256", "benchmark_sha256", "target"):
            comparison = copy.deepcopy(self.comparison)
            comparison[field] = "lge_mi" if field == "target" else "0" * 64
            with self.subTest(field=field):
                self.assert_rejected(comparison)
        for field in ("candidate_artifact_hashes", "parent_artifact_hashes"):
            for artifact in registry.ARTIFACTS:
                comparison = copy.deepcopy(self.comparison)
                comparison[field][artifact] = "0" * 64
                with self.subTest(field=field, artifact=artifact):
                    self.assert_rejected(comparison)

    def test_evaluator_must_succeed_and_be_independent(self):
        for field, value in (("actor", "model-owner"), ("actor", "MODEL-OWNER"), ("actor", ""),
                             ("role", "candidate_owner"), ("role", "trainer"),
                             ("status", "FAILED"), ("status", False), ("status", True)):
            comparison = copy.deepcopy(self.comparison)
            comparison["evaluator"][field] = value
            with self.subTest(field=field, value=value):
                self.assert_rejected(comparison)

    def test_owner_cannot_approve_own_policy(self):
        policy = registry.approve_policy(self.root, self.definition, approver="MODEL-OWNER")
        self.assert_rejected(policy=policy)
        registry.select_baseline(self.root, self.baseline["id"])
        with self.assertRaises(registry.RegistryError):
            registry.activate(self.root, self.candidate["id"], self.store(policy=policy), self.baseline["id"])

    def test_disjoint_flags_and_prediction_measurement_are_literal(self):
        for field in ("validation_test_patient_disjoint", "training_evaluation_patient_disjoint"):
            for value in (False, 1, "true", None, float("nan")):
                comparison = copy.deepcopy(self.comparison)
                comparison[field] = value
                with self.subTest(field=field, value=value):
                    self.assert_rejected(comparison)
        for field, values in {"source": ("self_report", "qc", "training"),
                              "self_reported": (True, 0, "false", None),
                              "candidate_predictions_sha256": (None, ""),
                              "parent_predictions_sha256": ("bad",),
                              "reference_sha256": (False,)}.items():
            for value in values:
                comparison = copy.deepcopy(self.comparison)
                comparison["measurement"][field] = value
                with self.subTest(field=field, value=value):
                    self.assert_rejected(comparison)

    def test_nonfinite_and_boolean_metrics_and_empty_denominators_fail(self):
        for field in ("candidate", "parent"):
            for value in (True, False, float("nan"), float("inf"), -float("inf"), None, "0.8", 10 ** 400):
                comparison = copy.deepcopy(self.comparison)
                comparison["metrics"][self.target]["synthetic_overlap"][field] = value
                with self.subTest(field=field, value=value):
                    self.assert_rejected(comparison)
        for field in ("candidate_count", "parent_count"):
            for value in (0, -1, True, False, 4.0, "4", None, 3):
                comparison = copy.deepcopy(self.comparison)
                comparison["metrics"][self.target]["synthetic_overlap"][field] = value
                with self.subTest(field=field, value=value):
                    self.assert_rejected(comparison)
        comparison = copy.deepcopy(self.comparison)
        comparison["metrics"][self.target]["synthetic_overlap"].update(candidate=1e308, parent=-1e308)
        self.assert_rejected(comparison)

    def test_missing_extra_or_mismatched_metric_keys_fail(self):
        variants = []
        for key in self.comparison:
            comparison = copy.deepcopy(self.comparison)
            del comparison[key]
            variants.append(comparison)
        comparison = copy.deepcopy(self.comparison)
        comparison["metrics"]["lge_myo"] = comparison["metrics"].pop(self.target)
        variants.append(comparison)
        for operation in ("missing", "extra", "empty", "extra_value"):
            comparison = copy.deepcopy(self.comparison)
            metrics = comparison["metrics"][self.target]
            if operation == "missing":
                del metrics["synthetic_overlap"]
            elif operation == "extra":
                metrics["unapproved"] = metrics["synthetic_overlap"]
            elif operation == "empty":
                metrics.clear()
            else:
                metrics["synthetic_overlap"]["unapproved"] = 3
            variants.append(comparison)
        for index, comparison in enumerate(variants):
            with self.subTest(index=index):
                self.assert_rejected(comparison)

    def test_policy_requires_explicit_finite_rules_and_segmentation(self):
        for field in ("direction", "kind", "min_delta", "max_regression"):
            definition = copy.deepcopy(self.definition)
            del definition["metrics"]["synthetic_overlap"][field]
            with self.subTest(missing=field), self.assertRaises(registry.RegistryError):
                registry.approve_policy(self.root, definition, approver="board")
        for field, values in {"direction": ("higher", False), "kind": ("invented",),
                              "min_delta": (-1, False, float("nan")),
                              "max_regression": (-1, True, float("inf"))}.items():
            for value in values:
                definition = copy.deepcopy(self.definition)
                definition["metrics"]["synthetic_overlap"][field] = value
                with self.subTest(field=field, value=value), self.assertRaises(registry.RegistryError):
                    registry.approve_policy(self.root, definition, approver="board")
        definition = copy.deepcopy(self.definition)
        definition["metrics"] = {"qc_score": definition["metrics"]["qc_score"]}
        with self.assertRaises(registry.RegistryError):
            registry.approve_policy(self.root, definition, approver="board")
        definition["metrics"]["qc_score"]["kind"] = "segmentation"
        with self.assertRaises(registry.RegistryError):
            registry.approve_policy(self.root, definition, approver="board")

    def test_no_improvement_below_delta_and_qc_only_gain_fail(self):
        comparison = copy.deepcopy(self.comparison)
        metrics = comparison["metrics"][self.target]
        for metric in metrics.values():
            metric["candidate"] = metric["parent"]
        self.assert_rejected(comparison)
        metrics["qc_score"]["candidate"] = 100
        self.assert_rejected(comparison)
        metrics["synthetic_overlap"]["candidate"] += 0.0625
        self.assert_rejected(comparison)
        definition = copy.deepcopy(self.definition)
        definition["metrics"]["synthetic_overlap"]["min_delta"] = 0
        policy = registry.approve_policy(self.root, definition, approver="board")
        metrics["synthetic_overlap"]["candidate"] = metrics["synthetic_overlap"]["parent"]
        self.assert_rejected(comparison, policy=policy)

    def test_regression_limits_and_min_delta_boundary(self):
        comparison = copy.deepcopy(self.comparison)
        metrics = comparison["metrics"][self.target]
        metrics["synthetic_overlap"]["candidate"] = 0.625
        metrics["synthetic_error"]["candidate"] = 0.625
        self.assertTrue(registry.assess_candidate(self.candidate, comparison, self.policy)["passed"])
        metrics["synthetic_error"]["candidate"] = 0.75
        result = registry.assess_candidate(self.candidate, comparison, self.policy)
        self.assertIn("regression:synthetic_error", result["reasons"])

    def test_preview_is_not_authorization_and_assessment_does_not_activate(self):
        preview = registry.assess_candidate(self.candidate, self.comparison, self.policy)
        self.assertTrue(preview["passed"])
        registry.select_baseline(self.root, self.baseline["id"])
        for fake in ({"passed": True}, preview, {"passed": True, "assessment_id": "0" * 64}):
            with self.subTest(fake=fake), self.assertRaises(registry.RegistryError):
                registry.activate(self.root, self.candidate["id"], fake, self.baseline["id"])
        self.store()
        self.assertEqual(registry.resolve_active(self.root, self.target)["candidate_id"], self.baseline["id"])

    def test_caller_cannot_upgrade_failed_stored_assessment(self):
        comparison = copy.deepcopy(self.comparison)
        comparison["evaluator"]["status"] = "FAILED"
        assessment = self.store(comparison)
        self.assertFalse(assessment["passed"])
        assessment.update(passed=True, reasons=[])
        registry.select_baseline(self.root, self.baseline["id"])
        with self.assertRaisesRegex(registry.RegistryError, "revalidation"):
            registry.activate(self.root, self.candidate["id"], assessment, self.baseline["id"])

    def test_mutating_input_receipt_and_verdict_does_not_change_stored_evidence(self):
        assessment = self.store()
        self.comparison["evaluator"]["status"] = "FAILED"
        assessment["passed"] = False
        registry.select_baseline(self.root, self.baseline["id"])
        event = registry.activate(self.root, self.candidate["id"], assessment, self.baseline["id"])
        self.assertEqual(event["action"], "ACTIVATE")

    def test_assessment_cannot_be_reused_for_another_candidate(self):
        other = self.make_candidate("other-attempt")
        registry.select_baseline(self.root, self.baseline["id"])
        with self.assertRaisesRegex(registry.RegistryError, "assessment candidate mismatch"):
            registry.activate(self.root, other["id"], self.store(), self.baseline["id"])

    def test_missing_policy_is_not_replaced_by_caller_approval(self):
        with self.assertRaises(registry.RegistryError):
            registry.record_assessment(self.root, self.candidate["id"], self.comparison, policy_id="0" * 64)
        fake = dict(self.policy, approved=True)
        self.assert_rejected(policy=fake)

    def test_activation_rechecks_candidate_parent_evidence_and_policy(self):
        assessment = self.store()
        registry.select_baseline(self.root, self.baseline["id"])
        paths = [self.root / "candidates" / self.candidate["id"] / name for name in registry.ARTIFACTS]
        paths += [self.root / "candidates" / self.baseline["id"] / "weights",
                  self.root / "assessments" / (assessment["assessment_id"] + ".json"),
                  self.root / "policies" / (self.policy["id"] + ".json")]
        for path in paths:
            original = path.read_bytes()
            with self.subTest(path=path):
                self.corrupt(path, b"tampered")
                with self.assertRaises(registry.RegistryError):
                    registry.activate(self.root, self.candidate["id"], assessment, self.baseline["id"])
                self.corrupt(path, original)
        self.assertEqual(registry.resolve_active(self.root, self.target)["candidate_id"], self.baseline["id"])

    def test_activation_recomputes_even_content_addressed_false_verdict(self):
        comparison = copy.deepcopy(self.comparison)
        comparison["measurement"]["self_reported"] = True
        assessment = self.store(comparison)
        evidence = registry._read_record(self.root, "assessments", assessment["assessment_id"])
        del evidence["id"]
        evidence["result"]["passed"] = True
        evidence["result"]["reasons"] = []
        forged = registry._record(evidence)
        # Simulate a bad control-layer record; a hash-valid verdict still is not authority.
        with registry._locked(self.root):
            registry._store(self.root, "assessments", forged)
        registry.select_baseline(self.root, self.baseline["id"])
        with self.assertRaisesRegex(registry.RegistryError, "revalidation"):
            registry.activate(self.root, self.candidate["id"], {"assessment_id": forged["id"]}, self.baseline["id"])

    def test_activate_resolve_and_explicit_rollback(self):
        event = self.select_and_activate()
        active = registry.resolve_active(self.root, self.target)
        self.assertEqual(active["candidate_id"], self.candidate["id"])
        self.assertEqual(active["selection"], event)
        for name, reference in active["artifacts"].items():
            path = Path(reference["path"])
            self.assertTrue(path.is_absolute() and path.is_relative_to(self.root))
            self.assertEqual(sha(path.read_bytes()), reference["sha256"])
            self.assertEqual(path.name, name)
        rollback = registry.rollback(self.root, self.target, expected_active_id=self.candidate["id"])
        self.assertEqual(rollback["candidate_id"], self.baseline["id"])
        self.assertEqual(rollback["previous_event_id"], event["id"])
        self.assertEqual(registry.resolve_active(self.root, self.target)["candidate_id"], self.baseline["id"])
        self.assertTrue((self.root / "history" / (event["id"] + ".json")).is_file())

    def test_stale_parent_and_rollback_conflicts_preserve_pointer(self):
        assessment = self.store()
        registry.select_baseline(self.root, self.baseline["id"])
        pointer = self.root / "active" / (self.target + ".json")
        original = pointer.read_bytes()
        with self.assertRaises(registry.ConcurrentUpdate):
            registry.activate(self.root, self.candidate["id"], assessment, "0" * 64)
        self.assertEqual(pointer.read_bytes(), original)
        registry.activate(self.root, self.candidate["id"], assessment, self.baseline["id"])
        with self.assertRaises(registry.ConcurrentUpdate):
            registry.activate(self.root, self.candidate["id"], assessment, self.baseline["id"])
        with self.assertRaises(registry.ConcurrentUpdate):
            registry.rollback(self.root, self.target, expected_active_id=self.baseline["id"])

    def test_failed_atomic_swap_preserves_old_pointer_and_retry_works(self):
        assessment = self.store()
        registry.select_baseline(self.root, self.baseline["id"])
        pointer = self.root / "active" / (self.target + ".json")
        original = pointer.read_bytes()
        real_replace = os.replace

        def fail_pointer(source, destination):
            if Path(destination) == pointer:
                raise OSError("synthetic interrupted swap")
            return real_replace(source, destination)

        with mock.patch.object(registry.os, "replace", side_effect=fail_pointer):
            with self.assertRaises(OSError):
                registry.activate(self.root, self.candidate["id"], assessment, self.baseline["id"])
        self.assertEqual(pointer.read_bytes(), original)
        self.assertFalse(list(pointer.parent.glob(".pending-*")))
        registry.activate(self.root, self.candidate["id"], assessment, self.baseline["id"])
        self.assertEqual(registry.resolve_active(self.root, self.target)["candidate_id"], self.candidate["id"])

    def test_resolver_rejects_tampering_and_path_edits(self):
        self.select_and_activate()
        for name in registry.ARTIFACTS:
            path = self.root / "candidates" / self.candidate["id"] / name
            original = path.read_bytes()
            self.corrupt(path, b"bad")
            with self.subTest(name=name), self.assertRaises(registry.RegistryError):
                registry.resolve_active(self.root, self.target)
            self.corrupt(path, original)
        candidate = copy.deepcopy(self.candidate)
        candidate["artifacts"]["weights"]["path"] = "../../outside"
        candidate.pop("id")
        candidate = registry._record(candidate)
        with self.assertRaises(registry.RegistryError):
            registry._candidate_shape(candidate)

    def test_concurrent_process_activations_have_exactly_one_winner(self):
        registry.select_baseline(self.root, self.baseline["id"])
        other = self.make_candidate("concurrent-alternative")
        assessments = [self.store(), self.store(self.make_comparison(other), candidate=other)]
        context = multiprocessing.get_context("spawn")
        ready, results, start = context.Queue(), context.Queue(), context.Event()
        processes = [context.Process(target=activation_worker,
                                    args=(str(self.root), candidate["id"], assessment,
                                          self.baseline["id"], ready, start, results))
                     for candidate, assessment in zip((self.candidate, other), assessments)]
        try:
            for process in processes:
                process.start()
            for _ in processes:
                self.assertTrue(ready.get(timeout=20))
            start.set()
            outcomes = [results.get(timeout=20) for _ in processes]
            self.assertCountEqual(outcomes, ["activated", "conflict"])
            for process in processes:
                process.join(20)
                self.assertEqual(process.exitcode, 0)
            self.assertIn(registry.resolve_active(self.root, self.target)["candidate_id"],
                          {self.candidate["id"], other["id"]})
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join(10)
            ready.close()
            results.close()

    def test_writer_crash_releases_os_lock(self):
        context = multiprocessing.get_context("spawn")
        ready = context.Queue()
        process = context.Process(target=lock_worker, args=(str(self.root), ready))
        try:
            process.start()
            self.assertTrue(ready.get(timeout=20))
            with self.assertRaises(registry.ConcurrentUpdate):
                registry.select_baseline(self.root, self.baseline["id"])
        finally:
            if process.is_alive():
                process.terminate()
            process.join(10)
            ready.close()
        registry.select_baseline(self.root, self.baseline["id"])

    def test_duplicate_json_keys_and_nonfinite_records_rejected(self):
        path = self.root / "policies" / (self.policy["id"] + ".json")
        for data in (b'{"id": "x", "id": "y"}', b'{"number": NaN}'):
            self.corrupt(path, data)
            with self.assertRaises(registry.RegistryError):
                self.store()

    def test_python39_grammar_and_postponed_annotations(self):
        for path in (Path(registry.__file__), Path(__file__)):
            source = path.read_text(encoding="utf-8")
            ast.parse(source, feature_version=(3, 9))
            self.assertIn("from __future__ import annotations", source)


if __name__ == "__main__":
    unittest.main()
