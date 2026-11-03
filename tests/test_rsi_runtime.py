import json
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from app.control_checkpoint import create_checkpoint, digest, recover_rollback, restore_release
from app.database import Database
from app.rsi_kernel import canonical_hash
from app.rsi_runtime import (assess_action, freeze_parent, inheritance,
                             paired_evaluation, reconcile_promotion,
                             score_predictions, write_record)


class PairedEvolutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "project"
        self.workspace = Path(self.temp.name) / "candidate"
        self.root.mkdir()
        write_record(self.root / "configs/rsi.json", {"mode": "observe"})
        code = self.workspace / "orchestrator/scripts"
        code.mkdir(parents=True)
        (code / "__init__.py").touch()
        self.module = code / "solver.py"
        self.module.write_text("def run_evaluation(data):\n    return {'predictions': {'a': 'ok', 'b': None}}\n")
        freeze_parent(self.workspace, self.root)
        data = {"tasks": [{"id": "a"}, {"id": "b"}],
                "references": {"a": "ok", "b": "ok"}}
        write_record(self.root / "runtime/rsi/benchmarks/test/input.json", data)
        self.contract = {
            "schema_version": "rsi-benchmark-v1", "benchmark_id": "synthetic-v1",
            "module": "scripts.solver", "scorer": "reference_predictions_v1",
            "input_path": "runtime/rsi/benchmarks/test/input.json", "input_sha256": canonical_hash(data),
            "scope": "synthetic_control_test",
            "metrics": {"correct": {"direction": "higher", "report_field": "correct"},
                        "wrong": {"direction": "lower", "report_field": "wrong"}},
        }
        write_record(self.root / "configs/rsi_benchmarks/lge_myo.json", self.contract)
        self.db = Database(self.root / "runtime/workflow.sqlite")
        self.db.initialize()
        dataset = self.db.register_dataset("test", self.root / "synthetic")
        self.run = self.db.create_run(dataset, "analysis", {})
        self.action = {"action_id": "test_a", "workflow_stage": "lge_myo", "owner_role": "qc_code_optimizer"}

    def assess(self):
        return assess_action(self.db, self.root, self.workspace, run_id=self.run,
                             action=self.action, changed_files=["scripts/solver.py"])

    def test_real_subprocess_pair_and_inheritance(self):
        self.module.write_text("def run_evaluation(data):\n    assert 'references' not in data\n    return {'predictions': {'a': 'ok', 'b': 'ok'}, 'accuracy': 999}\n")
        result = self.assess()
        self.assertEqual("PROMOTE", result["decision"])
        self.assertEqual(1, result["baseline_metrics"]["correct"])
        self.assertEqual(2, result["candidate_metrics"]["correct"])
        self.assertFalse(result["blocks_promotion"])
        memory = inheritance(self.db, stage="lge_myo")
        self.assertEqual("PROMOTE", memory["variants"][0]["decision"])
        self.assertTrue(memory["ready"])
        self.assertFalse(inheritance(self.db, stage="cmr_paired")["ready"])

    def test_no_gain_and_regression_retain_parent(self):
        self.assertEqual("RETAIN_PARENT", self.assess()["decision"])
        self.module.write_text("def run_evaluation(data):\n    return {'predictions': {'a': 'wrong', 'b': 'ok'}}\n")
        result = self.assess()
        self.assertEqual("RETAIN_PARENT", result["decision"])
        self.assertIn("wrong", result["regressed_metrics"])

    def test_enforce_blocks_no_gain(self):
        write_record(self.root / "configs/rsi.json", {"mode": "enforce"})
        self.assertTrue(self.assess()["blocks_promotion"])

    def test_changed_benchmark_is_detected(self):
        write_record(self.root / self.contract["input_path"], {"references": {"a": "other"}})
        result = self.assess()
        self.assertEqual("NEEDS_EVIDENCE", result["decision"])
        self.assertIn("benchmark_input_changed", result["reasons"][0])

    def test_missing_contract_is_recorded_without_fabricated_score(self):
        (self.root / "configs/rsi_benchmarks/lge_myo.json").unlink()
        result = self.assess()
        self.assertEqual("NEEDS_EVIDENCE", result["decision"])
        self.assertEqual({}, result["baseline_metrics"])
        self.assertEqual("NEEDS_EVIDENCE", inheritance(self.db)["variants"][0]["decision"])

    def test_solver_cannot_supply_own_score_or_shrink_denominator(self):
        scored = score_predictions({"references": {"a": "ok", "b": "ok"}},
                                   {"predictions": {"a": "ok"}, "accuracy": 1.0})
        self.assertEqual(0.5, scored["accuracy"])
        self.assertEqual(1, scored["abstained"])
        with self.assertRaises(ValueError):
            score_predictions({"references": {"a": "ok"}}, {"predictions": {"x": "ok"}})

    def test_private_references_never_written_to_solver_input(self):
        paired_evaluation(self.root, self.workspace, "lge_myo")
        data = json.loads((self.workspace / "rsi_input.json").read_text())
        self.assertNotIn("references", data)

    def test_parent_tampering_before_validation_is_detected(self):
        parent = self.workspace / "rsi_parent/orchestrator/scripts/solver.py"
        parent.write_text("def run_evaluation(data):\n    return {'predictions': {}}\n")
        result = self.assess()
        self.assertEqual("NEEDS_EVIDENCE", result["decision"])
        self.assertIn("frozen_parent_modified", result["reasons"][0])

    def test_shared_input_file_tampering_cannot_change_candidate_input(self):
        parent = self.workspace / "rsi_parent/orchestrator/scripts/solver.py"
        # This behavior is part of the frozen baseline, not a post-freeze edit.
        source = ("def run_evaluation(data):\n"
                  "    from pathlib import Path\n"
                  "    Path('../../../rsi_input.json').write_text('{}')\n"
                  "    return {'predictions': {'a': 'ok'}}\n")
        parent.write_text(source)
        # Recreate the fixture freeze to pin this baseline legitimately.
        from app.rsi_runtime import parent_record_path
        pin = parent_record_path(self.workspace, self.root)
        pinned = json.loads(pin.read_text())
        from app.rsi_runtime import tree_digest
        pinned["parent_digest"] = tree_digest(self.workspace / "rsi_parent")
        write_record(pin, pinned)
        self.module.write_text("def run_evaluation(data):\n    assert len(data['tasks']) == 2\n    return {'predictions': {'a': 'ok', 'b': 'ok'}}\n")
        self.assertEqual("PROMOTE", self.assess()["decision"])

    def test_reassessment_does_not_select_a_new_parent(self):
        first = self.assess()
        from app.rsi_runtime import tree_digest
        digest_value = tree_digest(self.workspace / "rsi_parent")
        self.db.register_evolution_variant(
            variant_id="other_promoted", parent_variant_id=None, source_run_id=None,
            action_id=None, workflow_stage="lge_myo", code_commit=None,
            continuity_revision=None, artifact_path=None, status="PROMOTED",
            metadata={"tree_digest": digest_value})
        again = self.assess()
        self.assertEqual(first["parent_variant_id"], again["parent_variant_id"])
        self.assertEqual(first["child_variant_id"], again["child_variant_id"])

    def test_role_inheritance_filters_before_limit(self):
        self.assess()
        for i in range(10):
            self.db.register_evolution_variant(
                variant_id=f"other{i}", parent_variant_id=None, source_run_id=None,
                action_id=None, workflow_stage="lge_myo", code_commit=None,
                continuity_revision=None, artifact_path=None, status="CANDIDATE", metadata={})
            self.db.register_evolution_variant(
                variant_id=f"child{i}", parent_variant_id=f"other{i}", source_run_id=None,
                action_id=None, workflow_stage="lge_myo", code_commit=None,
                continuity_revision=None, artifact_path=None, metadata={"owner_role": "rule_curator"})
        own = inheritance(self.db, role="qc_code_optimizer")
        self.assertTrue(own["ready"])
        self.assertEqual(1, len(own["variants"]))
        self.assertFalse(inheritance(self.db, role="not_used")["ready"])

    def test_action_and_variant_commit_atomically_and_reconcile_after_crash(self):
        selection = self.assess()
        self.db.save_maintenance_action(self.run, {
            **self.action, "action_class": "isolated_code_experiment"}, status="SUCCEEDED")
        with self.assertRaises(KeyError):
            self.db.update_maintenance_action(
                self.run, "test_a", status="PROMOTED", evolution_variant_id="not_a_variant")
        self.assertEqual("SUCCEEDED", self.db.get_maintenance_action(self.run, "test_a")["status"])
        artifact = self.root / "runtime/promotions/run_test/action_test"
        installed = self.root / "scripts/solver.py"
        installed.parent.mkdir(parents=True)
        installed.write_text("installed source")
        write_record(artifact / "promotion_manifest.json", {
            "status": "PROMOTED", "promoted": [{"owner": "orchestrator", "relative": "scripts/solver.py",
                                                   "new_hash": digest(installed)}]})
        write_record(self.workspace / "rsi_promotion_journal.json", {
            "run_id": self.run, "action_id": "test_a", "artifact": str(artifact), "status": "PREPARED"})
        write_record(self.workspace / "autopilot_validation.json", {"passed": True, "promotion_scope": "production_path"})
        recovery = reconcile_promotion(self.db, self.root, self.root / "unused_skill", self.workspace,
                                       run_id=self.run, action_id="test_a")
        self.assertEqual("PROMOTED", recovery["status"])
        self.assertEqual("PROMOTED", self.db.get_maintenance_action(self.run, "test_a")["status"])
        variants = {row["variant_id"]: row for row in self.db.list_evolution_variants()}
        variant = variants[selection["child_variant_id"]]
        self.assertEqual("PROMOTED", variant["status"])
        self.assertEqual(str(artifact / "promotion_manifest.json"), variant["rollback_manifest_path"])

    def test_incomplete_filesystem_promotion_does_not_resume_execution(self):
        self.assess()
        write_record(self.workspace / "rsi_promotion_journal.json", {
            "run_id": self.run, "action_id": "test_a", "artifact": str(self.root / "runtime/promotions/missing")})
        recovery = reconcile_promotion(self.db, self.root, self.root / "unused_skill", self.workspace,
                                       run_id=self.run, action_id="test_a")
        self.assertEqual("RECOVERY_REQUIRED", recovery["status"])
        (self.workspace / "rsi_selection.json").unlink()
        recovery = reconcile_promotion(self.db, self.root, self.root / "unused_skill", self.workspace,
                                       run_id=self.run, action_id="test_a")
        self.assertEqual("RECOVERY_REQUIRED", recovery["status"])


class CheckpointTests(unittest.TestCase):
    def test_rollback_rehearsal_and_concurrent_edit_protection(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            root = base / "project"
            root.mkdir()
            subprocess.run(["git", "init", str(root)], check=True, capture_output=True)
            subprocess.run(["git", "-C", str(root), "-c", "user.name=test", "-c", "user.email=test@test",
                            "commit", "--allow-empty", "-m", "baseline"], check=True, capture_output=True)
            (root / "app").mkdir()
            old = root / "app/old.py"
            old.write_text("old")
            db = Database(root / "runtime/workflow.sqlite")
            db.initialize()
            db.register_dataset("before", root / "synthetic")
            checkpoint = base / "checkpoint"
            result = create_checkpoint(root, checkpoint)
            self.assertTrue(result["database_saved"])
            self.assertNotIn("runtime/workflow.sqlite", json.loads((checkpoint / "manifest.json").read_text())["files"])
            old.write_text("new")
            added = root / "app/new.py"
            added.write_text("new module")
            installed = {"app/old.py": digest(old), "app/new.py": digest(added)}
            subprocess.run(["git", "-C", str(root), "add", "app/old.py", "app/new.py"],
                           check=True, capture_output=True)
            subprocess.run(["git", "-C", str(root), "-c", "user.name=test", "-c", "user.email=test@test",
                            "commit", "-m", "release"], check=True, capture_output=True)
            from scripts.rsi_control import record_release
            record_release(root, checkpoint, "HEAD", root / "runtime/release.json")
            self.assertEqual(installed, json.loads((root / "runtime/release.json").read_text())["installed"])
            db.register_dataset("after", root / "synthetic_after")
            self.assertEqual("PREVIEW", restore_release(root, checkpoint, installed)["status"])
            old.write_text("concurrent user work")
            self.assertEqual("CONFLICT", restore_release(root, checkpoint, installed, execute=True)["status"])
            self.assertTrue(added.exists())
            old.write_text("new")
            self.assertEqual("RESTORED", restore_release(root, checkpoint, installed, execute=True)["status"])
            self.assertEqual("old", old.read_text())
            self.assertFalse(added.exists())
            with db.connect() as con:
                self.assertEqual(2, con.execute("SELECT count(*) FROM datasets").fetchone()[0])

    def test_deleted_file_restoration_and_io_failure_compensation(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            root, checkpoint = base / "repo", base / "checkpoint"
            (root / "app").mkdir(parents=True)
            (checkpoint / "source/app").mkdir(parents=True)
            first, second = root / "app/a.py", root / "app/b.py"
            for target in (first, second):
                target.write_text("old")
                shutil.copy2(target, checkpoint / "source/app" / target.name)
            write_record(checkpoint / "manifest.json", {"root": str(root), "files": {
                "app/a.py": digest(first), "app/b.py": digest(second)}})
            first.unlink()
            second.write_text("new")
            installed = {"app/a.py": None, "app/b.py": digest(second)}
            import os
            original = os.replace
            def fail_second(source, destination):
                if Path(destination) == second:
                    raise OSError("injected second-file failure")
                return original(source, destination)
            with patch("app.control_checkpoint.os.replace", side_effect=fail_second):
                with self.assertRaises(OSError):
                    restore_release(root, checkpoint, installed, execute=True)
            self.assertFalse(first.exists())
            self.assertEqual("new", second.read_text())
            self.assertEqual("RESTORED", restore_release(root, checkpoint, installed, execute=True)["status"])
            self.assertEqual("old", first.read_text())
            self.assertEqual("old", second.read_text())

    def test_interrupted_rollback_can_be_replayed(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            root, checkpoint = base / "repo", base / "checkpoint"
            (root / "app").mkdir(parents=True)
            (checkpoint / "source/app").mkdir(parents=True)
            first, second = root / "app/a.py", root / "app/b.py"
            files = {}
            for target in (first, second):
                target.write_text("old")
                shutil.copy2(target, checkpoint / "source/app" / target.name)
                files["app/" + target.name] = digest(target)
                target.write_text("new")
            write_record(checkpoint / "manifest.json", {"root": str(root), "files": files})
            installed = {"app/a.py": digest(first), "app/b.py": digest(second)}
            import os
            original = os.replace
            def interrupt_second(source, destination):
                if Path(destination) == second:
                    raise KeyboardInterrupt("simulate process interruption")
                return original(source, destination)
            with patch("app.control_checkpoint.os.replace", side_effect=interrupt_second):
                with self.assertRaises(KeyboardInterrupt):
                    restore_release(root, checkpoint, installed, execute=True)
            self.assertEqual("old", first.read_text())
            self.assertEqual("new", second.read_text())
            journal = next(checkpoint.glob("rollback_*/journal.json"))
            self.assertEqual("PREVIEW", recover_rollback(root, journal)["status"])
            self.assertEqual("RESTORED", recover_rollback(root, journal, execute=True)["status"])
            self.assertEqual("old", second.read_text())


if __name__ == "__main__":
    unittest.main()
