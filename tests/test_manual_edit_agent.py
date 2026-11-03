from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from scripts.analyze_manual_edit_queue import (
    deterministic_prompt_plan,
    enforce_prompt_policy,
    process_one,
    recover_stale_claims,
)


class ManualEditAgentTests(unittest.TestCase):
    def test_deterministic_router_handles_regional_adjacent_gap(self) -> None:
        plan = deterministic_prompt_plan({
            "target": "myo", "neighbors": [{"offset": -1}, {"offset": 1}],
            "operator_instruction": "结合前后相邻帧，只补右下局部缺口，不要整体膨胀",
        })
        self.assertIsNotNone(plan)
        self.assertEqual(plan["operations"], [{
            "operation": "fill_from_adjacent_consensus", "radius": 3,
            "region": "right_lower",
        }])

    def test_prompt_policy_rejects_cmr_largest_component_shortcut(self) -> None:
        request = {
            "modality": "cmr", "target": "myo",
            "current": {"components": 2}, "neighbors": [{}, {}],
            "allowed_operations": ["remove_components_below_area"],
            "qc_context": {"forbidden_operations": ["keep_largest_component"],
                           "hard_constraints": ["preserve two dominant components"]},
        }
        plan = enforce_prompt_policy(request, {
            "decision": "apply_operations", "target": "myo",
            "operations": [{"operation": "keep_largest_component"}],
            "rationale": "错误捷径", "expected_effect": "单成分", "risk_flags": [],
            "confidence": 0.9,
        })
        self.assertEqual(plan["decision"], "manual_required")
        self.assertEqual(plan["operations"], [])
        self.assertIn("operation_not_allowed:keep_largest_component", plan["risk_flags"])

    def test_prompt_policy_requires_bilateral_support_for_consensus_fill(self) -> None:
        request = {
            "modality": "cmr", "target": "myo",
            "current": {"components": 1}, "neighbors": [{}],
            "allowed_operations": ["fill_from_adjacent_consensus"],
            "qc_context": {"forbidden_operations": [], "hard_constraints": []},
        }
        plan = enforce_prompt_policy(request, {
            "decision": "apply_operations", "target": "myo",
            "operations": [{"operation": "fill_from_adjacent_consensus", "radius": 3}],
            "rationale": "补缺口", "expected_effect": "补全", "risk_flags": [],
            "confidence": 0.8,
        })
        self.assertEqual(plan["decision"], "manual_required")
        self.assertIn("bilateral_neighbor_support_missing", plan["risk_flags"])

    def test_stale_running_claim_is_recovered_on_worker_startup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "work"
            job = work / "prompt_jobs/job-1"
            job.mkdir(parents=True)
            running = job / "agent_request.running"
            running.write_text(json.dumps({"job_id": "job-1"}), encoding="utf-8")
            (job / "local_request.json").write_text(
                json.dumps({"job_id": "job-1", "status": "PROCESSING"}), encoding="utf-8"
            )
            self.assertEqual(recover_stale_claims(work, stale_seconds=0), 1)
            self.assertTrue((job / "agent_request.json").is_file())
            local = json.loads((job / "local_request.json").read_text(encoding="utf-8"))
            self.assertEqual(local["status"], "QUEUED")
            self.assertEqual(local["recovery"], "stale_claim")

    def test_deidentified_queue_is_consumed_to_structured_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work = root / "work"
            edit = work / "edits/edit-1"
            edit.mkdir(parents=True)
            request = {
                "schema_version": "manual-edit-v1",
                "edit_id": "edit-1",
                "case_hash": "0123456789abcdef",
                "modality": "cmr",
                "target": "myo",
                "operation": "add",
                "suggested_pattern": "fill_internal_gap",
                "before": {"holes": 2},
                "after": {"holes": 1},
            }
            (edit / "agent_request.json").write_text(json.dumps(request), encoding="utf-8")
            providers = root / "providers.yaml"
            providers.write_text("providers:\n  mock:\n    adapter: mock\n", encoding="utf-8")
            agents = root / "agents.yaml"
            agents.write_text(
                "agents:\n  cmr_specialist:\n    provider: mock\n    model: mock\n",
                encoding="utf-8",
            )
            self.assertTrue(process_one(work, providers, agents))
            output = json.loads((edit / "agent_analysis.json").read_text(encoding="utf-8"))
            self.assertEqual(output["role"], "cmr_specialist")
            self.assertEqual(output["analysis"]["likely_edit_intent"], "fill_internal_gap")
            serialized = (work / "analyzed_training_corpus.jsonl").read_text(encoding="utf-8")
            self.assertNotIn("patient", serialized.lower())

    def test_prompt_job_uses_specialist_and_creates_visual_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            work = root / "work"
            patient, volume = "PRIVATE_CASE", "1.nii.gz"
            import csv
            import nibabel as nib
            import numpy as np
            for relative in ["images/cmr", "labels/cmr_checked/cmr_Myo_mask"]:
                (dataset / relative / patient).mkdir(parents=True, exist_ok=True)
            (dataset / "labels/check/manifest_backup").mkdir(parents=True)
            image = np.zeros((16, 16, 3), dtype=np.float32)
            mask = np.zeros_like(image, dtype=np.uint8)
            mask[3:13, 3:13, :] = 1
            mask[6:10, 6:10, :] = 0
            nib.save(nib.Nifti1Image(image, np.eye(4)), str(dataset / "images/cmr" / patient / volume))
            nib.save(nib.Nifti1Image(mask, np.eye(4)), str(dataset / "labels/cmr_checked/cmr_Myo_mask" / patient / volume))
            with (dataset / "labels/check/manifest_backup/cmr_middle_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["patient", "volume", "slice_index", "issues"])
                writer.writeheader(); writer.writerow({"patient": patient, "volume": volume, "slice_index": 2, "issues": "gap"})
            from app.manual_annotation import frame_metadata, queue_prompt_repair
            meta = frame_metadata(dataset, "cmr_middle_manifest", 0)
            queued = queue_prompt_repair(dataset, work, {
                "pool": "cmr_middle_manifest", "index": 0, "target": "myo",
                "prompt": "补全小缺口", "source_hash": meta["source_hashes"]["myo"],
            })
            providers = root / "providers.yaml"
            providers.write_text("providers:\n  mock:\n    adapter: mock\n", encoding="utf-8")
            agents = root / "agents.yaml"
            agents.write_text("agents:\n  cmr_specialist:\n    provider: mock\n    model: mock\n", encoding="utf-8")
            # Exercise two actual processes, matching the deployed --watch and
            # explicit --job-id entrypoints. No model credentials are needed.
            project=Path(__file__).resolve().parents[1]
            job = work / "prompt_jobs" / queued["job_id"]

            def diagnostics():
                paths = [root / "watch.log", root / "explicit.log"] + [
                    job / name for name in ("local_request.json", "agent_error.json",
                                             "agent_plan.json", "agent_request.json", "claim.json")]
                return json.dumps({str(path.relative_to(root)): path.read_text(
                    encoding="utf-8", errors="replace") if path.exists() else "<missing>"
                    for path in paths}, ensure_ascii=False, indent=2)

            command=[sys.executable,"-u",str(project/"scripts/analyze_manual_edit_queue.py"),
                     "--work-root",str(work),"--providers",str(providers),
                     "--agents",str(project/"configs/agents.yaml")]
            with (root/"watch.log").open("w") as watch_log,(root/"explicit.log").open("w") as explicit_log:
                watch=subprocess.Popen(command+["--watch","--poll-seconds","1"],cwd=project,stdout=watch_log,stderr=subprocess.STDOUT)
                try:
                    explicit=subprocess.Popen(command+["--job-id",queued["job_id"]],cwd=project,stdout=explicit_log,stderr=subprocess.STDOUT)
                    self.assertEqual(explicit.wait(timeout=30),0,diagnostics())
                    self.assertIsNone(watch.poll(),diagnostics())
                    claim=json.loads((work/"prompt_jobs"/queued["job_id"]/"claim.json").read_text())
                    self.assertEqual(claim["pid"],explicit.pid)
                finally:
                    watch.terminate()
                    try:watch.wait(timeout=5)
                    except subprocess.TimeoutExpired:watch.kill();watch.wait(timeout=5)
            self.assertTrue((job / "candidate.png").is_file(), diagnostics())
            plan = json.loads((job / "agent_plan.json").read_text())
            self.assertGreaterEqual(plan["planning_latency_ms"], 0)
            self.assertEqual(json.loads((job / "local_request.json").read_text())["status"], "READY")
            original=json.loads((job/"agent_request.json").read_text())
            self.assertEqual(original["operator_instruction"],"补全小缺口")
            self.assertFalse(process_one(work,providers,agents,job_id=queued["job_id"]))
            refined=queue_prompt_repair(dataset,work,{"pool":"cmr_middle_manifest","index":0,"target":"myo",
                "source_hash":meta["source_hashes"]["myo"],"mode":"refine","previous_job_id":queued["job_id"]})
            refined_input=json.loads((work/"prompt_jobs"/refined["job_id"]/"agent_request.json").read_text())
            self.assertEqual(refined_input["operator_instruction"],original["operator_instruction"])

    @unittest.skipUnless(sys.platform.startswith("linux"), "claim identity uses Linux /proc")
    def test_watch_recovery_waits_for_claim_publication(self) -> None:
        """Hold a real child between rename and claim publication during recovery."""
        import fcntl
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work = root / "work"
            job_id = "20261028T010101_prompt_012345abcd"
            job = work / "prompt_jobs" / job_id
            job.mkdir(parents=True)
            request = {"modality": "cmr", "target": "myo", "operator_instruction": "synthetic prompt"}
            (job / "agent_request.json").write_text(json.dumps(request), encoding="utf-8")
            (job / "local_request.json").write_text(json.dumps({
                "job_id": job_id, "status": "QUEUED", "execution_scope": "interactive_prompt"}), encoding="utf-8")
            # The executable name preserves the existing /proc worker identity contract.
            bootstrap = root / "analyze_manual_edit_queue.py"
            bootstrap.write_text('''
import json, sys, time
from pathlib import Path
mode, root, project, job_id = sys.argv[1:]
root = Path(root)
sys.path.insert(0, project)
from app import prompt_dispatch
from scripts.analyze_manual_edit_queue import claim_one, recover_stale_claims
def wait(name):
    deadline = time.monotonic() + 15
    while not (root / name).exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(name)
        time.sleep(.01)
if mode == "claim":
    original = prompt_dispatch.register_claim
    def delayed(folder):
        (root / "claim-window").touch()
        wait("publish")
        original(folder)
    prompt_dispatch.register_claim = delayed
    claim_one(root / "work", job_id)
    (root / "claimed").touch()
    wait("finish")
else:
    (root / "recovery-started").touch()
    result = recover_stale_claims(root / "work", stale_seconds=0)
    (root / "recovered.json").write_text(json.dumps(result))
''', encoding="utf-8")
            project = Path(__file__).resolve().parents[1]
            children = []

            def wait(name):
                deadline = time.monotonic() + 15
                while not (root / name).exists():
                    self.assertLess(time.monotonic(), deadline, "child marker missing: " + name)
                    time.sleep(.01)

            with (root / "children.log").open("w", encoding="utf-8") as log:
                try:
                    for mode, marker in (("claim", "claim-window"), ("recover", "recovery-started")):
                        children.append(subprocess.Popen([
                            sys.executable, "-B", str(bootstrap), mode, str(root), str(project), job_id],
                            cwd=project, stdout=log, stderr=subprocess.STDOUT))
                        wait(marker)
                    self.assertTrue((job / "agent_request.running").exists())
                    self.assertFalse((job / "claim.json").exists())
                    with (job / "dispatch.lock").open("a") as handle:
                        with self.assertRaises(BlockingIOError, msg="rename/publication must hold the job lock"):
                            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    (root / "publish").touch()
                    wait("claimed")
                    self.assertEqual(children[1].wait(timeout=15), 0)
                    self.assertEqual(json.loads((root / "recovered.json").read_text()), 0)
                    self.assertEqual(json.loads((job / "agent_request.running").read_text()), request)
                    self.assertFalse((job / "agent_request.json").exists())
                    self.assertEqual(json.loads((job / "claim.json").read_text())["pid"], children[0].pid)
                    (root / "finish").touch()
                    self.assertEqual(children[0].wait(timeout=15), 0)
                except Exception as exc:
                    log.flush()
                    state = {name: (root / name).read_text() if (root / name).exists() else "<missing>"
                             for name in ("recovered.json", "claimed")}
                    raise AssertionError(f"{exc}\n{state}\n" + (
                        root / "children.log").read_text(errors="replace")) from exc
                finally:
                    (root / "publish").touch()
                    (root / "finish").touch()
                    for child in children:
                        if child.poll() is None:
                            child.terminate()
                        try:
                            child.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            child.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
