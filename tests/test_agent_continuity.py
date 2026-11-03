from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from agents.evidence import build_evidence_package
from app.agent_continuity import build_agent_continuity
from app.database import Database
from app.existing_qc import snapshot_existing_qc


class AgentContinuityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.db = Database(self.root / "workflow.sqlite")
        self.db.initialize()
        self.dataset = self.root / "dataset"
        manifest = self.dataset / "labels/check/manifest_backup/lge_issue_manifest.csv"
        manifest.parent.mkdir(parents=True)
        manifest.write_text("patient,modality,issues\nP1,lge,myo_topology\n", encoding="utf-8")
        self.dataset_id = self.db.register_dataset("demo", self.dataset)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _run(self) -> int:
        return self.db.create_run(self.dataset_id, "analysis", {"dataset_root": str(self.dataset)})

    def test_capsule_survives_model_switch_and_quantifies_trend(self) -> None:
        runs = [self._run(), self._run()]
        models = ["qwen-old", "qwen-new"]
        for index, (run_id, model) in enumerate(zip(runs, models), 1):
            invocation = f"inv-{index}"
            self.db.begin_agent_invocation(
                invocation, run_id=run_id, thread_id=f"run-{run_id}:analysis",
                role="lge_myo_specialist", provider="aliyun",
                requested_model=model, input_hash=f"hash-{index}", input_data={"safe": True},
            )
            self.db.record_agent_invocation_event(invocation, {
                "attempt": 1, "model": model, "status": "SUCCEEDED",
                "latency_ms": 100 * index, "total_tokens": 10 * index,
            })
            self.db.finish_agent_invocation(invocation, status="SUCCEEDED")
        for value in (0.5, 0.5, 0.5, 0.8, 0.8, 0.8):
            self.db.save_agent_scorecard(runs[-1], "lge_myo_specialist", {
                "composite_score": value,
            })
        self.db.post_agent_message(
            runs[-1], thread_id="run-2:analysis", sender_role="failure_triage",
            recipient_role="lge_myo_specialist", message_type="failure_analysis",
            payload={"bounded": True},
        )
        history = [{"run_id": runs[-1], "actions": [{
            "action_id": "A1", "owner_role": "qc_code_optimizer",
            "action_class": "isolated_code_experiment", "status": "PROMOTED",
            "workflow_stage": "lge_myo", "objective": "build stage consumer",
        }]}]

        capsule = build_agent_continuity(self.db, iteration_history=history)
        role = capsule["roles"]["lge_myo_specialist"]
        self.assertEqual(role["current_model"], "qwen-new")
        self.assertEqual(role["model_history"], ["qwen-old", "qwen-new"])
        self.assertEqual(role["model_switches"], 1)
        self.assertEqual(role["successful_invocations"], 2)
        self.assertEqual(role["score"]["delta"], 0.3)
        self.assertEqual(role["handoffs_received"], 1)
        self.assertTrue(capsule["inheritance_ready"])
        self.assertEqual(capsule["recent_progress"][0]["action_id"], "A1")

    def test_latest_event_ignores_retry_history_and_keeps_calls_without_events(self) -> None:
        run_id = self._run()
        for name in ("retry", "no-event"):
            self.db.begin_agent_invocation(
                name, run_id=run_id, thread_id="analysis", role="lge_myo_specialist",
                provider="aliyun", requested_model="requested", input_hash=name,
                input_data={"safe": True},
            )
        self.db.record_agent_invocation_event("retry", {
            "attempt": 1, "model": "failed-model", "status": "FAILED",
            "total_tokens": 999, "latency_ms": 999,
        })
        self.db.record_agent_invocation_event("retry", {
            "attempt": 2, "model": "successful-model", "status": "SUCCEEDED",
            "total_tokens": 17, "latency_ms": 123,
        })
        self.db.finish_agent_invocation("retry", status="SUCCEEDED")
        self.db.finish_agent_invocation("no-event", status="FAILED")
        role = build_agent_continuity(self.db)["roles"]["lge_myo_specialist"]
        self.assertEqual(role["invocations"], 2)
        self.assertEqual(role["successful_invocations"], 1)
        self.assertEqual(role["failed_invocations"], 1)
        self.assertEqual(role["total_tokens"], 17)
        self.assertEqual(role["average_latency_ms"], 123)
        self.assertNotIn("failed-model", role["model_history"])

    def test_snapshot_and_evidence_deliver_same_capsule_to_replacement_model(self) -> None:
        continuity = build_agent_continuity(self.db, iteration_history=[{
            "run_id": 1, "actions": [{
                "action_id": "A1", "owner_role": "qc_code_optimizer",
                "action_class": "rule_draft", "status": "PROMOTED",
            }],
        }])
        snap = snapshot_existing_qc(
            self.db, self.root, self.dataset,
            iteration_history=[], role_continuity=continuity,
        )
        details = self.db.run_details(snap["analysis_run_id"])
        old_salt = os.environ.get("CMR_LGE_CASE_SALT")
        os.environ["CMR_LGE_CASE_SALT"] = "test-salt"
        try:
            evidence = build_evidence_package(details)
        finally:
            if old_salt is None:
                os.environ.pop("CMR_LGE_CASE_SALT", None)
            else:
                os.environ["CMR_LGE_CASE_SALT"] = old_salt
        self.assertEqual(evidence["role_continuity"]["revision"], continuity["revision"])
        self.assertEqual(
            evidence["role_continuity"]["inheritance_contract"],
            continuity["inheritance_contract"],
        )


if __name__ == "__main__":
    unittest.main()
