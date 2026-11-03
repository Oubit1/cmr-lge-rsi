from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.database import Database
from app.state_machine import RunState


class DatabaseTests(unittest.TestCase):
    def test_run_state_and_invalid_transition(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            dataset.mkdir()
            db = Database(root / "workflow.sqlite")
            db.initialize()
            dataset_id = db.register_dataset("demo", dataset)
            run_id = db.create_run(dataset_id, "audit", {})
            db.transition(run_id, RunState.QUEUED)
            db.transition(run_id, RunState.RUNNING)
            db.transition(run_id, RunState.QC_FAILED)
            self.assertEqual(db.get_run(run_id)["status"], "QC_FAILED")
            with self.assertRaises(ValueError):
                db.transition(run_id, RunState.PROMOTED)

    def test_retry_resets_run_timestamps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            dataset.mkdir()
            db = Database(root / "workflow.sqlite")
            db.initialize()
            dataset_id = db.register_dataset("demo", dataset)
            run_id = db.create_run(dataset_id, "audit", {})
            db.transition(run_id, RunState.QUEUED)
            db.transition(run_id, RunState.RUNNING)
            db.transition(run_id, RunState.ERROR)
            self.assertIsNotNone(db.get_run(run_id)["finished_at"])

            db.transition(run_id, RunState.QUEUED)
            queued = db.get_run(run_id)
            self.assertIsNone(queued["started_at"])
            self.assertIsNone(queued["finished_at"])

            self.assertTrue(db.claim_run(run_id))
            running = db.get_run(run_id)
            self.assertIsNotNone(running["started_at"])
            self.assertIsNone(running["finished_at"])

    def test_cancelled_and_superseded_are_terminal_not_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            dataset.mkdir()
            db = Database(root / "workflow.sqlite")
            db.initialize()
            dataset_id = db.register_dataset("demo", dataset)

            cancelled = db.create_run(dataset_id, "audit", {})
            db.transition(cancelled, RunState.QUEUED)
            db.transition(cancelled, RunState.CANCELLED)
            self.assertEqual(db.get_run(cancelled)["status"], "CANCELLED")
            self.assertIsNotNone(db.get_run(cancelled)["finished_at"])

            superseded = db.create_run(dataset_id, "audit", {})
            db.transition(superseded, RunState.QUEUED)
            db.transition(superseded, RunState.RUNNING)
            db.transition(superseded, RunState.SUPERSEDED)
            self.assertEqual(db.get_run(superseded)["status"], "SUPERSEDED")
            self.assertIsNotNone(db.get_run(superseded)["finished_at"])

    def test_existing_evidence_can_enter_failure_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            dataset.mkdir()
            db = Database(root / "workflow.sqlite")
            db.initialize()
            dataset_id = db.register_dataset("demo", dataset)
            run_id = db.create_run(dataset_id, "analysis", {})
            db.transition(run_id, RunState.EVIDENCE_READY)
            db.transition(run_id, RunState.FAILURE_ANALYSIS)
            self.assertEqual(db.get_run(run_id)["status"], "FAILURE_ANALYSIS")

    def test_agent_messages_form_a_persistent_role_addressed_inbox(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            dataset.mkdir()
            db = Database(root / "workflow.sqlite")
            db.initialize()
            dataset_id = db.register_dataset("demo", dataset)
            run_id = db.create_run(dataset_id, "analysis", {})
            message_id = db.post_agent_message(
                run_id, thread_id=f"run-{run_id}:A1", sender_role="supervisor",
                recipient_role="rule_curator", message_type="assigned_action",
                payload={"action_id": "A1"},
            )
            inbox = db.agent_inbox(run_id, "rule_curator", thread_id=f"run-{run_id}:A1")
            self.assertEqual(inbox[0]["message_id"], message_id)
            self.assertEqual(inbox[0]["sender_role"], "supervisor")
            self.assertEqual(inbox[0]["payload"], {"action_id": "A1"})

    def test_agent_invocation_preserves_input_and_actual_model_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            dataset.mkdir()
            db = Database(root / "workflow.sqlite")
            db.initialize()
            run_id = db.create_run(db.register_dataset("demo", dataset), "analysis", {})
            db.begin_agent_invocation(
                "trace-1", run_id=run_id, thread_id=f"run-{run_id}:analysis",
                role="cmr_specialist", provider="aliyun", requested_model="specialist",
                input_hash="abc", input_data={"case_id": "case_deadbeef", "_secret": "drop"},
            )
            db.record_agent_invocation_event("trace-1", {
                "attempt": 1, "model": "fallback", "status": "SUCCEEDED",
                "fallback_reason": "health fallback", "request_id": "req-1",
                "latency_ms": 12, "prompt_tokens": 3, "completion_tokens": 2,
                "total_tokens": 5,
            })
            db.finish_agent_invocation("trace-1", status="SUCCEEDED")
            details = db.run_details(run_id)
            trace = details["agent_invocations"][0]
            self.assertEqual(json.loads(trace["input_json"]), {"case_id": "case_deadbeef"})
            self.assertEqual(details["agent_invocation_events"][0]["actual_model"], "fallback")
            self.assertEqual(db.agent_invocation_event_count("trace-1"), 1)

    def test_same_dataset_runs_are_serialized(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            dataset.mkdir()
            db = Database(root / "workflow.sqlite")
            db.initialize()
            dataset_id = db.register_dataset("demo", dataset)
            first = db.create_run(dataset_id, "audit", {})
            second = db.create_run(dataset_id, "apply", {})
            db.transition(first, RunState.QUEUED)
            db.transition(second, RunState.QUEUED)

            self.assertTrue(db.claim_run(first))
            self.assertIsNone(db.next_queued_run_id())
            self.assertFalse(db.claim_run(second))

            db.transition(first, RunState.QC_FAILED)
            self.assertEqual(db.next_queued_run_id(), second)
            self.assertTrue(db.claim_run(second))

    def test_fail_open_tasks_closes_dangling_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            dataset.mkdir()
            db = Database(root / "workflow.sqlite")
            db.initialize()
            dataset_id = db.register_dataset("dataset", dataset)
            run_id = db.create_run(dataset_id, "audit", {})
            task_id = db.create_task(run_id, "qc", ["python", "qc.py"], root / "qc.log")
            db.update_task(task_id, "RUNNING")
            db.mark_task_process(task_id, 12345)
            db.touch_task(task_id)

            running = db.run_details(run_id)["tasks"][0]
            self.assertEqual(running["pid"], 12345)
            self.assertIsNotNone(running["heartbeat_at"])

            self.assertEqual(db.fail_open_tasks(run_id), 1)
            details = db.run_details(run_id)
            self.assertEqual(details["tasks"][0]["status"], "FAILED")
            self.assertEqual(details["tasks"][0]["exit_code"], -15)
            self.assertEqual(db.fail_open_tasks(run_id), 0)


if __name__ == "__main__":
    unittest.main()
