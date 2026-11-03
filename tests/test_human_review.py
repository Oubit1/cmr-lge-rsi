from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.database import Database
from scripts.board_server import submit_review


class HumanReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        dataset = root / "dataset"
        dataset.mkdir()
        self.db_path = root / "workflow.sqlite"
        self.db = Database(self.db_path)
        self.db.initialize()
        dataset_id = self.db.register_dataset("demo", dataset)
        self.run_id = self.db.create_run(dataset_id, "analysis", {})
        self.action = {
            "action_id": "A1-human-review-test",
            "owner_role": "cmr_specialist",
            "action_class": "isolated_code_experiment",
            "objective": "test only",
            "acceptance_metrics": ["no label writes"],
            "max_attempts": 3,
            "auto_execute": True,
        }
        self.db.save_maintenance_action(self.run_id, self.action, status="HUMAN_REQUIRED")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_defer_records_note_without_state_change(self) -> None:
        result = submit_review({
            "run_id": self.run_id,
            "action_id": self.action["action_id"],
            "decision": "defer",
            "reviewer": "doctor-a",
            "note": "等待补充证据",
            "confirmed": False,
        }, self.db_path)
        self.assertEqual(result["resulting_status"], "HUMAN_REQUIRED")
        self.assertEqual(
            self.db.get_maintenance_action(self.run_id, self.action["action_id"])["status"],
            "HUMAN_REQUIRED",
        )
        reviews = self.db.list_human_action_reviews(self.run_id, self.action["action_id"])
        self.assertEqual(len(reviews), 1)
        self.assertEqual(reviews[0]["note"], "等待补充证据")

    def test_retry_grants_exactly_one_additional_attempt(self) -> None:
        result = submit_review({
            "run_id": self.run_id,
            "action_id": self.action["action_id"],
            "decision": "retry_once",
            "reviewer": "doctor-a",
            "note": "允许按新证据再试一次",
            "confirmed": True,
        }, self.db_path)
        self.assertEqual(result["resulting_status"], "APPROVED")
        action = self.db.get_maintenance_action(self.run_id, self.action["action_id"])
        self.assertEqual(action["payload"]["max_attempts"], 4)
        self.assertEqual(len(action["payload"]["human_retry_history"]), 1)

    def test_operator_evidence_permit_is_consumed_exactly_once(self) -> None:
        self.db.review_maintenance_action(
            self.run_id, self.action["action_id"], decision="retry_once",
            reviewer="doctor-a", note="人工示例已经应用", confirmed=True,
        )
        marker = self.db.attach_operator_evidence_retry(
            self.run_id, self.action["action_id"],
            edit_id="edit-1", pool="cmr_middle_manifest",
        )
        self.assertFalse(marker["consumed"])
        consumed = self.db.consume_operator_evidence_retry(
            self.run_id, self.action["action_id"]
        )
        self.assertTrue(consumed["consumed"])
        with self.assertRaisesRegex(ValueError, "no unconsumed"):
            self.db.consume_operator_evidence_retry(
                self.run_id, self.action["action_id"]
            )

    def test_state_change_requires_explicit_confirmation(self) -> None:
        with self.assertRaisesRegex(ValueError, "confirmed=true"):
            submit_review({
                "run_id": self.run_id,
                "action_id": self.action["action_id"],
                "decision": "reject",
                "reviewer": "doctor-a",
                "note": "证据不足，拒绝",
                "confirmed": False,
            }, self.db_path)
        self.assertEqual(
            self.db.get_maintenance_action(self.run_id, self.action["action_id"])["status"],
            "HUMAN_REQUIRED",
        )

    def test_resolve_and_reject_are_audited_terminal_dispositions(self) -> None:
        result = self.db.review_maintenance_action(
            self.run_id,
            self.action["action_id"],
            decision="resolve_external",
            reviewer="doctor-a",
            note="外部验证已经完成",
            confirmed=True,
        )
        self.assertEqual(result["resulting_status"], "VALIDATED")
        with self.assertRaisesRegex(ValueError, "requires HUMAN_REQUIRED"):
            self.db.review_maintenance_action(
                self.run_id,
                self.action["action_id"],
                decision="reject",
                reviewer="doctor-a",
                note="不可重复处置",
                confirmed=True,
            )


if __name__ == "__main__":
    unittest.main()
