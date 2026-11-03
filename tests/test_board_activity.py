from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.board_server import (PAGE, agent_activity, batch_progress, batch_summary,
                                  active_label_review_tasks, candidate_gate_detail,
                                  rejected_batch_candidates, workflow_outcome_summary,
                                  reviewer_validation_backlog,
                                  aliyun_billing_status, maintenance_chat_context,
                                  board_control_authorized, control_workflow,
                                  manual_learning_status, release_linked_manual_example,
                                  resume_autopilot,
                                  AutopilotControlError)
from app.database import Database


class BoardActivityTests(unittest.TestCase):
    def test_applied_linked_example_releases_human_action_once(self) -> None:
        db_path = self.root / "feedback.sqlite"
        database = Database(db_path)
        database.initialize()
        dataset_id = database.register_dataset("demo", self.root / "dataset")
        run_id = database.create_run(dataset_id, "analysis", {})
        action_id = "S841_cmr_small_area_guard_delta_scan_exec"
        database.save_maintenance_action(run_id, {
            "action_id": action_id,
            "owner_role": "qc_code_optimizer",
            "action_class": "isolated_code_experiment",
            "objective": "CMR small area example",
            "acceptance_metrics": ["bounded"],
            "max_attempts": 2,
            "auto_execute": True,
        }, status="HUMAN_REQUIRED")
        applied = {
            "edit_id": "edit-1", "pool": "cmr_middle_manifest",
            "source_run_id": run_id, "source_action_id": action_id,
        }
        first = release_linked_manual_example(
            applied, "operator", db_path=db_path
        )
        self.assertEqual(first["resulting_status"], "APPROVED")
        self.assertEqual(database.get_maintenance_action(run_id, action_id)["status"],
                         "APPROVED")
        marker = database.get_maintenance_action(run_id, action_id)["payload"][
            "operator_evidence_retry"
        ]
        self.assertEqual(marker["edit_id"], "edit-1")
        self.assertEqual(marker["pool"], "cmr_middle_manifest")
        self.assertFalse(marker["consumed"])
        second = release_linked_manual_example(
            applied, "operator", db_path=db_path
        )
        self.assertTrue(second["already_released"])
        self.assertEqual(len(database.list_human_action_reviews(run_id, action_id)), 1)

    def test_reviewer_backlog_keeps_unmet_metrics_after_code_promotion(self) -> None:
        db = sqlite3.connect(":memory:")
        db.row_factory = sqlite3.Row
        db.execute("CREATE TABLE maintenance_actions (run_id INTEGER, action_id TEXT, status TEXT)")
        db.execute("CREATE TABLE agent_messages (message_id INTEGER PRIMARY KEY, run_id INTEGER, "
                   "thread_id TEXT, sender_role TEXT, message_type TEXT, payload_json TEXT)")
        db.execute("INSERT INTO maintenance_actions VALUES (734,'A2','PROMOTED')")
        db.execute("INSERT INTO agent_messages VALUES (1,734,'run-734:A2',"
                   "'independent_reviewer','implementation_review',?)",
                   (json.dumps({"decision": "accept_experiment",
                                "required_validation_metrics": ["Held-out concordance"]}),))
        self.assertEqual(reviewer_validation_backlog(db), [{
            "run_id": 734, "action_id": "A2", "status": "PROMOTED",
            "review_decision": "accept_experiment",
            "required_validation_metrics": ["Held-out concordance"],
        }])
        db.close()

    def test_reviewer_backlog_includes_pre_action_rejection(self) -> None:
        db = sqlite3.connect(":memory:")
        db.row_factory = sqlite3.Row
        db.execute("CREATE TABLE maintenance_actions (run_id INTEGER, action_id TEXT, status TEXT)")
        db.execute("CREATE TABLE agent_messages (message_id INTEGER PRIMARY KEY, run_id INTEGER, "
                   "thread_id TEXT, sender_role TEXT, message_type TEXT, payload_json TEXT)")
        db.execute("INSERT INTO maintenance_actions VALUES (735,'A2','HUMAN_REQUIRED')")
        db.execute("INSERT INTO agent_messages VALUES (1,735,'run-735:A2',"
                   "'independent_reviewer','action_review',?)",
                   (json.dumps({"decision": "needs_human_review",
                                "required_validation_metrics": ["Held-out false-route rate"]}),))
        self.assertEqual(reviewer_validation_backlog(db), [{
            "run_id": 735, "action_id": "A2", "status": "HUMAN_REQUIRED",
            "review_decision": "needs_human_review",
            "required_validation_metrics": ["Held-out false-route rate"],
        }])
        db.close()

    def test_code_promotion_does_not_claim_label_resolution(self) -> None:
        result = workflow_outcome_summary(
            [{"run_id": 734, "action_id": "A2_rule", "status": "PROMOTED"}],
            {"status": "ESCALATION_REQUIRED"},
            {"pending": 193, "checked": 144},
        )
        self.assertEqual(result["code_promoted_actions"], 1)
        self.assertEqual(result["latest_code_promotion"]["run_id"], 734)
        self.assertEqual(result["pending_labels"], 193)
        self.assertEqual(result["rule_shadow_validation"], "NOT_ESTABLISHED")
        self.assertEqual(result["label_resolution"], "NOT_ESTABLISHED")
        self.assertEqual(result["autopilot_status"], "ESCALATION_REQUIRED")

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.db = sqlite3.connect(":memory:")
        self.db.row_factory = sqlite3.Row
        self.db.execute(
            "create table agent_outputs (output_id integer, run_id integer, role text, "
            "provider text, model text, status text, created_at text)"
        )
        self.db.execute(
            "insert into agent_outputs values (1,7,'reviewer','aliyun','kimi','SUCCEEDED',"
            "'2026-10-21T12:00:00+00:00')"
        )
        self.manual = self.root / "manual"
        self.maintenance = self.root / "maintenance"

    def tearDown(self) -> None:
        self.db.close()
        self.temp.cleanup()

    @staticmethod
    def _write(path: Path, value: dict, mtime: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        os.utime(path, (mtime, mtime))

    def test_merges_three_sources_and_sorts_by_epoch(self) -> None:
        self._write(
            self.manual / "prompt_jobs/j1/agent_plan.json",
            {"role": "cmr_specialist", "model": "kimi", "plan": {
                "decision": "manual_required", "rationale": "needs operator"}},
            1_798_000_100,
        )
        self._write(
            self.maintenance / "run_000008/A1_x/maintenance_manifest.json",
            {"exit_code": 0, "result": {"executor": "llm:qwen", "ready_for_validation": True,
                                         "changed_files": ["a.py"], "summary": "done"}},
            1_798_000_200,
        )
        rows = agent_activity(self.db, manual_root=self.manual,
                              maintenance_root=self.maintenance)
        self.assertEqual({row["src"] for row in rows}, {"分析", "编辑", "维护"})
        self.assertEqual(rows[0]["src"], "维护")
        self.assertEqual(rows[0]["status"], "SUCCEEDED")
        self.assertEqual(rows[1]["status"], "MANUAL_REQUIRED")
        self.assertNotIn("epoch", rows[0])

    def test_ignores_partial_json_write(self) -> None:
        path = self.manual / "prompt_jobs/j1/agent_plan.json"
        path.parent.mkdir(parents=True)
        path.write_text('{"plan":', encoding="utf-8")
        rows = agent_activity(self.db, manual_root=self.manual,
                              maintenance_root=self.maintenance)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["src"], "分析")

    def test_batch_progress_prefers_freshest_valid_report(self) -> None:
        batch = self.root / "batch"
        self._write(batch / "cmr_middle_report.json",
                    {"pool": "cmr_middle", "scanned": 20, "flagged": 3, "auto_fixed": 1},
                    1_798_000_100)
        self._write(batch / "cmr_boundary_report.partial.json",
                    {"pool": "cmr_boundary", "scanned": 80, "flagged": 9,
                     "auto_fixed": 2, "changed_pixels_total": 17},
                    1_798_000_200)
        result = batch_progress(batch)
        self.assertEqual(result["status"], "RUNNING")
        self.assertEqual(result["pool"], "cmr_boundary")
        self.assertEqual(result["scanned"], 80)
        self.assertEqual(result["changed_pixels"], 17)

    def test_batch_summary_only_aggregates_complete_reports(self) -> None:
        batch = self.root / "batch"
        self._write(batch / "cmr_middle_report.json",
                    {"scanned": 20, "flagged": 3, "auto_fixed": 1,
                     "changed_pixels_total": 10}, 1_798_000_100)
        self._write(batch / "cmr_boundary_report.json",
                    {"scanned": 80, "flagged": 9, "auto_fixed": 2,
                     "changed_pixels_total": 17}, 1_798_000_200)
        self._write(batch / "cmr_boundary_report.partial.json",
                    {"scanned": 1000, "auto_fixed": 999}, 1_798_000_300)
        self._write(batch / "candidate_validation.json",
                    {"total": 3, "passed": 2, "rejected": 1}, 1_798_000_400)
        result = batch_summary(batch)
        self.assertEqual(result["report_count"], 2)
        self.assertEqual(result["scanned"], 100)
        self.assertEqual(result["auto_fixed"], 3)
        self.assertEqual(result["changed_pixels"], 27)
        self.assertEqual(result["gate_passed"], 2)
        self.assertEqual(result["gate_rejected"], 1)
        rejected = rejected_batch_candidates(batch)
        self.assertEqual(rejected, [])

    def test_candidate_gate_separates_label_tasks_from_rejected_diagnostics(self) -> None:
        batch = self.root / "batch"
        self._write(batch / "candidate_validation.json", {
            "total": 3, "passed": 1, "rejected": 2,
            "records": [
                {"pool": "cmr_middle_manifest", "index": 4,
                 "operation": ["restore_temporal_hole"], "passed": True,
                 "reasons": [], "metrics": {"changed_pixels": 12}},
                {"pool": "cmr_middle_manifest", "index": 5,
                 "operation": ["remove_small_components"], "passed": False,
                 "reasons": ["two_component_anatomy_is_legal"], "metrics": {}},
                {"pool": "cmr_middle_manifest", "index": 6,
                 "operation": ["remove_small_components"], "passed": False,
                 "reasons": ["two_component_anatomy_is_legal", "dominant_component_modified"],
                 "metrics": {}},
            ],
        }, 1_798_000_400)
        result = candidate_gate_detail(batch, rejected_limit=1)
        self.assertEqual(result["passed"], 1)
        self.assertEqual(result["rejected"], 2)
        self.assertEqual(len(result["passed_candidates"]), 1)
        self.assertEqual(len(result["rejected_candidates"]), 1)
        self.assertEqual(result["rejection_reasons"][0], {
            "reason": "two_component_anatomy_is_legal", "count": 2})

    def test_candidate_gate_resolves_current_sample_identity(self) -> None:
        batch = self.root / "batch"
        dataset = self.root / "dataset"
        manifest = dataset / "labels/check/manifest_backup/cmr_middle_manifest.csv"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(
            "patient,volume,frame_index,issues\nP1,1.nii.gz,10,hole\n",
            encoding="utf-8",
        )
        self._write(batch / "candidate_validation.json", {
            "total": 1, "passed": 1, "rejected": 0,
            "records": [{
                "pool": "cmr_middle_manifest", "index": 0,
                "operation": ["restore_temporal_hole"], "passed": True,
                "reasons": [], "metrics": {"changed_pixels": 12},
            }],
        }, 1_798_000_400)
        result = candidate_gate_detail(batch, dataset_root=dataset)
        self.assertEqual(
            result["passed_candidates"][0]["sample_name"],
            "P1/1.nii.gz#frame=10",
        )
        self.assertEqual(result["passed_candidates"][0]["review_status"], "work")

    def test_boundary_label_queue_only_includes_work_folder(self) -> None:
        dataset = self.root / "dataset"
        manifest = dataset / "labels/check/manifest_backup/cmr_boundary_manifest.csv"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(
            "patient,volume,slice_index,issues\nP1,1.nii.gz,1,boundary\nP2,2.nii.gz,1,boundary\n",
            encoding="utf-8",
        )
        for patient, name, folder in (("P1", "1.png", "待处理"), ("P2", "2.png", "垃圾桶")):
            path = dataset / "labels/check" / folder / "cmr" / patient / name
            path.parent.mkdir(parents=True)
            path.write_bytes(b"png")
        rows = [
            {"pool": "cmr_boundary_manifest", "index": 0},
            {"pool": "cmr_boundary_manifest", "index": 1},
            {"pool": "cmr_middle_manifest", "index": 9},
        ]
        self.assertEqual(active_label_review_tasks(rows, dataset), [rows[0]])

    def test_middle_candidate_is_not_a_human_task_until_it_enters_work(self) -> None:
        dataset = self.root / "dataset"
        manifest = dataset / "labels/check/manifest_backup/cmr_middle_manifest.csv"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(
            "patient,volume,frame_index,issues\nP1,1.nii.gz,10,hole\nP2,2.nii.gz,11,hole\n",
            encoding="utf-8",
        )
        for patient, name, folder in (("P1", "1.png", "待筛"), ("P2", "2.png", "待处理")):
            path = dataset / "labels/check" / folder / "cmr" / patient / name
            path.parent.mkdir(parents=True)
            path.write_bytes(b"png")
        rows = [
            {"pool": "cmr_middle_manifest", "index": 0},
            {"pool": "cmr_middle_manifest", "index": 1},
        ]
        self.assertEqual(active_label_review_tasks(rows, dataset), [rows[1]])

    def test_board_names_medical_review_and_maintenance_queues_separately(self) -> None:
        self.assertIn("标签人工复核队列", PAGE)
        self.assertIn("维护代码实验的人工裁定（诊断，不是标签队列）", PAGE)
        self.assertIn("多智能体工作流", PAGE)
        self.assertIn("输出质检", PAGE)
        self.assertIn("记忆已接续", PAGE)
        self.assertIn("数据处理进度", PAGE)
        self.assertIn("阿里云可用余额", PAGE)
        self.assertIn("Autopilot / Check Loop", PAGE)
        self.assertIn("项目维护助手", PAGE)
        self.assertIn("/api/maintenance-chat", PAGE)
        self.assertIn("启动闭环", PAGE)
        self.assertIn("onclick=\"controlWorkflow('start')\"", PAGE)
        self.assertIn("onclick=\"controlWorkflow('pause')\"", PAGE)
        self.assertIn("需要少量人工示例", PAGE)
        self.assertIn("已确认并进入知识链", PAGE)
        self.assertIn("/api/autopilot/control", PAGE)
        self.assertIn("framework_fix_available", PAGE)
        self.assertIn("应用修复并恢复", PAGE)
        self.assertIn("付费模型", PAGE)
        self.assertIn("医学判定、标签修改、删除或分桶", PAGE)
        self.assertNotIn("池 / 索引", PAGE)
        self.assertNotIn("打开池", PAGE)

    def test_framework_fix_resume_requires_new_commit(self) -> None:
        from scripts.board_server import framework_fix_available

        class Result:
            returncode = 0
            stdout = "new-commit\n"

        runner = lambda *args, **kwargs: Result()
        self.assertFalse(framework_fix_available({}, runner=runner))
        self.assertFalse(framework_fix_available({
            "paused_issue_fingerprint": "issue",
            "last_code_fingerprint": "new-commit",
        }, runner=runner))
        self.assertTrue(framework_fix_available({
            "paused_issue_fingerprint": "issue",
            "last_code_fingerprint": "old-commit",
        }, runner=runner))

    def test_aliyun_balance_snapshot_never_exposes_credentials(self) -> None:
        path = self.root / "private/aliyun_billing.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({
            "available_amount": "123.45", "cash_amount": "20.00",
            "coupon_balance": "103.45", "currency": "CNY",
            "updated_at": "2026-10-25 01:00:00",
            "access_key_secret": "must-not-leak",
        }), encoding="utf-8")
        result = aliyun_billing_status(path)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["available_amount"], "123.45")
        self.assertNotIn("access_key_secret", result)

    def test_aliyun_balance_reports_read_only_permissions_when_unconfigured(self) -> None:
        result = aliyun_billing_status(self.root / "missing.json")
        self.assertEqual(result["status"], "unconfigured")
        self.assertEqual(result["required_permissions"], [
            "bss:DescribeAcccount", "bss:DescribeCoupons"])

    def test_maintenance_chat_context_is_control_plane_only(self) -> None:
        context = maintenance_chat_context({
            "generated_at": "now",
            "state": {"status": "ESCALATION_REQUIRED", "paused": True,
                      "last_event": "ESCALATION_REQUIRED", "last_event_at": "01:00",
                      "cycles": 3, "stagnant": 3},
            "controllers": {"autopilot": "123", "worker": "-"},
            "boundary_review": {"pending": 61, "work": 3, "checked": 184, "trash": 458},
            "runs": [{"run_id": 9, "status": "NEEDS_REVIEW", "current_step": "review"}],
            "events": [{"event": "ESCALATION_REQUIRED", "detail": ""}],
            "agents": [{"who": "lge_mi_specialist", "run_id": 9, "status": "SUCCEEDED",
                        "time": "01:00", "model": "qwen", "detail": "control only"}],
        })
        self.assertEqual(context["check_buckets"]["pending"], 61)
        self.assertEqual(context["agent_branches"]["lge_mi_specialist"]["status"],
                         "SUCCEEDED")
        payload = json.dumps(context, ensure_ascii=False).lower()
        self.assertNotIn("access_key", payload)
        self.assertNotIn("patient", payload)
        self.assertIn("标签修改", payload)

    def test_dashboard_exposes_live_workflow_progress_and_review_buckets(self) -> None:
        self.assertNotIn('id="stage-progress"', PAGE)
        self.assertNotIn('id="runtime-progress"', PAGE)
        self.assertIn("function runningFor(d,role)", PAGE)
        self.assertIn("data-state=\"${state}\"", PAGE)
        self.assertIn("@keyframes nodeFlow", PAGE)
        self.assertIn("@keyframes wireFlow", PAGE)
        self.assertIn(".agent-node[data-state=\"active\"]:before", PAGE)
        self.assertIn("#agent-flow[data-state=\"active\"] .flow-arrow:before", PAGE)
        self.assertIn("document.getElementById('agent-flow').dataset.state", PAGE)
        self.assertIn('id="flow-wires"', PAGE)
        self.assertIn("function drawFlowWires()", PAGE)
        self.assertIn("['failure_triage','lge_myo_specialist']", PAGE)
        self.assertIn("['lge_myo_specialist','lge_mi_specialist','down']", PAGE)
        self.assertIn("['lge_mi_specialist','pairing_gate','down']", PAGE)
        self.assertIn("['pairing_gate','cmr_specialist','down']", PAGE)
        self.assertIn("['cmr_specialist','qc_code_optimizer','curve',0]", PAGE)
        self.assertIn("['qc_code_optimizer','rule_curator']", PAGE)
        self.assertIn("['rule_curator','skill_maintainer']", PAGE)
        self.assertIn("['skill_maintainer','qc_code_optimizer','loop']", PAGE)
        self.assertIn("['skill_maintainer','efficiency_auditor','down']", PAGE)
        self.assertIn("['efficiency_auditor','supervisor']", PAGE)
        self.assertIn("['supervisor','independent_reviewer','down']", PAGE)
        self.assertIn("['independent_reviewer','executor','down']", PAGE)
        self.assertNotIn("['lge_mi_specialist','skill_maintainer']", PAGE)
        self.assertIn("data-role=\"${esc(role)}\"", PAGE)
        self.assertIn("grid-template-columns:repeat(2", PAGE)
        self.assertIn('data-stage="domain"', PAGE)
        self.assertIn('② 数据依赖链', PAGE)
        self.assertIn('grid-template-columns:minmax(105px,.58fr)', PAGE)
        self.assertIn('data-stage="domain"] .parallel{grid-template-columns:1fr}', PAGE)
        self.assertIn('data-stage="optimization"] .parallel{grid-template-columns:repeat(3', PAGE)
        self.assertIn("grid-template-areas:'qc rules skill' 'maintenance audit .'", PAGE)
        self.assertIn('③ 当前阶段 QC 优化', PAGE)
        self.assertIn('知识 / Skill 同步', PAGE)
        self.assertIn('function shortModel(value)', PAGE)
        self.assertIn('.node-meta{display:block', PAGE)
        self.assertIn('@keyframes nodeFlow{0%{background-position:130% 0}', PAGE)
        self.assertIn('.agent-node[data-state="ok"]{background:#10171f;border-color:#3a4958', PAGE)
        self.assertIn('roleIconPaths', PAGE)
        self.assertIn('class="node-icon"', PAGE)
        self.assertIn(".flow-shell .agent-node{position:relative;z-index:3", PAGE)
        self.assertIn(".flow-wires{position:absolute;z-index:1", PAGE)
        self.assertIn("function updateConnectionWarning()", PAGE)
        self.assertIn("Date.now()-lastGoodAt>=15000", PAGE)
        self.assertIn("#offline.confirmed", PAGE)
        self.assertIn("stamp=`${elapsedSeconds", PAGE)
        self.assertNotIn("word=`运行 ${elapsedSeconds", PAGE)
        self.assertIn("review_status=${state}", PAGE)

    def test_board_control_requires_token_for_public_host(self) -> None:
        self.assertTrue(board_control_authorized("127.0.0.1:8765", "", ""))
        self.assertTrue(board_control_authorized("localhost:8765", "", ""))
        self.assertTrue(board_control_authorized("[::1]:8765", "", ""))
        self.assertFalse(board_control_authorized("board.example.com", "", ""))
        self.assertFalse(board_control_authorized("board.example.com", "wrong", "secret"))
        self.assertTrue(board_control_authorized("board.example.com", "secret", "secret"))

    def test_resume_autopilot_uses_fixed_audited_commands(self) -> None:
        state = self.root / "autopilot_state.json"
        audit = self.root / "control.jsonl"
        state.write_text(json.dumps({
            "status": "ESCALATION_REQUIRED",
            "paused_issue_fingerprint": "abc",
            "escalated_for": 834,
        }), encoding="utf-8")
        commands = []

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        def runner(command, **kwargs):
            commands.append(command)
            return Result()

        result = resume_autopilot(state_path=state, audit_path=audit, control_path=self.root/"operator.json", runner=runner, fingerprint_fn=lambda dataset:("abc",{}))
        self.assertEqual(result["status"], "resumed")
        self.assertEqual(commands[0], [
            "systemctl", "--user", "stop", "cmr-lge-autopilot.service"])
        self.assertIn("--resume-after-fix", commands[1])
        self.assertEqual({command[-1] for command in commands[2:]}, {
            "cmr-lge-autopilot.service", "cmr-lge-check-loop.service", "cmr-lge-edit-agent.service"})
        self.assertTrue(all(command[3] == "start" for command in commands[2:]))
        self.assertFalse(json.loads((self.root/"operator.json").read_text())["paused"])
        events = [json.loads(line)["event"] for line in audit.read_text(
            encoding="utf-8").splitlines()]
        self.assertEqual(events, ["BOARD_AUTOPILOT_RESUME_REQUESTED",
            "BOARD_WORKFLOW_START_REQUESTED", "BOARD_WORKFLOW_STARTED", "BOARD_AUTOPILOT_RESUMED"])

    def test_resume_autopilot_does_not_restart_service_when_resume_times_out(self) -> None:
        state = self.root / "autopilot_state_timeout.json"
        audit = self.root / "control_timeout.jsonl"
        state.write_text(json.dumps({
            "status": "ESCALATION_REQUIRED",
            "paused_issue_fingerprint": "abc",
            "escalated_for": 834,
        }), encoding="utf-8")
        commands = []

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        def runner(command, **kwargs):
            commands.append(command)
            if "--resume-after-fix" in command:
                raise subprocess.TimeoutExpired(command, 120)
            return Result()

        with self.assertRaises(AutopilotControlError):
            resume_autopilot(state_path=state, audit_path=audit, control_path=self.root/"operator.json", runner=runner, fingerprint_fn=lambda dataset:("abc",{}))
        self.assertFalse(any("start" in command for command in commands))

    def test_control_latch_is_persisted_before_service_commands_and_failed_start_pauses_all(self):
        state=self.root/"latch_state.json";control=self.root/"latch_operator.json";audit=self.root/"latch_audit.jsonl"
        state.write_text(json.dumps({"status":"RUNNING"}))
        commands=[]
        def runner(command,**kwargs):
            commands.append(command)
            if command[3]=="start":
                self.assertFalse(json.loads(control.read_text())["paused"])
                return type("Result",(),{"returncode":1,"stderr":"synthetic start failure","stdout":""})()
            self.assertTrue(json.loads(control.read_text())["paused"])
            return type("Result",(),{"returncode":0,"stderr":"","stdout":""})()
        with self.assertRaises(AutopilotControlError):
            control_workflow("start",state_path=state,control_path=control,audit_path=audit,runner=runner)
        self.assertTrue(json.loads(control.read_text())["paused"])
        self.assertEqual(3,sum(command[3]=="stop" for command in commands))

    def test_resume_with_new_knowledge_uses_normal_start_without_obsolete_same_fingerprint_command(self):
        state=self.root/"new_knowledge_state.json";control=self.root/"new_knowledge_operator.json";audit=self.root/"new_knowledge_audit.jsonl"
        state.write_text(json.dumps({"paused_issue_fingerprint":"old","status":"ESCALATION_REQUIRED"}))
        commands=[]
        def runner(command,**kwargs):
            commands.append(command)
            return type("Result",(),{"returncode":0,"stderr":"","stdout":""})()
        result=resume_autopilot(state_path=state,control_path=control,audit_path=audit,runner=runner,fingerprint_fn=lambda dataset:("new",{}))
        self.assertEqual("resumed",result["status"])
        self.assertFalse(any("--resume-after-fix" in command for command in commands))
        self.assertFalse(json.loads(control.read_text())["paused"])
        self.assertEqual(3,sum("start" in command for command in commands))

    def test_workflow_pause_and_start_control_all_processing_services(self) -> None:
        state = self.root / "autopilot_state.json"
        control = self.root / "operator_state.json"
        audit = self.root / "control.jsonl"
        state.write_text(json.dumps({"status": "RUNNING"}), encoding="utf-8")
        commands = []

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        def runner(command, **kwargs):
            commands.append(command)
            return Result()

        paused = control_workflow(
            "pause", state_path=state, control_path=control,
            audit_path=audit, runner=runner,
        )
        self.assertEqual(paused["status"], "paused")
        self.assertTrue(json.loads(control.read_text(encoding="utf-8"))["paused"])
        self.assertEqual([command[3] for command in commands], ["stop"] * 3)
        self.assertTrue(all(command[2] == "--no-block" for command in commands))

        commands.clear()
        started = control_workflow(
            "start", state_path=state, control_path=control,
            audit_path=audit, runner=runner,
        )
        self.assertEqual(started["status"], "started")
        self.assertFalse(json.loads(control.read_text(encoding="utf-8"))["paused"])
        self.assertEqual([command[3] for command in commands], ["start"] * 3)
        self.assertTrue(all(command[2] == "--no-block" for command in commands))

    def test_workflow_start_waits_for_human_when_fingerprint_is_unchanged(self) -> None:
        state = self.root / "autopilot_state.json"
        control = self.root / "operator_state.json"
        audit = self.root / "control.jsonl"
        state.write_text(json.dumps({
            "status": "ESCALATION_REQUIRED",
            "paused_issue_fingerprint": "same",
        }), encoding="utf-8")

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        with patch("scripts.autopilot.issue_fingerprint", return_value=("same", {})):
            result = control_workflow(
                "start", state_path=state, control_path=control,
                audit_path=audit, runner=lambda *args, **kwargs: Result(),
            )
        self.assertEqual(result["status"], "waiting_for_human")
        self.assertIn("1–2", result["message"])

    def test_manual_learning_counts_only_applied_signed_examples(self) -> None:
        manual = self.root / "runtime/manual_annotations"
        corpus = manual / "analyzed_training_corpus.jsonl"
        corpus.parent.mkdir(parents=True)
        rows = []
        for edit_id, status in (("staged", "STAGED"), ("applied", "APPLIED")):
            edit_dir = manual / "edits" / edit_id
            edit_dir.mkdir(parents=True)
            (edit_dir / "provenance.json").write_text(json.dumps({
                "edit_id": edit_id,
                "status": status,
                "applied_by": "operator" if status == "APPLIED" else None,
                "applied_hash": "sha256:ok" if status == "APPLIED" else None,
            }), encoding="utf-8")
            rows.append(json.dumps({"edit_id": edit_id, "modality": "cmr", "target": "myo"}))
        corpus.write_text("\n".join(rows) + "\n", encoding="utf-8")
        result = manual_learning_status(self.root)
        self.assertEqual(result["approved_examples"], 1)
        self.assertEqual(result["edit_statuses"], {"STAGED": 1, "APPLIED": 1})

    def test_label_review_queue_uses_readable_responsive_cards(self) -> None:
        self.assertIn('.review-list{max-height:360px;overflow-y:auto;overflow-x:hidden', PAGE)
        self.assertIn('@media(max-width:620px){.review-item{grid-template-columns:1fr}', PAGE)
        self.assertIn('function renderLabelReview(review)', PAGE)
        self.assertIn('review.map((x,i)=>`<article class="review-item">', PAGE)
        self.assertIn('onclick="openLabelReview(${i})"', PAGE)
        self.assertIn('const review=d.label_review_tasks||[];renderLabelReview(review);', PAGE)
        self.assertNotIn('<th>样本</th><th>状态</th><th>修改</th>', PAGE)

    def test_editor_verifies_and_highlights_agent_candidate_before_canvas_replace(self) -> None:
        editor = (Path(__file__).parents[1] / "static/annotation_editor.html").read_text(
            encoding="utf-8"
        )
        self.assertIn("async function loadPromptMask", editor)
        self.assertIn("cache:'no-store'", editor)
        self.assertIn("actualDelta===0", editor)
        self.assertIn("$('show-diff').checked=true", editor)
        self.assertIn("候选已载入画布，尚未写入正式标签", editor)


if __name__ == "__main__":
    unittest.main()
