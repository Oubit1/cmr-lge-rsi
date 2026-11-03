from __future__ import annotations

import base64
import csv
import io
import json
import threading
import tempfile
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

import nibabel as nib
import numpy as np
from PIL import Image
from app.label_executor_client import request_label_apply
from scripts.label_write_executor import ExecutorServer
from scripts.board_server import apply_staged_edit_local
from scripts import board_server

from app.manual_annotation import (
    apply_edit,
    edit_status,
    execute_prompt_repair,
    frame_metadata,
    list_candidates,
    png_layer,
    prompt_candidate_png,
    prompt_repair_status,
    queue_prompt_repair,
    save_edit,
    set_boundary_review_status,
    set_item_excluded,
)


def png_data(mask: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray((mask.astype(np.uint8) * 255)).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


class ManualAnnotationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "dataset"
        self.work = Path(self.temp.name) / "work"
        patient, volume = "PATIENT_NAME_001", "1.0.nii.gz"
        for relative in [
            "images/cmr", "labels/cmr_checked/cmr_Myo_mask",
            "labels/check/manifest_backup",
        ]:
            # 注意：不能再用 "check" in relative 判断 —— cmr_checked 含 "check" 子串
            if relative == "labels/check/manifest_backup":
                (self.root / relative).mkdir(parents=True, exist_ok=True)
            else:
                (self.root / relative / patient).mkdir(parents=True, exist_ok=True)
        image = np.zeros((16, 16, 3), dtype=np.float32)
        image[3:13, 3:13, :] = 100
        mask = np.zeros_like(image, dtype=np.uint8)
        mask[4:12, 4:12, :] = 1
        mask[6:10, 6:10, :] = 0
        affine = np.eye(4)
        nib.save(nib.Nifti1Image(image, affine), str(self.root / "images/cmr" / patient / volume))
        nib.save(nib.Nifti1Image(mask, affine), str(self.root / "labels/cmr_checked/cmr_Myo_mask" / patient / volume))
        manifest = self.root / "labels/check/manifest_backup/cmr_middle_manifest.csv"
        with manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["patient", "volume", "slice_index", "issues", "evidence"])
            writer.writeheader()
            writer.writerow({
                "patient": patient, "volume": volume, "slice_index": 2,
                "issues": "cmr_myo_gap", "evidence": "synthetic test",
            })
        self.patient = patient
        self.source = self.root / "labels/cmr_checked/cmr_Myo_mask" / patient / volume

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_candidate_listing_metadata_and_png_layers(self) -> None:
        listing = list_candidates(self.root, "cmr_middle_manifest")
        self.assertEqual(listing["total"], 1)
        self.assertEqual(listing["items"][0]["targets"], ["myo"])
        self.assertEqual(listing["items"][0]["layer_index"], 0)
        self.assertEqual(listing["items"][0]["total_layers"], 1)
        self.assertEqual(
            listing["items"][0]["sample_name"],
            "PATIENT_NAME_001/1.0.nii.gz#frame=2",
        )
        self.assertEqual(listing["items"][0]["paths"]["image"], str(
            self.root / "images/cmr/PATIENT_NAME_001/1.0.nii.gz"
        ))
        self.assertEqual(listing["items"][0]["paths"]["myo"], str(self.source))
        self.assertNotIn("mi", listing["items"][0]["paths"])
        meta = frame_metadata(self.root, "cmr_middle_manifest", 0)
        self.assertEqual((meta["height"], meta["width"]), (16, 16))
        self.assertEqual(len(meta["source_hashes"]["myo"]), 64)
        layer = Image.open(io.BytesIO(png_layer(
            self.root, "cmr_middle_manifest", 0, "myo", 0
        )))
        self.assertEqual(layer.size, (16, 16))

    def test_exclusion_is_reversible_frame_disposition_not_label_edit(self) -> None:
        before = self.source.read_bytes()
        with self.assertRaises(ValueError):
            set_item_excluded(self.root, "cmr_middle_manifest", 0,
                              excluded=True, reason="other")
        event = set_item_excluded(self.root, "cmr_middle_manifest", 0,
                                  excluded=True, reason="out_of_range")
        self.assertTrue(event["excluded"])
        self.assertEqual(list_candidates(self.root, "cmr_middle_manifest")["total"], 0)
        shown = list_candidates(self.root, "cmr_middle_manifest", include_excluded=True)
        self.assertTrue(shown["items"][0]["excluded"])
        meta = frame_metadata(self.root, "cmr_middle_manifest", 0)
        edited = np.asarray(nib.load(str(self.source)).get_fdata())[..., 1] > .5
        edited[3, 8] = True
        with self.assertRaisesRegex(ValueError, "removed from the editor queue"):
            save_edit(self.root, self.work, {
                "pool": "cmr_middle_manifest", "index": 0, "target": "myo",
                "mask_png": png_data(edited), "source_hash": meta["source_hashes"]["myo"],
                "reviewer": "operator",
            })
        set_item_excluded(self.root, "cmr_middle_manifest", 0, excluded=False)
        self.assertEqual(list_candidates(self.root, "cmr_middle_manifest")["total"], 1)
        self.assertEqual(self.source.read_bytes(), before)

    def test_manual_edit_can_generate_its_own_note_from_delta_and_tools(self) -> None:
        meta = frame_metadata(self.root, "cmr_middle_manifest", 0)
        edited = np.asarray(nib.load(str(self.source)).get_fdata())[..., 1] > .5
        edited[3, 8] = True
        staged = save_edit(self.root, self.work, {
            "pool": "cmr_middle_manifest", "index": 0, "target": "myo",
            "mask_png": png_data(edited), "source_hash": meta["source_hashes"]["myo"],
            "reviewer": "operator", "reason_code": "auto",
            "tool_trace": ["add", "undo", "add", "invalid"],
        })
        self.assertIn("增补", staged["review_note"])
        self.assertIn("局部归一化对比度", staged["review_note"])
        self.assertIn("add×2", staged["review_note"])
        provenance = json.loads((self.work / "edits" / staged["edit_id"] /
                                 "provenance.json").read_text())
        self.assertEqual(provenance["review_note_source"], "auto_delta")
        self.assertEqual(provenance["tool_trace"], ["add", "undo", "add"])
        self.assertEqual(provenance["reason_code"], staged["suggested_pattern"])
        self.assertEqual(provenance["status"], "STAGED")

    def test_boundary_review_folders_drive_spatial_queue_state(self) -> None:
        boundary = self.root / "labels/check/manifest_backup"
        boundary.mkdir(parents=True, exist_ok=True)
        with (boundary / "cmr_boundary_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["patient", "volume", "slice_index", "issues"])
            writer.writeheader()
            writer.writerow({"patient": self.patient, "volume": "1.0.nii.gz", "slice_index": 1,
                             "issues": "boundary"})
            writer.writerow({"patient": self.patient, "volume": "1.0.nii.gz", "slice_index": 2,
                             "issues": "boundary"})
        check_png = self.root / "labels/check/cmr/cmr_Myo_mask" / self.patient / "1.0.png"
        check_png.parent.mkdir(parents=True)
        Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(check_png)

        pending = list_candidates(
            self.root, "cmr_boundary_manifest", review_status="pending", group_spatial=True
        )
        self.assertEqual(pending["total"], 1)
        result = set_boundary_review_status(
            self.root, "cmr_boundary_manifest", pending["items"][0]["index"], "work"
        )
        self.assertEqual(result["status"], "work")
        self.assertEqual(result["counts"]["work"], 1)
        self.assertEqual(list_candidates(
            self.root, "cmr_boundary_manifest", review_status="pending", group_spatial=True
        )["total"], 0)
        work = list_candidates(
            self.root, "cmr_boundary_manifest", review_status="work", group_spatial=True
        )
        self.assertEqual(work["total"], 1)
        self.assertEqual(work["items"][0]["review_status"], "work")
        self.assertTrue((self.root / "labels/check/待处理/cmr"
                         / self.patient / "1.0.png").is_file())
        self.assertFalse(check_png.exists())

    def test_same_state_confirmation_hides_reviewed_trash_without_changing_inventory(self) -> None:
        boundary = self.root / "labels/check/manifest_backup"
        boundary.mkdir(parents=True, exist_ok=True)
        with (boundary / "cmr_boundary_manifest.csv").open(
                "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["patient", "volume", "slice_index", "issues"]
            )
            writer.writeheader()
            writer.writerow({"patient": self.patient, "volume": "1.0.nii.gz",
                             "slice_index": 0, "issues": "boundary"})
        trash_png = self.root / "labels/check/垃圾桶/cmr" / self.patient / "1.0.png"
        trash_png.parent.mkdir(parents=True)
        Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(trash_png)
        before = list_candidates(
            self.root, "cmr_boundary_manifest", review_status="trash",
            group_spatial=True, hide_reviewed=True,
        )
        self.assertEqual(before["total"], 1)
        result = set_boundary_review_status(
            self.root, "cmr_boundary_manifest", before["items"][0]["index"], "trash"
        )
        self.assertTrue(result["review_confirmed"])
        after = list_candidates(
            self.root, "cmr_boundary_manifest", review_status="trash",
            group_spatial=True, hide_reviewed=True,
        )
        self.assertEqual(after["total"], 0)
        self.assertEqual(after["review_inventory_total"], 1)
        self.assertTrue(trash_png.is_file())

    def test_applied_boundary_edit_moves_spatial_slice_to_checked(self) -> None:
        boundary = self.root / "labels/check/manifest_backup"
        boundary.mkdir(parents=True, exist_ok=True)
        (boundary / "cmr_boundary_manifest.csv").write_text(
            "patient,volume,slice_index,issues\nPATIENT_NAME_001,1.0.nii.gz,2,boundary\n",
            encoding="utf-8",
        )
        check_png = self.root / "labels/check/cmr/cmr_Myo_mask/PATIENT_NAME_001/1.0.png"
        check_png.parent.mkdir(parents=True)
        Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(check_png)
        set_boundary_review_status(self.root, "cmr_boundary_manifest", 0, "work")
        meta = frame_metadata(self.root, "cmr_boundary_manifest", 0)
        edited = np.asarray(nib.load(str(self.source)).get_fdata())[..., 1] > 0.5
        edited[3, 8] = True
        staged = save_edit(self.root, self.work, {
            "pool": "cmr_boundary_manifest", "index": 0, "target": "myo",
            "mask_png": png_data(edited), "source_hash": meta["source_hashes"]["myo"],
            "reviewer": "doctor", "note": "边界层人工修正", "reason_code": "other",
        })
        applied = apply_edit(
            self.root, self.work, staged["edit_id"], reviewer="doctor", confirmation="APPLY"
        )
        self.assertEqual(applied["boundary_review_status"], "checked")
        self.assertTrue((self.root / "labels/check/checked/cmr"
                         / self.patient / "1.0.png").is_file())
        self.assertFalse((self.root / "labels/check/待处理/cmr"
                          / self.patient / "1.0.png").exists())

    def test_staged_boundary_edit_uses_separate_socket_and_audit(self) -> None:
        (self.root / "labels/check/manifest_backup/cmr_boundary_manifest.csv").write_text(
            "patient,volume,slice_index,issues\nPATIENT_NAME_001,1.0.nii.gz,2,boundary\n",
            encoding="utf-8",
        )
        check_png = self.root / "labels/check/cmr/cmr_Myo_mask/PATIENT_NAME_001/1.0.png"
        check_png.parent.mkdir(parents=True)
        Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(check_png)
        set_boundary_review_status(self.root, "cmr_boundary_manifest", 0, "work")
        meta = frame_metadata(self.root, "cmr_boundary_manifest", 0)
        edited = np.asarray(nib.load(str(self.source)).get_fdata())[..., 1] > 0.5
        edited[3, 8] = True
        staged = save_edit(self.root, self.work, {
            "pool": "cmr_boundary_manifest", "index": 0, "target": "myo",
            "mask_png": png_data(edited), "source_hash": meta["source_hashes"]["myo"],
            "reviewer": "synthetic-operator", "note": "synthetic boundary edit",
        })
        socket_path = self.work / "executor.sock"
        audit_log = self.work / "executor-audit.jsonl"
        with ExecutorServer(socket_path, self.root, self.work, audit_log) as server:
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:
                applied = request_label_apply(
                    dataset_root=self.root, work_root=self.work,
                    edit_id=staged["edit_id"], reviewer="synthetic-operator",
                    confirmation="APPLY", socket_path=socket_path,
                )
            finally:
                server.shutdown()
                worker.join(timeout=5)
        self.assertEqual(applied["status"], "APPLIED")
        self.assertEqual(applied["boundary_review_status"], "checked")
        self.assertTrue(Path(applied["backup"]).is_file())
        self.assertTrue((self.root / "labels/check/checked/cmr"
                         / self.patient / "1.0.png").is_file())
        self.assertIn('"ok": true', audit_log.read_text(encoding="utf-8"))

    def test_board_direct_apply_requires_confirmation_and_audits(self) -> None:
        (self.root / "labels/check/manifest_backup/cmr_boundary_manifest.csv").write_text(
            "patient,volume,slice_index,issues\nPATIENT_NAME_001,1.0.nii.gz,2,boundary\n",
            encoding="utf-8",
        )
        check_png = self.root / "labels/check/cmr/cmr_Myo_mask/PATIENT_NAME_001/1.0.png"
        check_png.parent.mkdir(parents=True)
        Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(check_png)
        set_boundary_review_status(self.root, "cmr_boundary_manifest", 0, "work")
        meta = frame_metadata(self.root, "cmr_boundary_manifest", 0)
        edited = np.asarray(nib.load(str(self.source)).get_fdata())[..., 1] > 0.5
        edited[3, 8] = True
        staged = save_edit(self.root, self.work, {
            "pool": "cmr_boundary_manifest", "index": 0, "target": "myo",
            "mask_png": png_data(edited), "source_hash": meta["source_hashes"]["myo"],
            "reviewer": "synthetic-operator", "note": "synthetic board edit",
        })
        before = self.source.read_bytes()
        audit_log = self.work / "board-audit.jsonl"
        params = dict(dataset_root=self.root, work_root=self.work, audit_path=audit_log,
                      edit_id=staged["edit_id"], reviewer="synthetic-operator")
        with self.assertRaises(OSError):
            apply_staged_edit_local(confirmation="APPLY", **{**params, "audit_path": self.work})
        self.assertEqual(self.source.read_bytes(), before)
        with self.assertRaises(ValueError):
            apply_staged_edit_local(confirmation="NO", **params)
        self.assertEqual(self.source.read_bytes(), before)
        with mock.patch.object(board_server, "DATASET", self.root), \
             mock.patch.object(board_server, "MANUAL_WORK", self.work), \
             mock.patch.object(board_server, "LABEL_APPLY_AUDIT", audit_log), \
             ThreadingHTTPServer(("127.0.0.1", 0), board_server.Handler) as server:
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:
                request = urllib.request.Request(
                    f"http://127.0.0.1:{server.server_port}/api/editor/apply",
                    data=json.dumps({"edit_id": staged["edit_id"],
                                     "reviewer": "synthetic-operator",
                                     "confirmation": "APPLY"}).encode("utf-8"),
                    headers={"Content-Type": "application/json",
                             "X-Requested-With": "cmr-lge-board"},
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=5) as response:
                    self.assertEqual(response.status, 201)
                    applied = json.load(response)
            finally:
                server.shutdown()
                worker.join(timeout=5)
        self.assertEqual(applied["status"], "APPLIED")
        self.assertEqual(applied["boundary_review_status"], "checked")
        self.assertTrue(Path(applied["backup"]).is_file())
        self.assertTrue((self.root / "labels/check/checked/cmr"
                         / self.patient / "1.0.png").is_file())
        records = [json.loads(line) for line in audit_log.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([record["event"] for record in records],
                         ["requested", "rejected", "requested", "applied"])

    def test_stage_is_non_mutating_and_writes_anonymous_corpus(self) -> None:
        before = self.source.read_bytes()
        meta = frame_metadata(self.root, "cmr_middle_manifest", 0)
        edited = np.asarray(nib.load(str(self.source)).get_fdata())[..., 1] > 0.5
        edited[3, 8] = True
        result = save_edit(self.root, self.work, {
            "pool": "cmr_middle_manifest", "index": 0, "target": "myo",
            "mask_png": png_data(edited), "source_hash": meta["source_hashes"]["myo"],
            "reviewer": "doctor", "note": "补全局部缺口", "reason_code": "fill_internal_gap",
            "request_agent_analysis": True,
        })
        self.assertEqual(result["status"], "STAGED")
        self.assertEqual(before, self.source.read_bytes())
        corpus_text = (self.work / "training_corpus.jsonl").read_text(encoding="utf-8")
        self.assertNotIn(self.patient, corpus_text)
        self.assertNotIn(str(self.root), corpus_text)
        corpus = json.loads(corpus_text)
        self.assertEqual(corpus["constraints"]["production_label_modified"], False)
        self.assertIn("prompt_gt", corpus["prompt_supervision"])
        self.assertIn("心肌标注", result["prompt_gt"])
        self.assertTrue((self.work / "agent_queue.jsonl").exists())
        self.assertEqual(edit_status(self.work, result["edit_id"])["agent_status"], "QUEUED")

    def test_apply_requires_confirmation_and_creates_backup(self) -> None:
        meta = frame_metadata(self.root, "cmr_middle_manifest", 0)
        edited = np.asarray(nib.load(str(self.source)).get_fdata())[..., 1] > 0.5
        edited[3, 8] = True
        staged = save_edit(self.root, self.work, {
            "pool": "cmr_middle_manifest", "index": 0, "target": "myo",
            "mask_png": png_data(edited), "source_hash": meta["source_hashes"]["myo"],
            "reviewer": "doctor", "note": "补全局部缺口", "reason_code": "fill_internal_gap",
            "source_run_id": 840,
            "source_action_id": "S841_cmr_small_area_guard_delta_scan_exec",
        })
        with self.assertRaisesRegex(ValueError, "APPLY"):
            apply_edit(self.root, self.work, staged["edit_id"], reviewer="doctor", confirmation="yes")
        applied = apply_edit(
            self.root, self.work, staged["edit_id"], reviewer="doctor", confirmation="APPLY"
        )
        self.assertEqual(applied["status"], "APPLIED")
        self.assertEqual(applied["source_run_id"], 840)
        self.assertEqual(applied["source_action_id"],
                         "S841_cmr_small_area_guard_delta_scan_exec")
        self.assertTrue(Path(applied["backup"]).is_file())
        self.assertTrue(np.asarray(nib.load(str(self.source)).get_fdata())[3, 8, 1] > 0.5)

    def test_prompt_repair_is_deidentified_staged_and_bounded(self) -> None:
        meta = frame_metadata(self.root, "cmr_middle_manifest", 0)
        queued = queue_prompt_repair(self.root, self.work, {
            "pool": "cmr_middle_manifest", "index": 0, "target": "myo",
            "prompt": "补全较小的内部缺口", "source_hash": meta["source_hashes"]["myo"],
        })
        job = self.work / "prompt_jobs" / queued["job_id"]
        serialized = (job / "agent_request.json").read_text(encoding="utf-8")
        self.assertNotIn(self.patient, serialized)
        self.assertNotIn(str(self.root), serialized)
        result = execute_prompt_repair(self.work, queued["job_id"], {
            "decision": "apply_operations", "target": "myo",
            "operations": [{"operation": "fill_polygon",
                            "points": [[450, 50], [550, 50], [550, 180], [450, 180]]}],
            "rationale": "bounded test", "expected_effect": "fill", "risk_flags": [],
            "confidence": 0.8,
        })
        self.assertEqual(result["status"], "READY")
        self.assertTrue(prompt_candidate_png(self.work, queued["job_id"]))
        status = prompt_repair_status(self.work, queued["job_id"])
        self.assertTrue(status["candidate_ready"])
        self.assertEqual(status["pool"], "cmr_middle_manifest")
        self.assertEqual(status["manifest_index"], 0)
        self.assertEqual(status["target"], "myo")
        self.assertEqual(status["source_hash"], meta["source_hashes"]["myo"])
        self.assertEqual(meta["source_hashes"]["myo"], frame_metadata(
            self.root, "cmr_middle_manifest", 0
        )["source_hashes"]["myo"])

    def test_prompt_rejects_patient_identifiers(self) -> None:
        meta = frame_metadata(self.root, "cmr_middle_manifest", 0)
        with self.assertRaisesRegex(ValueError, "identifier"):
            queue_prompt_repair(self.root, self.work, {
                "pool": "cmr_middle_manifest", "index": 0, "target": "myo",
                "prompt": f"修正 {self.patient} 的标签",
                "source_hash": meta["source_hashes"]["myo"],
            })

    def test_manual_delta_becomes_prompt_supervision_and_next_round_context(self) -> None:
        meta = frame_metadata(self.root, "cmr_middle_manifest", 0)
        queued = queue_prompt_repair(self.root, self.work, {
            "pool": "cmr_middle_manifest", "index": 0, "target": "myo",
            "prompt": "删除右上方多余区域", "source_hash": meta["source_hashes"]["myo"],
        })
        job = self.work / "prompt_jobs" / queued["job_id"]
        (job / "agent_plan.json").write_text(json.dumps({
            "role": "cmr_specialist", "model": "mock", "plan": {
                "decision": "manual_required", "operations": [], "rationale": "无法定位"
            }
        }), encoding="utf-8")
        edited = np.asarray(nib.load(str(self.source)).get_fdata())[..., 1] > 0.5
        edited[4:6, 10:12] = False
        staged = save_edit(self.root, self.work, {
            "pool": "cmr_middle_manifest", "index": 0, "target": "myo",
            "mask_png": png_data(edited), "source_hash": meta["source_hashes"]["myo"],
            "reviewer": "doctor", "note": "人工删除右上伪区域",
            "reason_code": "shrink_oversegmentation", "edit_origin": "prompt_then_manual",
            "prompt_job_id": queued["job_id"],
        })
        supervision = json.loads((self.work / "edits" / staged["edit_id"] /
                                  "prompt_supervision.json").read_text(encoding="utf-8"))
        self.assertEqual(supervision["operator_prompt"], "删除右上方多余区域")
        self.assertEqual(supervision["agent_decision"], "manual_required")
        self.assertGreater(supervision["human_delta"]["removed"]["pixels"], 0)

        next_job = queue_prompt_repair(self.root, self.work, {
            "pool": "cmr_middle_manifest", "index": 0, "target": "myo",
            "prompt": "按人工示例精确删除右上区域", "source_hash": meta["source_hashes"]["myo"],
            "previous_job_id": queued["job_id"],
        })
        request = json.loads((self.work / "prompt_jobs" / next_job["job_id"] /
                              "agent_request.json").read_text(encoding="utf-8"))
        self.assertEqual(len(request["iteration_history"]), 1)
        self.assertEqual(request["iteration_history"][0]["agent_decision"], "manual_required")
        self.assertTrue(request["human_demonstrations"])
        self.assertIn("mask_grid", request["neighbors"][0])

    def test_refine_reuses_original_prompt_and_failed_candidate(self) -> None:
        meta = frame_metadata(self.root, "cmr_middle_manifest", 0)
        first = queue_prompt_repair(self.root, self.work, {
            "pool": "cmr_middle_manifest", "index": 0, "target": "myo",
            "prompt": "删除右上方多余区域", "source_hash": meta["source_hashes"]["myo"],
        })
        first_dir = self.work / "prompt_jobs" / first["job_id"]
        local_path = first_dir / "local_request.json"
        local = json.loads(local_path.read_text(encoding="utf-8"))
        local["status"] = "READY"
        local_path.write_text(json.dumps(local), encoding="utf-8")
        (first_dir / "agent_plan.json").write_text(json.dumps({
            "plan": {"decision": "apply_operations", "operations": [], "rationale": "初次候选"}
        }), encoding="utf-8")
        original = np.asarray(nib.load(str(self.source)).get_fdata())[..., 1] > 0.5
        candidate_mask = original.copy()
        candidate_mask[0, 0] = ~candidate_mask[0, 0]
        from PIL import Image
        Image.fromarray((candidate_mask * 255).astype(np.uint8)).save(first_dir / "candidate.png")
        second = queue_prompt_repair(self.root, self.work, {
            "pool": "cmr_middle_manifest", "index": 0, "target": "myo",
            "mode": "refine", "prompt": "忽略这句新提示词",
            "source_hash": meta["source_hashes"]["myo"],
            "previous_job_id": first["job_id"],
        })
        request = json.loads((self.work / "prompt_jobs" / second["job_id"] /
                              "agent_request.json").read_text(encoding="utf-8"))
        self.assertEqual(request["mode"], "refine")
        self.assertEqual(request["operator_instruction"], "删除右上方多余区域")
        self.assertIn("上一轮候选结果不符合要求", request["iteration_history"][0]["operator_feedback"])
        self.assertEqual(request["previous_candidate"]["changed_pixels"], 1)

    def test_refine_requires_ready_candidate_without_creating_job(self) -> None:
        meta = frame_metadata(self.root, "cmr_middle_manifest", 0)
        first = queue_prompt_repair(self.root, self.work, {
            "pool": "cmr_middle_manifest", "index": 0, "target": "myo",
            "prompt": "删除右上方多余区域", "source_hash": meta["source_hashes"]["myo"],
        })
        jobs_before = set((self.work / "prompt_jobs").iterdir())
        with self.assertRaisesRegex(ValueError, "completed previous candidate"):
            queue_prompt_repair(self.root, self.work, {
                "pool": "cmr_middle_manifest", "index": 0, "target": "myo",
                "mode": "refine", "source_hash": meta["source_hashes"]["myo"],
                "previous_job_id": first["job_id"],
            })
        self.assertEqual(set((self.work / "prompt_jobs").iterdir()), jobs_before)

    def test_prompt_request_contains_cmr_qc_priors_and_compact_temporal_context(self) -> None:
        meta = frame_metadata(self.root, "cmr_middle_manifest", 0)
        queued = queue_prompt_repair(self.root, self.work, {
            "pool": "cmr_middle_manifest", "index": 0, "target": "myo",
            "prompt": "结合前后帧只补局部缺口", "source_hash": meta["source_hashes"]["myo"],
        })
        request = json.loads((self.work / "prompt_jobs" / queued["job_id"] /
                              "agent_request.json").read_text(encoding="utf-8"))
        policy = request["qc_context"]
        self.assertIn("keep_largest_component", policy["forbidden_operations"])
        self.assertIn("fill_from_adjacent_consensus", request["allowed_operations"])
        self.assertNotIn("frame_table", request)
        self.assertLessEqual(len(request["frame_window"]), 7)
        self.assertTrue(policy["bilateral_neighbor_evidence_available"])

    def test_cmr_component_cleanup_preserves_two_dominant_components(self) -> None:
        raw = np.zeros((16, 16, 3), dtype=np.uint8)
        raw[2:7, 2:7, :] = 1
        raw[9:14, 9:14, :] = 1
        raw[1, 14, :] = 1
        nib.save(nib.Nifti1Image(raw, np.eye(4)), str(self.source))
        meta = frame_metadata(self.root, "cmr_middle_manifest", 0)
        queued = queue_prompt_repair(self.root, self.work, {
            "pool": "cmr_middle_manifest", "index": 0, "target": "myo",
            "prompt": "删除无支持的小碎片，保留两个主成分",
            "source_hash": meta["source_hashes"]["myo"],
        })
        outcome = execute_prompt_repair(self.work, queued["job_id"], {
            "decision": "apply_operations", "target": "myo",
            "operations": [{"operation": "remove_components_below_area", "max_area": 4}],
            "rationale": "仅删除第三成分", "expected_effect": "保留双环", "risk_flags": [],
            "confidence": 0.9,
        })
        self.assertEqual(outcome["status"], "READY")
        candidate = np.asarray(Image.open(io.BytesIO(
            prompt_candidate_png(self.work, queued["job_id"])
        ))) > 0
        self.assertTrue(candidate[2:7, 2:7].all())
        self.assertTrue(candidate[9:14, 9:14].all())
        self.assertFalse(candidate[1, 14])

    def test_no_change_prompt_does_not_offer_a_fake_candidate(self) -> None:
        meta = frame_metadata(self.root, "cmr_middle_manifest", 0)
        queued = queue_prompt_repair(self.root, self.work, {
            "pool": "cmr_middle_manifest", "index": 0, "target": "myo",
            "prompt": "证据不足则不修改", "source_hash": meta["source_hashes"]["myo"],
        })
        execute_prompt_repair(self.work, queued["job_id"], {
            "decision": "no_change", "target": "myo", "operations": [],
            "rationale": "insufficient evidence", "expected_effect": "none",
            "risk_flags": ["visual_review_required"], "confidence": 0.7,
        })
        status = prompt_repair_status(self.work, queued["job_id"])
        self.assertEqual(status["status"], "NO_CHANGE")
        self.assertFalse(status["candidate_ready"])

    def test_adjacent_consensus_repairs_only_a_local_gap(self) -> None:
        raw = np.asarray(nib.load(str(self.source)).get_fdata()).astype(np.uint8)
        # Current frame has a small defect; both temporal neighbors retain the
        # corresponding myocardial support.
        raw[4:6, 7:9, 1] = 0
        nib.save(nib.Nifti1Image(raw, np.eye(4)), str(self.source))
        meta = frame_metadata(self.root, "cmr_middle_manifest", 0)
        queued = queue_prompt_repair(self.root, self.work, {
            "pool": "cmr_middle_manifest", "index": 0, "target": "myo",
            "prompt": "结合前后相邻帧补全上方局部缺口",
            "source_hash": meta["source_hashes"]["myo"],
        })
        outcome = execute_prompt_repair(self.work, queued["job_id"], {
            "decision": "apply_operations", "target": "myo",
            "operations": [{"operation": "fill_from_adjacent_consensus", "radius": 3}],
            "rationale": "双侧相邻帧支持局部补全", "expected_effect": "fill local gap",
            "risk_flags": ["visual_review_required"], "confidence": 0.8,
        })
        self.assertEqual(outcome["status"], "READY")
        candidate = np.asarray(Image.open(io.BytesIO(
            prompt_candidate_png(self.work, queued["job_id"])
        ))) > 0
        self.assertTrue(candidate[4:6, 7:9].all())
        self.assertEqual(int((candidate ^ (raw[..., 1] > 0)).sum()), 4)

    def test_adjacent_consensus_can_be_restricted_to_requested_region(self) -> None:
        raw = np.asarray(nib.load(str(self.source)).get_fdata()).astype(np.uint8)
        raw[4, 10, 1] = 0
        raw[11, 10, 1] = 0
        nib.save(nib.Nifti1Image(raw, np.eye(4)), str(self.source))
        meta = frame_metadata(self.root, "cmr_middle_manifest", 0)
        queued = queue_prompt_repair(self.root, self.work, {
            "pool": "cmr_middle_manifest", "index": 0, "target": "myo",
            "prompt": "结合前后帧只补右下缺口",
            "source_hash": meta["source_hashes"]["myo"],
        })
        outcome = execute_prompt_repair(self.work, queued["job_id"], {
            "decision": "apply_operations", "target": "myo",
            "operations": [{"operation": "fill_from_adjacent_consensus",
                            "radius": 3, "region": "right_lower"}],
            "rationale": "只补右下", "expected_effect": "局部补全",
            "risk_flags": [], "confidence": 0.9,
        })
        self.assertEqual(outcome["status"], "READY")
        candidate = np.asarray(Image.open(io.BytesIO(
            prompt_candidate_png(self.work, queued["job_id"])
        ))) > 0
        self.assertFalse(candidate[4, 10])
        self.assertTrue(candidate[11, 10])

    def test_polygon_operation_creates_a_reviewable_candidate(self) -> None:
        meta = frame_metadata(self.root, "cmr_middle_manifest", 0)
        queued = queue_prompt_repair(self.root, self.work, {
            "pool": "cmr_middle_manifest", "index": 0, "target": "myo",
            "prompt": "用多边形补上方一个小区域", "source_hash": meta["source_hashes"]["myo"],
        })
        outcome = execute_prompt_repair(self.work, queued["job_id"], {
            "decision": "apply_operations", "target": "myo",
            "operations": [{
                "operation": "fill_polygon",
                "points": [[470, 120], [530, 120], [530, 190], [470, 190]],
            }],
            "rationale": "局部多边形补全", "expected_effect": "fill polygon",
            "risk_flags": ["visual_review_required"], "confidence": 0.7,
        })
        self.assertEqual(outcome["status"], "READY")
        self.assertGreater(outcome["changed_pixels"], 0)


if __name__ == "__main__":
    unittest.main()
