from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.check_inventory import build_check_inventory


class CheckInventoryTests(unittest.TestCase):
    def test_reports_folder_state_and_keeps_shadow_signal_non_authoritative(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            runtime = Path(tmp) / "runtime"
            check = root / "labels/check"
            for folder, name in (("待筛", "pending"), ("checked", "accepted"),
                                 ("垃圾桶", "discarded")):
                path = check / folder / "cmr/PATIENT" / f"{name}.png"
                path.parent.mkdir(parents=True)
                path.write_bytes(b"synthetic")
            run = runtime / "check_iterations/run1"
            run.mkdir(parents=True)
            (runtime / "check_iterations/latest.json").write_text(json.dumps({
                "report_path": str(run / "evaluation.json"), "created_at": "test",
                "pending": 1, "work": 0, "checked": 1, "trash": 1,
            }))
            (run / "evaluation.json").write_text(json.dumps({"records": [
                {"key": "PATIENT/pending.nii.gz", "frames": [1], "quality_delta": 0.4},
            ]}))
            (run / "spatial_boundary_shadow.json").write_text(json.dumps({
                "control_gate": "FAIL", "records": [
                    {"state": "待筛", "key": "PATIENT/pending.nii.gz",
                     "decision": "spatial_boundary_candidate"},
                    {"state": "checked", "key": "PATIENT/accepted.nii.gz",
                     "decision": "unresolved"},
                ]}))
            (runtime / "promotion").mkdir()
            (runtime / "promotion/promotion_log.jsonl").write_text(json.dumps({
                "key": "PATIENT/accepted.nii.gz", "action": "direct_promoted",
            }) + "\n")
            result = build_check_inventory(root, runtime)
            self.assertEqual(result["counts"], {
                "pending": 1, "work": 0, "checked": 1, "trash": 1,
            })
            self.assertTrue(result["shadow_current"])
            self.assertEqual(result["shadow_control_gate"], "FAIL")
            pending = next(row for row in result["rows"] if row["state"] == "pending")
            self.assertTrue(pending["shadow_boundary"])
            self.assertEqual(pending["candidate_frames"], 1)
            checked = next(row for row in result["rows"] if row["state"] == "checked")
            self.assertEqual(checked["source"], "自动直通晋升")
            trash = next(row for row in result["rows"] if row["state"] == "trash")
            self.assertIn("未单独核验", trash["source"])
            audit = runtime / "check_iterations/auto_apply.jsonl"
            audit.write_text(json.dumps({"event": "applied", "key": "PATIENT/accepted.nii.gz",
                                         "route": "checked", "timestamp": "test"}) + "\n")
            attributed = build_check_inventory(root, runtime)
            self.assertEqual(attributed["auto_apply"], {"applied": 1, "last_at": "test"})
            checked = next(row for row in attributed["rows"] if row["state"] == "checked")
            self.assertIn("自动筛选／评分晋升", checked["source"])

            new_item = check / "待筛/cmr/PATIENT/new.png"
            new_item.write_bytes(b"synthetic")
            stale = build_check_inventory(root, runtime)
            self.assertFalse(stale["shadow_current"])
            self.assertIsNone(next(row for row in stale["rows"]
                                   if row["key"] == "PATIENT/pending.nii.gz")["shadow_boundary"])


if __name__ == "__main__":
    unittest.main()
