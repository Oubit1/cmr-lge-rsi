import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from app.rsi_kernel import canonical_hash
from app.segmentation_evidence import collect_evidence
from app.segmentation_status import status


class AcceptanceEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.dataset = self.root / "data"
        self.dataset.mkdir()

    def write(self, relative, content):
        p = self.root / relative
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(content), encoding="utf-8")

    def transaction(self, stage, myo="a", mi="b"):
        txid = stage + "-transaction"
        hashes = {"image": "i", "label": myo if stage == "lge_myo" else mi,
                  "mi" if stage == "lge_myo" else "myo": mi if stage == "lge_myo" else myo}
        receipt = {"transaction_id": txid, "case_id": "synthetic-case", "status": "CHECKED",
                   "after_hashes": hashes, "score_after": 80, "calibration_id": "version-a"}
        self.write("runtime/lge_execution/transactions/" + txid + ".json",
                   {"transaction_id": txid, "state": "COMMITTED", "dataset": str(self.dataset),
                    "stage": stage, "relative": "synthetic/1.nii.gz", "after_sha256": hashes["label"], "receipt": receipt})
        path = "runtime/lge_execution/receipts/" + stage + "/" + canonical_hash("synthetic-case") + ".json"
        self.write(path, receipt)
        return path

    def test_reconciles_joint_myo_mi_without_invalidating_myo(self):
        self.transaction("lge_myo", mi="old")
        self.transaction("lge_mi", mi="new")
        rows, _ = collect_evidence(self.root, self.dataset)
        myo = next(r for r in rows if r["target"] == "lge_myo")
        self.assertEqual(myo["hashes"]["mi"], "new")
        self.assertEqual(myo["provenance"]["paired_mi_record_id"], "lge_mi-transaction")

    def test_later_rejection_invalidates_old_journal(self):
        path = self.transaction("lge_myo")
        self.write(path, {"status": "NEEDS_REVIEW"})
        rows, meta = collect_evidence(self.root, self.dataset)
        self.assertEqual(rows, [])
        self.assertEqual(meta["evidence_debt"]["superseded_lge_receipt"], 1)

    def test_legacy_cmr_not_fabricated_into_current_image_acceptance(self):
        self.write("runtime/check_iterations/auto_apply.jsonl", {"event": "applied", "key": "synthetic/1.nii.gz",
                   "route": "checked", "source_sha_after": "label"})
        rows, meta = collect_evidence(self.root, self.dataset)
        self.assertEqual(rows, [])
        self.assertEqual(meta["evidence_debt"]["cmr_legacy_image_hash_missing"], 1)

    def test_later_ledger_rejection_not_hidden_by_old_acceptance(self):
        self.transaction("lge_myo")
        self.write("runtime/dataset_acceptance/receipts.jsonl", {"dataset": str(self.dataset),
             "key": "synthetic/1.nii.gz", "target": "lge_myo", "status": "REVOKED"})
        rows, _ = collect_evidence(self.root, self.dataset)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "REVOKED")

    def test_board_has_no_patient_rows(self):
        self.write("runtime/dataset_release/summary.json", {"status": "AUDITED", "rows": [{"patient": "private"}],
                  "summary": {"complete_pairs": 3}})
        self.assertNotIn("private", json.dumps(status(self.root)))
        self.assertEqual(status(self.root)["summary"]["complete_pairs"], 3)

    def test_latest_negative_decision_overrides_acceptance_and_is_not_counted_accepted(self):
        for state in ("NEEDS_REVIEW", "PENDING", "ERROR", "INELIGIBLE"):
            with self.subTest(state=state):
                self.transaction("lge_myo")
                path = self.root / "runtime/dataset_acceptance/receipts.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                base = {"dataset": str(self.dataset), "key": "synthetic/1.nii.gz", "target": "lge_myo"}
                path.write_text("\n".join(json.dumps(dict(base, status=s))
                                          for s in ("ACCEPTED", state)) + "\n", encoding="utf-8")
                rows, meta = collect_evidence(self.root, self.dataset)
                self.assertEqual([r["status"] for r in rows], [state])
                self.assertEqual(meta["normalized_acceptances"], 0)
                self.assertEqual(meta["normalized_decisions"], 1)

    def test_first_audit_creates_runtime_output(self):
        from scripts.dataset_release import run
        with patch("scripts.dataset_release.collect_evidence", return_value=([], {})), \
             patch("scripts.dataset_release.frozen_patients", return_value=set()), \
             patch("scripts.dataset_release.audit_pairs", return_value={"summary": {"releasable_pairs": 0}}):
            result = run(self.root, self.dataset)
        self.assertEqual(result["status"], "AUDITED")
        self.assertTrue((self.root / "runtime/dataset_release/summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
