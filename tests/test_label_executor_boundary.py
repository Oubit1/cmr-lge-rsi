from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.label_executor_client import LabelExecutorUnavailable, request_label_apply


ROOT = Path(__file__).resolve().parents[1]


class LabelExecutorBoundaryTests(unittest.TestCase):
    def test_board_uses_operator_confirmed_local_apply_with_audit(self):
        source = (ROOT / "scripts/board_server.py").read_text(encoding="utf-8")
        self.assertIn("apply_staged_edit_local(", source)
        self.assertIn("result = apply_edit(dataset_root, work_root, edit_id,", source)
        self.assertIn('audit("requested")', source)
        self.assertIn('audit("applied",', source)
        self.assertIn('if args.host not in {"127.0.0.1", "::1", "localhost"}', source)

    def test_missing_executor_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            work = root / "work"
            dataset.mkdir()
            work.mkdir()
            with self.assertRaises(LabelExecutorUnavailable):
                request_label_apply(
                    dataset_root=dataset, work_root=work, edit_id="edit-1",
                    reviewer="operator", confirmation="APPLY",
                    socket_path=root / "missing.sock",
                )


if __name__ == "__main__":
    unittest.main()
