from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.database import Database
from app.orchestrator import WorkflowOrchestrator


class OrchestratorTests(unittest.TestCase):
    def test_audit_command_is_non_mutating_and_skips_annotation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            dataset.mkdir()
            skill = root / "skill"
            (skill / "scripts").mkdir(parents=True)
            (skill / "scripts/run_full_qc_pipeline.py").write_text("", encoding="utf-8")
            python = root / "python"
            python.write_text("", encoding="utf-8")
            db = Database(root / "workflow.sqlite")
            db.initialize()
            orch = WorkflowOrchestrator(db, root, skill, python)
            run_id = orch.submit(dataset_root=dataset, raw_input=None, execute_mode="audit",
                                 device="auto", workers=4, run_annotation=False)
            command = orch.build_pipeline_command(db.get_run(run_id))
            self.assertIn("--audit-only", command)
            self.assertIn("--skip-annotation", command)


if __name__ == "__main__":
    unittest.main()
