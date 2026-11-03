from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agents.governance import implementation_review_approved
from agents.implementation_review import review_implementation
from app.database import Database
from contracts.compat import model_validate
from providers.mock import MockProvider


PROJECT = Path(__file__).resolve().parents[1]


class AgentHandoffTests(unittest.TestCase):
    def test_owner_result_requires_persisted_post_implementation_review(self):
        class AcceptingReviewer(MockProvider):
            async def generate_structured(self, **kwargs):
                if kwargs["output_schema"].__name__ == "ReviewDecision":
                    return model_validate(kwargs["output_schema"], {
                        "decision": "accept_experiment",
                        "reasons": ["isolated tests and deterministic gate passed"],
                        "required_validation_metrics": ["false_repair_rate"],
                        "risk_flags": [], "confidence": 0.9,
                    })
                return await super().generate_structured(**kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            dataset.mkdir()
            db = Database(root / "workflow.sqlite")
            db.initialize()
            run_id = db.create_run(db.register_dataset("demo", dataset), "analysis", {})
            action = {
                "action_id": "A1", "owner_role": "rule_curator",
                "action_class": "rule_draft", "objective": "bounded rule",
                "acceptance_metrics": ["false repair"], "priority": 1,
                "max_attempts": 1, "auto_execute": True, "review_approved": True,
            }
            with patch(
                "agents.implementation_review.ProviderRegistry.get",
                return_value=AcceptingReviewer(),
            ):
                decision = review_implementation(
                    db=db, run_id=run_id, action=action,
                    implementation={"changed_files": ["experiments/a.py"],
                                    "tests_run": ["tests.test_a"]},
                    validation={"passed": True, "checks": ["compile", "unit"]},
                    providers_path=PROJECT / "configs/providers.yaml",
                    agents_path=PROJECT / "configs/agents.yaml",
                )
            self.assertEqual(decision.decision, "accept_experiment")
            # Accepting an experiment while still requesting false-repair
            # evidence is not authorization to promote it.
            self.assertFalse(implementation_review_approved(db, run_id, "A1", "rule_curator"))
            reviewer_inbox = db.agent_inbox(
                run_id, "independent_reviewer", thread_id=f"run-{run_id}:A1"
            )
            owner_inbox = db.agent_inbox(run_id, "rule_curator", thread_id=f"run-{run_id}:A1")
            self.assertEqual(reviewer_inbox[-1]["message_type"], "implementation_result")
            self.assertEqual(owner_inbox[-1]["message_type"], "implementation_review")
            self.assertEqual(owner_inbox[-1]["reply_to"], reviewer_inbox[-1]["message_id"])


if __name__ == "__main__":
    unittest.main()
