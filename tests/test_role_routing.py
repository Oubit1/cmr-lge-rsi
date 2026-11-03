from __future__ import annotations

import unittest

from app.role_routing import (
    ROLE_RESPONSIBILITIES,
    assert_role_owns,
    role_for_label_task,
    validate_agent_config,
)


class RoleRoutingTests(unittest.TestCase):
    def test_label_tasks_have_distinct_specialist_owners(self) -> None:
        self.assertEqual(role_for_label_task({"modality": "cmr", "target": "myo"}),
                         "cmr_specialist")
        self.assertEqual(role_for_label_task({"modality": "lge", "target": "myo"}),
                         "lge_myo_specialist")
        self.assertEqual(role_for_label_task({"modality": "lge", "target": "mi"}),
                         "lge_mi_specialist")

    def test_invalid_cross_modality_target_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            role_for_label_task({"modality": "cmr", "target": "mi"})

    def test_role_cannot_absorb_another_roles_work(self) -> None:
        with self.assertRaises(RuntimeError):
            assert_role_owns("qc_code_optimizer", "cmr_prompt_repair")

    def test_profile_requires_all_read_only_roles(self) -> None:
        profile = {role: {"provider": "mock", "model": role, "can_write": False}
                   for role in ROLE_RESPONSIBILITIES}
        validate_agent_config(profile)
        profile["cmr_specialist"]["can_write"] = True
        with self.assertRaises(RuntimeError):
            validate_agent_config(profile)


if __name__ == "__main__":
    unittest.main()
