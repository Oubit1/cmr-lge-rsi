from __future__ import annotations

import unittest

from experiments.llm_maintenance import resolve_provider_role


class AgentRoleRoutingTests(unittest.TestCase):
    def test_explicit_lge_alias(self):
        agents = {"lge_myo_specialist": {}, "qc_code_optimizer": {}}
        self.assertEqual(resolve_provider_role(agents, "lge_specialist"),
                         "lge_myo_specialist")

    def test_unknown_owner_fails_instead_of_silent_optimizer_fallback(self):
        with self.assertRaisesRegex(ValueError, "unconfigured maintenance owner"):
            resolve_provider_role({"qc_code_optimizer": {}}, "invented_role")

    def test_known_role_stays_its_own_role(self):
        agents = {"cmr_specialist": {}, "qc_code_optimizer": {}}
        self.assertEqual(resolve_provider_role(agents, "cmr_specialist"),
                         "cmr_specialist")


if __name__ == "__main__":
    unittest.main()
