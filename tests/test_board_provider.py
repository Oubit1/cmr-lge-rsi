from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.board_provider import subscription_active
from scripts import board_server


class BoardProviderRoutingTests(unittest.TestCase):
    def test_subscription_role_selection_reads_active_profiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "configs").mkdir()
            (root / "configs/agents.aliyun-roles.yaml").write_text(
                "agents:\n  workflow_maintainer:\n    provider: sub\n  efficiency_auditor:\n    provider: sub\n", encoding="utf-8")
            (root / "configs/providers.yaml").write_text(
                "providers:\n  sub:\n    adapter: codex_cli\n", encoding="utf-8")
            self.assertTrue(subscription_active("workflow_maintainer", root))
            self.assertTrue(subscription_active("efficiency_auditor", root))

    def test_missing_profiles_mean_subscription_inactive(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(subscription_active("workflow_maintainer", Path(tmp)))

    def test_malformed_profile_is_not_silently_treated_as_aliyun(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "configs").mkdir()
            (root / "configs/agents.aliyun-roles.yaml").write_text("agents: [broken", encoding="utf-8")
            (root / "configs/providers.yaml").write_text("providers: {}", encoding="utf-8")
            with self.assertRaises(Exception):
                subscription_active("workflow_maintainer", root)

    def test_auto_translation_and_brief_do_not_call_subscription_model(self):
        with patch("app.board_provider.subscription_active", return_value=True), \
             patch("urllib.request.urlopen", side_effect=AssertionError("must not call provider")), \
             patch.object(board_server, "_trans_loaded", True), \
             patch.object(board_server, "_trans_cache", {}):
            self.assertEqual(board_server.translate_texts(["unchanged source"]), ["unchanged source"])
            self.assertEqual(board_server._llm_brief({"attempts": 1}), (None, None))

    def test_maintenance_subscription_route_ignores_aliyun_credentials(self):
        result = {"choices": [{"message": {"content": "订阅回复"}}], "model": "configured"}
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(board_server, "MAINTENANCE_CHAT_AUDIT", Path(tmp)/"chat.jsonl"), \
             patch("app.board_provider.subscription_active", return_value=True), \
             patch("app.board_provider.generate_reply", return_value=result) as call, \
             patch.dict(os.environ, {}, clear=True), \
             patch.object(board_server, "maintenance_chat_context", return_value={}):
            reply = board_server.maintenance_chat([{"role": "user", "content": "status?"}], snapshot={})
        self.assertEqual(reply["reply"], "订阅回复")
        self.assertEqual(reply["model"], "configured")
        self.assertIn("worked_for", reply)
        self.assertIn("boundary", reply)
        self.assertEqual(call.call_args.args[0], "workflow_maintainer")


if __name__ == "__main__":
    unittest.main()
