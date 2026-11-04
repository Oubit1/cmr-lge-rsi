from __future__ import annotations

import asyncio
import json
import os
import signal
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from contracts.models import FailureAnalysis
from providers.codex_cli import CodexCliProvider
from providers.registry import ProviderRegistry


class CodexCliProviderTests(unittest.TestCase):
    def test_structured_invocation_args_schema_and_telemetry(self):
        output = {"failure_type": "myo_topology", "affected_modalities": ["lge"],
                  "evidence_cases": [], "hypothesis": "h", "suggested_experiment": "e",
                  "expected_metrics": [], "confidence": .8, "requires_human_review": True}
        seen = {}

        class Proc:
            returncode = 0
            async def communicate(self, data):
                seen["prompt"] = data.decode()
                Path(seen["args"][seen["args"].index("--output-last-message") + 1]).write_text(
                    json.dumps({"response_json": json.dumps(output)}), encoding="utf-8")
                return (b'{"type":"turn.completed","usage":{"input_tokens":9,"output_tokens":4}}\n', b"")
            async def wait(self): pass

        class AuthProc:
            returncode = 0
            async def communicate(self, data=None): return (b"Logged in using ChatGPT", b"")
            async def wait(self): pass

        calls = []
        async def spawn(*args, **kwargs):
            if len(args) > 1 and args[1:3] == ("login", "status"):
                calls.append(args)
                return AuthProc()
            seen.update(args=args, kwargs=kwargs)
            return Proc()

        events = []
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"CODEX_CLI_PATH": "fake-codex", "OPENAI_API_KEY": "must-not-leak", "CMR_LGE_AGENT_CALL_LOG": str(Path(tmp)/"calls.jsonl")}), \
             patch("providers.codex_cli.asyncio.create_subprocess_exec", spawn):
            result = asyncio.run(CodexCliProvider(reasoning_effort="low").generate_structured(
                system_prompt="system intact", input_data={"value": 12, "_invocation_id": "trace",
                "_request_budget_callback": lambda: None, "_call_event_callback": events.append},
                output_schema=FailureAnalysis, model="gpt-6.1-sol", timeout_seconds=3))
        self.assertEqual(result.failure_type, "myo_topology")
        args = seen["args"]
        self.assertIn("--json", args)
        self.assertIn("--ignore-user-config", args)
        self.assertIn("--ignore-rules", args)
        self.assertIn("--skip-git-repo-check", args)
        for config in ('features.shell_tool=false', 'features.unified_exec=false',
                       'features.multi_agent=false', 'web_search="disabled"',
                       'model_reasoning_effort="low"'):
            self.assertIn(config, args)
        self.assertIn("--sandbox", args)
        self.assertEqual(args[args.index("--sandbox") + 1], "read-only")
        self.assertIn("--output-schema", args)
        self.assertIn("system intact", seen["prompt"])
        self.assertIn("Required result schema", seen["prompt"])
        self.assertEqual(events[0]["total_tokens"], 13)
        self.assertEqual(events[0]["status"], "SUCCEEDED")
        self.assertEqual(calls[0][1:], ("login", "status"))
        self.assertNotIn("OPENAI_API_KEY", seen["kwargs"]["env"])
        self.assertNotIn("OPENAI_API_KEY", calls and seen["kwargs"]["env"])

    def test_registry_selects_codex_adapter(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "providers.yaml"
            config.write_text("providers:\n  subscription:\n    adapter: codex_cli\n", encoding="utf-8")
            self.assertIsInstance(ProviderRegistry(config).get("subscription"), CodexCliProvider)

    def test_registry_loads_scalar_reasoning_effort(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "providers.yaml"
            config.write_text("providers:\n  codex_routine:\n    adapter: codex_cli\n    reasoning_effort: low\n", encoding="utf-8")
            provider = ProviderRegistry(config).get("codex_routine")
            self.assertEqual(provider.reasoning_effort, "low")

    def _run_with_spawns(self, side_effect, **kwargs):
        async def spawn(*args, **spawn_kwargs):
            if args[1:3] == ("login", "status"):
                return side_effect("auth", args, spawn_kwargs)
            return side_effect("model", args, spawn_kwargs)
        env = {"CODEX_CLI_PATH": "fake-codex", **kwargs.pop("env", {})}
        with patch.dict(os.environ, env), patch("providers.codex_cli.asyncio.create_subprocess_exec", spawn):
            return asyncio.run(CodexCliProvider(reasoning_effort="low").generate_structured(
                system_prompt="s", input_data={"_request_budget_callback": kwargs.pop("budget", lambda: None)},
                output_schema=FailureAnalysis, model="test-model", timeout_seconds=kwargs.pop("timeout", 1)))

    def test_rejects_non_chatgpt_login_without_model_spawn(self):
        spawned = []
        class Proc:
            returncode = 0
            async def communicate(self, data=None): return b"Logged in using API key", None
        def fake(kind, args, spawn_kwargs):
            spawned.append(kind)
            return Proc()
        with self.assertRaisesRegex(RuntimeError, "ChatGPT account"):
            self._run_with_spawns(fake)
        self.assertEqual(spawned, ["auth"])

    def test_schema_validation_failure_is_not_reported_as_success(self):
        events = []
        class Auth:
            returncode = 0
            async def communicate(self, data=None): return b"Logged in using ChatGPT", None
            def kill(self): pass
            async def wait(self): self.returncode = -9
        class Model:
            returncode = 0
            async def communicate(self, data):
                Path(self.path).write_text("{}", encoding="utf-8")
                return b"", b""
        async def spawn(*args, **kw):
            if args[1:3] == ("login", "status"): return Auth()
            proc = Model(); proc.path = args[args.index("--output-last-message") + 1]; return proc
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"CODEX_CLI_PATH":"fake-codex", "CMR_LGE_AGENT_CALL_LOG":str(Path(tmp)/"calls.jsonl")}), patch("providers.codex_cli.asyncio.create_subprocess_exec", spawn):
            with self.assertRaisesRegex(RuntimeError, "invalid structured output"):
                asyncio.run(CodexCliProvider().generate_structured(system_prompt="s", input_data={"_request_budget_callback":lambda:None,"_call_event_callback":events.append}, output_schema=FailureAnalysis, model="m"))
        self.assertEqual(events[-1]["status"], "ERROR")

    def test_budget_rejection_does_not_spawn_model(self):
        kinds = []
        class Auth:
            returncode = 0
            async def communicate(self, data=None): return b"Logged in using ChatGPT", None
        async def spawn(*args, **kw):
            kinds.append("auth" if args[1:3] == ("login", "status") else "model")
            return Auth()
        with patch.dict(os.environ, {"CODEX_CLI_PATH":"fake-codex"}), patch("providers.codex_cli.asyncio.create_subprocess_exec", spawn):
            with self.assertRaisesRegex(RuntimeError, "budget"):
                asyncio.run(CodexCliProvider().generate_structured(system_prompt="s", input_data={"_request_budget_callback":lambda: (_ for _ in ()).throw(RuntimeError("budget denied"))}, output_schema=FailureAnalysis, model="m"))
        self.assertEqual(kinds, ["auth"])

    def test_model_timeout_kills_process(self):
        spawned = []
        class Auth:
            returncode = 0
            async def communicate(self, data=None): return b"Logged in using ChatGPT", None
        class Model:
            pid = 123
            returncode = None
            async def communicate(self, data): raise asyncio.TimeoutError
            async def wait(self): self.returncode = -9
        async def spawn(*args, **kw):
            if args[1:3] == ("login", "status"): return Auth()
            proc = Model(); spawned.append(proc); return proc
        patches = [patch.dict(os.environ, {"CODEX_CLI_PATH":"fake-codex"}),
                   patch("providers.codex_cli.asyncio.create_subprocess_exec", spawn)]
        if os.name == "nt":
            terminator = patch("providers.codex_cli.subprocess.run")
        else:
            terminator = patch("providers.codex_cli.os.killpg")
        with patches[0], patches[1], terminator as kill:
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                asyncio.run(CodexCliProvider().generate_structured(system_prompt="s", input_data={"_request_budget_callback":lambda:None}, output_schema=FailureAnalysis, model="m", timeout_seconds=1))
        if os.name == "nt":
            kill.assert_called_once_with(["taskkill", "/PID", "123", "/T", "/F"], capture_output=True)
        else:
            kill.assert_called_once_with(123, signal.SIGKILL)
        self.assertEqual(spawned[0].returncode, -9)


if __name__ == "__main__":
    unittest.main()
