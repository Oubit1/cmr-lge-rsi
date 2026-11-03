from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from contracts.models import FailureAnalysis
from providers.openai_compatible import OpenAICompatibleProvider


class ProviderHttpTests(unittest.TestCase):
    def test_missing_key_or_aliyun_base_url_fails_before_network(self):
        with patch.dict(os.environ, {}, clear=True):
            provider = OpenAICompatibleProvider("openai", "OPENAI_API_KEY")
            with self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY"):
                provider._credentials()
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "secret"}, clear=True):
            provider = OpenAICompatibleProvider(
                "aliyun", "DASHSCOPE_API_KEY", base_url_env="DASHSCOPE_BASE_URL"
            )
            with self.assertRaisesRegex(RuntimeError, "DASHSCOPE_BASE_URL"):
                provider._credentials()

    def test_openai_compatible_request_and_structured_validation(self):
        seen = {}
        callback_events = []
        output = {
            "failure_type": "myo_topology",
            "affected_modalities": ["lge"],
            "evidence_cases": ["case_123"],
            "hypothesis": "test",
            "suggested_experiment": "dry run",
            "expected_metrics": ["issue_recall"],
            "confidence": 0.8,
            "requires_human_review": True,
        }

        async def fake_post(endpoint, api_key, payload, timeout_seconds):
            seen.update(endpoint=endpoint, api_key=api_key, payload=payload, timeout=timeout_seconds)
            return {"id": "call-1", "choices": [{"message": {"content": json.dumps(output)}}],
                    "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}}

        with tempfile.TemporaryDirectory() as tmp:
            telemetry = os.path.join(tmp, "calls.jsonl")
            with patch.dict(os.environ, {"TEST_PROVIDER_KEY": "test-secret",
                                         "CMR_LGE_AGENT_CALL_LOG": telemetry}), patch.object(
                OpenAICompatibleProvider, "_post", staticmethod(fake_post)
            ):
                provider = OpenAICompatibleProvider(
                    "test", "TEST_PROVIDER_KEY", base_url="https://provider.example/v1", json_mode=True
                )
                result = asyncio.run(provider.generate_structured(
                    system_prompt="test", input_data={
                        "anonymous": True, "role": "failure_triage",
                        "_invocation_id": "trace-1",
                        "_call_event_callback": callback_events.append,
                    },
                    output_schema=FailureAnalysis, model="test-model", timeout_seconds=5,
                ))
            with open(telemetry, encoding="utf-8") as handle:
                event = json.loads(handle.read())
            self.assertEqual(event["status"], "SUCCEEDED")
            self.assertEqual(event["role"], "failure_triage")
            self.assertEqual(event["total_tokens"], 18)
            self.assertEqual(event["invocation_id"], "trace-1")
            self.assertEqual(callback_events[0]["request_id"], "call-1")
        self.assertEqual(result.failure_type, "myo_topology")
        self.assertEqual(seen["endpoint"], "https://provider.example/v1/chat/completions")
        self.assertEqual(seen["api_key"], "test-secret")
        self.assertEqual(seen["payload"]["response_format"], {"type": "json_object"})

    def test_aliyun_structured_calls_disable_thinking_by_default(self):
        seen = []
        output = {
            "failure_type": "myo_topology", "affected_modalities": ["cmr"],
            "evidence_cases": [], "hypothesis": "test", "suggested_experiment": "shadow",
            "expected_metrics": [], "confidence": 0.8, "requires_human_review": False,
        }

        async def fake_post(endpoint, api_key, payload, timeout_seconds):
            seen.append(payload)
            return {"choices": [{"message": {"content": json.dumps(output)}}]}

        with tempfile.TemporaryDirectory() as tmp:
            env = {"DASHSCOPE_API_KEY": "secret", "DASHSCOPE_BASE_URL": "https://example/v1",
                   "CMR_LGE_AGENT_CALL_LOG": os.path.join(tmp, "calls.jsonl")}
            with patch.dict(os.environ, env, clear=True), patch.object(
                OpenAICompatibleProvider, "_post", staticmethod(fake_post)
            ):
                provider = OpenAICompatibleProvider(
                    "aliyun", "DASHSCOPE_API_KEY", base_url_env="DASHSCOPE_BASE_URL"
                )
                asyncio.run(provider.generate_structured(
                    system_prompt="test", input_data={"role": "rule_curator"},
                    output_schema=FailureAnalysis, model="deepseek-v4.1-flash",
                ))
                asyncio.run(provider.generate_structured(
                    system_prompt="test",
                    input_data={"role": "rule_curator", "_enable_thinking": True},
                    output_schema=FailureAnalysis, model="deepseek-v4.1-flash",
                ))
        self.assertIs(seen[0]["enable_thinking"], False)
        self.assertIs(seen[1]["enable_thinking"], True)

    def test_transport_error_falls_back_once_and_keeps_requested_model_auditable(self):
        seen_models = []
        output = {
            "failure_type": "myo_topology",
            "affected_modalities": ["cmr"],
            "evidence_cases": ["case_123"],
            "hypothesis": "test",
            "suggested_experiment": "dry run",
            "expected_metrics": ["issue_recall"],
            "confidence": 0.8,
            "requires_human_review": False,
        }

        async def fake_post(endpoint, api_key, payload, timeout_seconds):
            seen_models.append(payload["model"])
            if payload["model"] == "specialist-model":
                raise TimeoutError("specialist timeout")
            return {"id": "fallback-1",
                    "choices": [{"message": {"content": json.dumps(output)}}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}}

        with tempfile.TemporaryDirectory() as tmp:
            telemetry = os.path.join(tmp, "calls.jsonl")
            env = {"TEST_PROVIDER_KEY": "test-secret",
                   "ALIYUN_MODEL": "base-model",
                   "CMR_LGE_AGENT_CALL_LOG": telemetry}
            with patch.dict(os.environ, env, clear=True), patch.object(
                OpenAICompatibleProvider, "_post", staticmethod(fake_post)
            ):
                provider = OpenAICompatibleProvider(
                    "test", "TEST_PROVIDER_KEY", base_url="https://provider.example/v1"
                )
                result = asyncio.run(provider.generate_structured(
                    system_prompt="test",
                    input_data={"role": "cmr_specialist", "_max_output_tokens": 77},
                    output_schema=FailureAnalysis,
                    model="specialist-model",
                    timeout_seconds=5,
                ))
            with open(telemetry, encoding="utf-8") as handle:
                events = [json.loads(line) for line in handle]

        self.assertEqual(result.failure_type, "myo_topology")
        self.assertEqual(seen_models, ["specialist-model", "base-model"])
        self.assertEqual([row["status"] for row in events], ["ERROR", "SUCCEEDED"])
        self.assertTrue(all(row["requested_model"] == "specialist-model" for row in events))
        self.assertIn("runtime fallback", events[-1]["fallback_reason"])


if __name__ == "__main__":
    unittest.main()
