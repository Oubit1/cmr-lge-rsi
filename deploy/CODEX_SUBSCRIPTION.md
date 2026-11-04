# Official Codex CLI subscription provider

The `codex_cli` adapter runs the locally authenticated official Codex CLI (`codex exec`). It does not use an API key or fall back to an API provider. Select it explicitly in a provider configuration:

```yaml
providers:
  codex_subscription:
    adapter: codex_cli
    binary_env: CODEX_CLI_PATH # optional; defaults to codex on PATH
```

Choose an available model through the official `model/list` endpoint before activation. Example role profiles are in `configs/agents.codex-subscription.yaml` and `configs/providers.codex-subscription.yaml`. They use Luna for routine roles, Sol for specialists and implementation, and Astra for independent review. Availability is account-dependent.

The adapter uses a strict `response_json` string envelope with `--output-schema`, decodes the contained domain result and validates it against the original Pydantic contract. This accommodates existing optional fields and open dictionaries without changing clinical contracts. The original schema, system prompt and public role input are passed to the model.

Each call uses an ephemeral read-only temporary directory, ignores user configuration/rules, disables shell, execution, subagent and web tools, and removes API-key environment variables. Authentication must report ChatGPT. It does not copy, extract or expose account tokens. Role history, approved rules, casebook and Skill context continue to come from the project's persisted handoffs, not the CLI's conversation memory.

The provider records the existing metadata-only call event, including invocation trace, elapsed time, success/error, and token counts when emitted in Codex JSONL events. It never records prompt contents or CLI stderr. Timeout kills the CLI process (and process group on POSIX). Structured output is checked against the requested schema. This adapter has no automatic API, legacy-model, or cross-provider fallback.

Offline coverage is in `tests/test_codex_cli_provider.py`; it mocks process creation and does not invoke a model.

Budgets count CLI invocations; the official CLI may perform internal transport retries. Token usage comes from completed-turn events. A successful invocation or context acknowledgement is not evidence of better labels: run the existing independent benchmark, implementation review and audited application chain before claiming clinical progress. Preserve a versioned accepted dataset for doctor review separately from any migration trial.

Official references: [subscription authentication](https://learn.chatgpt.com/docs/auth?surface=cli), [non-interactive execution](https://learn.chatgpt.com/docs/non-interactive-mode), [model availability](https://learn.chatgpt.com/docs/app-server). Eligible Pro subscribers may separately claim [API credits](https://developers.openai.com/api/docs/guides/chatgpt-api-credits); that uses a project API key and is a different billing route from this adapter.

## Bounded migration validation (2026-11-04)

An actual production trial completed eight model invocations, including an owner repair and independent implementation review. A failed full-regression result now returns to the original approved owner when attempts remain. Exhausted attempt counts still require an explicit, audited extension; automatic recovery does not increase the budget or skip tests. Synthetic provider tests use isolated telemetry files.

The promoted generator reduced defect false acceptance from 2/3 to 1/3 on the small frozen LGE Myo challenge, with clean acceptance unchanged. These challenge results do not establish clinical generalization. The subsequent bounded ten-item application trial produced **zero accepted items and zero label writes**: seven unsupported snapshot shapes, two paired snapshot geometry mismatches, and one candidate without scored improvement. A promoted implementation is therefore not evidence that production annotation progress or full autonomous RSI has passed acceptance.

The synchronized source passed 2,290 offline regression tests (one skipped). Keep the existing immutable automatic-QC dataset distinct from newly corrected output, and collect physician verdicts separately. Reusing previously accepted data must not be presented as new improvement from the migrated models.
