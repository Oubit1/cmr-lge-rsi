# Multi-agent governance

Use the server orchestrator for residual failures, reusable QC improvements and
long-running automation. The deterministic skill remains the only production
label executor.

## Roles

- `supervisor`: routes work, limits attempts, defines acceptance metrics and stops low-value loops.
- `failure_triage`: identifies the dominant residual failure class.
- `cmr_specialist`, `lge_myo_specialist`, `lge_mi_specialist`: modality evidence review.
- `qc_code_optimizer`: proposes bounded implementation experiments.
- `rule_curator`: consolidates repeated cases into narrow rules and contraindications.
- `skill_maintainer`: proposes concise routing, tool and instruction updates.
- `efficiency_auditor`: evaluates runtime, compute, model calls and manual escalation.
- `independent_reviewer`: can reject or escalate the supervisor plan.

Agents exchange validated JSON and do not share a writable production workspace.
Approved maintenance is implemented by the role-configured provider model in a disposable snapshot that
excludes datasets, labels, images, credentials, weights and vendor code.

The exchange must be a persisted, role-addressed message thread. A runnable action requires, in order:

1. a Supervisor `assigned_action` message to the action's named owner;
2. an Independent Reviewer `action_review=accept_experiment` message;
3. implementation by that owner role only, with snapshot-write capability but no production-write capability;
4. deterministic validation in the disposable snapshot;
5. an owner `implementation_result` message and a second Independent Reviewer `implementation_review` approval.

The latest reviewer message is authoritative. A later rejection revokes an earlier approval. CLI, daemon and
special-purpose dispatch scripts must enforce the same thread; no standalone JSON file, framework maintainer,
Codex session or fallback role may synthesize the missing approval or implement a domain action on the owner's
behalf. Ownership is fixed: `isolated_code_experiment -> qc_code_optimizer`, `rule_draft -> rule_curator`, and
`skill_draft -> skill_maintainer`. Framework maintainers may repair this control plane and its tests only.
An action class without an explicitly registered executor (for example `known_tool`) must become
`HUMAN_REQUIRED` before any provider call. Never route it through a snapshot-maintenance role merely because
`auto_execute=true`; adding an executor is a separate control-plane change with its own tests and permissions.

Every provider dispatch must write an invocation record before network I/O. The record contains the role-addressed
thread, requested model, exact de-identified public input and input hash. Provider attempts append actual model,
fallback reason, request ID, status, latency and token counts under the same invocation ID. A prompt, handoff file or
model claim is not a substitute for this trace.

For hybrid reasoning models used for schema-bound maintenance output, configure the provider for deterministic
structured output (including disabling optional thinking when the provider requires that). A schema failure consumes
one bounded owner attempt; it must not cause a framework maintainer to produce the missing domain result.

Repository `AGENTS.md` defines the framework-maintainer role but is only a model instruction. Production label paths
must be owned by a separate Unix identity. The board stages candidates and talks to a narrow Unix-socket executor;
it must not import or call the production writer. If that executor is absent, apply fails closed while staging remains
available. Never claim hard isolation while Codex, Autopilot and the label writer share the same writable identity.

Provider health is a hard prerequisite. An inaccessible model is shown as unhealthy
and temporarily falls back to the configured base model so one bad model ID cannot
stall the full loop. Deterministic safety rejections and timed-out maintenance calls
are bounded and persisted; they must not be retried forever.

## Evaluation and stopping

Clinical gates take precedence: no new critical QC issues, false-repair rate at
or below 1%, Myo Dice drop at most 0.002 and MI Dice drop at most 0.003. A useful
candidate must improve target issue recall by at least 0.05. Also record unresolved
middle-slice reduction, runtime, GPU hours, LLM calls and manual-escalation rate.

Each supervisor action has one owner, explicit metrics and at most three attempts.
Stop and retain the unresolved check set when metrics do not improve, safety
regresses, evidence is ambiguous, or required data are missing. Promotion never
means automatic production merge.
