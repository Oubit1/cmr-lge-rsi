# RSI kernel v1 — reversible introduction

Date: 2026-10-27 (Asia/Shanghai)

## What this release delivers

The existing production loop executes named role handoffs and promotes approved
maintenance code. This release adds a paired-selection mechanism to that loop:

1. Copy the exact parent code and Skill before the Owner runs.
2. Pin the parent tree hash and ancestry in the controller's runtime directory.
3. Let the configured Owner edit the disposable candidate.
4. Keep the existing deterministic validation and independent Reviewer.
5. Execute parent and child in separate interpreters on identical input bytes.
6. Score predictions outside the candidate using frozen reference answers.
7. Compare required metrics; archive selection, lineage and actual action status.
8. Feed measured successes and failed hypotheses into the next model's role memory.

`configs/rsi.json` selects `off`, `observe`, or `enforce`. This release defaults
to **observe**. Observe records the decision without replacing existing approval
rules. Enforce also blocks promotion unless the paired selection says PROMOTE.
Changing RSI policy, kernel files or benchmark contracts inside an Owner patch
is blocked even in observe mode. The configured roles continue to own domain
reasoning, thresholds, reference selection and label operations.

## Evidence boundaries

| Evidence | Meaning |
| --- | --- |
| Unit tests pass | Framework behavior is tested, including failure recovery |
| Paired selection PROMOTE | A child improved declared benchmark metrics without a declared regression |
| Action PROMOTED | Existing Reviewer and production gates accepted installed code |
| Production audit / stage report | Actual dataset progress, checked/trash/abstention distribution |
| Human sampling of results | Independent evidence of domain reliability |

These are separate facts. In particular, an observed selection or an installed
experiment is not a medical acceptance decision.

Missing metrics are never filled with zero. Changed benchmark identity or input,
modified parent code, nonnumeric/NaN/infinite metrics, failed constraints, no
quality improvement, and regression all receive explicit decisions/reasons.
The solver cannot increase its score by reporting its own accuracy or omitting
hard samples. Missing predictions count as abstentions in the fixed denominator.

The runner uses clean subprocesses without provider credentials. Same-UID
processes are **not** an OS security sandbox. Parent hashes detect changes;
they do not prevent a same-UID program from accessing arbitrary server files.
Existing snapshot permissions, role approval and production-write boundaries
remain necessary.

## Benchmark contract

The controller reads `configs/rsi_benchmarks/<workflow_stage>.json` and frozen
data under `runtime/rsi/benchmarks/`. Neither is supplied by the Owner's score
file. Example for a synthetic control test (not a medical production benchmark):

```json
{
  "schema_version": "rsi-benchmark-v1",
  "benchmark_id": "synthetic-control-v1",
  "scope": "synthetic_control_test",
  "module": "scripts.example_solver",
  "scorer": "reference_predictions_v1",
  "input_path": "runtime/rsi/benchmarks/synthetic/input.json",
  "input_sha256": "canonical JSON SHA-256 computed by app.rsi_kernel.canonical_hash",
  "metrics": {
    "correct": {"direction": "higher", "report_field": "correct"},
    "wrong": {"direction": "lower", "report_field": "wrong"}
  }
}
```

Frozen input has task data and a `references` mapping keyed by sample ID.
`run_evaluation(data)` returns `{"predictions": {"sample_id": "answer"}}`.
The runner removes `references` before supplying input to either solver.
Metrics come from the controller's reference scorer, not from arbitrary scores
declared by the solver. Each child records the contract hash and both metrics.

No production LGE reference benchmark is invented by this release. Until a
real existing evaluator is adapted and its reference contract is independently
registered, actions record NEEDS_EVIDENCE/benchmark_contract_missing. Observe
mode preserves the old approved path. Enforce would block that path, so it must
not be enabled solely to make the system appear more autonomous.

## External memory and ancestry

The additive SQLite tables are `evolution_variants`, `evolution_evaluations` and
`evolution_events`. A candidate ID includes its pinned parent, run, action and
content digest. Reassessment does not rediscover a different parent. A promoted
variant matching the next starting tree becomes its parent; otherwise a baseline
node describes the actual source tree. This preserves existing uncommitted code.

Role memory filters by owner before applying the eight-item limit. It includes
the hypothesis, changed files, parent ID, benchmark, measured metrics, selection
reason and actual action status. Global oversight roles receive a shared capsule.
Provider/model replacement uses these persisted capsules through the existing
`role_memory` handoff. It does not depend on a vendor chat session.

This is a version archive for successive maintenance attempts. Automatic search
over old branches, diversity-based parent sampling, validated case-to-Skill
distillation, and production canary rollback are not implemented by this release.
They must use the same independent evaluation contract when introduced.

## Rollback and recovery

Before installing, create a checkpoint outside the repository:

```bash
python scripts/rsi_control.py checkpoint /srv/rsi/cmr_control_checkpoints/<name>
```

The checkpoint saves selected source files (including current WIP), tracked Git
diff/status, commit ID and a consistent SQLite backup checked with quick_check.
It excludes dataset labels, runtime/maintenance trees and provider `.env` files.
The directory is created private to the user. This does not rotate credentials.

Save a release manifest with `installed` mapping each touched source path to its
post-install SHA-256 (or null for a deleted release file). Normal rollback:

```bash
python scripts/rsi_control.py rollback <checkpoint> --release-manifest <release.json>
python scripts/rsi_control.py rollback <checkpoint> --release-manifest <release.json> --execute
```

Create/update that manifest with `record-release <checkpoint> --commit HEAD
--output <release.json>`. It uses the cumulative committed diff from the
checkpoint baseline, and verifies that live touched files match the selected
commit. For this deployment the final manifest is
`/srv/rsi/cmr_control_checkpoints/pre_rsi_20261027_0005/release_rsi_v1.json`.

The first command previews. Restoration refuses concurrent edits, stages the
displaced code before writing, records a recovery journal and compensates if a
later operation fails. Deleted files are restored; additions from this release
are removed after saving their contents. The live database is **not restored**:
new human reviews and audit records survive code rollback. New SQLite tables can
remain unused by the old code. The database copy is disaster recovery material,
not a normal release rollback command.

Journals are atomically replaced and fsynced. If the process is interrupted
between file replacement and completion bookkeeping, resume from before/after
hashes using `python scripts/rsi_control.py recover-rollback <journal.json>`
(preview), followed by the same command with `--execute`. Replaying keeps
already-restored files and verifies remaining files before changing them.

Promotion now records a filesystem journal before installing code. Maintenance
action and evolution status commit in the same DB transaction. If the process
stops between filesystem promotion and DB commit, the next tick verifies the
installed hashes and completed validation, then reconciles bookkeeping. An
incomplete or changed installation emits PROMOTION_RECOVERY_REQUIRED and is not
restarted as another paid Owner job. The maintainer must repair that concrete
installation fault before normal operation resumes.
The check spans outstanding actions across runs before choosing a new run or
requesting a new analysis, so a newer run cannot skip an older partial install.

## Verification and activation

Tests use synthetic data only. They exercise paired subprocess execution, hidden
references, unchanged parent/input, missing evidence, no-gain/regression
selection, durable lineage, role filtering, atomic DB updates, crash
reconciliation, rollback conflicts, deletions and failure compensation.

`python scripts/rsi_control.py status` reads policy and memory. `/api/state.rsi`
exposes the same archive capsule through the existing board state response.
Installing this release does not start services or clear operator pause.

The pre-RSI checkpoint for this deployment is:
`/srv/rsi/cmr_control_checkpoints/pre_rsi_20261027_0005`.
It captures baseline commit `b0e39a713bacf847234b79fded6746bd7cf5b2cb`, 385 source
files and a consistent database backup. Existing dirty `app/check_auto_apply.py`
and `scripts/shadow_spatial_boundary.py` are outside this release's edits.

## Research/code references

- DGM: https://github.com/jennyzzt/dgm — archived variants and empirical selection.
- SICA: https://github.com/MaximeRobeyns/self_improving_coding_agent — evaluated
  self-modification and retained experiment records.
- ExpeL: https://github.com/LeapLabTHU/ExpeL — external experience retrieval.

This release adapts those mechanisms to the repository's existing role handoffs;
it does not vendor those projects or imply their benchmark results transfer to
CMR/LGE labels.

## Phase 2: reviewed-pair bridge (2026-10-27, supersedes observe-only status)

The registered `reviewed-pairs-v4` generation connects existing gold doctor-pair
features and the existing scorer to independent paired evaluation. Quasi-gold
is excluded. Patient IDs define a deterministic calibration/validation split
shared across targets; no persisted patient ID appears in both sets. Existing
score, grade, confidence and identifiers are excluded from solver features.

| Target | Calibration pairs | Held-out pairs | Controlled mismatch pairs |
| --- | ---: | ---: | ---: |
| CMR myocardium | 4833 | 1194 | 53 existing injections, feature and raw replay |
| LGE myocardium | 4859 | 1205 | 36 temporary injections |
| LGE infarction | 4859 | 1205 | 24 temporary injections |

Calibration uses 747 persisted patients, validation 185. An additional 32 held-out
raw CMR image/mask pairs exercise extraction, rather than only replaying cached
features. LGE injections use the repository's existing deterministic perturbation
tool on temporary copies; they are controlled mismatches, not clinically
adjudicated defect labels. CMR controls bind exact configured gold label paths,
not filename suffixes. Source masks remain unchanged.

The three contracts activate through one atomically replaced `active.json`,
with a generation digest and per-input hashes. Each comparison reads one whole
generation. Parent and candidate get independent asset copies, a fresh Python
interpreter, no provider credentials, and a normal Python I/O audit guard.
Private references stay in the controller. This is integrity protection, not an
OS sandbox against arbitrary hostile same-UID native code.

`configs/rsi.json` is now `enforce` for the registered scoring scope. Any change
to that scorer or its imported local dependencies evaluates all three targets.
Unrelated components retain their existing gates. Unresolved dynamic imports
disable scope exemption. Calibration, interslice context and report/input
builders cannot evolve under this narrower contract; they require an expanded
benchmark. Selection requires an improvement without measured regressions,
control discrimination, the existing quality gate, full regression and
independent Reviewer approval. Constant high scores cannot satisfy this gate.

Historical replay of existing fix `402797c` -> current scorer raises raw CMR
retention from 28/32 to 32/32 with no measured regression. Identical-source
comparison must retain the parent. These are offline framework verification,
not newly generated Owner actions, new medical repairs or proof of clinical
accuracy. Current measured weaknesses include LGE myocardium fragmentation and
hole-fill controls (0/12 detected each); these aggregate results and training
profile are delivered to configured roles, while private validation records
and per-record scores remain local. The registered benchmark now makes future
role-produced scoring changes measurable; it does not demonstrate that an
autonomous role has already improved those weaknesses.

Commands (from the repository, no provider calls or service startup):

```sh
python scripts/rsi_control.py status
python scripts/rsi_control.py verify-quality --output runtime/rsi/verifications/new_unique_name
python scripts/rsi_control.py verify-quality --output runtime/rsi/verifications/history_unique_name --baseline-ref 402797c
# To build a genuinely new immutable generation, choose a fresh version:
python scripts/rsi_control.py prepare-quality --version reviewed-pairs-next
```

Release checkpoint: `/srv/rsi/cmr_control_checkpoints/pre_quality_bridge_20261027`
(baseline `f70c8fc`, 393 source files and consistent DB backup). The cumulative
release manifest is `release_quality_bridge.json` in that directory. Use the
rollback preview procedure above with this checkpoint/manifest; preserve live
review/audit data. Board deployment never clears operator pause or starts
Autopilot, Check Loop or Edit Agent.

## Scoring installation verification and recovery

An enforced all-target scoring promotion now closes a durable consumer gate
before source installation. Autopilot takes an exclusive installation lock;
the automatic CMR/LGE consumer holds its shared side for the complete scan/apply
operation. No consumer can enter halfway through an install. The consumer also
reloads the scorer before rebinding apply functions, avoiding service-start code
remaining active after promotion.

After installation, the controller compares the actually installed source to
the frozen parent in another disposable workspace. It requires the same
benchmark and reproducible parent/candidate measurements as the accepted
selection. Candidate regression with a reproducible parent restores only the
release's source files using the existing compare-and-swap rollback journal.
Changed user files are never overwritten. A baseline/environment or benchmark
mismatch is recorded as recovery required rather than falsely blaming the
candidate. An interrupted rollback replays from hashes before any further
model-analysis round; rejected installation evidence is available in role
inheritance. Consumer gate status is exposed as `/api/state.rsi.installation`.

This is scoring-code installation verification, not clinical production
canary monitoring or restoration of label writes. Calibration/context evolution,
LGE label production writing and long-term clinical outcome monitoring still
need their own measured contracts. Existing LGE consumer reports enforce
`label_write=false` and zero production writes: an aggregate LGE report is not
evidence that those masks have been repaired. Do not remove that constraint
without implementing candidate validation, holdout exclusion and apply audit.

Checkpoint for this release:
`/srv/rsi/cmr_control_checkpoints/pre_postpromotion_monitor_20261027`,
baseline `65e5f8b`, 396 source files plus consistent DB. The release manifest is
`release_postpromotion_monitor.json` in that checkpoint directory.

## LGE execution bridge (supersedes the read-only production limitation)

`configs/lge_execution.json` activates a separate governed candidate producer
and trusted validation/apply/receipt controller. The legacy zero-write report
adapter cannot authorize applications in this mode. See
[LGE_EXECUTION.md](LGE_EXECUTION.md) for the exact interfaces, independent
grade-C policy, coupled Myo/MI checks, patient holdout, source locks, crash
recovery, role handoff and UI counts. Domain algorithms remain owned by the
configured roles; the maintainer did not implement a clinical repair strategy.

Offline doctor-reference copies verified actual grading and controlled mask
restoration through this executor: two Myo writes in a temporary dataset,
two MI acceptances, one frozen case excluded, zero repeat writes, unchanged
original source hashes and zero model calls. The candidate was an oracle test
reference, not an autonomous domain-agent result or evidence of generalization.
