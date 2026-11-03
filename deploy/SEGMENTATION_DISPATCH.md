# Prepared-plan Autopilot dispatch

This integration dispatches existing prepared, independently reviewed training
plans. It is **not autonomous network design or full clinical RSI**. A specialist
can return a typed plan over configured artifacts; it cannot generate a new job
or architecture artifact through this lane. New job/architecture hashes still
require an operator configuration update.

`segmentation_training` is a dedicated action class. Target ownership is
`cmr_myo -> cmr_specialist`, `lge_myo -> lge_myo_specialist`, and
`lge_mi -> lge_mi_specialist`. The supervisor must carry the actual specialist
plan unchanged. Dispatch requires the original supervisor assignment, a
hash-bound independent action review, and an independent implementation review
of the exact typed plan. It uses `segmentation_host.make_controller`, the same
trusted host as `scripts/run_segmentation_cycle.py`; it never enters QC code
repair/promotion or changes source labels, clinical thresholds, or review buckets.
Normal Autopilot's existing operator/consumer/workflow gates remain in effect.

## Explicit opt-in

There is no enabled default configuration. An operator supplies
`configs/segmentation_dispatch.json` with these fields:

| Field | Contract |
| --- | --- |
| `schema` | `segmentation-dispatch-v1` |
| `enabled` | Boolean; missing config or false disables dispatch |
| `max_actions` | Integer 1–3, maximum distinct actions in the existing run-budget session |
| `walltime_seconds` | Integer 1–21600; also capped by the existing global deadline |
| `automatic_reviews` | Optional boolean, default false; enables the configured independent reviewer for evaluation/activation |
| `plan_inputs` | 1–16 exact prepared input objects |

Each input object contains `target`, absolute `registry_root`, `parent_id`,
`policy_id`, and the four `{path, sha256}` bindings `job`, `architecture`,
`dataset_manifest`, `cohort`. These are trusted deployment inputs, never HTTP
arguments. Configuration alone is not approval. The existing cycle preflight
rechecks files, policy, splits and hashes; the host independently requires the
parent's actual archived model source bundle before training.

No control path arms, extends, or resumes a global budget. Pause, exhaustion,
missing control records and an expired deadline block work. Private durable
claims prevent silent retries after an interrupted review or execution. Raising
`max_actions` does not increase an already established session allowance.

## Independent review requests

Waiting evaluation/activation produces deduplicated DB messages addressed to
`independent_reviewer`: `segmentation_evaluation_request` and
`segmentation_activation_request`. Their bindings include exact plan/report/
candidate/policy identities and candidate artifact hashes; activation also binds
the computed assessment. Missing baseline sources produce a
`segmentation_source_request`, never a guessed vendor path.

With `automatic_reviews=true`, sufficient independently stored provenance
allows one configured reviewer invocation per phase within the existing budget.
The actual response produces `segmentation_evaluation_evidence` or
`segmentation_activation_review`, which the read-only host consumes from the DB.
The dispatcher does not approve itself or accept a returned boolean as approval.

Reference provenance, baseline exposure, and split evidence must already exist
as exact independent DB records defined by `segmentation_host`. **Their production
is still external to this dispatch lane.** Unknown baseline exposure remains
blocked as `MANUAL_EVIDENCE_REQUIRED`; the model is not asked to infer it. With
automatic reviews disabled, requests remain `MANUAL_REVIEW_REQUESTED` until an
independent workflow supplies the actual response. Rejected/unresolved reviews
remain `MANUAL_REVIEW_REQUIRED`; repeated ticks do not repurchase the same review.

## Board integration

`app.segmentation_dispatch.readiness(ROOT)` is a read-only, path-free projection:

- `enabled`, `status`, `reason_code`, `missing` explain disabled/blocked/paused work.
- `queue`: `inspected`, `has_more`, `approved`, `running`, `waiting_approval`;
  counts cover up to 100 pending actions. `approved` includes authorized running
  actions. Unavailable counters are null, never fabricated zeroes.
- `current`: null or `{status, target, review_status?}` from persisted metadata.

It never scans clinical files or starts/resumes anything. Board wiring is owned
separately. A disabled card should display `CONFIG_MISSING`/`CONFIG_DISABLED`,
not imply that an enabled training worker simply has not started.

## Remaining artifact-generation phase

Autonomous proposals of new network artifacts need a separate trusted boundary:
fixed released dataset/cohort/policy and resource grants; an isolated writable
Owner workspace; trusted snapshotting and complete source manifests of generated
job/architecture artifacts; and independent review bound to those exact snapshots.
Removing this dispatcher's input allowlist without that producer/snapshot boundary
would weaken approval and is not part of this change.
