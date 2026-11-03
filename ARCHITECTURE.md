# Architecture and trust boundaries

## Components

1. `WorkflowOrchestrator` owns persistent run state and invokes the existing
   CMR/LGE skill as a subprocess.
2. Deterministic Python QC tools remain the only components that calculate or
   apply label changes.
3. Analysis agents receive a de-identified aggregate evidence package and return
   Pydantic-validated objects. They cannot write production labels or code.
4. A top-level supervisor assigns bounded actions, retry limits and acceptance metrics.
5. Maintenance agents cover QC code, rule/casebook, skill and efficiency work.
6. Codex may implement reviewer-approved actions only in disposable snapshots;
   snapshots exclude datasets, images, labels, credentials, model weights and vendor code.
7. `WorktreeManager` isolates proposed code changes in a real Git repository.
8. `promotion_gate.py` compares fixed-validation metrics against versioned YAML
   thresholds. It never merges code by itself.

## Governed analysis graph

```text
QC_FAILED
  -> failure_triage
  -> [cmr_specialist, lge_myo_specialist, lge_mi_specialist] in parallel
  -> [qc_code_optimizer, rule_curator, skill_maintainer, efficiency_auditor] in parallel
  -> supervisor (priorities, owners, budgets, acceptance metrics)
  -> independent_reviewer
  -> isolated Codex maintenance snapshots when explicitly auto-executable
  -> fixed validation and promotion gate
  -> EXPERIMENT_PROPOSED / NEEDS_REVIEW / PROMOTED / REJECTED
```

Agent-to-agent communication is limited to serialized structured outputs. There
is no shared chat room. Coding agents receive a fresh isolated snapshot per action.

## Evaluation hierarchy

- Clinical safety: false-repair rate and new critical QC issues are hard gates.
- Segmentation non-regression: CMR/LGE Myo Dice and MI Dice use configured tolerances.
- Utility: target issue recall and reduction in unresolved middle slices.
- Efficiency: wall time, GPU hours, LLM calls and manual-escalation rate.
- Agent quality: schema validity, evidence coverage, actionability, risk awareness
  and confidence calibration. Scorecards never override the clinical gate.

## State ownership

- SQLite is the source of truth for run, task, artifact, agent-output and
  experiment state.
- Dataset directories remain the source of truth for medical-image artifacts.
- Git is the source of truth for algorithm code and experiment diffs.
- Environment variables or an external secret manager are the source of truth
  for credentials. Secrets must never enter the first three stores.

## Privacy boundary

The first-version evidence package contains issue counts, modality labels and
salted case IDs only. It excludes patient names, absolute paths, DICOM headers
and source images. Adding image crops later requires a separate de-identification
test and explicit data-governance approval for each external provider.

## Recovery

- Workers atomically claim `QUEUED` runs.
- Annotation runs select the GPU with the most free memory (lowest utilization
  as tie-breaker) and hold a filesystem lock for that physical device.
- A normal failure enters `ERROR` and can be requeued with `retry`.
- A worker killed outside the orchestrator may leave a run in `RUNNING`; an
  operator must verify that no process remains before using `recover-stale`.
- Existing skill-level backups continue to protect label writes in explicit
  `apply` mode.

## Current limits

- No external provider is called by default.
- Agents may modify code/rule/skill copies only in isolated snapshots. They do not
  merge or promote those changes automatically.
- Production thresholds, model weights and labels remain deterministic executor concerns.
- No image is sent to an external model.
- SQLite supports a small number of workers; migrate to PostgreSQL before
  deploying many concurrent servers.
