# LGE candidate validation, application and audit

The service pipeline is now `governed candidate producer -> independent
validation -> bounded source application -> durable receipts -> refreshed
Myo/MI counts`. The previous `run_evaluation(data)` adapter remains available
only for legacy installations without `configs/lge_execution.json`; its report
cannot satisfy the new execution protocol.

## Responsibility and startup

Codex maintains the controller and tests, not repair algorithms or clinical
decisions. The configured domain roles implement
`scripts/lge_candidate_generator.py` in their reviewed Owner snapshot. The file
is intentionally not supplied by the framework maintainer. A candidate producer
must have an actual `PROMOTED` action, matching installed source/imported local
dependencies, a passing validation record and full regression. An arbitrary
file on disk or `PROMOTED_SHADOW` action is insufficient.

When the configured producer is missing, evidence explicitly names the producer
interface and planning/execution guards require that target rather than another
read-only report adapter. Owner tests and the independent implementation review
remain required. Promotion exercises both Myo and MI synthetic file contracts.
The fixture gate establishes interface behavior, not medical accuracy.

The existing board Start/Pause controls start/stop the configured services.
Check Loop `--auto-apply` permits execution; the operator-pause record still
blocks writes. Plain `run_lge_feedback_consumer.py` is preview-only; `--execute`
is for the authorized service, never a maintainer shortcut against live labels.
Installing this release does not resume services or make paid model calls.

## Producer contract

```python
def generate_candidates(data: dict, output_dir: str) -> list[dict]:
    ...
```

`data` contains `workflow_stage`, a training-only `calibration_profile`, and
bounded `records`. Each record has a salted `case_id`, existing issue codes and
numeric measurements, snapshot `paths` for image/label/paired mask, and the
source hash. No live source destination, API token or frozen reference answer
is supplied. Return exactly one decision per supplied case:

- `repair`: `case_id`, nonempty `reason`, and `candidate` relative to output_dir;
- `accept_existing`: `case_id` and nonempty `reason`;
- `needs_review`: `case_id` and a specific nonempty `reason`.

Candidates are binary NIfTI files with exactly the source shape and affine.
The producer can reuse local static app/scripts/experiments tools; dynamic
imports disable governance verification. It runs in a fresh credential-free
Python process, reads supplied copies and writes only candidate output. Normal
Python filesystem/process/network access is guarded; this is not an OS sandbox
against hostile same-UID native code. No producer-reported score or bucket
count authorizes a label write. Unsupported discards remain review items;
the maintainer does not invent an LGE exclusion rule.

## Independent validation and physical writes

Authoritative working masks are the same roots used by the editor:
`labels/lge_checked/lge_Myo_labels` and `labels/lge_checked/lge_MI_labels`.
Paths must stay inside the configured dataset. Source identities resolve from
the exact snapshotted LGE manifest; duplicate physical-label identities fail
instead of applying competing candidates. Missing files stay visible.

The controller loads the immutable doctor-calibrated profile, hashes the scorer
and reloads its actual current source. It extracts features from the staged
candidate bytes, not supplied JSON scores. Repairs require the existing C-grade
threshold (70) and positive improvement; no-change acceptance requires 70.
A Myo repair cannot reduce the paired MI score. An MI candidate requires a
passing current Myo score. MI receipts bind the exact Myo bytes and become
pending if Myo changes; an independently validated MI change does not erase
an unchanged Myo acceptance. New scorer/profile versions invalidate automatic
acceptances for reevaluation.

The first service snapshot freezes patient-level holdout membership (at most
two patients), shared across targets. It is hash- and identity-salt-bound;
automatic producers never receive those cases. Unknown later patients are
review items until the split is explicitly reconciled. Actual signed editor
applies with matching image/paired-mask hashes can resolve manual items,
including holdout reviews; staged drafts and model claims cannot.

Each automatic write retains the original bytes, persists a PREPARED journal,
rechecks all source/dependency hashes and atomically replaces only the target.
Editor applies and service writes share a cross-process label lock. Failed
apply/audit I/O propagates for recovery rather than being called a bad mask.
Recovery distinguishes before/after hashes and finishes missing receipt/audit
bookkeeping without applying twice. Other source changes cause a conflict,
never overwrite. Normal code rollback does not restore these patient labels.

## Progress, feedback and scheduling

`runtime/lge_execution/receipts/<stage>/` determines checked counts.
`transactions/` stores backups and journals; `apply_audit.jsonl` distinguishes
repair writes, unchanged acceptance, failed candidates and recovered commits.
Preview output is not counted as a completed application.

Failed candidates are parked against source hashes, calibration and producer
release. They are retried when human intervention or a new governed producer
provides changed evidence. Failure reason counts and review links reach the
roles and board. A large residual requires producer improvement; a bounded
small residual and frozen reviews stay visible while the next dependency
stage proceeds. The controller never forces those items into checked/trash.

The service processes LGE before any CMR refresh/application. CMR evidence
refresh cannot block LGE planning. Actual LGE receipts/source changes enter
the global progress fingerprint; moving failures to manual review alone is
not counted as a physical repair. While a fresh executing consumer has eligible
pending work, Autopilot waits for the local batch rather than generating
unnecessary paid maintenance rounds. Later CMR requires the separately persisted
paired-slice selection gate.

The board exposes `lge_execution` and displays separate Myo/MI pending,
manual-review and checked counts. Historical CMR four-bucket totals remain CMR
totals, not a substitute for LGE execution counts.

## Verification and rollback

`tests/test_lge_execution.py` covers isolated writes, paired dependencies,
holdout, stale sources, governance, human receipts, replay/crash recovery,
stage transitions, progress fingerprint and consumer ordering.

An optional offline probe uses copies of existing gold doctor pairs and an
oracle test candidate after a known controlled perturbation:

```sh
python tests/lge_reference_execution_probe.py --root /path/to/repository \
  --output runtime/lge_execution/reference_protocol_probe.json
```

It creates a disposable dataset internally. This verifies the real grader and
physical executor, not domain-agent reasoning, clinical generalization or a
production run. Original reference hashes must remain unchanged and model calls
must be zero. Do not fabricate production Owner/Reviewer records from the probe.

Release checkpoint:
`/srv/rsi/cmr_control_checkpoints/pre_lge_execution_20261027`, baseline
`6571cb4`, 398 source files plus consistent SQLite backup. Record the cumulative
release manifest as `release_lge_execution.json` there and use the normal
rollback preview procedure. Preserve live human reviews, receipts and labels.

Final integration checks: calibration identity includes the scorer's local
dependency fingerprint. A running Check Loop re-execs its own process before
any consumer if that generation changes, so accepted helper/package updates
cannot leave old grading functions cached. No inactive service is started by
this mechanism. Owner context includes the exact candidate protocol, training-only
profile and actual existing LGE tool signatures; replacement models do not have
to reconstruct these interfaces from an old provider conversation.

The final regression is 1803 tests passing (one skipped). The live read-only
input check found 289 Myo cases and 697 MI cases, all with available authoritative
image/mask pairs. These are task inputs, not a clinical claim that every case
requires repair. The candidate generator is not yet promoted; after operator
resume, the configured roles must implement and review it through the supplied
contract. The framework maintainer has not supplied that domain algorithm.

Read-only deployment checks also confirm writable target directories for all
289/697 inputs. Atomic replacement retains the original file permission bits;
backup files are restricted to 0600. No new privileged service or sudo is required.

If actual MI validation finds the paired Myo below the existing grade threshold,
the controller records an upstream Myo request and returns the dependency chain
to that queue, even if the old QC manifest did not flag Myo. Only the Myo stage
can repair it. The changed Myo hash then unblocks a fresh MI attempt. The
multi-stage recovery is covered by an end-to-end temporary-file test; it does
not require a maintainer to manually redispatch the sample.

## 有界真实角色试跑（2026-10-27）

用户已明确授权一轮付费真实试跑，最多应用10例；这覆盖历史的全面暂停限制，
不授权无限轮次或Codex代改标签。`autopilot.py --bounded-run <record.json>`
仅创建一个新分析轮次，然后沿用原有Owner/Reviewer/确定性验证处理该轮动作。
记录持久化run_id、步骤和终态；中断分析无run_id时拒绝重买，终态不可重复执行。
控制器不启动标签消费者、不写标签，也不解除常驻服务暂停。
应用另由一次`check_feedback_loop.py --once --auto-apply`服务承担（本轮为LGE
上游阶段，单批最多10）；不得循环启动该消费者扩大此次授权。
真实试跑结果及是否实际应用，以主交接最新记录和审计为准。

2026-10-27实跑补充：路径合同现在同时提供`paths.image/label/myo/mi`及对应
`snapshot_image_path/snapshot_mask_path/snapshot_myo_path/snapshot_mi_path`别名，
均指向隔离副本。输入契约版本进入失败回执身份；版本变化重新排队旧失败。
真实生成器崩溃作为执行故障交回维护链，不能算医生人工复核。
单轮record遇到与当前已晋级版本匹配、execute零写入的真实producer_failure，可按原动作
剩余max_attempts回传同一个Owner；已拒绝的修复快照若有真实验证失败也可使用剩余次数，
不得新开分析或扩大总应用10例的授权。每版重新独立评审并绑定实际源哈希，
二维/三维单层/三维多层接口探针和Owner测试错误一起反馈；全部尝试耗尽须明确停下。
这类零应用技术失败后可重试同批，原审计和各次application记录必须保留。

运行修复模型可由角色配置`runtime_repair_models`选择既有替代模型，身份/权限与行动计划
保持原Owner；重试提供同动作上一未晋级快照的代码/哈希作为只读上下文，不自动安装。
显式`--extend-bounded-owner-budget`只增加当前真实失败快照的一次尝试，不能由常驻tick
自动调用，必须有新的操作员授权。每次授权和实际模型调用均可审计，总应用上限仍10。
