---
name: cmr-lge-label-qc-workflow
description: Generic deterministic tools for local cardiac MRI label QC; no patient casebook, inference weights or clinical acceptance data are distributed.
---

# CMR/LGE local QC tools

Use the orchestrator's role handoffs, independent reviewer, frozen benchmarks,
source hashes and audited executor. Framework maintainers must not make clinical
adjudications or alter individual labels to substitute for configured roles.

The `tools/` and `scripts/` directories contain the generic deterministic QC and
pipeline implementations. Refer to [QC rules](references/qc_rules.md) for their
software conventions. CMR temporal-axis checks differ from spatial LGE checks.
Rules require validation against local physician-reviewed pairs before deployment.

Patient-specific legacy patches, real examples and replay coordinates are absent.
The public pattern catalogue is de-identified and contains no clinical exemplars.
Model inference is optional and requires separately supplied, appropriately
licensed implementations/checkpoints; these are not bundled in the public tree.

See the root README and docs/DEPLOYMENT.md for installation and local paths.
Human confirmed edits should enter the governed feedback chain; never commit
raw edits, prompts, images, labels or provider transcripts to this repository.
