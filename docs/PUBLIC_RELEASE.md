# Public source release

This repository starts from a new Git history. The original private repository
and production deployment are not made public or rewritten.

Included: orchestration source, current generic agent-produced modules and tests,
UI, protocol contracts, generic QC tools and data-free configuration templates.

Excluded: all original runtime/artifact directories, databases, run logs, provider
transcripts, backups, worktree snapshots, images/masks, patient lists and matching
records, physician gold pairs, calibrated profiles, frozen benchmark assets,
case-specific patches, pretrained weights and private deployment handoff logs.
Casebook entries retain generic patterns but remove real case examples.

Private usernames, institution/dataset names, server paths and case identifiers
in retained files have been replaced or removed. Source defaults now use local
paths or environment variables. No identifier reversal table is distributed.

The old bundled inference `vendor/` trees are excluded because redistribution
provenance/licenses were not established in this release. If used locally, supply
compatible implementations at the paths expected by `auto_annotate_dataset.py`,
plus your own licensed checkpoints; see the optional inference dependencies.
They are not needed to run the dashboard, editor, QC tooling or synthetic tests.

`runtime/` and `data/` remain private even after a local run. The public audit
checks content and file names in addition to `.gitignore`. Future changes still
need review: no automated scanner can prove all possible sensitive data absent.

## Reviewed interface media

The operator subsequently supplied a dashboard screenshot and two screen recordings
for public documentation. Only reviewed derivatives are included. The editor clip
uses original seconds 36–52: the entire case list is cropped away and the identity
bar is covered with an opaque rectangle on every frame. Both videos are silent
and re-encoded without source metadata. Original files remain local.

The identity mask was checked in every decoded editor frame (383 frames, no
nonuniform interior detected). This checks the visible redaction geometry, not
clinical correctness or anonymization of arbitrary future inputs. The remaining
UI and image viewport were visually reviewed; no personal identity text is shown.
The recording-tool watermark is retained. These are illustrative UI excerpts,
not source medical data or a clinical evaluation dataset.

`approved-media.json` binds these specific assets to SHA-256 and byte counts.
Any replacement must be reviewed and the manifest updated; other binary or
clinical files remain blocked by the publication audit.
