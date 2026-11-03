# Paired dataset and segmentation model lifecycle

The deliverable is a versioned **five-file bundle**: CMR image + CMR Myo mask +
LGE image + LGE Myo mask + LGE MI mask. A QC PNG, a populated `checked` folder,
successful code promotion, or a high aggregate score is not itself that release.

## Implemented control interfaces

1. `scripts/dataset_release.py --dataset /private/dataset` reconciles existing
   committed acceptances, current file hashes, geometry and frozen exclusions.
   The report separates file-complete bundles from releasable bundles. It never
   changes source labels or assigns a new clinical verdict. The optional
   `--freeze /private/new-release` copies only the accepted subset to an immutable
   version, with original provenance in a private manifest.
2. `app.segmentation_jobs.compile_job` compiles that frozen release and an explicit
   patient split into a training job. Private original patient groups persist
   across releases; CMR frames and both LGE targets cannot move between splits.
3. `scripts/train_segmentation_candidate.py --job /private/job.json` preflights a
   job; `--execute` runs one bounded attempt through the existing vendor trainer.
   Train/validation files are copied and hashed. Test reference paths never enter
   the worker payload. Actual processes have persistent status, cancellation and
   deadlines. Failed attempts are not silently retried.
4. `app.segmentation_evaluation.compare_predictions` measures parent/candidate
   predictions against a complete frozen reference cohort. It reports patient
   macro Dice, positive-reference Dice, empty-reference false positives and full
   coverage. Missing/changed predictions fail the comparison; QC scores are not
   segmentation accuracy.
5. `app.segmentation_registry` records architecture/config/checkpoint/data hashes,
   parent versions and training attempts. Independent evidence and an explicitly
   approved metric policy are needed for activation; the prior pointer remains
   recoverable. Importing a legacy baseline makes no improvement claim.
6. `scripts/segment_from_registry.py` resolves a verified model/config together
   and predicts into a separate new candidate directory. It does not install
   predictions into accepted labels. MI pins the Myo weights it was evaluated
   with. Changed architectures require a compatible inference adapter.
   `--model-id` evaluates an immutable candidate before activation. Trained
   candidates also bind the complete vendor Python source manifest.
7. `app.segmentation_cycle.run_cycle` connects a hash-bound plan, actual bounded
   training, candidate registration, paired prediction/evaluation, assessment and
   activation. Supervisor/Owner/reviewer identities remain separate. Missing
   reference/exposure evidence or an exact stored approval leaves an explicit
   waiting stage. Restart does not silently repeat a failed training attempt.
   Trusted prediction/evidence adapters are capabilities supplied by the host,
   never callable names or approval flags supplied by a model.

The board and subsequent agent evidence packages expose aggregate dataset/model
status. They do not scan NIfTI files synchronously during HTTP refresh.

## Dataset controls on the board

The **成套数据与分割模型** section has three asynchronous controls:

- **核验配对** refreshes file, geometry, evidence and exclusion counts.
- **验收下一批** checks at most 100 eligible bundles with the existing frozen
  scorer and existing acceptance floor. It does not relabel the result as doctor
  gold. Pending/work/trash entries and frozen patients remain excluded. Sparse
  phase CMR scores cannot certify an entire cine volume. Unchanged failures are
  remembered instead of repeatedly consumed.
- **导出通过集** refreshes the audit and copies its accepted subset into a new
  private version. It neither overwrites the source nor edits bucket membership.

These controls remain usable while paid model processing is paused; they perform
only the explicitly requested deterministic data job. Progress and errors are
persisted independently of the HTTP request, with a finite job deadline. They do
not enable training or spend provider credits.

The initial annotation pipeline can opt into reviewed model selections with
`--model-registry /private/registry --orchestrator-root /private/controller`.
Absent that explicit configuration it retains legacy weights. A configured but
missing/corrupt selection fails rather than silently substituting another model.
Initial annotation still feeds the existing QC stages; selecting a model does
not declare its new predictions accepted.

## Responsibility and current limits

Network architecture, loss, augmentation, training schedule and clinical
acceptance criteria belong to the configured domain roles and independent
reviewers. The framework only validates and executes their approved artifacts.
Existing private trainers are reused; no replacement network or loss was authored
by the framework maintainer. A new architecture cannot be activated merely by
replacing a weight filename.

The mechanical cycle is **not an autonomous experiment designer**. The existing
Autopilot handles QC/code actions; its regular tick does not yet synthesize
segmentation plans or supply the cycle's trusted prediction/reference adapters.
It therefore cannot yet run the complete new loop unattended. Registry actor strings and hashes are
not authentication. Controller-only mutation APIs must not be exposed directly to
model output or an unauthenticated web endpoint.

The existing calibration benchmark measures QC/tool behavior. A segmentation
promotion benchmark additionally needs doctor reference masks, patient assignments
and a documented distinction between validation for repeated selection and an
untouched final test. Old weights' training exposure is not inferred from names.

Historical CMR apply logs often bind only a mask hash. They are reported as
`cmr_legacy_image_hash_missing`, not converted into image-bound approval by
hashing today's image. Single-frame human edits cannot certify a whole cine
volume. The dataset acceptance ledger must be written by a configured validator
or confirmed review workflow after rechecking the actual image/label pair.

Manual corrections, automatically QC-accepted labels and doctor references
retain distinct provenance. Automatic acceptance is not silently relabeled as
doctor gold. A source update invalidates a dependent live acceptance; old frozen
releases remain intact and new training uses a new version.

Pairing currently follows the pre-existing patient/relative-filename convention.
The exporter checks within-modality geometry; it does not invent anatomical
registration between CMR and LGE. Inputs requiring a separate slice matching
table need that established mapping first.

Exports are **private research datasets**, not public anonymized datasets. Random
output names do not remove identifying NIfTI headers. Do not publish releases,
private manifests, model checkpoints or training logs to this source repository.

## Validation and rollback

Synthetic tests cover stale receipts, partial frame reviews, incomplete bundles,
affine/hash drift, frozen exclusions, train/test leakage, subprocess failures,
concurrent activation and rollback. Actual vendor-engine smoke tests on synthetic
images prove adapter execution only; they do not establish clinical accuracy or
successful RSI improvement. Runtime evidence stays private.

Code rollback must compare expected installed hashes and preserve new runtime
records, dataset versions, human edits and source labels. Model rollback changes
the selected model version; it never rolls back the patient dataset.

## External design references

- [nnU-Net manual splits](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/manual_data_splits.md)
- [nnU-Net training and model-selection workflow](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md)
- [MONAI Bundle workflow properties](https://monai.readthedocs.io/en/stable/mb_properties.html)

These inform explicit data splits and separation of training, evaluation and
inference. They are not claims that nnU-Net/MONAI or new clinical architectures
have been installed or validated in this project.
