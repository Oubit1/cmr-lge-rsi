# LGE QC tools

## Main entry points

- `../scripts/run_full_qc_pipeline.py`: complete new-dataset workflow: NIfTI/DICOM intake, annotation, LGE/CMR audit, learned CMR repair rules, LGE safe repair, final audit and check outputs.
- `../scripts/run_lge_qc_workflow.py`: complete dataset audit and conservative safe repair.
- `casebook_qc.py`: match remaining MI candidates to the accumulated casebook rules and produce a decision table. Default is `dry-run`; use `apply-safe` only after inspecting its CSV.
- `cmr_qc.py`: audit and conservatively repair CMR myocardial masks slice-by-slice.
- `run_cmr_qc_batch.py`: run `cmr_qc.py` in patient chunks for large datasets.
- `aggregate_cmr_qc.py`: merge chunk manifests and regenerate flat CMR check PNGs.
- `apply_cmr_case_rule.py`: apply an explicitly reviewed, volume-scoped CMR rule with backup and case record; it is not a blind batch repair tool.
- `match_cmr_apex_base_rule.py`: match the reviewed apex/base largest-component pattern against remaining CMR volumes; always run `--mode dry-run` before `--mode apply --render`.
- `match_cmr_to_lge_rule.py`: use a nearest-position, single-component LGE Myo mask as conservative CMR geometry evidence; always dry-run before applying.
- `apply_cmr_lge_shape_case.py`: apply a reviewed per-slice LGE shape-coverage correction when a false single-component CMR mask has a thin bridge; never use it as an unconditional batch operation.
- `match_cmr_lge_shape_rule.py`: batch-match the learned false-bridge/shape-overflow pattern with conservative LGE coverage and adjacent-frame thresholds.
- `match_cmr_temporal_lge_rule.py`: unify CMR frame position using first-frame/LGE reference and temporally supported neighboring masks; always dry-run before applying.
- `match_cmr_ring_gap_rule.py`: detect and conservatively repair small ring-wall gaps. Restrict it with `--manifest` to an existing CMR issue manifest; do not run it unrestricted over all frames because normal ring curvature can be overcalled. The full pipeline supplies this restriction automatically.
- `apply_cmr_ring_gap_case.py`: apply a manually confirmed small ring-wall closure with backup and check output.
- `isolate_boundary_issue_slices.py`: split residual checks into patient-level spatial boundary and middle-SAX pools. For cine NIfTI files, the third axis is treated as time; it is never used as the apex/base layer index.
- `reclassify_cmr_anatomical_boundaries.py` (experimental, not promoted): shadow-test basal multi-chamber/outflow-tract routing missed by fixed stack-end counts. It requires one reliable trusted SAX core and proposes only an untrusted contiguous prefix/suffix. Production use remains `dry-run`; `apply` is reserved for an explicitly approved validation run and never edits masks.
- `suppress_supported_cmr_small_area.py`: remove small-area-only false positives from the middle CMR review pool when temporal topology and paired LGE independently support a valid ring. It never edits masks and should be run in `dry-run` before `apply`.
- `repair_cmr_temporal_missing_holes.py`: restore a missing physiological blood-pool hole in a 1–3-frame cine run when valid rings on both temporal sides provide highly overlapping hole support. It backs up every modified CMR volume; always run `dry-run` first.
- `repair_cmr_temporal_open_ring.py`: complete a 1–3-frame C-shaped/open myocardial ring by adding only the intersection of two highly concordant closed temporal support rings. This is distinct from restoring a hole in a solid mask; always run `dry-run` first.
- `render_check_panels.py`: render CMR as a three-panel standalone red-Myo check and LGE as a six-panel green-Myo/red-MI combined check. It groups CMR frames by volume and supports parallel workers.
- `myo_stack_trajectory_qc.py`: shared CMR/LGE patient-level LV-location QC. It learns the robust centre trajectory from majority valid rings and removes only an unsupported competing component with a unique high-confidence match. LGE MI is clipped after Myo repair. Always dry-run before apply.
- `render_myo_trajectory_candidates.py`: render the final cross-modality trajectory candidates and unresolved cases into a flat review pool; use `--clean` to prevent stale PNGs.
- `apply_reviewed_lge_stack_case.py`: apply a clinician-reviewed JSON action specification to one LGE stack. It supports boundary clearing, LV selection by reference trajectory, largest-MI retention, directional MI clearing, adjacent-supported Myo-ring completion and light Myo smoothing; all writes are backed up and logged.
- `finalize_paired_dataset.py`: audit the final paired dataset and, only in explicit `quarantine` mode, move unusable non-SAX/irreparable levels as complete recoverable bundles. For iteratively corrected datasets, `lge_checked` is authoritative and `lge_original` is provenance only; CMR image+Myo and LGE image+checked Myo+MI must keep identical relative keys and modality-consistent shapes.

For a large CMR dataset:

```bash
python /home/rsi/.codex/skills/cmr-lge-label-qc-workflow/tools/run_cmr_qc_batch.py \
  --root /path/to/Processed_dataset --mode audit --chunk-size 40 --no-render
```

Then pass the completed `cmr_qc_runs/<timestamp>` directories to
`aggregate_cmr_qc.py` with repeated `--run` arguments. Only after reviewing the
manifest should `run_cmr_qc_batch.py --mode apply-safe` be used.

Example:

```bash
python /home/rsi/.codex/skills/cmr-lge-label-qc-workflow/tools/casebook_qc.py \
  --root /path/to/Processed_dataset --mode dry-run
```

## Legacy case patches

`legacy_case_patches/` contains copies of the historical patient-specific scripts, including the former `apply_geng_zhang_hui_068y_mi_qc.py`. They are preserved for provenance and comparison only. They have hard-coded patient names and dataset paths and must not be run on a new dataset as a batch tool.

`legacy_scanners/` contains older scanners, check-generation utilities, and the former three-center analysis script. New work should use the main entry points above; use a legacy utility only when reproducing an old run.

其中 `check_multicenter_label_rules.py` 只用于对多个已验证中心重新统计标签特征、检查基线和重新推导阈值，不参与新数据集的日常自动校正。它的核心 QC 结论已经固化在 `references/qc_rules.md` 和 `casebook_cases.json` 中。

The machine-readable representative-case index is `casebook_cases.json`. It maps the confirmed examples to general patterns; it does not replay patient-specific coordinates.
