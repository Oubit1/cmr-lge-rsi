# Artifact layout

Given `ROOT`:

```text
ROOT/
  images/lge/<patient>/*.nii.gz
  images/cmr/<patient>/*.nii.gz                 # optional for LGE-only QC
  labels/lge_checked/lge_Myo_labels/<patient>/*.nii.gz
  labels/lge_checked/lge_MI_labels/<patient>/*.nii.gz
  label_iteration/
    qc_runs/<timestamp>/audit.csv
    qc_runs/<timestamp>/summary.json
    qc_runs/<timestamp>/checks/patient_layer.png
    backups/<timestamp>/<relative-label-path>
    modifications_<timestamp>.jsonl
  labels/check/
    lge_issue_slices/issue_manifest.csv
    lge_issue_slices/*.png
```

The check directories are flat: use `patient_layer.png`, never a nested patient directory. A check image is a review artifact, not a label source. The NIfTI masks remain the source of truth.

For CMR boundary disposition, a lightweight state index may additionally live
under `labels/check/{待筛,待处理,垃圾桶,checked}/cmr_boundary/<patient>/`.
Each boundary check PNG must exist in exactly one of these folders, with no
duplicate retained in the former canonical check path. This moves review PNGs,
never masks or NIfTI data. `待处理` means salvageable but unfinished;
`checked` is reserved for completed review/repair. Moving a PNG link to
`垃圾桶` records disposition only; final exclusion still requires recoverable
paired-bundle quarantine of all CMR/LGE images and labels.

Minimum modification log fields:

`timestamp`, `modality`, `patient`, `slice`, `label`, `rule`, `action`, `before_pixels`, `after_pixels`, `changed_pixels`, `backup`, `evidence`, `status`.

For reproducibility, retain the command line, thresholds, package versions when available, and the complete audit summary in the run directory.
