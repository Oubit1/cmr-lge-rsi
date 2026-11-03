# LGE/CINE QC rules

## Indexing and geometry

- Use `images/lge/<patient>/*.nii.gz` as the primary slice index. Sort slices numerically by the numeric stem, not lexicographically.
- Expected labels are `labels/lge_checked/lge_Myo_labels/<patient>/<slice>.nii.gz`, `labels/lge_checked/lge_MI_labels/<patient>/<slice>.nii.gz`, and optionally `labels/cmr/cmr_Myo_mask/<patient>/<slice>.nii.gz`.
- Check shape and affine compatibility before comparing masks. A missing or incompatible label is an unresolved data problem.

## Myocardium

- The normal myocardial mask is one connected annular region; the central blood-pool hole is normal and must not be filled.
- Apply the same topology check to `labels/cmr/cmr_Myo_mask` when present, but do not assume CMR and LGE filenames or slice counts are identical.
- A disconnected internal bridge/gap is suspicious when a closing proposal is supported by both adjacent slices. The default support test is 3-pixel dilation of both neighbours and at least 35% of proposed added pixels supported by each/combined neighbourhood.
- At apex/base, one-sided support may be valid. Do not force a complete ring when the slice is a genuine boundary slice.
- When repairing Myo, re-evaluate MI containment afterwards.
- **Isolated non-MI secondary component cleanup**: Spurious disconnected speckles outside the heart (e.g. chest wall, liver) or inside the blood pool that contain 0 MI pixels and have area <= 100px (or < 12% ratio of main LV ring) are safely stripped using `clean_lge_myo_topology.py`.
- **Coupled tiny artifact cleanup**: A tiny secondary Myo component (<= 30px) that contains only a tiny unsupported MI artifact (<= 20px) is safely removed along with the MI component.

## CMR myocardium branch

- CMR Myo is audited independently with `tools/cmr_qc.py`; it has no MI containment check.
- Before applying any apex/base rule, identify axis semantics. In datasets where each CMR file is one spatial slice and the third dimension contains 20–50 cine phases, `slice_index` is a temporal frame and must never be used as the patient-level apex/base index. Sort the patient's CMR files numerically by spatial-position filename and classify all temporal frames of the first/last two files together. Long-axis/three-chamber views at these stack ends belong in the boundary isolation branch rather than the middle-SAX repair branch.
- **Experimental, not promoted:** numeric stack rank alone can miss basal multi-chamber/outflow-tract anatomy. `tools/reclassify_cmr_anatomical_boundaries.py` tests a patient-level trusted-SAX-core rule in `dry-run`: ring topology in at least 60% of cine frames, at least three contiguous core volumes, and exactly one trusted core segment. Only a contiguous prefix/suffix outside that core is proposed as boundary; isolated internal failures and multi-segment/ambiguous stacks remain middle. It must not run in `apply` from the production pipeline until a fixed clinician-adjudicated validation and threshold-sensitivity gate pass. It never modifies CMR images or Myo labels.
- CMR masks may intentionally contain two major disconnected myocardial rings (typically LV and RV). One or two major components are valid; a third or additional component is a review candidate.
- The expected blood-pool holes are normal, including one hole per ventricular ring. Up to two holes are therefore acceptable; a third or additional hole is a review candidate. Empty masks, non-2D masks, and masks touching the image border are review candidates rather than automatic deletions.
- Adjacent slices are sorted by numeric slice position. Area jumps greater than 3-fold relative to the mean of the two neighbours, or centroid drift over 20 pixels, generate check candidates.
- A tiny CMR component (default <=20 pixels) may be removed automatically only when neither adjacent slice has a component within 15 pixels of its centroid. All other multi-component or gap cases remain for manual review.
- A small myocardial area is a warning, not an automatic rejection, because apical layers can be legitimately small.
- A fixed pixel-area threshold must not flag an otherwise valid middle-SAX cine frame by itself. Suppress `cmr_myo_small_area` when it is the only issue and the frame plus both temporal neighbours each have one component, one physiological hole and no border contact; at least 90% of frames have the same topology; current/adjacent mean area ratio is 0.75–1.33; normalized area is at least 0.4%; centroid drift is at most 8 pixels; and paired LGE Myo is independently a valid single ring. Record the decision as `accepted_no_change`; do not alter the mask.
- A run of one to three solid CMR masks with no blood-pool hole is a repair candidate when it is bracketed in time by valid single-ring masks whose hole Dice is at least 0.85 and whose conservative hole intersection is at least 30 pixels. Remove only that intersection from the affected frames, require the repaired mask to remain one component with one hole, and back up the source volume. Use `repair_cmr_temporal_missing_holes.py`; longer or one-sided runs remain manual review.
- A different zero-hole pattern is an open C-shaped ring caused by a missing wall segment. A 1--3-frame run may be completed with `repair_cmr_temporal_open_ring.py` only when both bracketing temporal frames are closed single rings, support-hole Dice is at least 0.85, support-mask Dice is at least 0.70, and adding the intersection of the two supporting myocardium masks contributes no more than 35% of their mean area. The result must be one component with exactly one physiological hole. Do not use the solid-mask hole-subtraction repair for this pattern.
- A local wall notch/gap may be filled only when a 5x5 closing adds at most 100 pixels, at least 20% of added pixels are supported by adjacent temporal frames, the result remains one component, and the central blood-pool hole is preserved. This rule is evaluated only on the current CMR issue manifest. It must not be run across all images.
- For a CMR frame with a partial LV ring, first compare the previous/next z-slices and temporal frames, then the same-position LGE Myo. Keep the largest anatomically continuous ring; preserve genuine apex/base open boundaries and leave ambiguous cases unresolved.
- Use `tools/apply_cmr_qc_fast.py` to run multi-process parallel safe repair across large cohorts to prevent timeouts.
- The parallel implementation must emit at most one copy of a given volume/frame/rule row. Cross-slice passes execute once per volume, never once per frame. A sudden candidate inflation proportional to cine frame count is a software-QC failure, not a clinical finding; stop promotion and inspect loop nesting.

## Shared CMR/LGE myocardial trajectory rule

- Apply the same patient-level LV-location principle to both CMR and LGE Myo: derive a robust normalized LV-centre trajectory from the majority of trustworthy layers/frames, rather than deciding from one image or choosing the largest component blindly.
- A trustworthy reference has exactly one major myocardial component, one physiological blood-pool hole, and no image-border contact. At least 55% of the patient's units must be trustworthy, at least three nearby references are required, and reference-centre spread must be <=12 pixels.
- A multi-component mask is automatically corrected only when one valid ring lies <=12 pixels from the expected trajectory, the next component is at least 10 pixels farther away, and the competing location is supported by fewer than two nearby spatial/temporal units. Keep only the trajectory-consistent ring.
- For LGE, clip MI to the corrected Myo after component removal. For CMR, preserve a persistent second ventricular ring when temporal or adjacent-layer support exists.
- The first and last two spatial layers are excluded from automatic trajectory correction because apex/base and long-axis boundary anatomy may be incomplete.
- A single-component centroid outlier, absent mask, open ring, or ambiguous two-component competition is review-only. Never translate a mask, synthesize a new ring, or copy a neighbouring label solely to satisfy the trajectory.
- Use `tools/myo_stack_trajectory_qc.py` in `dry-run` before `apply`. After all repairs, run it again and render unresolved rows with `tools/render_myo_trajectory_candidates.py` into `labels/check/myo_trajectory_review_slices`.

## Infarct mask

- MI must be contained in Myo. Pixels outside Myo are removed only by the safe repair mode and are logged.
- A tiny component (default <=20 pixels) is removed only if neither adjacent slice has MI within 15 pixels of its centroid. If adjacent slices support it, keep it for review/possible merge.
- Extended tiny component removal: For slices with components=2 where the smaller component area is in (20, 40] px, area ratio < 0.15, and no adjacent-layer MI exists within 15px, the small component is safely removed (`tools/remove_small_unsupported_mi.py`).
- A nearby small component can be merged/expanded only when the source LGE image has comparable high signal, adjacent layers support the same territory, and the proposed closing is strictly clipped to Myo.
- **Same-territory threshold gap merge**: Components separated by a 1-3 pixel sub-pixel thresholding gap within the same myocardial territory are merged using a conservative 3x3 closing constrained to Myo with added pixels <= 30 (`tools/merge_nearby_mi_components.py`).
- Two separated MI regions are valid when both have convincing high signal and layer-to-layer continuity. Never connect them merely to satisfy connected-component QC.
- A component far from the main region, very small, not bright, and absent above/below is an artifact candidate and may be removed by safe mode.
- A high MI/Myo ratio is not itself an error. Compare the original high-signal boundary and adjacent ratios; smooth single-component transmural patterns are accepted without label change. Ambiguous cases stay in the review manifest.
- For a reviewed stack with competing ventricular components, select the LV component by continuity of centroid and shape with at least two validated reference layers; clip MI to the selected Myo before any MI component decision. Never choose the largest ring blindly when a second ventricle is present.
- A clinician-confirmed MI termination layer may define a directional cutoff: beyond the last layer with convincing LGE contrast, clear MI while retaining or repairing Myo independently. This cutoff is case-specific until matched against additional reviewed stacks.
- An open Myo ring between two valid layers may be completed from the intersection of one-pixel dilations of both adjacent Myo masks only when the result has one component and one physiological hole and the added area stays within the reviewed case limit. This is a topology repair, not evidence for MI extension.

## Review categories

`missing_label`, `shape_or_affine_mismatch`, `myo_topology`, `mi_outside_myo`, `mi_tiny_unsupported`, `mi_multiple_components`, `mi_fraction_high`, and `manual_review` are mutually informative issue tags. A row can have multiple tags.

## Check-image convention

- Standalone CMR myocardium checks use three panels: original image, semi-transparent red Myo overlay, and red Myo contour. They contain no MI legend or MI panel.
- Combined LGE checks use six panels: original/Myo overlay/Myo+MI overlay on the first row and original/Myo contour/Myo+MI contour on the second row. Myo is green and MI is red.
- These two layouts intentionally match the historical verified check images. Never render CMR Myo with the combined LGE palette or show an empty MI panel for CMR.

## Casebook principle

Every accepted manual correction records: patient, slice, issue, image evidence, adjacent-layer evidence, action (`remove`, `merge`, `expand`, `shrink`, `fill_myo`, or `no_change`), before/after pixels, backup, and timestamp. A new rule is not generalized from one image until it is tested against the remaining candidates.

## LGE Patient-Level Cross-Slice Continuity & Blood-Pool Exclusion Rule

When inspecting and repairing middle/boundary LGE slices for a patient:
1. **Endocardial Blood-Pool Exclusion**: Near apical cuts, the LV cavity collapses. Myocardial segmentations often dilate into high-signal cavity blood pool (intensity > 500). Because blood pool is hyperintense, the downstream MI network segments this inner rim as circumferential infarct ($MI/Myo > 0.70$). The fix prunes inner border voxels with cavity-like intensity, stopping the false-positive cascade.
2. **Z-Axis Coronary Territory Continuity**: True infarct follows coronary distributions (LAD, LCX, RCA). In slices with multi-component MI (`mi_multiple_components`), small secondary components (< 100 px) in an opposing wall with 0 z-overlap on adjacent slices ($z \pm 1$) are isolated false-positive artifacts and are pruned.
3. **Remote Myocardium Constraint**: In circumferential flood cases, calculate patient remote myocardium baseline ($\mu_{\text{remote}} + 3\times\text{SD}$). MI is constrained to true hyperenhancement, eliminating misclassified null/negative myocardium.
