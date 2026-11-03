# Manual edit feedback loop

## Dashboard routing contract

The `HUMAN_REQUIRED` **处理** button must route to the action's issue pool and
offer two explicit entries:

1. **Agent prompt repair** (default): the operator describes only the desired
   edit, without names, IDs, volume names or paths.  A modality-specific agent
   receives de-identified topology, component area and adjacent-layer/frame
   summaries and returns at most three allowlisted morphology operations.  The
   deterministic local executor renders a candidate in the editor.  It does not
   write a production label.
2. **Manual editor fallback**: if the refreshed candidate is still wrong, keep
   it as the starting mask and enable brush/polygon add and erase.  Record this
   route as `prompt_then_manual`; a direct editor entry is `manual`, while a
   visually accepted prompt candidate is `prompt`.

The operator may save either result to staging. Applying it still requires the
normal explicit confirmation and recoverable backup. Prompt success/failure,
the prompt job ID, and the eventual manual delta are training evidence for
maintaining the label-edit agent, not automatic authorization for a global QC
rule.

The editor interaction is conversation-first. A follow-up prompt carries the
previous de-identified plan and operator feedback, adjacent layer/frame mask
descriptors, and a small set of recent matching human demonstrations. A
`manual_required` response must remain in the conversation and must not switch
the UI into manual mode automatically. The operator either refines the prompt
or explicitly opens the drawing drawer.

Prompt repair v2 has three additional deterministic tools:

- `erase_polygon` / `fill_polygon`: precise local edits using 3--24 normalized
  vertices instead of global dilation/erosion;
- `fill_from_adjacent_consensus`: centroid-align both adjacent masks, intersect
  their support, and add only consensus pixels within 1--6 pixels of the current
  contour. Use this for an explicitly requested local gap repair based on both
  neighboring frames/layers; do not substitute global `binary_dilate`.

Prompt repair v3 makes the QC context binding rather than advisory. Every
request carries target-specific evidence priority, allowed/forbidden operations,
and hard constraints. The executor must enforce them again after planning:

- CMR Myo preserves up to two dominant components and physiological cavities;
  do not use largest-component selection, global dilation/erosion, or automatic
  hole filling. A local gap fill requires support from both adjacent masks.
- LGE Myo preserves the blood-pool cavity and any component containing MI.
- LGE MI remains inside Myo after every operation; multiple territories and a
  high MI fraction are not defects by themselves, and adjacent-layer support
  protects a small component from automatic removal.

For explicit, low-ambiguity instructions, route locally without a cloud call:
bilateral adjacent-mask support plus a named quadrant and local gap maps to
regional `fill_from_adjacent_consensus`; MI outside Myo maps to
`clip_mi_to_myo`. All other plans remain specialist-owned and bounded by the
prompt timeout. Keep only the current +/-3 frame/layer window and at most two
issue-matched human demonstrations in the cloud context. Persist planning
latency, evidence used, and constraints checked, and expose those fields to the
operator so a candidate is auditable rather than merely plausible.

The worker must put every prompt job into a visible terminal state. Provider
calls have a bounded outer timeout, prompt jobs retry at most once, and an
abandoned `agent_request.running` is reclaimed when the single systemd worker
starts. The editor stops polling immediately for `ERROR` or `STALLED`. Entering
manual mode also stops polling, preserves the current canvas, and prevents a
late Agent candidate from overwriting an in-progress manual correction.

Use this workflow when rule-based repair remains ambiguous and a clinician or
operator needs pixel-level control.

## Editor contract

The orchestrator serves `/editor` from `scripts/board_server.py`. It maps each
issue-manifest row back to the NIfTI image and label and supports CMR Myo, LGE
Myo, and LGE MI editing. CMR Myo is red; LGE Myo is green and MI is red. The
previous/next control is evidence only; edits are permitted only on the current
layer/frame.

Saving an edit creates, under
`cmr_lge_orchestrator/runtime/manual_annotations/edits/<edit_id>/`:

- `candidate_label.nii.gz`: staged replacement preserving geometry/header;
- `added.png` and `removed.png`: exact binary delta;
- `provenance.json`: local identity/path, reviewer note, source hash and state;
- `agent_request.json`: optional de-identified analysis request.

The production label is unchanged while the state is `STAGED`. Applying an edit
requires the literal confirmation `APPLY`, verifies the source hash, and writes a
recoverable backup under `label_iteration/manual_edits/backups/<edit_id>/` before
replacement. Re-run modality QC and check rendering after every applied edit.

## Learning record

`training_corpus.jsonl` contains one de-identified record per staged edit:
modality/target, existing issue types, operation (add/remove/mixed), topology
before/after, added/removed component geometry, and polar image-coordinate
location relative to the myocardial centroid. These coordinates are not AHA
segments unless image orientation has been independently standardized.

Every staged manual delta also writes `prompt_supervision.json` beside the edit
and appends `prompt_supervision_corpus.jsonl`. It aligns the original operator
instruction and failed agent plan (when present) with the true added/removed
geometry and a deterministic concise `prompt_gt`. Future prompt requests may
retrieve only matching modality/target examples from this corpus. Never place
patient identity, paths, raw images, raw masks, or free-text reviewer notes in
this supervision record.

Free-text reviewer notes, patient names, images, masks, and paths must never be
placed in `agent_request.json` or sent to a cloud model. The agent worker is:

```bash
python /srv/rsi/project/scripts/analyze_manual_edit_queue.py --watch
```

It routes CMR Myo to `cmr_specialist`, LGE Myo to `lge_myo_specialist`, and LGE
MI to `lge_mi_specialist`. Structured output must describe the operation and its
myocardial-coordinate location, propose a compact casebook/prompt lesson, list
contraindications, and state uncertainty.

Run it through `cmr-lge-edit-agent.service` in production so the API secrets are
loaded and only one consumer owns `.running` claims. A prompt candidate remains
staging-only even when the Agent decision is `apply_operations`; production
labels still require visual acceptance, staging, and literal `APPLY`.

## Promotion boundary

One edit is a case example, not a production rule. Consolidate repeated analyses
by target and suggested pattern, then test a candidate prompt/rule on the fixed
clean and defect-injection sets. Promotion still requires the normal gate:
improved target recall, bounded clean false-positive/false-repair rate, no new
critical QC failures, and independent review. Never train and evaluate on the
same manual edits. Keep a fixed holdout of accepted edits for forward testing.
