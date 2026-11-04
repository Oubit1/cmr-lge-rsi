"""LGE myocardium candidate generator.

Reads snapshot image and mask paths from supplied records, produces
repair / accept_existing / needs_review proposals, and writes candidate
binary NIfTI files only under output_dir.  The controller owns scoring,
holdout, apply and audit.

Contract: generate_candidates(data: dict, output_dir: str) -> list[dict]
"""
from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict, List, Optional

import nibabel as nib
import numpy as np

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _load_nifti(path: str) -> np.ndarray:
    """Return a numpy array from a NIfTI file."""
    img = nib.load(path)
    return np.asarray(img.dataobj, dtype=np.int16)

def _save_nifti(data: np.ndarray, affine: np.ndarray, path: str) -> None:
    """Write a binary int16 NIfTI file."""
    img = nib.Nifti1Image(data.astype(np.int16), affine)
    nib.save(img, path)

def _structure_for_ndim(ndim: int, connectivity: int = 8) -> np.ndarray:
    """Return a connectivity structure whose rank matches the mask."""
    if ndim == 2:
        return np.ones((3, 3), dtype=bool)
    if ndim == 3:
        return np.ones((3, 3, 3), dtype=bool)
    shape = tuple([3] * ndim)
    return np.ones(shape, dtype=bool)

def _connected_component_count(mask: np.ndarray, connectivity: int = 8) -> int:
    """Count components using scipy or the parent's 2-D flood-fill."""
    try:
        from scipy.ndimage import label as scipy_label
        structure = _structure_for_ndim(mask.ndim, connectivity)
        labeled, n = scipy_label(mask > 0, structure=structure)
        return n
    except ImportError:
        if mask.ndim != 2:
            raise RuntimeError("scipy is required for 3-D masks")
        visited = np.zeros_like(mask, dtype=bool)
        count = 0
        rows, cols = mask.shape
        for r in range(rows):
            for c in range(cols):
                if mask[r, c] > 0 and not visited[r, c]:
                    count += 1
                    stack = [(r, c)]
                    while stack:
                        cr, cc = stack.pop()
                        if visited[cr, cc]:
                            continue
                        visited[cr, cc] = True
                        for dr in (-1, 0, 1):
                            for dc in (-1, 0, 1):
                                nr, nc = cr + dr, cc + dc
                                if 0 <= nr < rows and 0 <= nc < cols:
                                    if mask[nr, nc] > 0 and not visited[nr, nc]:
                                        stack.append((nr, nc))
        return count

def _largest_component(mask: np.ndarray, connectivity: int = 8) -> np.ndarray:
    """Return a binary mask containing only the largest connected component."""
    try:
        from scipy.ndimage import label as scipy_label
        structure = _structure_for_ndim(mask.ndim, connectivity)
        labeled, n = scipy_label(mask > 0, structure=structure)
        if n == 0:
            return np.zeros_like(mask, dtype=np.int16)
        sizes = np.bincount(labeled.ravel())[1:]
        largest_label = np.argmax(sizes) + 1
        return (labeled == largest_label).astype(np.int16)
    except ImportError:
        if mask.ndim != 2:
            raise RuntimeError("scipy is required for 3-D masks")
        visited = np.zeros_like(mask, dtype=bool)
        rows, cols = mask.shape
        best_mask = np.zeros_like(mask, dtype=np.int16)
        best_size = 0
        for r in range(rows):
            for c in range(cols):
                if mask[r, c] > 0 and not visited[r, c]:
                    comp = np.zeros_like(mask, dtype=bool)
                    stack = [(r, c)]
                    size = 0
                    while stack:
                        cr, cc = stack.pop()
                        if visited[cr, cc]:
                            continue
                        visited[cr, cc] = True
                        comp[cr, cc] = True
                        size += 1
                        for dr in (-1, 0, 1):
                            for dc in (-1, 0, 1):
                                nr, nc = cr + dr, cc + dc
                                if 0 <= nr < rows and 0 <= nc < cols:
                                    if mask[nr, nc] > 0 and not visited[nr, nc]:
                                        stack.append((nr, nc))
                    if size > best_size:
                        best_size = size
                        best_mask = comp.astype(np.int16)
        return best_mask

def _snapshot_path(rec: dict, name: str) -> Optional[str]:
    paths = rec.get('paths')
    paths = paths if isinstance(paths, dict) else {}
    alias = 'mask' if name == 'label' else name
    path = paths.get(name) or rec.get(f'snapshot_{alias}_path')
    return path if isinstance(path, str) and path else None

def _acceptance_reason(mask: np.ndarray, profile: Any) -> Optional[str]:
    # Training evidence: n=4859, holes q01=q99=1; no reference answers.
    # Scope this prior explicitly: generic single-component masks need not ring.
    if not isinstance(profile, dict):
        return None
    targets = profile.get('targets')
    target = targets.get('lge_myo') if isinstance(targets, dict) else None
    if not isinstance(target, dict):
        return None
    metrics = target.get('metrics')
    holes = metrics.get('holes') if isinstance(metrics, dict) else None
    if not isinstance(holes, dict) or holes.get('q01') != 1 or holes.get('q99') != 1:
        return None
    squeezed = np.squeeze(mask > 0)
    planes = ([squeezed] if squeezed.ndim == 2 else
              [mask[:, :, z] > 0 for z in range(mask.shape[2])])
    for plane in planes:
        if not plane.any():
            continue
        count = _connected_component_count(np.pad(~plane, 1, constant_values=True)) - 1
        if count != 1:
            return f'myo_single_component_cavity_review:holes={count}'
    return None

def generate_candidates(data: dict, output_dir: str) -> list[dict]:
    """Read snapshots; return one disposition per record, writing only repairs."""
    records: List[Dict[str, Any]] = data.get('records', []) if isinstance(data, dict) else []
    if not isinstance(data, dict) or not isinstance(records, list):
        return []
    stage = data.get('workflow_stage', 'lge_myo')
    sources = {os.path.realpath(p) for rec in records if isinstance(rec, dict)
               for name in ('image', 'label', 'myo', 'mi')
               for p in [_snapshot_path(rec, name)] if p}
    results: List[Dict[str, Any]] = []
    for index, raw_rec in enumerate(records):
        rec = raw_rec if isinstance(raw_rec, dict) else {}
        case_id = rec.get('case_id', 'unknown')
        result = dict(case_id=case_id, decision='needs_review',
                      reason='empty_mask', candidate_filename=None)
        results.append(result)
        if stage not in ('lge_myo', 'lge_mi') or not isinstance(raw_rec, dict):
            result['reason'] = 'unsupported_input'
            continue
        image_path = _snapshot_path(rec, 'image')
        mask_path = _snapshot_path(rec, 'label')
        measurements = rec.get('measurements') or {}
        measurements = measurements if isinstance(measurements, dict) else {}
        if not image_path or not mask_path:
            result['reason'] = 'missing_snapshot_paths'
            continue
        try:
            if not os.path.isfile(image_path) or not os.path.isfile(mask_path):
                result['reason'] = 'snapshot_file_not_found'
                continue
            mask = _load_nifti(mask_path)
            image = _load_nifti(image_path)
            img_obj = nib.load(image_path)
            mask_obj = nib.load(mask_path)
            affine = mask_obj.affine
            if mask.ndim not in (2, 3) or not all(mask.shape) or image.shape != mask.shape:
                raise ValueError('unsupported_snapshot_shape')
            if not np.allclose(img_obj.affine, affine, rtol=0, atol=1e-5):
                raise ValueError('snapshot_affine_mismatch')
            pair_path = _snapshot_path(rec, 'mi' if stage == 'lge_myo' else 'myo')
            if pair_path:
                pair = _load_nifti(pair_path)
                pair_obj = nib.load(pair_path)
                if pair.shape != mask.shape or not np.allclose(
                        pair_obj.affine, affine, rtol=0, atol=1e-5):
                    raise ValueError('paired_snapshot_geometry_mismatch')
            components = _connected_component_count(mask > 0)
            if components == 1:
                reason = (_acceptance_reason(mask, data.get('calibration_profile'))
                          if stage == 'lge_myo' else None)
                result.update(decision='needs_review' if reason else 'accept_existing',
                              reason=reason or 'single_component_mask')
                continue
            if components == 0:
                continue
            repaired = _largest_component(mask > 0)
        except Exception as exc:
            result['reason'] = f'nifti_load_error:{exc}'
            continue
        candidate_filename = f'candidate_{index:06d}_repaired.nii.gz'
        try:
            candidate_path = os.path.realpath(os.path.join(output_dir, candidate_filename))
            root = os.path.realpath(output_dir)
            if os.path.commonpath([root, candidate_path]) != root or candidate_path in sources:
                raise ValueError('candidate_path_conflicts_with_snapshot_boundary')
            _ensure_dir(output_dir)
            _save_nifti(repaired, affine, candidate_path)
        except Exception as exc:
            result['reason'] = f'candidate_write_error:{exc}'
            continue
        reason = f'repaired_multiple_components:original_components={components}'
        myo_components_meas = measurements.get('myo_components')
        if myo_components_meas is not None:
            reason += f',measured_myo_components={myo_components_meas}'
        result.update(decision='repair', reason=reason, candidate_filename=candidate_filename)
    return results

def run_evaluation(data: dict) -> dict:
    """Pure measurement report; snapshot fitness requires controller replay."""
    records = copy.deepcopy(data.get('records', []))
    for rec in records:
        rec.update(decision='needs_review', reason='snapshot_replay_not_executed')
    return {
        'workflow_stage': data.get('workflow_stage'), 'records': records, 'status': 'UNAVAILABLE',
        'reason': 'independent_snapshot_replay_required',
        'coverage': {'input_records': len(records), 'snapshot_replays': 0},
        'decision_counts': {'repair': 0, 'accept_existing': 0, 'needs_review': len(records)},
        'frozen_split': copy.deepcopy(data.get('frozen_split') or {'status': 'UNAVAILABLE'}),
        'holdout_excluded': 0, 'production_writes': 0, 'candidate_writes': 0,
        'metrics': {'defect_false_acceptance': None, 'clean_acceptance': None,
                    'concordance': None, 'precision': None, 'ci': None},
        'limitations': ['Measurements alone cannot establish mask correctness or non-regression.'],
    }

def main() -> None:
    """Thin CLI that reads JSON from stdin and prints JSON to stdout."""
    import sys
    raw = sys.stdin.read()
    data = json.loads(raw) if raw.strip() else {}
    output_dir = data.get('output_dir', 'output/candidates')
    results = generate_candidates(data, output_dir)
    print(json.dumps(results, indent=2, sort_keys=True))

if __name__ == '__main__':
    main()
