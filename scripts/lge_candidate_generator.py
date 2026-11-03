"""LGE myocardium candidate generator.

Reads snapshot image and mask paths from supplied records, produces
repair / accept_existing / needs_review proposals, and writes candidate
binary NIfTI files only under output_dir.  The controller owns scoring,
holdout, apply and audit.

Contract: generate_candidates(data: dict, output_dir: str) -> list[dict]
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import nibabel as nib
import numpy as np

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


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
    """Return a connectivity structure array whose rank matches *ndim*.

    For 2-D masks a (3,3) cross is returned; for 3-D masks a (3,3,3)
    cross is returned.  This avoids the scipy ``RuntimeError: structure
    and input must have equal rank``.
    """
    if ndim == 2:
        return np.ones((3, 3), dtype=bool)
    if ndim == 3:
        return np.ones((3, 3, 3), dtype=bool)
    shape = tuple([3] * ndim)
    return np.ones(shape, dtype=bool)


def _connected_component_count(mask: np.ndarray, connectivity: int = 8) -> int:
    """Return the number of connected components in a binary mask.

    Uses scipy.ndimage.label when available; falls back to a simple
    flood-fill for 2-D slices.
    """
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


# ---------------------------------------------------------------------------
# public entrypoint
# ---------------------------------------------------------------------------


def generate_candidates(data: dict, output_dir: str) -> list[dict]:
    """Generate repair / accept_existing / needs_review proposals.

    Parameters
    ----------
    data : dict
        Must contain a "records" key with a list of record dicts.
        Each record must have at least:
          - case_id : str
          - snapshot_image_path : str   (NIfTI image)
          - snapshot_mask_path  : str   (NIfTI mask)
        Optional keys:
          - issue_codes : list[str]
          - measurements : dict
    output_dir : str
        Directory where candidate NIfTI files are written.

    Returns
    -------
    list[dict]
        One dict per record with keys:
          - case_id
          - decision : "repair" | "accept_existing" | "needs_review"
          - reason : str
          - candidate_filename : str | None   (relative to output_dir, only for repair)
    """
    records: List[Dict[str, Any]] = data.get("records", []) if isinstance(data, dict) else []
    _ensure_dir(output_dir)

    results: List[Dict[str, Any]] = []

    for rec in records:
        case_id = rec.get("case_id", "unknown")
        image_path = rec.get("snapshot_image_path")
        mask_path = rec.get("snapshot_mask_path")
        issue_codes = rec.get("issue_codes") or []
        measurements = rec.get("measurements") or {}

        if not image_path or not mask_path:
            results.append(
                {
                    "case_id": case_id,
                    "decision": "needs_review",
                    "reason": "missing_snapshot_paths",
                    "candidate_filename": None,
                }
            )
            continue

        if not os.path.isfile(image_path) or not os.path.isfile(mask_path):
            results.append(
                {
                    "case_id": case_id,
                    "decision": "needs_review",
                    "reason": "snapshot_file_not_found",
                    "candidate_filename": None,
                }
            )
            continue

        try:
            mask = _load_nifti(mask_path)
            image = _load_nifti(image_path)
            img_obj = nib.load(image_path)
            affine = img_obj.affine
        except Exception as exc:
            results.append(
                {
                    "case_id": case_id,
                    "decision": "needs_review",
                    "reason": f"nifti_load_error:{exc}",
                    "candidate_filename": None,
                }
            )
            continue

        components = _connected_component_count(mask > 0)
        myo_components_meas = measurements.get("myo_components")

        if components == 1:
            results.append(
                {
                    "case_id": case_id,
                    "decision": "accept_existing",
                    "reason": "single_component_mask",
                    "candidate_filename": None,
                }
            )
            continue

        if components > 1:
            repaired = _largest_component(mask > 0)
            candidate_filename = f"{case_id}_repaired.nii.gz"
            candidate_path = os.path.join(output_dir, candidate_filename)
            try:
                _save_nifti(repaired, affine, candidate_path)
            except Exception as exc:
                results.append(
                    {
                        "case_id": case_id,
                        "decision": "needs_review",
                        "reason": f"candidate_write_error:{exc}",
                        "candidate_filename": None,
                    }
                )
                continue

            reason = (
                f"repaired_multiple_components:original_components={components}"
            )
            if myo_components_meas is not None:
                reason += f",measured_myo_components={myo_components_meas}"

            results.append(
                {
                    "case_id": case_id,
                    "decision": "repair",
                    "reason": reason,
                    "candidate_filename": candidate_filename,
                }
            )
            continue

        results.append(
            {
                "case_id": case_id,
                "decision": "needs_review",
                "reason": "empty_mask",
                "candidate_filename": None,
            }
        )

    return results


# ---------------------------------------------------------------------------
# CLI entrypoint for production replay gate
# ---------------------------------------------------------------------------


def main() -> None:
    """Thin CLI that reads JSON from stdin and prints JSON to stdout."""
    import sys

    raw = sys.stdin.read()
    data = json.loads(raw) if raw.strip() else {}
    output_dir = data.get("output_dir", "output/candidates")
    results = generate_candidates(data, output_dir)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
