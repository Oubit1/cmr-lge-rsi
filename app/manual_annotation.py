from __future__ import annotations

import base64
import csv
import hashlib
import hmac
import io
import json
import math
import os
import re
import shutil
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from app.role_routing import role_for_label_task


POOL_MANIFESTS = {
    "cmr_boundary_manifest": "cmr_boundary_manifest.csv",
    "cmr_middle_manifest": "cmr_middle_manifest.csv",
    "lge_boundary_manifest": "lge_boundary_manifest.csv",
    "lge_issue_manifest": "lge_issue_manifest.csv",
    "lge_middle_manifest": "lge_middle_manifest.csv",
    "trajectory_manifest": "trajectory_manifest.csv",
}
TARGET_DIRS = {
    ("cmr", "myo"): "labels/cmr_checked/cmr_Myo_mask",
    ("lge", "myo"): "labels/lge_checked/lge_Myo_labels",
    ("lge", "mi"): "labels/lge_checked/lge_MI_labels",
}
BOUNDARY_REVIEW_DIRS = {
    "pending": "待筛",
    "work": "待处理",
    "trash": "垃圾桶",
    "checked": "checked",
}
CMR_REVIEW_SUBDIR = "cmr"
_WRITE_LOCK = threading.Lock()


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stack(array: np.ndarray) -> tuple[np.ndarray, int | None]:
    arr = np.squeeze(array)
    if arr.ndim == 2:
        return arr[..., None], None
    if arr.ndim != 3:
        raise ValueError(f"expected a 2-D/3-D NIfTI, got {arr.shape}")
    candidates = [
        i for i, n in enumerate(arr.shape)
        if n <= 64 and all(arr.shape[j] > n for j in range(3) if j != i)
    ]
    axis = candidates[0] if len(candidates) == 1 else int(np.argmin(arr.shape))
    return np.moveaxis(arr, axis, -1), axis


def _restore(stack: np.ndarray, original: np.ndarray, axis: int | None) -> np.ndarray:
    arr = stack[..., 0] if axis is None else np.moveaxis(stack, -1, axis)
    if arr.shape == original.shape:
        return arr
    squeezed = np.squeeze(original)
    if arr.shape == squeezed.shape and original.ndim == 3 and original.shape[-1] == 1:
        return arr[..., None]
    raise ValueError(f"cannot restore {arr.shape} to {original.shape}")


def _numeric_name(path: Path) -> tuple[int, float | str]:
    stem = path.name[:-7] if path.name.endswith(".nii.gz") else path.stem
    try:
        return 0, float(stem)
    except ValueError:
        return 1, stem


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest(dataset_root: Path, pool: str) -> list[dict[str, str]]:
    if pool not in POOL_MANIFESTS:
        raise ValueError(f"unknown issue pool: {pool}")
    path = dataset_root / "labels/check/manifest_backup" / POOL_MANIFESTS[pool]
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _row_identity(pool: str, row: dict[str, str]) -> dict[str, Any]:
    modality = (row.get("modality") or ("lge" if pool.startswith("lge") else "cmr")).lower()
    volume = row.get("volume") or row.get("slice") or ""
    frame = int(row.get("frame_index") or row.get("slice_index") or 1)
    return {
        "modality": modality,
        "patient": row.get("patient") or "",
        "volume": volume,
        "frame": frame,
    }


def _paths(dataset_root: Path, ident: dict[str, Any], target: str) -> tuple[Path, Path]:
    key = (ident["modality"], target)
    if key not in TARGET_DIRS:
        raise ValueError(f"target {target!r} is unavailable for {ident['modality']}")
    relative = Path(ident["patient"]) / ident["volume"]
    image = dataset_root / "images" / ident["modality"] / relative
    label = dataset_root / TARGET_DIRS[key] / relative
    for path in (image, label):
        if not path.is_file() or dataset_root not in path.resolve().parents:
            raise FileNotFoundError(path)
    return image, label


def _sample_reference(dataset_root: Path, ident: dict[str, Any]) -> dict[str, Any]:
    """Return local-only paths for copying from the browser review list."""
    relative = Path(ident["patient"]) / ident["volume"]
    targets = ["myo"] if ident["modality"] == "cmr" else ["myo", "mi"]
    return {
        "sample_name": f'{ident["patient"]}/{ident["volume"]}#frame={ident["frame"]}',
        "paths": {
            "image": str(dataset_root / "images" / ident["modality"] / relative),
            **{
                target: str(dataset_root / TARGET_DIRS[(ident["modality"], target)] / relative)
                for target in targets
            },
        },
    }


def _boundary_review_png(dataset_root: Path, ident: dict[str, Any], status: str) -> Path:
    if status not in BOUNDARY_REVIEW_DIRS:
        raise ValueError(f"unknown boundary review status: {status}")
    stem = ident["volume"][:-7] if ident["volume"].endswith(".nii.gz") else Path(ident["volume"]).stem
    return (dataset_root / "labels/check" / BOUNDARY_REVIEW_DIRS[status]
            / CMR_REVIEW_SUBDIR / ident["patient"] / f"{stem}.png")


def boundary_review_status(dataset_root: Path, ident: dict[str, Any]) -> str:
    matches = [status for status in BOUNDARY_REVIEW_DIRS
               if _boundary_review_png(dataset_root, ident, status).exists()]
    if len(matches) > 1:
        raise ValueError(f"boundary review state conflict: {ident['patient']}/{ident['volume']}")
    return matches[0] if matches else "pending"


def boundary_review_counts(dataset_root: Path) -> dict[str, int]:
    check = dataset_root / "labels/check"
    return {
        status: sum(1 for _ in (check / folder / CMR_REVIEW_SUBDIR).rglob("*.png"))
        for status, folder in BOUNDARY_REVIEW_DIRS.items()
    }


def _boundary_review_confirmations(dataset_root: Path) -> set[tuple[str, str, str]]:
    """Return latest same-state operator confirmations by spatial slice.

    Inventory state and review progress are intentionally separate: confirming
    an item that is already in trash keeps it in trash, but removes it from the
    outstanding human-review queue.
    """
    path = dataset_root / "labels/check/boundary_review_history.jsonl"
    latest: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
            key = str(event["relative_key"])
            latest[key] = event
        except (KeyError, TypeError, json.JSONDecodeError):
            continue
    confirmed: set[tuple[str, str, str]] = set()
    for key, event in latest.items():
        status = str(event.get("status") or "")
        if status and event.get("previous_status") == status:
            patient, separator, volume = key.partition("/")
            if separator and patient and volume:
                confirmed.add((patient, volume, status))
    return confirmed


def set_boundary_review_status(dataset_root: Path, pool: str, index: int,
                               status: str, reviewer: str = "local_operator") -> dict[str, Any]:
    if pool != "cmr_boundary_manifest":
        raise ValueError("boundary status is available only for cmr_boundary_manifest")
    if status not in BOUNDARY_REVIEW_DIRS:
        raise ValueError(f"unknown boundary review status: {status}")
    _, ident = candidate(dataset_root, pool, index)
    source = (dataset_root / "labels/check/cmr/cmr_Myo_mask" / ident["patient"]
              / _boundary_review_png(dataset_root, ident, status).name)
    with _WRITE_LOCK:
        current = boundary_review_status(dataset_root, ident)
        existing = _boundary_review_png(dataset_root, ident, current)
        if not existing.exists() and not source.is_file():
            raise FileNotFoundError(source)
        target = _boundary_review_png(dataset_root, ident, status)
        target.parent.mkdir(parents=True, exist_ok=True)
        if existing.exists() and existing != target:
            os.replace(existing, target)
        elif not target.exists():
            os.replace(source, target)
        event = {
            "timestamp": utcnow(), "pool": pool, "index": index,
            "relative_key": f'{ident["patient"]}/{ident["volume"]}',
            "previous_status": current, "status": status,
            "review_confirmed": current == status,
            "reviewer": str(reviewer or "local_operator")[:80],
        }
        with (dataset_root / "labels/check/boundary_review_history.jsonl").open(
            "a", encoding="utf-8"
        ) as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    return {**event, "counts": boundary_review_counts(dataset_root)}


def _excluded_items(dataset_root: Path) -> dict[tuple[str, int], dict[str, Any]]:
    path = dataset_root / "labels/check/manual_exclusions.jsonl"
    result: dict[tuple[str, int], dict[str, Any]] = {}
    if not path.is_file():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
            key = (event["pool"], int(event["index"]))
            result[key] = event
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            continue
    return result


def _require_active_item(dataset_root: Path, pool: str, index: int,
                         ident: dict[str, Any]) -> None:
    event = _excluded_items(dataset_root).get((pool, index))
    if event and (event.get("patient"), event.get("volume"), event.get("frame")) != (
            ident["patient"], ident["volume"], ident["frame"]):
        raise ValueError("manifest changed since exclusion; reconcile audit first")
    if event and event.get("excluded"):
        raise ValueError("review item was removed from the editor queue; restore it first")


def set_item_excluded(dataset_root: Path, pool: str, index: int, *, excluded: bool,
                      reason: str = "", detail: str = "",
                      reviewer: str = "local_operator") -> dict[str, Any]:
    """Operator disposition for one manifest frame; labels remain untouched."""
    if pool not in POOL_MANIFESTS:
        raise ValueError("unknown issue pool")
    _, ident = candidate(dataset_root, pool, index)
    allowed = {"out_of_range", "unreadable", "doctor_review", "other"}
    if excluded and reason not in allowed:
        raise ValueError("invalid exclusion reason")
    detail = str(detail).strip()
    if len(detail) > 240 or (excluded and reason == "other" and not detail):
        raise ValueError("custom exclusion reason required (max 240 characters)")
    event = {"timestamp": utcnow(), "pool": pool, "index": index,
             "patient": ident["patient"], "volume": ident["volume"],
             "frame": ident["frame"], "excluded": bool(excluded),
             "reason": reason if excluded else "", "detail": detail if excluded else "",
             "reviewer": str(reviewer or "local_operator")[:80]}
    path = dataset_root / "labels/check/manual_exclusions.jsonl"
    with _WRITE_LOCK:
        prior = _excluded_items(dataset_root).get((pool, index))
        if prior and (prior.get("patient"), prior.get("volume"), prior.get("frame")) != (
                ident["patient"], ident["volume"], ident["frame"]):
            raise ValueError("manifest changed since exclusion; reconcile audit first")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    return event


def list_candidates(dataset_root: Path, pool: str, *, offset: int = 0,
                    limit: int = 100, review_status: str | None = None,
                    group_spatial: bool = False, include_excluded: bool = False,
                    hide_reviewed: bool = False) -> dict[str, Any]:
    rows = _manifest(dataset_root, pool)
    offset = max(0, int(offset))
    limit = max(1, min(int(limit), 500))
    all_items = []
    excluded = _excluded_items(dataset_root)
    confirmations = (_boundary_review_confirmations(dataset_root)
                     if pool == "cmr_boundary_manifest" else set())
    seen: set[tuple[str, str]] = set()
    for index, row in enumerate(rows):
        ident = _row_identity(pool, row)
        disposition = excluded.get((pool, index))
        if disposition and (disposition.get("patient"), disposition.get("volume"), disposition.get("frame")) != (
                ident["patient"], ident["volume"], ident["frame"]):
            raise ValueError("manifest changed since exclusion; reconcile audit first")
        if disposition and disposition.get("excluded") and not include_excluded:
            continue
        state = boundary_review_status(dataset_root, ident) if pool == "cmr_boundary_manifest" else None
        if review_status and review_status != "all" and state != review_status:
            continue
        review_confirmed = bool(
            state and (ident["patient"], ident["volume"], state) in confirmations
        )
        if hide_reviewed and review_confirmed:
            continue
        spatial_key = (ident["patient"], ident["volume"])
        if group_spatial and spatial_key in seen:
            continue
        seen.add(spatial_key)
        targets = ["myo"] if ident["modality"] == "cmr" else ["myo", "mi"]
        all_items.append({
            "index": index,
            **ident,
            **_sample_reference(dataset_root, ident),
            **({"review_status": state} if state else {}),
            "review_confirmed": review_confirmed,
            "excluded": bool(disposition and disposition.get("excluded")),
            "exclusion_reason": disposition.get("reason", "") if disposition else "",
            "issues": row.get("issues") or row.get("decision") or "",
            "evidence": row.get("evidence") or "",
            "targets": targets,
        })
    items = all_items[offset:offset + limit]
    layer_names: dict[tuple[str, str], list[str]] = {}
    for item in items:
        key = (item["modality"], item["patient"])
        if key not in layer_names:
            image_root = (dataset_root / "images" / key[0]).resolve()
            image_dir = (image_root / key[1]).resolve()
            layer_names[key] = (
                [p.name for p in sorted(image_dir.glob("*.nii*"), key=_numeric_name)]
                if image_dir.parent == image_root and image_dir.is_dir() else []
            )
        names = layer_names[key]
        item["layer_index"] = names.index(item["volume"]) if item["volume"] in names else None
        item["total_layers"] = len(names) or None
    result = {"pool": pool, "offset": offset, "limit": limit,
              "total": len(all_items), "items": items}
    if pool == "cmr_boundary_manifest":
        result["review_counts"] = boundary_review_counts(dataset_root)
        if review_status in {"trash", "checked"}:
            result["review_inventory_total"] = result["review_counts"].get(review_status, 0)
    return result


def candidate(dataset_root: Path, pool: str, index: int) -> tuple[dict[str, str], dict[str, Any]]:
    rows = _manifest(dataset_root, pool)
    if index < 0 or index >= len(rows):
        raise IndexError(f"candidate index out of range: {index}")
    row = rows[index]
    return row, _row_identity(pool, row)


def _resolve_plane(dataset_root: Path, pool: str, index: int, target: str,
                   neighbor: int = 0, *, layer_neighbor: int = 0,
                   clamp: bool = False) -> tuple[Path, Path, nib.Nifti1Image, np.ndarray,
                                                np.ndarray, int | None, int, dict[str, Any]]:
    _, ident = candidate(dataset_root, pool, index)
    image_path, label_path = _paths(dataset_root, ident, target)
    image_nii = nib.load(str(image_path))
    label_nii = nib.load(str(label_path))
    image_raw = np.asarray(image_nii.get_fdata())
    label_raw = np.asarray(label_nii.get_fdata())
    image_stack, _ = _stack(image_raw)
    label_stack, label_axis = _stack(label_raw)
    frame = max(0, ident["frame"] - 1)

    if ident["modality"] == "cmr":
        if layer_neighbor:
            files = sorted((dataset_root / "images/cmr" / ident["patient"]).glob("*.nii*"), key=_numeric_name)
            names = [p.name for p in files]
            try:
                pos = names.index(ident["volume"]) + int(layer_neighbor)
                if clamp:
                    pos = min(max(0, pos), len(files) - 1)
                if not 0 <= pos < len(files):raise IndexError("CMR layer outside stack")
                next_ident = dict(ident, volume=files[pos].name, frame=1)
            except (ValueError, IndexError):
                raise IndexError("neighboring CMR layer is outside the patient stack")
            ident = next_ident
            image_path, label_path = _paths(dataset_root, ident, target)
            image_nii = nib.load(str(image_path))
            label_nii = nib.load(str(label_path))
            image_raw = np.asarray(image_nii.get_fdata())
            label_raw = np.asarray(label_nii.get_fdata())
            image_stack, _ = _stack(image_raw)
            label_stack, label_axis = _stack(label_raw)
        k = max(0, ident["frame"] - 1) + int(neighbor)
        if not 0 <= k < image_stack.shape[-1]:
            if clamp:
                k = min(max(0, k), image_stack.shape[-1] - 1)
            else:
                raise IndexError("neighboring cine frame is outside the volume")
    elif neighbor or layer_neighbor:
        files = sorted((dataset_root / "images/lge" / ident["patient"]).glob("*.nii*"), key=_numeric_name)
        names = [p.name for p in files]
        try:
            pos = names.index(ident["volume"]) + int(neighbor) + int(layer_neighbor)
            if clamp:
                pos = min(max(0, pos), len(files) - 1)
            if not 0 <= pos < len(files):raise IndexError("LGE layer outside stack")
            next_image = files[pos]
        except (ValueError, IndexError):
            raise IndexError("neighboring LGE layer is outside the patient stack")
        next_ident = dict(ident, volume=next_image.name, frame=1)
        image_path, label_path = _paths(dataset_root, next_ident, target)
        image_nii = nib.load(str(image_path))
        label_nii = nib.load(str(label_path))
        image_raw = np.asarray(image_nii.get_fdata())
        label_raw = np.asarray(label_nii.get_fdata())
        image_stack, _ = _stack(image_raw)
        label_stack, label_axis = _stack(label_raw)
        ident = next_ident
        k = 0
    else:
        k = min(frame, image_stack.shape[-1] - 1)

    if image_stack.shape[:2] != label_stack.shape[:2]:
        raise ValueError("image and label dimensions differ")
    return image_path, label_path, label_nii, label_raw, image_stack[..., k], label_axis, k, ident


def frame_metadata(dataset_root: Path, pool: str, index: int) -> dict[str, Any]:
    row, ident = candidate(dataset_root, pool, index)
    _, label_path = _paths(dataset_root, ident, "myo")
    nii = nib.load(str(label_path))
    stack, _ = _stack(np.asarray(nii.get_fdata()))
    total_layers = None
    layer_index = None
    if ident.get("volume"):
        rel = "images/cmr" if ident["modality"] == "cmr" else "images/lge"
        d = dataset_root / rel / ident["patient"]
        if d.is_dir():
            files = sorted(d.glob("*.nii*"), key=_numeric_name)
            names = [p.name for p in files]
            total_layers = len(files)
            if ident["volume"] in names:
                layer_index = names.index(ident["volume"])
    return {
        "pool": pool,
        "index": index,
        **ident,
        "total_frames": int(stack.shape[-1]) if ident["modality"] == "cmr" else None,
        "total_layers": total_layers,
        "layer_index": layer_index,
        "width": int(stack.shape[1]),
        "height": int(stack.shape[0]),
        "targets": ["myo"] if ident["modality"] == "cmr" else ["myo", "mi"],
        "issues": row.get("issues") or row.get("decision") or "",
        "evidence": row.get("evidence") or "",
        "source_hashes": {
            target: _sha256(_paths(dataset_root, ident, target)[1])
            for target in (["myo"] if ident["modality"] == "cmr" else ["myo", "mi"])
        },
    }


def png_layer(dataset_root: Path, pool: str, index: int, layer: str,
              neighbor: int = 0, *, layer_neighbor: int = 0) -> bytes:
    target = "mi" if layer == "mi" else "myo"
    _, _, _, _, image, _, k, ident = _resolve_plane(
        dataset_root, pool, index, target, neighbor, layer_neighbor=layer_neighbor, clamp=True
    )
    if layer == "image":
        finite = np.isfinite(image)
        values = image[finite]
        if not len(values):
            rendered = np.zeros(image.shape, dtype=np.uint8)
        else:
            lo, hi = np.percentile(values, [1, 99])
            rendered = np.zeros(image.shape, dtype=np.uint8) if hi <= lo else np.rint(
                np.clip((image - lo) / (hi - lo), 0, 1) * 255
            ).astype(np.uint8)
    else:
        _, label_path = _paths(dataset_root, ident, target)
        stack, _ = _stack(np.asarray(nib.load(str(label_path)).get_fdata()))
        rendered = ((stack[..., min(k, stack.shape[-1] - 1)] > 0.5) * 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(rendered).save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def _decode_mask(data_url: str, shape: tuple[int, int]) -> np.ndarray:
    if not data_url.startswith("data:image/png;base64,"):
        raise ValueError("edited mask must be a PNG data URL")
    raw = base64.b64decode(data_url.split(",", 1)[1], validate=True)
    if len(raw) > 4 * 1024 * 1024:
        raise ValueError("edited mask PNG is too large")
    image = Image.open(io.BytesIO(raw)).convert("L")
    if image.size != (shape[1], shape[0]):
        raise ValueError(f"edited mask shape {image.size} does not match {(shape[1], shape[0])}")
    return np.asarray(image) >= 128


def _mask_grid(mask: np.ndarray, size: int = 52) -> str:
    """Downsampled ASCII occupancy map: '#'=label, '.'=background, rows top to bottom."""
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    bh = max(1, h // size)
    bw = max(1, w // size)
    rows = []
    for y in range(0, h, bh):
        line = []
        for x in range(0, w, bw):
            line.append("#" if mask[y:y + bh, x:x + bw].any() else ".")
        rows.append("".join(line))
    return "\n".join(rows)


def _components_detail(mask: np.ndarray) -> list:
    mask = np.asarray(mask, dtype=bool)
    labels, count = ndi.label(mask, structure=np.ones((3, 3)))
    h = max(1, mask.shape[0] - 1)
    w = max(1, mask.shape[1] - 1)
    out = []
    for label_id in range(1, min(int(count), 8) + 1):
        ys, xs = np.nonzero(labels == label_id)
        if not len(ys):
            continue
        out.append({
            "area": int(len(ys)),
            "bbox_norm": [int(round(1000 * xs.min() / w)), int(round(1000 * ys.min() / h)),
                          int(round(1000 * xs.max() / w)), int(round(1000 * ys.max() / h))],
            "centroid_norm": [int(round(1000 * xs.mean() / w)), int(round(1000 * ys.mean() / h))],
            "region": _region_tag(float(ys.mean()) / h, float(xs.mean()) / w),
        })
    out.sort(key=lambda item: -item["area"])
    return out


def _region_mask(shape: tuple, region: str) -> np.ndarray:
    h, w = shape
    ny = (np.arange(h) / max(1, h - 1))[:, None]
    nx = (np.arange(w) / max(1, w - 1))[None, :]
    if region == "center":
        return (nx >= 0.3) & (nx <= 0.7) & (ny >= 0.3) & (ny <= 0.7)
    if region == "left_upper":
        return (nx < 0.5) & (ny < 0.5)
    if region == "right_upper":
        return (nx >= 0.5) & (ny < 0.5)
    if region == "left_lower":
        return (nx < 0.5) & (ny >= 0.5)
    if region == "right_lower":
        return (nx >= 0.5) & (ny >= 0.5)
    return np.zeros((h, w), dtype=bool)


def _region_tag(ny: float, nx: float) -> str:
    """Quadrant tag in operator-screen orientation (row 0 = top)."""
    if 0.3 <= nx <= 0.7 and 0.3 <= ny <= 0.7:
        return "center"
    if ny < 0.5:
        return "left_upper" if nx < 0.5 else "right_upper"
    return "left_lower" if nx < 0.5 else "right_lower"


def _mask_stats(mask: np.ndarray) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    labels, components = ndi.label(mask, structure=np.ones((3, 3)))
    holes = ndi.binary_fill_holes(mask) & ~mask
    _, hole_count = ndi.label(holes, structure=np.ones((3, 3)))
    yy, xx = np.where(mask)
    return {
        "area": int(mask.sum()),
        "components": int(components),
        "holes": int(hole_count),
        "touches_border": bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()),
        "centroid_xy": [round(float(xx.mean()), 3), round(float(yy.mean()), 3)] if len(xx) else None,
        "shape": [int(mask.shape[0]), int(mask.shape[1])],
        "component_regions": _component_regions(mask, labels, components),
    }


def _component_regions(mask: np.ndarray, labels: np.ndarray, count: int) -> list:
    out = []
    hh = max(1, mask.shape[0] - 1)
    ww = max(1, mask.shape[1] - 1)
    for label_id in range(1, min(int(count), 8) + 1):
        comp = labels == label_id
        cy, cx = ndi.center_of_mass(comp)
        out.append({"region": _region_tag(float(cy) / hh, float(cx) / ww),
                    "area": int(comp.sum())})
    out.sort(key=lambda item: -item["area"])
    return out


def _component_descriptors(mask: np.ndarray) -> list[dict[str, Any]]:
    labels, count = ndi.label(np.asarray(mask, dtype=bool), structure=np.ones((3, 3)))
    output = []
    height, width = mask.shape
    for label_id in range(1, count + 1):
        yy, xx = np.where(labels == label_id)
        if not len(xx):
            continue
        output.append({
            "area": int(len(xx)),
            "centroid_normalized_xy": [round(float(xx.mean()) / width, 4), round(float(yy.mean()) / height, 4)],
            "touches_border": bool(
                (xx == 0).any() or (xx == width - 1).any()
                or (yy == 0).any() or (yy == height - 1).any()
            ),
        })
    return sorted(output, key=lambda item: item["area"], reverse=True)[:12]


def _edit_location(edit: np.ndarray, myo: np.ndarray) -> dict[str, Any]:
    yy, xx = np.where(edit)
    my, mx = np.where(myo)
    if not len(xx):
        return {"pixels": 0}
    cx = float(mx.mean()) if len(mx) else edit.shape[1] / 2.0
    cy = float(my.mean()) if len(my) else edit.shape[0] / 2.0
    ex, ey = float(xx.mean()), float(yy.mean())
    angle = (math.degrees(math.atan2(-(ey - cy), ex - cx)) + 360.0) % 360.0
    radial = math.hypot(ex - cx, ey - cy)
    myo_radius = float(np.percentile(np.hypot(mx - cx, my - cy), 95)) if len(mx) else 1.0
    return {
        "pixels": int(len(xx)),
        "centroid_normalized_xy": [round(ex / edit.shape[1], 4), round(ey / edit.shape[0], 4)],
        "relative_to_myo_center_xy": [round((ex - cx) / edit.shape[1], 4), round((ey - cy) / edit.shape[0], 4)],
        "polar_angle_deg_image_coordinates": round(angle, 2),
        "radial_fraction_of_myo": round(radial / max(myo_radius, 1.0), 3),
        "bbox_xyxy": [int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())],
        "components": int(ndi.label(edit, structure=np.ones((3, 3)))[1]),
    }


def _anonymous_id(patient: str, work_root: Path) -> str:
    salt = os.getenv("CMR_LGE_CASE_SALT")
    if not salt:
        salt_path = work_root / ".case_salt"
        if not salt_path.exists():
            salt_path.write_text(os.urandom(32).hex(), encoding="ascii")
            salt_path.chmod(0o600)
        salt = salt_path.read_text(encoding="ascii").strip()
    return hmac.new(salt.encode(), patient.encode(), hashlib.sha256).hexdigest()[:16]


def _suggest_pattern(target: str, added: np.ndarray, removed: np.ndarray,
                     before: dict[str, Any], after: dict[str, Any]) -> str:
    add_n, remove_n = int(added.sum()), int(removed.sum())
    if remove_n and not add_n and after["components"] < before["components"]:
        return "remove_unsupported_component"
    if add_n and not remove_n and after["holes"] < before["holes"]:
        return "fill_internal_gap"
    if add_n and not remove_n:
        return "expand_undersegmented_region"
    if remove_n and not add_n:
        return "shrink_or_remove_oversegmentation"
    if target == "mi" and add_n and remove_n:
        return "reshape_scar_to_signal_boundary"
    return "mixed_manual_refinement"


def _location_phrase(location: dict[str, Any]) -> str:
    centroid = location.get("centroid_normalized_xy") or []
    if len(centroid) != 2:
        return "目标区域"
    x, y = float(centroid[0]), float(centroid[1])
    vertical = "上方" if y < 0.4 else ("下方" if y > 0.6 else "中部")
    horizontal = "左侧" if x < 0.4 else ("右侧" if x > 0.6 else "中央")
    return horizontal + vertical


def _prompt_ground_truth(target: str, pattern: str, added: dict[str, Any],
                         removed: dict[str, Any], before: dict[str, Any],
                         after: dict[str, Any]) -> str:
    """Create a concise, de-identified instruction aligned to the human delta."""
    label = "心肌" if target == "myo" else "梗死"
    clauses = []
    if int(removed.get("pixels") or 0):
        clauses.append(f"删除{_location_phrase(removed)}约{removed['pixels']}像素的多余{label}标注")
    if int(added.get("pixels") or 0):
        action = "补全" if pattern == "fill_internal_gap" else "增补"
        clauses.append(f"{action}{_location_phrase(added)}约{added['pixels']}像素的{label}标注")
    topology = (
        f"连通域 {before.get('components', 0)}→{after.get('components', 0)}，"
        f"内部洞 {before.get('holes', 0)}→{after.get('holes', 0)}"
    )
    return "；".join(clauses or [f"按人工结果微调{label}边界"]) + f"；保持其余区域不变（{topology}）"


def _image_context(image: np.ndarray, changed: np.ndarray) -> dict[str, Any]:
    """Descriptive local signal evidence only; never infer clinical correctness."""
    finite = np.isfinite(image)
    if not finite.any() or not changed.any():
        return {}
    lo, hi = np.percentile(image[finite], [1, 99])
    if hi <= lo:
        return {}
    signal = np.clip((np.nan_to_num(image, nan=lo) - lo) / (hi - lo), 0, 1)
    ring = ndi.binary_dilation(changed, iterations=3) & ~changed
    if not ring.any():
        return {}
    gradient = np.hypot(ndi.sobel(signal, axis=0), ndi.sobel(signal, axis=1))
    threshold = float(np.percentile(gradient[finite], 80))
    return {
        "local_contrast": round(float(np.median(signal[changed]) - np.median(signal[ring])), 3),
        "edge_fraction": round(float(np.mean(gradient[changed] >= threshold)), 3),
        "normalization": "image p1-p99; edge threshold image p80",
    }


def _prompt_supervision(work_root: Path, prompt_job_id: str, *, target: str,
                        pattern: str, added: dict[str, Any], removed: dict[str, Any],
                        before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    supervision: dict[str, Any] = {
        "prompt_gt": _prompt_ground_truth(target, pattern, added, removed, before, after),
        "human_delta": {"added": added, "removed": removed, "before": before, "after": after},
    }
    if not prompt_job_id:
        return supervision
    job_dir = (work_root / "prompt_jobs" / prompt_job_id).resolve()
    if work_root.resolve() not in job_dir.parents:
        return supervision
    request = _read_json_object(job_dir / "agent_request.json")
    plan = _read_json_object(job_dir / "agent_plan.json")
    if request:
        supervision["operator_prompt"] = request.get("operator_instruction")
    if plan:
        agent_plan = plan.get("plan") if isinstance(plan.get("plan"), dict) else plan
        supervision["agent_decision"] = agent_plan.get("decision")
        supervision["agent_operations"] = agent_plan.get("operations") or []
        supervision["agent_rationale"] = agent_plan.get("rationale")
    return supervision


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}


def _similar_prompt_examples(work_root: Path, modality: str, target: str,
                             issue_types: list[str] | None = None,
                             limit: int = 2) -> list[dict[str, Any]]:
    """Retrieve recent de-identified human-delta demonstrations."""
    path = work_root / "prompt_supervision_corpus.jsonl"
    if not path.exists():
        return []
    matches = []
    wanted = {str(item).strip() for item in (issue_types or []) if str(item).strip()}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()[-500:]
    except OSError:
        return []
    for line in reversed(lines):
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if item.get("modality") != modality or item.get("target") != target:
            continue
        item_issues = {str(value).strip() for value in item.get("issue_types") or []}
        overlap = len(wanted & item_issues)
        matches.append((overlap, {
            "issue_types": item.get("issue_types") or [],
            "suggested_pattern": item.get("suggested_pattern"),
            "prompt_gt": (item.get("prompt_supervision") or {}).get("prompt_gt"),
            "human_delta": (item.get("prompt_supervision") or {}).get("human_delta"),
        }))
        if len(matches) >= max(limit * 4, limit):
            break
    matches.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in matches[:limit]]


def _prompt_qc_context(modality: str, target: str, pool: str,
                       issue_types: list[str], current: dict[str, Any],
                       neighbor_count: int) -> dict[str, Any]:
    """Compact, executable QC prior for one prompt-repair request."""
    common = [
        "Only edit the current frame/layer and only what the operator requested.",
        "Prefer the smallest local delta supported by supplied geometry.",
        "If required evidence is absent or conflicting, return manual_required.",
    ]
    if modality == "cmr":
        hard = common + [
            "One or two dominant myocardial components can be valid LV/RV anatomy; never discard either dominant component.",
            "Up to two blood-pool holes can be physiological; never fill them merely to simplify topology.",
            "Small area alone can be valid at apex/base or end systole and does not authorize expansion.",
            "A local gap repair requires support from both adjacent temporal frames and must preserve the central cavity.",
        ]
        if "boundary" in pool:
            hard.append("This is a boundary/non-SAX review pool; never force an incomplete apex/base into a complete ring.")
        allowed = [
            "remove_components_below_area", "binary_close",
            "remove_components_in_region", "erase_region_pixels",
            "erase_box", "fill_box", "erase_circle", "fill_circle",
            "erase_polygon", "fill_polygon", "fill_from_adjacent_consensus",
        ]
        forbidden = ["keep_largest_component", "fill_internal_holes_below_area",
                     "binary_dilate", "binary_erode"]
        evidence_priority = ["operator_instruction", "both_temporal_neighbors",
                             "current_topology", "local_mask_geometry"]
    elif target == "myo":
        hard = common + [
            "Preserve the physiological LV blood-pool hole and annular topology.",
            "Remove a secondary Myo component only when it is small, unsupported and contains no MI.",
            "Do not force a complete ring at a genuine apex/base boundary.",
            "After any Myo edit, MI containment must remain valid.",
        ]
        allowed = [
            "remove_components_below_area", "binary_close",
            "remove_components_in_region", "erase_region_pixels",
            "erase_box", "fill_box", "erase_circle", "fill_circle",
            "erase_polygon", "fill_polygon", "fill_from_adjacent_consensus",
        ]
        forbidden = ["keep_largest_component", "fill_internal_holes_below_area",
                     "binary_dilate", "binary_erode", "clip_mi_to_myo"]
        evidence_priority = ["operator_instruction", "both_spatial_neighbors",
                             "current_topology", "mi_containment"]
    else:
        hard = common + [
            "MI must remain inside Myo.",
            "Separated MI territories may be real; never connect or delete them only to reduce component count.",
            "High MI/Myo fraction alone is not an error.",
            "Remove a tiny MI component only when neither adjacent layer supports its territory.",
        ]
        allowed = [
            "remove_components_below_area", "binary_close", "clip_mi_to_myo",
            "remove_components_in_region", "erase_region_pixels",
            "erase_box", "fill_box", "erase_circle", "fill_circle",
            "erase_polygon", "fill_polygon", "fill_from_adjacent_consensus",
        ]
        forbidden = ["keep_largest_component", "fill_internal_holes_below_area",
                     "binary_dilate", "binary_erode"]
        evidence_priority = ["operator_instruction", "myocardium_constraint",
                             "both_spatial_neighbors", "current_component_geometry"]
    return {
        "task_scope": f"{modality}_{target}_{'boundary' if 'boundary' in pool else 'middle_or_trajectory'}",
        "issue_types": issue_types,
        "hard_constraints": hard,
        "evidence_priority": evidence_priority,
        "allowed_operations": allowed,
        "forbidden_operations": forbidden,
        "current_topology": {key: current.get(key) for key in (
            "area", "components", "holes", "touches_border"
        )},
        "bilateral_neighbor_evidence_available": neighbor_count == 2,
    }


def save_edit(dataset_root: Path, work_root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    pool = str(payload.get("pool") or "")
    index = int(payload.get("index", -1))
    target = str(payload.get("target") or "")
    reviewer = str(payload.get("reviewer") or "").strip()
    note = str(payload.get("note") or "").strip()
    reason_code = str(payload.get("reason_code") or "other").strip()
    source_run_raw = payload.get("source_run_id")
    source_action_id = str(payload.get("source_action_id") or "").strip()
    source_run_id: int | None = None
    if source_run_raw not in (None, ""):
        source_run_id = int(source_run_raw)
        if source_run_id <= 0:
            raise ValueError("source_run_id must be positive")
    if bool(source_run_id) != bool(source_action_id):
        raise ValueError("source_run_id and source_action_id must be provided together")
    if source_action_id and not re.fullmatch(r"[A-Za-z0-9_.:-]{1,240}", source_action_id):
        raise ValueError("invalid source_action_id")
    if not reviewer:
        raise ValueError("reviewer is required")
    if len(note) < 3:
        note = ""
    row, ident = candidate(dataset_root, pool, index)
    _require_active_item(dataset_root, pool, index, ident)
    _, label_path, label_nii, label_raw, image_frame, label_axis, k, _ = _resolve_plane(
        dataset_root, pool, index, target, 0
    )
    label_stack, _ = _stack(label_raw)
    original = label_stack[..., k] > 0.5
    edited = _decode_mask(str(payload.get("mask_png") or ""), original.shape)
    supplied_hash = str(payload.get("source_hash") or "")
    actual_hash = _sha256(label_path)
    if supplied_hash != actual_hash:
        raise ValueError("source label changed after the editor opened; reload before saving")

    if target == "mi":
        _, myo_path = _paths(dataset_root, ident, "myo")
        myo_stack, _ = _stack(np.asarray(nib.load(str(myo_path)).get_fdata()))
        myo = myo_stack[..., min(k, myo_stack.shape[-1] - 1)] > 0.5
        edited &= myo
    else:
        myo = edited
    added, removed = edited & ~original, original & ~edited
    if not added.any() and not removed.any():
        raise ValueError("edited mask is unchanged")

    edit_id = datetime.now().strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:10]
    edit_dir = work_root / "edits" / edit_id
    edit_dir.mkdir(parents=True, exist_ok=False)
    candidate_stack = label_stack.copy()
    candidate_stack[..., k] = edited.astype(candidate_stack.dtype)
    candidate_raw = _restore(candidate_stack, label_raw, label_axis)
    header = label_nii.header.copy()
    header.set_data_dtype(np.uint8)
    candidate_path = edit_dir / "candidate_label.nii.gz"
    nib.save(nib.Nifti1Image(candidate_raw.astype(np.uint8), label_nii.affine, header), str(candidate_path))
    Image.fromarray((added * 255).astype(np.uint8)).save(edit_dir / "added.png")
    Image.fromarray((removed * 255).astype(np.uint8)).save(edit_dir / "removed.png")

    before, after = _mask_stats(original), _mask_stats(edited)
    pattern = _suggest_pattern(target, added, removed, before, after)
    if reason_code == "auto":
        reason_code = pattern
    added_location = _edit_location(added, myo)
    removed_location = _edit_location(removed, myo)
    prompt_job_id = str(payload.get("prompt_job_id") or "")
    supervision = _prompt_supervision(
        work_root, prompt_job_id, target=target, pattern=pattern,
        added=added_location, removed=removed_location, before=before, after=after,
    )
    image_context = _image_context(image_frame, added | removed)
    tool_trace = [str(x) for x in (payload.get("tool_trace") or [])
                  if isinstance(x, str) and x in {"add", "erase", "poly-add", "poly-erase", "undo", "redo", "reset"}][:100]
    auto_note = supervision["prompt_gt"]
    if image_context:
        auto_note += (f"；局部归一化对比度 {image_context['local_contrast']:+.3f}，"
                      f"改动像素处高梯度占比 {image_context['edge_fraction']:.1%}"
                      "（仅影像描述，非临床判断）")
    if tool_trace:
        counts = {name: tool_trace.count(name) for name in dict.fromkeys(tool_trace)}
        auto_note += "；工具动作：" + "、".join(f"{name}×{count}" for name, count in counts.items())
    note_source = "manual" if note else "auto_delta"
    note = note or auto_note
    corpus = {
        "schema_version": "manual-edit-v1",
        "edit_id": edit_id,
        "created_at": utcnow(),
        "case_hash": _anonymous_id(ident["patient"], work_root),
        "modality": ident["modality"],
        "target": target,
        "frame_index": ident["frame"],
        "issue_types": (row.get("issues") or row.get("decision") or "").split(";"),
        "reason_code": reason_code,
        "tool_trace": tool_trace,
        "image_context": image_context,
        "edit_origin": str(payload.get("edit_origin") or "manual"),
        "prompt_job_id": prompt_job_id or None,
        "source_run_id": source_run_id,
        "source_action_id": source_action_id or None,
        "operation": "mixed" if added.any() and removed.any() else ("add" if added.any() else "remove"),
        "suggested_pattern": pattern,
        "before": before,
        "after": after,
        "added": added_location,
        "removed": removed_location,
        "prompt_supervision": supervision,
        "constraints": {
            "mi_clipped_to_myo": target == "mi",
            "coordinate_semantics": "image coordinates relative to myocardial centroid; not AHA segments",
            "production_label_modified": False,
        },
    }
    provenance = {
        **corpus,
        "reviewer": reviewer,
        "review_note": note,
        "review_note_source": note_source,
        "pool": pool,
        "manifest_index": index,
        "source_label": str(label_path),
        "source_hash": actual_hash,
        "candidate_label": str(candidate_path),
        "status": "STAGED",
    }
    (edit_dir / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (edit_dir / "prompt_supervision.json").write_text(
        json.dumps(supervision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    queue_item = {
        **corpus,
        "task": "analyze_manual_mask_delta",
        "requested_outputs": [
            "anatomical_location_summary", "edit_intent", "reusable_casebook_pattern",
            "prompt_improvement", "contraindications", "confidence",
        ],
    }
    if payload.get("request_agent_analysis") is True:
        (edit_dir / "agent_request.json").write_text(
            json.dumps(queue_item, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    with _WRITE_LOCK:
        with (work_root / "training_corpus.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(corpus, ensure_ascii=False) + "\n")
        with (work_root / "prompt_supervision_corpus.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(corpus, ensure_ascii=False) + "\n")
        if payload.get("request_agent_analysis") is True:
            with (work_root / "agent_queue.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(queue_item, ensure_ascii=False) + "\n")
    return {
        "edit_id": edit_id,
        "status": "STAGED",
        "candidate_label": str(candidate_path),
        "added_pixels": int(added.sum()),
        "removed_pixels": int(removed.sum()),
        "suggested_pattern": pattern,
        "prompt_gt": supervision["prompt_gt"],
        "review_note": note,
        "before": before,
        "after": after,
        "added_location": added_location,
        "removed_location": removed_location,
        "agent_analysis_queued": payload.get("request_agent_analysis") is True,
    }


def queue_prompt_repair(dataset_root: Path, work_root: Path,
                        payload: dict[str, Any]) -> dict[str, Any]:
    """Queue a de-identified prompt-to-operations request.

    The local request keeps the source mapping.  Only ``agent_request.json`` is
    consumed by the cloud worker and it intentionally contains no patient,
    source path, image, or mask pixels.
    """
    pool = str(payload.get("pool") or "")
    index = int(payload.get("index", -1))
    target = str(payload.get("target") or "")
    mode = str(payload.get("mode") or "render")
    prompt = str(payload.get("prompt") or "").strip()
    if target not in {"myo", "mi"}:
        raise ValueError("target must be myo or mi")
    if mode not in {"render", "refine"}:
        raise ValueError("unknown prompt repair mode")
    if mode == "render" and (len(prompt) < 3 or len(prompt) > 2000):
        raise ValueError("repair prompt must contain 3..2000 characters")
    row, ident = candidate(dataset_root, pool, index)
    _require_active_item(dataset_root, pool, index, ident)
    if ident["modality"] == "cmr" and target != "myo":
        raise ValueError("CMR only supports the myocardial target")
    _, label_path, _, label_raw, _, _, k, _ = _resolve_plane(
        dataset_root, pool, index, target, 0
    )
    source_hash = _sha256(label_path)
    if str(payload.get("source_hash") or "") != source_hash:
        raise ValueError("source label changed after the editor opened; reload before prompting")
    stack, _ = _stack(label_raw)
    current = stack[..., k] > 0.5
    frame_table = []
    for kk in range(min(int(stack.shape[-1]), 40)):
        fr = stack[..., kk] > 0.5
        lab, cnt = ndi.label(fr, structure=np.ones((3, 3)))
        holes_arr = ndi.binary_fill_holes(fr) & ~fr
        _, hc = ndi.label(holes_arr, structure=np.ones((3, 3)))
        frame_table.append({"frame": kk, "area": int(fr.sum()),
                            "components": int(cnt), "holes": int(hc)})
    myo_stats = None
    if target == "mi":
        _, myo_path = _paths(dataset_root, ident, "myo")
        myo_stack, _ = _stack(np.asarray(nib.load(str(myo_path)).get_fdata()))
        myo_stats = _mask_stats(myo_stack[..., min(k, myo_stack.shape[-1] - 1)] > 0.5)
    work_root.mkdir(parents=True, exist_ok=True)
    case_hash = _anonymous_id(ident["patient"], work_root)
    iteration_history = []
    previous_candidate = None
    previous_job_id = str(payload.get("previous_job_id") or "").strip()
    if mode == "refine" and not previous_job_id:
        raise ValueError("refine requires a previous candidate")
    if previous_job_id:
        previous_dir = (work_root / "prompt_jobs" / previous_job_id).resolve()
        if work_root.resolve() not in previous_dir.parents:
            raise ValueError("invalid previous job")
        previous_request = _read_json_object(previous_dir / "agent_request.json")
        if not previous_request:
            raise ValueError("上一轮任务的提示词上下文缺失，请先重新渲染一次")
        previous_local = _read_json_object(previous_dir / "local_request.json")
        previous_plan_doc = _read_json_object(previous_dir / "agent_plan.json")
        previous_plan = (previous_plan_doc.get("plan")
                         if isinstance(previous_plan_doc.get("plan"), dict)
                         else previous_plan_doc)
        if (previous_request.get("case_hash") != case_hash
                or previous_request.get("target") != target
                or previous_local.get("pool") != pool
                or previous_local.get("manifest_index") != index
                or previous_local.get("frame_index") != ident["frame"]
                or previous_local.get("source_hash") != source_hash):
            raise ValueError("previous job is not for this frame and source label")
        if mode == "refine":
            if previous_local.get("status") != "READY" or not (previous_dir / "candidate.png").is_file() or not previous_plan:
                raise ValueError("refine requires a completed previous candidate")
            prompt = str(previous_request.get("operator_instruction") or "").strip()
            with Image.open(previous_dir / "candidate.png") as prior_image:
                prior_mask = np.asarray(prior_image.convert("L")) > 127
            if prior_mask.shape != current.shape:
                raise ValueError("previous candidate shape changed")
            previous_candidate = {"stats": _mask_stats(prior_mask),
                                  "mask_grid": _mask_grid(prior_mask, size=44),
                                  "changed_pixels": int(np.count_nonzero(prior_mask != current))}
        iteration_history.append({
            "operator_instruction": previous_request.get("operator_instruction"),
            "agent_decision": previous_plan.get("decision"),
            "agent_operations": previous_plan.get("operations") or [],
            "agent_rationale": previous_plan.get("rationale"),
            "operator_feedback": ("上一轮候选结果不符合要求；请保留原提示词，反思上轮计划与候选几何，"
                                  "自行调整并仅修改有依据的局部区域。" if mode == "refine" else prompt),
        })
    if len(prompt) < 3 or len(prompt) > 2000:
        raise ValueError("repair prompt must contain 3..2000 characters")
    forbidden = [ident["patient"].lower(), "/data", ".nii", "patient="]
    if any(token and token in prompt.lower() for token in forbidden):
        raise ValueError("repair prompt contains a patient identifier or source path; describe only the edit")
    neighbors=[];requested_layers=[];current_layer_number=None;layer_count=None
    if ident["modality"] == "lge":
        metadata=frame_metadata(dataset_root,pool,index)
        if metadata.get("layer_index") is not None:
            current_layer_number=int(metadata["layer_index"])+1
            layer_count=int(metadata["total_layers"])
            requested_layers=sorted({int(v) for v in re.findall(r"(?:第)?(\d+)\s*(?=层|(?:[、,，]|和|及)\s*(?:第)?\d+\s*层)",prompt)}-{current_layer_number})[:4]
    offsets={-1,1}
    if current_layer_number is not None:
        offsets.update(number-current_layer_number for number in requested_layers if 1<=number<=layer_count)
    for delta in sorted(offsets):
        try:
            _,_,_,neighbor_raw,_,_,nk,_=_resolve_plane(dataset_root,pool,index,target,delta)
            neighbor_stack,_=_stack(neighbor_raw);neighbor_mask=neighbor_stack[...,nk]>0.5
            item={"offset":delta,**_mask_stats(neighbor_mask),
                  "components_detail":_components_detail(neighbor_mask),"mask_grid":_mask_grid(neighbor_mask,size=36)}
            if current_layer_number is not None:item["layer_number"]=current_layer_number+delta
            neighbors.append(item)
        except (IndexError,FileNotFoundError,ValueError):continue
    provided_layers=sorted(n["layer_number"] for n in neighbors if "layer_number" in n)
    reference_context={"axis":"layer" if ident["modality"]=="lge" else "frame",
        "current_layer_number":current_layer_number,"requested_layers":requested_layers,
        "provided_layers":provided_layers,"missing_layers":[n for n in requested_layers if n not in provided_layers],
        "target":target,"explicit_reference_limit":4}
    job_id = datetime.now().strftime("%Y%m%dT%H%M%S") + "_prompt_" + uuid.uuid4().hex[:10]
    job_dir = work_root / "prompt_jobs" / job_id
    local = {
        "job_id": job_id,
        "created_at": utcnow(),
        "dataset_root": str(dataset_root),
        "pool": pool,
        "manifest_index": index,
        "target": target,
        "source_label": str(label_path),
        "source_hash": source_hash,
        "frame_index": ident["frame"],
        "status": "QUEUED",
        "execution_scope": "interactive_prompt",
    }
    issue_types = [
        value.strip() for value in (row.get("issues") or row.get("decision") or "").split(";")
        if value.strip()
    ]
    frame_window = [
        item for item in frame_table
        if abs(int(item["frame"]) - int(k)) <= 3
    ]
    frame_summary = {}
    if frame_table:
        areas = [int(item["area"]) for item in frame_table]
        frame_summary = {
            "frames": len(frame_table),
            "area_min_median_max": [min(areas), int(round(float(np.median(areas)))), max(areas)],
            "zero_hole_frames": [int(item["frame"]) for item in frame_table if int(item["holes"]) == 0][:12],
            "three_plus_component_frames": [
                int(item["frame"]) for item in frame_table if int(item["components"]) >= 3
            ][:12],
        }
    current_stats = {**_mask_stats(current),
                     "component_descriptors": _component_descriptors(current),
                     "mask_grid": _mask_grid(current, size=44),
                     "components_detail": _components_detail(current)}
    qc_context = _prompt_qc_context(
        ident["modality"], target, pool, issue_types, current_stats, sum(n["offset"] in {-1,1} for n in neighbors)
    )
    request = {
        "schema_version": "prompt-repair-v2",
        "mode": mode,
        "job_id": job_id,
        "case_hash": case_hash,
        "modality": ident["modality"],
        "target": target,
        "issue_types": issue_types,
        "operator_instruction": prompt,
        "frame_window": frame_window,
        "frame_summary": frame_summary,
        "current_frame_index": int(k),
        "current": current_stats,
        "neighbors": neighbors,
        "reference_context": reference_context,
        "qc_context": qc_context,
        "iteration_history": iteration_history,
        "previous_candidate": previous_candidate,
        "human_demonstrations": _similar_prompt_examples(
            work_root, ident["modality"], target, issue_types
        ),
        "myocardium_constraint": myo_stats,
        "allowed_operations": qc_context["allowed_operations"],
        "constraints": {
            "max_operations": 3,
            "max_morphology_radius": 2,
            "max_adjacent_consensus_radius": 6,
            "require_qc_context_compliance": True,
            "visual_review_required": True,
            "production_label_modified": False,
        },
    }
    assigned_role = role_for_label_task(request)
    local["assigned_role"] = assigned_role
    request["assigned_role"] = assigned_role
    job_dir.mkdir(parents=True, exist_ok=False)
    (job_dir / "local_request.json").write_text(
        json.dumps(local, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (job_dir / "agent_request.json").write_text(
        json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    from app.prompt_dispatch import grant
    grant(work_root,job_id)
    return {"job_id": job_id, "status": "QUEUED", "assigned_role": assigned_role}


def _bounded_holes(mask: np.ndarray, max_area: int) -> np.ndarray:
    holes = ndi.binary_fill_holes(mask) & ~mask
    labels, count = ndi.label(holes, structure=np.ones((3, 3)))
    fill = np.zeros_like(mask, dtype=bool)
    for label_id in range(1, count + 1):
        component = labels == label_id
        if int(component.sum()) <= max_area:
            fill |= component
    return mask | fill


def _shift_to_centroid(mask: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Integer-align one adjacent mask to the current frame without interpolation."""
    if not mask.any() or not reference.any():
        return mask.copy()
    src_y, src_x = ndi.center_of_mass(mask)
    dst_y, dst_x = ndi.center_of_mass(reference)
    return ndi.shift(
        mask.astype(np.uint8),
        shift=(int(round(dst_y - src_y)), int(round(dst_x - src_x))),
        order=0,
        mode="constant",
        cval=0,
        prefilter=False,
    ) > 0


def _adjacent_consensus(dataset_root: Path, pool: str, index: int,
                        target: str, original: np.ndarray) -> np.ndarray:
    supports = []
    for delta in (-1, 1):
        try:
            _, _, _, raw, _, _, nk, _ = _resolve_plane(
                dataset_root, pool, index, target, delta
            )
            stack, _ = _stack(raw)
            neighbor = stack[..., nk] > 0.5
            if neighbor.shape == original.shape and neighbor.any():
                supports.append(_shift_to_centroid(neighbor, original))
        except (IndexError, FileNotFoundError, ValueError):
            continue
    if len(supports) < 2:
        raise ValueError("adjacent consensus requires valid support on both sides")
    return np.logical_and(supports[0], supports[1])


def _polygon_mask(shape: tuple[int, int], points: Any) -> np.ndarray:
    if not isinstance(points, list) or not 3 <= len(points) <= 24:
        raise ValueError("polygon operation requires 3..24 vertices")
    height, width = shape
    vertices = []
    for point in points:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError("polygon vertices must be [x,y]")
        x, y = int(point[0]), int(point[1])
        if not 0 <= x <= 1000 or not 0 <= y <= 1000:
            raise ValueError("polygon vertices must use normalized 0..1000 coordinates")
        vertices.append((round(x / 1000 * (width - 1)), round(y / 1000 * (height - 1))))
    canvas = Image.new("1", (width, height), 0)
    ImageDraw.Draw(canvas).polygon(vertices, fill=1)
    return np.asarray(canvas, dtype=bool)


def execute_prompt_repair(work_root: Path, job_id: str,
                          plan: dict[str, Any]) -> dict[str, Any]:
    job_dir = (work_root / "prompt_jobs" / job_id).resolve()
    if work_root.resolve() not in job_dir.parents:
        raise ValueError("invalid prompt job id")
    local_path = job_dir / "local_request.json"
    local = json.loads(local_path.read_text(encoding="utf-8"))
    dataset_root = Path(local["dataset_root"])
    _, ident = candidate(dataset_root, local["pool"], int(local["manifest_index"]))
    _, label_path, _, label_raw, _, _, k, _ = _resolve_plane(
        dataset_root, local["pool"], int(local["manifest_index"]), local["target"], 0
    )
    if _sha256(label_path) != local["source_hash"]:
        raise ValueError("source label changed while prompt repair was queued")
    stack, _ = _stack(label_raw)
    original = stack[..., k] > 0.5
    edited = original.copy()
    operation_names = [str(item.get("operation") or "") for item in plan.get("operations", [])[:3]]
    if plan.get("decision") == "apply_operations":
        for operation in plan.get("operations", [])[:3]:
            name = operation.get("operation")
            if name == "keep_largest_component":
                raise ValueError("largest-component selection lacks trajectory support; manual review required")
            elif name == "remove_components_below_area":
                maximum = max(1, min(int(operation.get("max_area") or 32), 500))
                labels, count = ndi.label(edited, structure=np.ones((3, 3)))
                protected_ids: set[int] = set()
                if ident["modality"] == "cmr" and count:
                    areas = [(label_id, int((labels == label_id).sum()))
                             for label_id in range(1, count + 1)]
                    protected_ids.update(label_id for label_id, _ in sorted(
                        areas, key=lambda item: item[1], reverse=True
                    )[:2])
                mi_support = None
                if ident["modality"] == "lge" and local["target"] == "myo":
                    _, mi_path = _paths(dataset_root, ident, "mi")
                    mi_stack, _ = _stack(np.asarray(nib.load(str(mi_path)).get_fdata()))
                    mi_support = mi_stack[..., min(k, mi_stack.shape[-1] - 1)] > 0.5
                adjacent_support = None
                if ident["modality"] == "lge" and local["target"] == "mi":
                    supports = []
                    for delta in (-1, 1):
                        try:
                            _, _, _, raw, _, _, nk, _ = _resolve_plane(
                                dataset_root, local["pool"], int(local["manifest_index"]),
                                "mi", delta,
                            )
                            neighbor_stack, _ = _stack(raw)
                            supports.append(neighbor_stack[..., nk] > 0.5)
                        except (IndexError, FileNotFoundError, ValueError):
                            continue
                    if supports:
                        adjacent_support = np.logical_or.reduce(supports)
                for label_id in range(1, count + 1):
                    component = labels == label_id
                    supported = label_id in protected_ids
                    if mi_support is not None and bool((component & mi_support).any()):
                        supported = True
                    if adjacent_support is not None and adjacent_support.any():
                        cy, cx = ndi.center_of_mass(component)
                        distance = ndi.distance_transform_edt(~adjacent_support)
                        supported = supported or float(distance[int(round(cy)), int(round(cx))]) <= 15.0
                    if int(component.sum()) <= maximum and not supported:
                        edited[component] = False
            elif name == "fill_internal_holes_below_area":
                if local["target"] == "myo":
                    raise ValueError("myocardial blood-pool holes are physiological; manual review required")
                edited = _bounded_holes(edited, max(1, min(int(operation.get("max_area") or 64), 500)))
            elif name in {"binary_close", "binary_dilate", "binary_erode"}:
                if ident["modality"] == "cmr" and name in {"binary_dilate", "binary_erode"}:
                    raise ValueError("global CMR dilation/erosion is not allowed; manual review required")
                radius = max(1, min(int(operation.get("radius") or 1), 2))
                yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
                structure = xx * xx + yy * yy <= radius * radius
                fn = {"binary_close": ndi.binary_closing,
                      "binary_dilate": ndi.binary_dilation,
                      "binary_erode": ndi.binary_erosion}[name]
                edited = fn(edited, structure=structure)
            elif name in {"remove_components_in_region", "keep_components_in_region"}:
                region = str(operation.get("region") or "")
                if region:
                    labels, count = ndi.label(edited, structure=np.ones((3, 3)))
                    if count:
                        hh = max(1, edited.shape[0] - 1)
                        ww = max(1, edited.shape[1] - 1)
                        keep = np.zeros(count + 1, dtype=bool)
                        for label_id in range(1, count + 1):
                            cy, cx = ndi.center_of_mass(labels == label_id)
                            inside = _region_tag(float(cy) / hh, float(cx) / ww) == region
                            keep[label_id] = inside if name == "keep_components_in_region" else (not inside)
                        edited = keep[labels]
            elif name in {"erase_box", "fill_box", "erase_circle", "fill_circle"}:
                H, W = edited.shape
                def _px(v, dim):
                    return max(0, min(dim - 1, int(round((v or 0) / 1000.0 * (dim - 1)))))
                if name.endswith("box"):
                    xa, xb = sorted((_px(operation.get("x0"), W), _px(operation.get("x1"), W)))
                    ya, yb = sorted((_px(operation.get("y0"), H), _px(operation.get("y1"), H)))
                    block = np.zeros_like(edited)
                    block[ya:yb + 1, xa:xb + 1] = True
                else:
                    ccx, ccy = _px(operation.get("cx"), W), _px(operation.get("cy"), H)
                    rr = max(1, _px(operation.get("r"), min(H, W)))
                    yy, xx = np.ogrid[:H, :W]
                    block = ((xx - ccx) ** 2 + (yy - ccy) ** 2) <= rr * rr
                edited = (edited & ~block) if name.startswith("erase") else (edited | block)
            elif name in {"erase_polygon", "fill_polygon"}:
                block = _polygon_mask(edited.shape, operation.get("points"))
                edited = (edited & ~block) if name.startswith("erase") else (edited | block)
            elif name == "fill_from_adjacent_consensus":
                radius = max(1, min(int(operation.get("radius") or 3), 6))
                consensus = _adjacent_consensus(
                    dataset_root, local["pool"], int(local["manifest_index"]),
                    local["target"], original,
                )
                # Adjacent evidence may add only pixels close to the current
                # contour. This repairs a local gap without copying an entire
                # neighboring anatomy into the current frame/layer.
                near_current = ndi.binary_dilation(
                    edited, structure=np.ones((3, 3)), iterations=radius
                )
                addition = consensus & near_current
                region = str(operation.get("region") or "")
                if region:
                    addition &= _region_mask(edited.shape, region)
                edited |= addition
            elif name == "erase_region_pixels":
                region = str(operation.get("region") or "")
                if region:
                    edited &= ~_region_mask(edited.shape, region)
            elif name == "clip_mi_to_myo" and local["target"] == "mi":
                _, myo_path = _paths(dataset_root, ident, "myo")
                myo_stack, _ = _stack(np.asarray(nib.load(str(myo_path)).get_fdata()))
                edited &= myo_stack[..., min(k, myo_stack.shape[-1] - 1)] > 0.5
    if local["target"] == "mi":
        _, myo_path = _paths(dataset_root, ident, "myo")
        myo_stack, _ = _stack(np.asarray(nib.load(str(myo_path)).get_fdata()))
        edited &= myo_stack[..., min(k, myo_stack.shape[-1] - 1)] > 0.5
    if local["target"] == "myo" and "binary_close" in operation_names:
        before_holes = int(_mask_stats(original)["holes"])
        after_holes = int(_mask_stats(edited)["holes"])
        if before_holes > 0 and after_holes < before_holes:
            raise ValueError("binary close removed a physiological blood-pool hole; manual review required")
    changed = edited ^ original
    changed_fraction = float(changed.sum()) / max(float(original.sum()), 1.0)
    if changed_fraction > 0.35:
        raise ValueError(f"prompt candidate changes {changed_fraction:.1%} of source area; manual review required")
    decision = str(plan.get("decision") or "manual_required")
    if decision == "apply_operations" and not changed.any():
        decision = "no_change"
    if decision == "apply_operations":
        Image.fromarray((edited * 255).astype(np.uint8)).save(job_dir / "candidate.png")
    local.update({
        "status": "READY" if decision == "apply_operations" else decision.upper(),
        "completed_at": utcnow(),
        "changed_pixels": int(changed.sum()),
        "changed_fraction": round(changed_fraction, 6),
    })
    from app.control_checkpoint import save_journal
    save_journal(local_path,local)
    return local


def validate_prompt_source(work_root: Path, job_id: str) -> None:
    from app.prompt_dispatch import job_folder
    folder=job_folder(work_root,job_id)
    local=json.loads((folder/"local_request.json").read_text(encoding="utf-8"))
    dataset=Path(local["dataset_root"]).resolve();source=Path(local["source_label"]).resolve()
    if not any(source.is_relative_to((dataset/relative).resolve()) for relative in TARGET_DIRS.values()):
        raise ValueError("prompt source is outside governed labels")
    if _sha256(source)!=local["source_hash"]:
        raise ValueError("当前标签已被其他流程更新，请刷新当前样本后重新提交")


def prompt_repair_status(work_root: Path, job_id: str) -> dict[str, Any]:
    job_dir = (work_root / "prompt_jobs" / job_id).resolve()
    if work_root.resolve() not in job_dir.parents:
        raise ValueError("invalid prompt job id")
    local_path=job_dir/"local_request.json"
    local=json.loads(local_path.read_text(encoding="utf-8"))
    fields=("job_id","status","changed_pixels","changed_fraction","assigned_role",
            "pool","manifest_index","target","source_hash")
    result={key:local.get(key) for key in fields}
    result["error"]=local.get("terminal_error")
    result["elapsed_seconds"]=max(0,int(time.time()-local_path.stat().st_mtime))
    active={"QUEUED","RETRYING","PROCESSING"}
    terminal={"READY","NO_CHANGE","MANUAL_REQUIRED","ERROR"}
    running=job_dir/"agent_request.running"
    dispatch=_read_json_object(job_dir/"dispatch.json")
    if dispatch:
        result["elapsed_seconds"]=max(0,int(time.time()-float(dispatch["launched_at"])))
    if result.get("status") in active:
        from app.prompt_dispatch import worker_alive
        alive=worker_alive(job_dir)
        if not alive and (not running.exists() or (job_dir/"claim.json").exists()):
            # A worker can publish its terminal result between our first read
            # and process exit. Reload before declaring a failure.
            latest=_read_json_object(local_path)
            if latest.get("status") in terminal:
                result.update({key:latest.get(key) for key in fields})
                result["error"]=latest.get("terminal_error")
            elif dispatch or (job_dir/"claim.json").exists():
                result.update(status="ERROR",error="提示词工作进程已退出，请重新渲染")
            elif result["elapsed_seconds"]>15:
                result.update(status="STALLED",error="提示词工作进程未启动，请重新渲染")
        # Legacy watch workers may own a running claim without a dispatch PID.
        # Their result remains pending until completion or the bounded timeout.
        if result.get("status") in active and running.exists() and time.time()-running.stat().st_mtime>150:
            result.update(status="STALLED",error="本轮提示词处理超时，请检查任务状态后重试")
    plan_doc=_read_json_object(job_dir/"agent_plan.json")
    if plan_doc:result["plan"]=plan_doc
    plan=plan_doc.get("plan") or plan_doc
    decision=plan.get("decision");result["decision"]=decision
    if result.get("status") in {"READY","NO_CHANGE","MANUAL_REQUIRED"}:
        if decision=="manual_required":result["status"]="MANUAL_REQUIRED"
        elif decision=="no_change" or (decision=="apply_operations" and not plan.get("operations")):
            result["status"]="NO_CHANGE"
    if result.get("status")=="ERROR" and not result.get("error"):
        result["error"]=_read_json_object(job_dir/"agent_error.json").get("error")
    result["candidate_ready"]=(job_dir/"candidate.png").exists() and result.get("status")=="READY"
    return result


def prompt_candidate_png(work_root: Path, job_id: str) -> bytes:
    job_dir = (work_root / "prompt_jobs" / job_id).resolve()
    if work_root.resolve() not in job_dir.parents:
        raise ValueError("invalid prompt job id")
    return (job_dir / "candidate.png").read_bytes()


def apply_edit(dataset_root: Path, work_root: Path, edit_id: str, *, reviewer: str,
               confirmation: str) -> dict[str, Any]:
    from app.rsi_postpromotion import file_lock
    with file_lock(dataset_root / "label_iteration/control/label_writer.lock", exclusive=True):
        return _apply_edit_locked(dataset_root, work_root, edit_id, reviewer=reviewer, confirmation=confirmation)


def _apply_edit_locked(dataset_root: Path, work_root: Path, edit_id: str, *, reviewer: str,
                       confirmation: str) -> dict[str, Any]:
    if not reviewer.strip():
        raise ValueError("reviewer is required")
    if confirmation != "APPLY":
        raise ValueError("type APPLY to confirm production-label replacement")
    edit_dir = (work_root / "edits" / edit_id).resolve()
    if work_root.resolve() not in edit_dir.parents:
        raise ValueError("invalid edit id")
    provenance_path = edit_dir / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("status") != "STAGED":
        raise ValueError(f"edit is not staged: {provenance.get('status')}")
    source = Path(provenance["source_label"]).resolve()
    candidate_path = Path(provenance["candidate_label"]).resolve()
    allowed_roots = [(dataset_root / relative).resolve()
                     for relative in TARGET_DIRS.values()]
    if (not any(root in source.parents for root in allowed_roots)
            or candidate_path != (edit_dir / "candidate_label.nii.gz").resolve()
            or not candidate_path.is_file()):
        raise ValueError("edit paths are outside the governed roots")
    if _sha256(source) != provenance["source_hash"]:
        raise ValueError("source label changed since staging; application aborted")
    applied_context = {}
    if provenance.get("modality") == "lge" and provenance.get("target") in {"myo", "mi"}:
        relative = source.relative_to((dataset_root / TARGET_DIRS[("lge", provenance["target"])]).resolve())
        image = dataset_root / "images/lge" / relative
        paired = dataset_root / TARGET_DIRS[("lge", "myo")] / relative
        applied_context = {"image": _sha256(image)}
        if provenance["target"] == "mi":
            applied_context["myo"] = _sha256(paired)
    backup = dataset_root / "label_iteration/manual_edits/backups" / edit_id / source.name
    backup.parent.mkdir(parents=True, exist_ok=False)
    shutil.copy2(source, backup)
    shutil.copy2(candidate_path, source)
    provenance.update({
        "status": "APPLIED",
        "applied_at": utcnow(),
        "applied_by": reviewer.strip(),
        "backup": str(backup),
        "applied_hash": _sha256(source),
    })
    if applied_context:
        provenance["applied_context"] = applied_context
    provenance_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    result = {
        "edit_id": edit_id, "status": "APPLIED", "backup": str(backup),
        "pool": provenance.get("pool"),
        "source_run_id": provenance.get("source_run_id"),
        "source_action_id": provenance.get("source_action_id"),
    }
    if provenance.get("pool") == "cmr_boundary_manifest":
        try:
            review = set_boundary_review_status(
                dataset_root, "cmr_boundary_manifest", int(provenance["manifest_index"]),
                "checked", reviewer=reviewer,
            )
            result["boundary_review_status"] = review["status"]
        except (ValueError, IndexError, FileNotFoundError, OSError) as exc:
            # The production edit is already applied and backed up. Report a
            # recoverable review-state mismatch without claiming apply failed.
            result["boundary_review_warning"] = str(exc)
    return result


def edit_status(work_root: Path, edit_id: str) -> dict[str, Any]:
    edit_dir = (work_root / "edits" / edit_id).resolve()
    if work_root.resolve() not in edit_dir.parents:
        raise ValueError("invalid edit id")
    provenance = json.loads((edit_dir / "provenance.json").read_text(encoding="utf-8"))
    result: dict[str, Any] = {
        "edit_id": edit_id,
        "status": provenance.get("status"),
        "suggested_pattern": provenance.get("suggested_pattern"),
    }
    analysis = edit_dir / "agent_analysis.json"
    error = edit_dir / "agent_error.json"
    if analysis.exists():
        result["agent_status"] = "ANALYZED"
        result["agent_analysis"] = json.loads(analysis.read_text(encoding="utf-8"))
    elif error.exists():
        result["agent_status"] = "ERROR_RETRYING"
        result["agent_error"] = json.loads(error.read_text(encoding="utf-8")).get("error")
    elif (edit_dir / "agent_request.running").exists():
        result["agent_status"] = "RUNNING"
    elif (edit_dir / "agent_request.json").exists():
        result["agent_status"] = "QUEUED"
    else:
        result["agent_status"] = "NOT_REQUESTED"
    return result
