from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import nibabel as nib
import numpy as np
import yaml
from scipy import ndimage as ndi


FORBIDDEN_DERIVED_TOKENS = ("warp", "warpp", "registr")


@dataclass(frozen=True)
class PairSpec:
    dataset: str
    tier: str
    target: str
    image: Path
    label: Path
    myo: Path | None
    relative_id: str
    case_id: str
    slice_position: float
    reviewed: bool


def _assert_original(path: Path) -> None:
    lowered = str(path).lower()
    if any(token in lowered for token in FORBIDDEN_DERIVED_TOKENS):
        raise ValueError(f"derived/registered path is forbidden: {path}")


def _files(root: Path) -> list[Path]:
    _assert_original(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    return sorted(root.rglob("*.nii.gz")) + sorted(root.rglob("*.nii"))


def _stem(path: Path) -> str:
    return path.name[:-7] if path.name.endswith(".nii.gz") else path.stem


def _case_and_position(relative: Path, prefix: str = "") -> tuple[str, float]:
    parent = relative.parent.as_posix()
    stem = _stem(relative)
    match = re.fullmatch(r"(.+)_slice(\d+)", stem, flags=re.IGNORECASE)
    if match:
        return match.group(1), float(match.group(2))
    try:
        position = float(stem)
    except ValueError:
        position = float(int(hashlib.sha256(stem.encode()).hexdigest()[:8], 16))
    case = parent if parent != "." else prefix or "flat"
    return case, position


def _resolve_image(image_root: Path, relative: Path, relative_prefix: str = "") -> Path:
    candidates = [image_root / relative]
    if relative_prefix:
        candidates.insert(0, image_root / relative_prefix / relative)
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]


def load_specs(config_path: Path) -> tuple[list[PairSpec], dict[str, Any]]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    specs: list[PairSpec] = []
    for dataset in config["datasets"]:
        name, tier = dataset["name"], dataset.get("tier", "gold")
        for target, item in dataset.get("targets", {}).items():
            image_root = Path(item["image_dir"])
            label_root = Path(item["label_dir"])
            myo_root = Path(item["myo_dir"]) if item.get("myo_dir") else None
            reviewed_root = Path(item["reviewed_dir"]) if item.get("reviewed_dir") else None
            prefix = str(item.get("relative_prefix") or "")
            for label in _files(label_root):
                relative = label.relative_to(label_root)
                image = _resolve_image(image_root, relative, prefix)
                myo = (myo_root / relative) if myo_root else None
                case, position = _case_and_position(relative, prefix)
                review_relative = Path(prefix) / relative if prefix else relative
                reviewed = bool(reviewed_root and (
                    (reviewed_root / review_relative).with_suffix("").with_suffix(".png").is_file()
                    or (reviewed_root / review_relative.parent / f"{_stem(relative)}.png").is_file()
                ))
                specs.append(PairSpec(
                    dataset=name, tier=tier, target=target, image=image, label=label,
                    myo=myo, relative_id=(Path(prefix) / relative).as_posix() if prefix else relative.as_posix(),
                    case_id=f"{name}:{case}", slice_position=position, reviewed=reviewed,
                ))
    return specs, config


def _stack(array: np.ndarray) -> np.ndarray:
    arr = np.squeeze(np.asarray(array))
    if arr.ndim == 2:
        return arr[..., None]
    if arr.ndim != 3:
        raise ValueError(f"expected 2-D/3-D data, got {arr.shape}")
    axis = int(np.argmin(arr.shape))
    return np.moveaxis(arr, axis, -1)


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    denominator = int(a.sum() + b.sum())
    return 1.0 if denominator == 0 else float(2 * np.logical_and(a, b).sum() / denominator)


def _mask_features(mask: np.ndarray) -> dict[str, float]:
    mask = np.asarray(mask, dtype=bool)
    labels, count = ndi.label(mask, structure=np.ones((3, 3)))
    areas = sorted((int((labels == i).sum()) for i in range(1, count + 1)), reverse=True)
    area = int(mask.sum())
    filled = ndi.binary_fill_holes(mask)
    holes = filled & ~mask
    hole_labels, hole_count = ndi.label(holes, structure=np.ones((3, 3)))
    hole_areas = [int((hole_labels == i).sum()) for i in range(1, hole_count + 1)]
    eroded = ndi.binary_erosion(mask, structure=np.ones((3, 3)), border_value=0)
    perimeter = int(np.logical_xor(mask, eroded).sum())
    yy, xx = np.where(mask)
    return {
        "area": float(area),
        "area_fraction": float(area / mask.size),
        "components": float(count),
        "largest_component_fraction": float(areas[0] / area) if area else 1.0,
        "second_component_fraction": float(areas[1] / area) if area and len(areas) > 1 else 0.0,
        "extra_after_two_fraction": float(sum(areas[2:]) / area) if area else 0.0,
        "tiny_component_fraction": float(sum(a for a in areas if a <= 8) / area) if area else 0.0,
        "holes": float(hole_count),
        "small_hole_fraction": float(sum(a for a in hole_areas if a <= 16) / max(area, 1)),
        "touches_border": float(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()),
        "compactness": float(4 * math.pi * area / max(perimeter * perimeter, 1)),
        "centroid_x": float(xx.mean() / mask.shape[1]) if len(xx) else math.nan,
        "centroid_y": float(yy.mean() / mask.shape[0]) if len(yy) else math.nan,
    }


def _image_features(image: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    image = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(image)
    values = image[finite]
    if not len(values):
        return {"finite_fraction": 0.0, "dynamic_range": 0.0,
                "boundary_gradient_ratio": 0.0, "inside_outside_z": 0.0}
    clean = np.where(finite, image, float(np.median(values)))
    lo, hi = np.percentile(values, [1, 99])
    gy, gx = np.gradient(clean)
    gradient = np.hypot(gx, gy)
    boundary = np.logical_xor(mask, ndi.binary_erosion(mask, structure=np.ones((3, 3))))
    global_gradient = float(np.median(gradient[finite])) + 1e-6
    boundary_gradient = float(np.median(gradient[boundary])) if boundary.any() else 0.0
    outside_ring = ndi.binary_dilation(mask, iterations=2) & ~mask
    inside = clean[mask]
    outside = clean[outside_ring]
    scale = float(np.median(np.abs(values - np.median(values))) * 1.4826) + 1e-6
    contrast = (float(np.median(inside)) - float(np.median(outside))) / scale if len(inside) and len(outside) else 0.0
    return {
        "finite_fraction": float(finite.mean()),
        "dynamic_range": float(hi - lo),
        "boundary_gradient_ratio": boundary_gradient / global_gradient,
        "inside_outside_z": contrast,
    }


def _mean(values: Iterable[float], default: float = math.nan) -> float:
    finite = [float(x) for x in values if np.isfinite(x)]
    return float(np.mean(finite)) if finite else default


def extract_pair(spec: PairSpec) -> dict[str, Any]:
    base = {
        "dataset": spec.dataset, "tier": spec.tier, "target": spec.target,
        "relative_id": spec.relative_id, "case_id": spec.case_id,
        "slice_position": spec.slice_position, "reviewed": spec.reviewed,
        "image_exists": spec.image.is_file(), "label_exists": spec.label.is_file(),
    }
    if not spec.image.is_file() or not spec.label.is_file():
        return {**base, "valid": False, "failure": "missing_image_or_label"}
    try:
        image_nii, label_nii = nib.load(str(spec.image)), nib.load(str(spec.label))
        image_raw = np.asanyarray(image_nii.dataobj)
        label_raw = np.asanyarray(label_nii.dataobj)
        image, label = _stack(image_raw), _stack(label_raw)
    except Exception as exc:
        return {**base, "valid": False, "failure": f"load_error:{type(exc).__name__}"}
    if image.shape[:2] != label.shape[:2]:
        return {**base, "valid": False, "failure": "shape_mismatch",
                "image_shape": list(image.shape), "label_shape": list(label.shape)}
    if label.shape[-1] not in (1, image.shape[-1]):
        return {**base, "valid": False, "failure": "frame_mismatch",
                "image_shape": list(image.shape), "label_shape": list(label.shape)}
    label = label > 0.5
    frames = image.shape[-1]
    if label.shape[-1] == 1 and frames > 1:
        label = np.repeat(label, frames, axis=-1)
    masks = [label[..., i] for i in range(frames)]
    active_indices = [i for i, mask in enumerate(masks) if mask.any()]
    sparse_phase = spec.target == "cmr_myo" and frames > 5 and 0 < len(active_indices) <= 2
    scored_indices = active_indices if sparse_phase else list(range(frames))
    scored_masks = [masks[i] for i in scored_indices]
    mask_features = [_mask_features(mask) for mask in scored_masks]
    image_features = [_image_features(image[..., i], masks[i]) for i in scored_indices]
    consecutive_dice = ([] if sparse_phase else
                        [_dice(masks[i - 1], masks[i]) for i in range(1, frames)])
    area_values = [item["area"] for item in mask_features]
    component_values = [item["components"] for item in mask_features]
    hole_values = [item["holes"] for item in mask_features]
    component_mode = max(set(component_values), key=component_values.count) if component_values else 0.0
    hole_mode = max(set(hole_values), key=hole_values.count) if hole_values else 0.0
    isolated_area_jumps = []
    for i in ([] if sparse_phase else range(1, frames - 1)):
        expected = (area_values[i - 1] + area_values[i + 1]) / 2.0
        isolated_area_jumps.append(abs(area_values[i] - expected) / max(expected, 1.0))
    centroid_jumps = []
    for left, right in zip(mask_features, mask_features[1:]):
        if all(np.isfinite([left["centroid_x"], left["centroid_y"], right["centroid_x"], right["centroid_y"]])):
            centroid_jumps.append(math.hypot(
                right["centroid_x"] - left["centroid_x"],
                right["centroid_y"] - left["centroid_y"],
            ))
    record = {
        **base, "valid": True, "failure": "", "frames": frames,
        "annotated_frames": len(active_indices), "sparse_phase": sparse_phase,
        "binary_fraction": float(np.mean(np.isclose(label_raw, 0) | np.isclose(label_raw, 1))),
        "affine_match": float(np.allclose(image_nii.affine, label_nii.affine, atol=1e-3)),
        "area": _mean((item["area"] for item in mask_features), 0.0),
        "area_fraction": _mean((item["area_fraction"] for item in mask_features), 0.0),
        "components": _mean((item["components"] for item in mask_features), 0.0),
        "largest_component_fraction": _mean((item["largest_component_fraction"] for item in mask_features), 1.0),
        "extra_after_two_fraction": _mean((item["extra_after_two_fraction"] for item in mask_features), 0.0),
        "tiny_component_fraction": _mean((item["tiny_component_fraction"] for item in mask_features), 0.0),
        "max_second_component_fraction": max((item["second_component_fraction"] for item in mask_features), default=0.0),
        "holes": _mean((item["holes"] for item in mask_features), 0.0),
        "small_hole_fraction": _mean((item["small_hole_fraction"] for item in mask_features), 0.0),
        "border_frame_fraction": _mean((item["touches_border"] for item in mask_features), 0.0),
        "compactness": _mean((item["compactness"] for item in mask_features), 0.0),
        "finite_fraction": _mean((item["finite_fraction"] for item in image_features), 0.0),
        "dynamic_range": _mean((item["dynamic_range"] for item in image_features), 0.0),
        "boundary_gradient_ratio": _mean((item["boundary_gradient_ratio"] for item in image_features), 0.0),
        "inside_outside_z": _mean((item["inside_outside_z"] for item in image_features), 0.0),
        "temporal_dice": _mean(consecutive_dice, 1.0),
        "temporal_dice_p05": (float(np.quantile(consecutive_dice, .05))
                               if consecutive_dice else math.nan),
        "temporal_centroid_jump": _mean(centroid_jumps, 0.0),
        "temporal_centroid_jump_max": max(centroid_jumps, default=0.0),
        "temporal_area_cv": (float(np.std(area_values) / max(np.mean(area_values), 1.0))
                             if not sparse_phase else math.nan),
        "isolated_area_jump": max(isolated_area_jumps, default=0.0),
        "component_mode_mismatch_fraction": float(np.mean([
            value > component_mode for value in component_values
        ])),
        "hole_mode_mismatch_fraction": float(np.mean([value != hole_mode for value in hole_values])),
        "empty_frame_fraction": float(np.mean([not mask.any() for mask in masks])),
    }
    if spec.target == "lge_mi":
        if spec.myo is None or not spec.myo.is_file():
            record.update({"myo_exists": False, "mi_containment": 0.0, "mi_signal_z": math.nan})
        else:
            myo = _stack(np.asanyarray(nib.load(str(spec.myo)).dataobj)) > 0.5
            if myo.shape[:2] != label.shape[:2]:
                record.update({"myo_exists": True, "mi_containment": 0.0, "mi_signal_z": math.nan})
            else:
                myo_mask = myo[..., 0]
                mi_mask = label[..., 0]
                record["myo_exists"] = True
                record["mi_containment"] = (1.0 if not mi_mask.any() else
                                            float(np.logical_and(mi_mask, myo_mask).sum() / mi_mask.sum()))
                values = image[..., 0]
                scar, normal = values[mi_mask], values[myo_mask & ~mi_mask]
                scale = float(np.median(np.abs(normal - np.median(normal))) * 1.4826) + 1e-6 if len(normal) else 1.0
                record["mi_signal_z"] = ((float(np.median(scar)) - float(np.median(normal))) / scale
                                         if len(scar) and len(normal) else math.nan)
    return record


PROFILE_METRICS = {
    "cmr_myo": ["area_fraction", "largest_component_fraction", "extra_after_two_fraction",
                "holes", "boundary_gradient_ratio", "temporal_dice",
                "temporal_dice_p05", "temporal_centroid_jump", "temporal_area_cv",
                "temporal_centroid_jump_max",
                "isolated_area_jump", "component_mode_mismatch_fraction",
                "hole_mode_mismatch_fraction"],
    "lge_myo": ["area_fraction", "largest_component_fraction", "extra_after_two_fraction",
                "holes", "boundary_gradient_ratio"],
    "lge_mi": ["area_fraction", "components", "largest_component_fraction",
               "boundary_gradient_ratio", "mi_signal_z", "mi_containment"],
}


def build_profile(records: list[dict[str, Any]]) -> dict[str, Any]:
    profile: dict[str, Any] = {"schema_version": "image-label-quality-v1", "targets": {}}
    for target, metrics in PROFILE_METRICS.items():
        subset = [r for r in records if r.get("valid") and r["target"] == target]
        target_profile: dict[str, Any] = {"n": len(subset), "metrics": {}}
        for metric in metrics:
            values = np.asarray([r.get(metric, math.nan) for r in subset], dtype=float)
            values = values[np.isfinite(values)]
            if not len(values):
                continue
            target_profile["metrics"][metric] = {
                "q01": float(np.quantile(values, .01)), "q05": float(np.quantile(values, .05)),
                "median": float(np.median(values)), "q95": float(np.quantile(values, .95)),
                "q99": float(np.quantile(values, .99)),
            }
        profile["targets"][target] = target_profile
    return profile


def _low_good(value: float, good: float, bad: float) -> float:
    if not np.isfinite(value):
        return math.nan
    if value <= good:
        return 100.0
    if value >= bad:
        return 0.0
    return 100.0 * (bad - value) / max(bad - good, 1e-8)


def _high_good(value: float, bad: float, good: float) -> float:
    if not np.isfinite(value):
        return math.nan
    if value >= good:
        return 100.0
    if value <= bad:
        return 0.0
    return 100.0 * (value - bad) / max(good - bad, 1e-8)


def _weighted(parts: dict[str, tuple[float, float]]) -> tuple[float, dict[str, float]]:
    used = {name: (score, weight) for name, (score, weight) in parts.items() if np.isfinite(score)}
    total = sum(weight for _, weight in used.values())
    return (sum(score * weight for score, weight in used.values()) / max(total, 1e-8),
            {name: round(score, 3) for name, (score, _) in used.items()})


def score_record(record: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    if not record.get("valid"):
        return {**record, "score": 0.0, "grade": "F", "confidence": 1.0,
                "subscores": {"integrity": 0.0}, "flags": [record.get("failure", "invalid")]}
    target = record["target"]
    p = profile["targets"].get(target, {}).get("metrics", {})
    integrity = 100 * min(record["binary_fraction"], record["finite_fraction"])
    if not record["affine_match"]:
        integrity -= 5
    if record["dynamic_range"] <= 0:
        integrity -= 40
    integrity = max(0.0, integrity)
    largest = (100.0 if record["components"] <= 2 else
               _high_good(record["largest_component_fraction"], .65, .90))
    extras = _low_good(record["extra_after_two_fraction"], .005, .10)
    holes = _low_good(record["holes"], 2.0, 5.0)
    topology = _weighted({"dominance": (largest, .45), "extra_components": (extras, .35), "holes": (holes, .20)})[0]
    boundary_floor = p.get("boundary_gradient_ratio", {}).get("q05", .8)
    image_support = _high_good(record["boundary_gradient_ratio"], max(0.0, boundary_floor * .4), boundary_floor)
    flags: list[str] = []
    if target == "cmr_myo":
        topology = _weighted({
            "base_topology": (topology, .55),
            "component_consistency": (_low_good(record["component_mode_mismatch_fraction"], 0.0, .06), .25),
            "hole_consistency": (_low_good(record["hole_mode_mismatch_fraction"], .02, .10), .20),
        })[0]
        event_consistency = min(
            _low_good(record["component_mode_mismatch_fraction"], 0.0, .05),
            _low_good(record["hole_mode_mismatch_fraction"], .02, .08),
            _low_good(record["isolated_area_jump"], .08, .30),
        )
        temporal = math.nan if record.get("sparse_phase") else _weighted({
            "event_consistency": (event_consistency, .40),
            "dice": (_high_good(record["temporal_dice_p05"], .45, .72), .30),
            "centroid_outlier": (_low_good(record["temporal_centroid_jump_max"], .05, .20), .20),
            "area_cv": (_low_good(record["temporal_area_cv"], .18, .55), .10),
        })[0]
        score, subs = _weighted({"integrity": (integrity, .15), "topology": (topology, .30),
                                 "temporal": (temporal, .30), "image_support": (image_support, .25)})
    elif target == "lge_myo":
        continuity = float(record.get("interslice_score", math.nan))
        score, subs = _weighted({"integrity": (integrity, .15), "topology": (topology, .35),
                                 "interslice": (continuity, .20), "image_support": (image_support, .30)})
    else:
        containment = _high_good(record.get("mi_containment", math.nan), .90, .995)
        scar_empty = record["area"] == 0
        if scar_empty:
            mi_signal = math.nan
            topology = 100.0
        else:
            topology = _weighted({
                "tiny_components": (_low_good(record["tiny_component_fraction"], .01, .15), .65),
                "component_count": (_low_good(record["components"], 3.0, 8.0), .35),
            })[0]
            signal_floor = p.get("mi_signal_z", {}).get("q05", 0.0)
            mi_signal = _high_good(record.get("mi_signal_z", math.nan), min(-.5, signal_floor - 1), max(.5, signal_floor))
        continuity = float(record.get("interslice_score", math.nan))
        score, subs = _weighted({"integrity": (integrity, .10), "topology": (topology, .15),
                                 "containment": (containment, .25), "interslice": (continuity, .20),
                                 "image_support": (mi_signal, .30)})
        if record["area"] > 0 and np.isfinite(record.get("mi_containment", math.nan)) and record["mi_containment"] < .995:
            flags.append("mi_outside_myo")
    if target != "lge_mi" and record["components"] > 2 and record["largest_component_fraction"] < .8:
        flags.append("fragmentation_candidate")
    if target == "lge_mi" and record["tiny_component_fraction"] > .08:
        flags.append("fragmentation_candidate")
    if record["border_frame_fraction"] > .5:
        flags.append("border_contact_review")
    if target == "cmr_myo" and record["border_frame_fraction"] > 0 and record["isolated_area_jump"] > .08:
        flags.append("border_clip_candidate")
    if image_support < 40:
        flags.append("weak_image_boundary_support")
    if record["empty_frame_fraction"] > .5 and target != "lge_mi" and not record.get("sparse_phase"):
        flags.append("mostly_empty_label")
    confidence = 0.98 if record["tier"] == "gold" else (0.98 if record["reviewed"] else .85)
    if not np.isfinite(record.get("interslice_score", math.nan)) and target.startswith("lge"):
        confidence -= .08
    if record.get("sparse_phase"):
        confidence -= .10
    grade = "A" if score >= 90 else "B" if score >= 80 else "C" if score >= 70 else "D" if score >= 60 else "F"
    return {**record, "score": round(score, 3), "grade": grade,
            "confidence": round(max(0.0, confidence), 3), "subscores": subs, "flags": flags}


def add_interslice_context(records: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        if record.get("valid") and record["target"].startswith("lge"):
            grouped.setdefault((record["case_id"], record["target"]), []).append(record)
    for group in grouped.values():
        group.sort(key=lambda item: item["slice_position"])
        for index, current in enumerate(group):
            neighbors = [group[j] for j in (index - 1, index + 1) if 0 <= j < len(group)]
            if len(neighbors) < 2:
                current["interslice_score"] = math.nan
                current["boundary_slice"] = True
                continue
            current["boundary_slice"] = False
            areas = [item["area"] for item in neighbors]
            if current["target"] == "lge_mi":
                if current["area"] == 0:
                    current["interslice_score"] = 20.0 if all(area > 0 for area in areas) else 100.0
                    current["sandwich_gap"] = all(area > 0 for area in areas)
                    continue
                if all(area == 0 for area in areas):
                    current["interslice_score"] = 50.0
                    current["sandwich_gap"] = False
                    continue
            expected = float(np.mean(areas))
            area_score = _low_good(abs(math.log((current["area"] + 1) / (expected + 1))), .25, 1.2)
            current["interslice_score"] = area_score
            if current["target"] == "lge_mi" and current["area"] == 0 and all(area > 0 for area in areas):
                current["interslice_score"] = min(current["interslice_score"], 20.0)
                current["sandwich_gap"] = True
            else:
                current["sandwich_gap"] = False


def write_reports(records: list[dict[str, Any]], profile: dict[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "calibration_profile.json").write_text(
        json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    flattened = []
    for record in records:
        row = {key: value for key, value in record.items() if key not in {"subscores", "flags"}}
        row["subscores"] = json.dumps(record.get("subscores", {}), ensure_ascii=False, sort_keys=True)
        row["flags"] = ";".join(record.get("flags", []))
        flattened.append(row)
    fields = sorted({key for row in flattened for key in row})
    with (output / "pair_scores.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(flattened)
    critical = {"mi_outside_myo", "fragmentation_candidate", "mostly_empty_label"}
    flagged = [row for row in flattened if row.get("grade") in {"D", "F"}
               or critical.intersection(str(row.get("flags") or "").split(";"))]
    with (output / "review_queue.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(flagged)
    summary: dict[str, Any] = {"schema_version": "image-label-quality-v1", "total": len(records), "groups": {}}
    for dataset in sorted({r["dataset"] for r in records}):
        for target in sorted({r["target"] for r in records if r["dataset"] == dataset}):
            subset = [r for r in records if r["dataset"] == dataset and r["target"] == target]
            valid_scores = [r["score"] for r in subset if r.get("valid")]
            summary["groups"][f"{dataset}:{target}"] = {
                "n": len(subset), "valid": sum(bool(r.get("valid")) for r in subset),
                "mean_score": round(float(np.mean(valid_scores)), 3) if valid_scores else None,
                "median_score": round(float(np.median(valid_scores)), 3) if valid_scores else None,
                "grades": {grade: sum(r.get("grade") == grade for r in subset) for grade in "ABCDF"},
                "reviewed": sum(bool(r.get("reviewed")) for r in subset),
            }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
