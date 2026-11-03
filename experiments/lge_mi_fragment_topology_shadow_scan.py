"""Read-only shadow scan for LGE MI secondary connected components.

Flags MI components whose area is < 5% of the dominant MI domain and that are
non-adjacent to the myocardium ring. Emits per-case counts, non-adjacency
fraction, mi_multiple_components/myo_topology overlap, suppression coverage
(empty / border / temporal-hole) and a module hash + audit diff.

No labels, no production code, no writes. Concordance/CI is reported as
unavailable when no independent reference labels are supplied.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

AREA_FRACTION_THRESHOLD = 0.05
MODULE_VERSION = "lge_mi_fragment_topology_shadow_scan/1.0"


def _label_components(mask: np.ndarray) -> np.ndarray:
    """4-connected labeling without scipy; returns int label map (0 = bg)."""
    mask = np.asarray(mask).astype(bool)
    labels = np.zeros(mask.shape, dtype=np.int32)
    current = 0
    h, w = mask.shape
    for i in range(h):
        for j in range(w):
            if not mask[i, j] or labels[i, j] != 0:
                continue
            current += 1
            stack = [(i, j)]
            labels[i, j] = current
            while stack:
                y, x = stack.pop()
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and labels[ny, nx] == 0:
                        labels[ny, nx] = current
                        stack.append((ny, nx))
    return labels


def _dilate(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    mask = np.asarray(mask).astype(bool)
    out = mask.copy()
    for _ in range(max(0, radius)):
        shifted = out.copy()
        shifted[1:, :] |= out[:-1, :]
        shifted[:-1, :] |= out[1:, :]
        shifted[:, 1:] |= out[:, :-1]
        shifted[:, :-1] |= out[:, 1:]
        out = shifted
    return out


def _touches_border(mask: np.ndarray) -> bool:
    if mask.size == 0:
        return False
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


@dataclass
class CaseResult:
    case_id: str
    mi_components: int = 0
    dominant_area: int = 0
    flagged_fragments: int = 0
    flagged_non_adjacent: int = 0
    mi_multiple_components: bool = False
    myo_topology_flag: bool = False
    overlap: bool = False
    suppression: Dict[str, int] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "mi_components": self.mi_components,
            "dominant_area": self.dominant_area,
            "flagged_fragments": self.flagged_fragments,
            "flagged_non_adjacent": self.flagged_non_adjacent,
            "mi_multiple_components": self.mi_multiple_components,
            "myo_topology_flag": self.myo_topology_flag,
            "overlap": self.overlap,
            "suppression": dict(self.suppression),
            "notes": list(self.notes),
        }


def scan_case(
    case_id: str,
    mi_mask: np.ndarray,
    myo_mask: np.ndarray,
    *,
    area_fraction_threshold: float = AREA_FRACTION_THRESHOLD,
    temporal_hole: bool = False,
) -> CaseResult:
    """Read-only scan of one case. Never mutates inputs."""
    mi = np.asarray(mi_mask).astype(bool)
    myo = np.asarray(myo_mask).astype(bool)
    result = CaseResult(case_id=case_id)
    result.suppression = {
        "empty_mi": 0,
        "empty_myo": 0,
        "border_fragment": 0,
        "temporal_hole": 0,
    }

    if not mi.any():
        result.suppression["empty_mi"] = 1
        result.notes.append("empty_mi_suppressed")
        return result
    if not myo.any():
        result.suppression["empty_myo"] = 1
        result.notes.append("empty_myo_suppressed")
        return result
    if temporal_hole:
        result.suppression["temporal_hole"] = 1
        result.notes.append("temporal_hole_suppressed")
        return result

    labels = _label_components(mi)
    n_components = int(labels.max())
    result.mi_components = n_components
    result.mi_multiple_components = n_components > 1

    if n_components == 0:
        return result

    areas = {lbl: int((labels == lbl).sum()) for lbl in range(1, n_components + 1)}
    dominant_label = max(areas, key=lambda k: areas[k])
    result.dominant_area = areas[dominant_label]
    threshold = area_fraction_threshold * result.dominant_area

    myo_ring = _dilate(myo, radius=1)
    result.myo_topology_flag = bool(myo.any())

    for lbl, area in areas.items():
        if lbl == dominant_label:
            continue
        if area >= threshold:
            continue
        comp = labels == lbl
        if _touches_border(comp):
            result.suppression["border_fragment"] += 1
            continue
        result.flagged_fragments += 1
        if not (comp & myo_ring).any():
            result.flagged_non_adjacent += 1

    result.overlap = bool(result.mi_multiple_components and result.myo_topology_flag)
    return result


def module_hash() -> str:
    src = inspect.getsource(inspect.getmodule(scan_case))
    return hashlib.sha256(src.encode("utf-8")).hexdigest()


def audit_diff(previous: Optional[Dict[str, Any]], current: Dict[str, Any]) -> Dict[str, Any]:
    """Structural diff between two scan reports (read-only)."""
    if previous is None:
        return {"baseline": "none", "changed_cases": [], "added_cases": list(current.get("cases", {}).keys())}
    prev_cases = previous.get("cases", {})
    cur_cases = current.get("cases", {})
    changed = [cid for cid in cur_cases if cid in prev_cases and prev_cases[cid] != cur_cases[cid]]
    added = [cid for cid in cur_cases if cid not in prev_cases]
    removed = [cid for cid in prev_cases if cid not in cur_cases]
    return {"baseline": previous.get("module_hash"), "changed_cases": changed, "added_cases": added, "removed_cases": removed}


def scan_cohort(
    cases: Sequence[Dict[str, Any]],
    *,
    area_fraction_threshold: float = AREA_FRACTION_THRESHOLD,
    reference_labels: Optional[Dict[str, Any]] = None,
    previous_report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Scan a cohort of cases. `cases` items: {case_id, mi_mask, myo_mask, temporal_hole?}."""
    results: Dict[str, Any] = {}
    total_flagged = 0
    total_non_adjacent = 0
    overlap_cases = 0
    suppression_totals = {"empty_mi": 0, "empty_myo": 0, "border_fragment": 0, "temporal_hole": 0}

    for case in cases:
        res = scan_case(
            case["case_id"],
            case["mi_mask"],
            case["myo_mask"],
            area_fraction_threshold=area_fraction_threshold,
            temporal_hole=bool(case.get("temporal_hole", False)),
        )
        results[res.case_id] = res.as_dict()
        total_flagged += res.flagged_fragments
        total_non_adjacent += res.flagged_non_adjacent
        if res.overlap:
            overlap_cases += 1
        for key in suppression_totals:
            suppression_totals[key] += res.suppression.get(key, 0)

    n_cases = len(cases)
    report: Dict[str, Any] = {
        "module_version": MODULE_VERSION,
        "module_hash": module_hash(),
        "area_fraction_threshold": area_fraction_threshold,
        "n_cases": n_cases,
        "cases": results,
        "totals": {
            "flagged_fragments": total_flagged,
            "flagged_non_adjacent": total_non_adjacent,
            "non_adjacent_fraction": (total_non_adjacent / total_flagged) if total_flagged else None,
            "non_adjacent_denominator": total_flagged,
            "overlap_cases": overlap_cases,
            "overlap_denominator": n_cases,
            "overlap_rate": (overlap_cases / n_cases) if n_cases else None,
            "suppression": suppression_totals,
        },
        "concordance": "unavailable_no_reference_labels" if reference_labels is None else "reference_labels_supplied",
        "ci": "unavailable_no_reference_labels" if reference_labels is None else "not_computed",
    }
    report["audit_diff"] = audit_diff(previous_report, report)
    return report


def report_to_json(report: Dict[str, Any]) -> str:
    return json.dumps(report, sort_keys=True, indent=2)
