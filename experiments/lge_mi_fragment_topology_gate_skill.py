"""Read-only LGE MI fragment topology gate skill (draft, shadow triage only).

Skill contract
--------------
1. Topology pre-gate: an LGE MI secondary component may only be considered a
   fragment candidate after the myocardial ring topology gate passes. If the
   ring is empty, unbounded, or the case is a temporal hole, the fragment
   verdict is suppressed and routed to shadow triage only.
2. Suppression conditions: empty MI, empty myocardium, border-touching
   fragment, temporal hole. Suppressed cases never emit a fragment flag.
3. Additive edits must preserve connected-component count and hole count and
   must not touch the image border. Any edit violating this is rejected.
4. No automatic label edits: this skill never writes, mutates or relabels
   production masks. It only emits shadow triage records.
5. Escalation requires real held-out evidence: a 20-volume held-out checklist
   with explicit denominators, module hash and audit diff. Absent that, the
   skill stays in shadow mode.

This module is read-only. It performs no writes to datasets, labels or code.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

SKILL_VERSION = "lge_mi_fragment_topology_gate_skill/1.0"
AREA_FRACTION_THRESHOLD = 0.05
HELDOUT_VOLUME_TARGET = 20

SUPPRESSION_CONDITIONS = (
    "empty_mi",
    "empty_myo",
    "border_fragment",
    "temporal_hole",
)

NO_AUTO_LABEL_EDIT_CLAUSE = (
    "This skill must never write, mutate, relabel or delete production masks or "
    "labels. All outputs are shadow triage records for human review only."
)


def _label_components(mask: Sequence[Sequence[bool]]) -> List[List[int]]:
    """4-connected labeling without scipy. Returns int label map (0 = bg)."""
    h = len(mask)
    w = len(mask[0]) if h else 0
    labels = [[0] * w for _ in range(h)]
    current = 0
    for i in range(h):
        for j in range(w):
            if not mask[i][j] or labels[i][j] != 0:
                continue
            current += 1
            stack = [(i, j)]
            labels[i][j] = current
            while stack:
                y, x = stack.pop()
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny][nx] and labels[ny][nx] == 0:
                        labels[ny][nx] = current
                        stack.append((ny, nx))
    return labels


def _count_holes(mask: Sequence[Sequence[bool]]) -> int:
    """Count background components fully enclosed by foreground (4-connectivity)."""
    h = len(mask)
    w = len(mask[0]) if h else 0
    if h == 0 or w == 0:
        return 0
    bg = [[not mask[i][j] for j in range(w)] for i in range(h)]
    labels = _label_components(bg)
    border_labels = set()
    for j in range(w):
        border_labels.add(labels[0][j])
        border_labels.add(labels[h - 1][j])
    for i in range(h):
        border_labels.add(labels[i][0])
        border_labels.add(labels[i][w - 1])
    holes = 0
    seen = set()
    for i in range(h):
        for j in range(w):
            lbl = labels[i][j]
            if lbl == 0 or lbl in border_labels or lbl in seen:
                continue
            seen.add(lbl)
            holes += 1
    return holes


def _touches_border(mask: Sequence[Sequence[bool]]) -> bool:
    h = len(mask)
    w = len(mask[0]) if h else 0
    if h == 0 or w == 0:
        return False
    if any(mask[0][j] for j in range(w)) or any(mask[h - 1][j] for j in range(w)):
        return True
    if any(mask[i][0] for i in range(h)) or any(mask[i][w - 1] for i in range(h)):
        return True
    return False


def _dilate(mask: Sequence[Sequence[bool]], radius: int = 1) -> List[List[bool]]:
    h = len(mask)
    w = len(mask[0]) if h else 0
    out = [[bool(mask[i][j]) for j in range(w)] for i in range(h)]
    for _ in range(max(0, radius)):
        nxt = [row[:] for row in out]
        for i in range(h):
            for j in range(w):
                if not out[i][j]:
                    continue
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = i + dy, j + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        nxt[ny][nx] = True
        out = nxt
    return out


@dataclass
class TopologyGateResult:
    case_id: str
    gate_passed: bool = False
    suppression_reason: Optional[str] = None
    mi_components: int = 0
    dominant_area: int = 0
    flagged_fragments: int = 0
    flagged_non_adjacent: int = 0
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "gate_passed": self.gate_passed,
            "suppression_reason": self.suppression_reason,
            "mi_components": self.mi_components,
            "dominant_area": self.dominant_area,
            "flagged_fragments": self.flagged_fragments,
            "flagged_non_adjacent": self.flagged_non_adjacent,
            "notes": list(self.notes),
        }


def topology_gate(
    case_id: str,
    mi_mask: Sequence[Sequence[bool]],
    myo_mask: Sequence[Sequence[bool]],
    *,
    temporal_hole: bool = False,
    area_fraction_threshold: float = AREA_FRACTION_THRESHOLD,
) -> TopologyGateResult:
    """Read-only topology pre-gate. Never mutates inputs."""
    res = TopologyGateResult(case_id=case_id)
    mi_any = any(any(row) for row in mi_mask) if mi_mask else False
    myo_any = any(any(row) for row in myo_mask) if myo_mask else False

    if not mi_any:
        res.suppression_reason = "empty_mi"
        res.notes.append("suppressed_empty_mi")
        return res
    if not myo_any:
        res.suppression_reason = "empty_myo"
        res.notes.append("suppressed_empty_myo")
        return res
    if temporal_hole:
        res.suppression_reason = "temporal_hole"
        res.notes.append("suppressed_temporal_hole")
        return res

    labels = _label_components(mi_mask)
    n_components = max((max(row) for row in labels), default=0)
    res.mi_components = n_components
    if n_components == 0:
        res.suppression_reason = "empty_mi"
        return res

    areas: Dict[int, int] = {}
    for row in labels:
        for lbl in row:
            if lbl:
                areas[lbl] = areas.get(lbl, 0) + 1
    dominant_label = max(areas, key=lambda k: areas[k])
    res.dominant_area = areas[dominant_label]
    threshold = area_fraction_threshold * res.dominant_area

    myo_ring = _dilate(myo_mask, radius=1)
    h = len(mi_mask)
    w = len(mi_mask[0]) if h else 0

    for lbl, area in areas.items():
        if lbl == dominant_label or area >= threshold:
            continue
        comp = [[labels[i][j] == lbl for j in range(w)] for i in range(h)]
        if _touches_border(comp):
            res.notes.append(f"suppressed_border_fragment_{lbl}")
            continue
        res.flagged_fragments += 1
        adjacent = any(
            comp[i][j] and myo_ring[i][j] for i in range(h) for j in range(w)
        )
        if not adjacent:
            res.flagged_non_adjacent += 1

    res.gate_passed = True
    return res


def additive_edit_invariants(
    before: Sequence[Sequence[bool]],
    after: Sequence[Sequence[bool]],
) -> Dict[str, Any]:
    """Check additive-edit invariants: component count, hole count, no border touch."""
    before_labels = _label_components(before)
    after_labels = _label_components(after)
    before_components = max((max(row) for row in before_labels), default=0)
    after_components = max((max(row) for row in after_labels), default=0)
    before_holes = _count_holes(before)
    after_holes = _count_holes(after)
    border_touch = _touches_border(after)
    return {
        "components_before": before_components,
        "components_after": after_components,
        "components_preserved": before_components == after_components,
        "holes_before": before_holes,
        "holes_after": after_holes,
        "holes_preserved": before_holes == after_holes,
        "touches_border": border_touch,
        "additive_edit_ok": (
            before_components == after_components
            and before_holes == after_holes
            and not border_touch
        ),
    }


def heldout_checklist(n_volumes: int) -> Dict[str, Any]:
    """Reproducible 20-volume held-out checklist with explicit denominators."""
    return {
        "target_volumes": HELDOUT_VOLUME_TARGET,
        "supplied_volumes": n_volumes,
        "denominator_ok": n_volumes >= HELDOUT_VOLUME_TARGET,
        "steps": [
            "1. Freeze module hash before scanning.",
            "2. Run topology_gate on each held-out volume; record gate_passed.",
            "3. Record flagged_fragments and flagged_non_adjacent per case.",
            "4. Denominator for non-adjacency fraction = total flagged_fragments.",
            "5. Denominator for overlap rate = number of held-out volumes.",
            "6. Record suppression counts per condition with denominator.",
            "7. Emit audit diff vs previous report; confirm zero production writes.",
            "8. Report concordance/CI as unavailable absent reference labels.",
        ],
        "suppression_conditions": list(SUPPRESSION_CONDITIONS),
        "no_auto_label_edit_clause": NO_AUTO_LABEL_EDIT_CLAUSE,
    }


def module_hash() -> str:
    src = inspect.getsource(inspect.getmodule(topology_gate))
    return hashlib.sha256(src.encode("utf-8")).hexdigest()


def audit_diff(previous: Optional[Dict[str, Any]], current: Dict[str, Any]) -> Dict[str, Any]:
    if previous is None:
        return {
            "baseline": "none",
            "changed_cases": [],
            "added_cases": sorted(current.get("cases", {}).keys()),
            "removed_cases": [],
            "production_writes": 0,
        }
    prev_cases = previous.get("cases", {})
    cur_cases = current.get("cases", {})
    changed = [c for c in cur_cases if c in prev_cases and prev_cases[c] != cur_cases[c]]
    added = [c for c in cur_cases if c not in prev_cases]
    removed = [c for c in prev_cases if c not in cur_cases]
    return {
        "baseline": previous.get("module_hash"),
        "changed_cases": changed,
        "added_cases": added,
        "removed_cases": removed,
        "production_writes": 0,
    }


def run_evaluation(data: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate supplied existing-qc-measurements-v1 records (read-only).

    Only aggregate area/cavity measurements are available. Fragment topology,
    non-adjacency, overlap and concordance/CI are reported as unavailable.
    """
    records = data.get("records", []) if isinstance(data, dict) else []
    n = len(records)
    area_unchanged = 0
    cavity_unchanged = 0
    area_changed_cases: List[str] = []
    cavity_changed_cases: List[str] = []
    quality_deltas: List[float] = []

    for rec in records:
        cid = rec.get("case_id")
        ba = rec.get("before_area")
        aa = rec.get("after_area")
        bc = rec.get("before_cavity")
        ac = rec.get("after_cavity")
        if ba is not None and aa is not None:
            if ba == aa:
                area_unchanged += 1
            else:
                area_changed_cases.append(cid)
        if bc is not None and ac is not None:
            if bc == ac:
                cavity_unchanged += 1
            else:
                cavity_changed_cases.append(cid)
        qd = rec.get("quality_delta")
        if qd is not None:
            quality_deltas.append(float(qd))

    return {
        "skill_version": SKILL_VERSION,
        "module_hash": module_hash(),
        "mode": "read_only_shadow",
        "n_records": n,
        "heldout_checklist": heldout_checklist(n),
        "additive_edit_invariants": {
            "area_unchanged": area_unchanged,
            "area_denominator": n,
            "area_changed_cases": area_changed_cases,
            "cavity_unchanged": cavity_unchanged,
            "cavity_denominator": n,
            "cavity_changed_cases": cavity_changed_cases,
        },
        "quality_delta": {
            "n": len(quality_deltas),
            "min": min(quality_deltas) if quality_deltas else None,
            "max": max(quality_deltas) if quality_deltas else None,
        },
        "unavailable": {
            "fragment_topology": "no masks or frame geometry supplied",
            "non_adjacency_fraction": "no masks supplied",
            "overlap_rate": "no masks supplied",
            "concordance": "no independent reference labels supplied",
            "ci": "no independent reference labels supplied",
        },
        "no_auto_label_edit_clause": NO_AUTO_LABEL_EDIT_CLAUSE,
        "writes": {"labels": 0, "code": 0, "datasets": 0},
    }


def report_to_json(report: Dict[str, Any]) -> str:
    return json.dumps(report, sort_keys=True, indent=2)
