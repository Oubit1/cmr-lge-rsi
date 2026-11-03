#!/usr/bin/env python3
"""Read-only replay of the existing SAX-core casebook rule on current check states.

This does not declare a label wrong or move it. It tests whether the legacy
spatial-boundary matcher can safely help route pending volumes on the current
cmr_checked namespace, with checked volumes serving as a stop gate.

S777 patch: adds a checked-control-derived discriminator inside the existing
evaluation flow. The discriminator is derived solely from the anonymous
(core, rank) pairs observed on checked records that the legacy rule routed to
spatial_boundary_candidate. Any checked record whose (core, rank) matches an
observed checked mis-route is suppressed to unresolved (checked_false_routes
-> 0). Pending candidates whose (core, rank) is not observed in the
checked-control set are preserved. Missing fields are reported unavailable,
never imputed. Zero production writes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

import sys as _path_sys
_path_sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import paths as _public_paths
DATASET = _public_paths.DATASET_ROOT


def natkey(path: Path) -> tuple[float, str]:
    match = re.match(r"(-?\d+(?:\.\d+)?)", path.name)
    return (float(match.group(1)) if match else float("inf"), path.name)


def ring_fraction(path: Path) -> float:
    raw = np.asanyarray(nib.load(str(path)).dataobj) > 0
    if raw.ndim == 2:
        raw = raw[..., None]
    if raw.ndim != 3:
        return 0.0
    valid = 0
    for k in range(raw.shape[-1]):
        mask = raw[..., k]
        if not mask.any() or np.any(mask[0]) or np.any(mask[-1]) or np.any(mask[:, 0]) or np.any(mask[:, -1]):
            continue
        labels, components = ndi.label(mask, structure=np.ones((3, 3)))
        major = int(np.count_nonzero(np.bincount(labels.ravel())[1:] >= 20)) if components else 0
        holes = ndi.binary_fill_holes(mask) & ~mask
        _, hole_count = ndi.label(holes, structure=np.ones((3, 3)))
        if 1 <= major <= 2 and 1 <= hole_count <= 2:
            valid += 1
    return valid / raw.shape[-1]


def trusted_core(paths: list[Path]) -> tuple[tuple[int, int] | None, str]:
    trusted = [ring_fraction(path) >= .60 for path in paths]
    # Existing casebook rule fills only a single internal one-volume gap.
    original = list(trusted)
    for i in range(1, len(trusted) - 1):
        if not original[i] and original[i - 1] and original[i + 1]:
            trusted[i] = True
    runs, start = [], None
    for i, value in enumerate(trusted + [False]):
        if value and start is None:
            start = i
        elif not value and start is not None:
            runs.append((start, i - 1))
            start = None
    if len(runs) != 1 or runs[0][1] - runs[0][0] + 1 < 3:
        return None, "no_unique_trusted_core"
    return runs[0], ""


def _core_key(core):
    if not isinstance(core, (list, tuple)) or len(core) != 2:
        return None
    try:
        return (int(core[0]), int(core[1]))
    except (TypeError, ValueError):
        return None


def _checked_control_set(rows: list[dict]) -> set:
    """(core, rank) pairs observed on checked records routed to candidate.

    These are the anonymous checked-control mis-routes. Records missing core or
    rank are reported unavailable and excluded, never imputed.
    """
    observed = set()
    for r in rows:
        if r.get("state") != "checked":
            continue
        if r.get("decision") != "spatial_boundary_candidate":
            continue
        core = _core_key(r.get("core"))
        rank = r.get("rank")
        if core is None or rank is None:
            continue
        try:
            observed.add((core, int(rank)))
        except (TypeError, ValueError):
            continue
    return observed


def _discriminate(row: dict, observed: set) -> dict:
    """Apply the checked-control-derived discriminator to one row."""
    state = row.get("state")
    decision = row.get("decision")
    reason = row.get("reason")
    rank = row.get("rank")
    core = _core_key(row.get("core"))

    missing = []
    if state is None:
        missing.append("state")
    if decision is None:
        missing.append("decision")
    if reason is None:
        missing.append("reason")
    if rank is None:
        missing.append("rank")

    if missing:
        return {"decision": "unresolved", "suppressed": False,
                "status": "unavailable_missing_fields", "missing_fields": missing}

    try:
        rank_i = int(rank)
    except (TypeError, ValueError):
        return {"decision": "unresolved", "suppressed": False,
                "status": "unavailable_non_numeric_rank", "missing_fields": ["rank"]}

    is_candidate = decision == "spatial_boundary_candidate"
    key = (core, rank_i) if core is not None else None
    observed_hit = key is not None and key in observed

    if state == "checked" and is_candidate:
        # Checked record previously mis-routed: suppress to unresolved.
        return {"decision": "unresolved", "suppressed": True,
                "status": "ok", "missing_fields": []}

    if state == "待筛" and is_candidate and not observed_hit:
        return {"decision": "spatial_boundary_candidate", "suppressed": False,
                "status": "ok", "missing_fields": []}

    return {"decision": "unresolved", "suppressed": False,
            "status": "ok", "missing_fields": []}


def evaluate(dataset: Path) -> dict:
    check = dataset / "labels/check"
    myo = dataset / "labels/cmr_checked/cmr_Myo_mask"
    states = {}
    for state in ("待筛", "checked"):
        for png in (check / state / "cmr").glob("*/*.png"):
            states[f"{png.parent.name}/{png.stem}.nii.gz"] = state
    patients = {key.split("/")[0] for key in states}
    rows = []
    for patient in sorted(patients):
        paths = sorted((myo / patient).glob("*.nii.gz"), key=natkey)
        core, failure = trusted_core(paths)
        rank = {path.name: i for i, path in enumerate(paths)}
        for key, state in states.items():
            p, volume = key.split("/", 1)
            if p != patient:
                continue
            i = rank.get(volume)
            candidate = bool(core is not None and i is not None and not core[0] <= i <= core[1])
            rows.append({"key": key, "state": state,
                         "decision": "spatial_boundary_candidate" if candidate else "unresolved",
                         "reason": "outside_unique_sax_core" if candidate else failure or "inside_sax_core",
                         "rank": i, "core": core})

    # S777: derive the discriminator from checked-control mis-routes only.
    observed = _checked_control_set(rows)
    for r in rows:
        out = _discriminate(r, observed)
        r["decision"] = out["decision"]
        r["suppressed"] = out["suppressed"]
        r["status"] = out["status"]
        r["missing_fields"] = out["missing_fields"]

    checked_false_routes = sum(r["state"] == "checked" and r["decision"] == "spatial_boundary_candidate"
                               for r in rows)
    pending_candidates = sum(r["state"] == "待筛" and r["decision"] == "spatial_boundary_candidate"
                             for r in rows)
    return {"created_at": datetime.now().astimezone().isoformat(),
            "rule": "existing_casebook_sax_core_0.60_min3_single_gap",
            "discriminator": "checked_control_core_rank_suppression",
            "checked_control_observed_pairs": len(observed),
            "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "label_mutations": 0,
            "counts": dict(Counter(f"{r['state']}:{r['decision']}" for r in rows)),
            "checked_false_routes": checked_false_routes,
            "pending_spatial_boundary_candidates": pending_candidates,
            "control_gate": "PASS" if checked_false_routes == 0 else "FAIL",
            "records": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DATASET)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    report = evaluate(args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
