#!/usr/bin/env python3
"""Match the reviewed apex/base CMR rule against remaining volumes.

Only volumes whose multi-component slices are locally supported by a dominant
component and whose surrounding volume is predominantly single-component are
eligible. Ambiguous CMR topology remains for manual review.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np

from cmr_qc import as_stack, components, load, render, restore_stack


def volume_match(stack: np.ndarray) -> tuple[bool, list[dict], str]:
    per_slice = []
    for k in range(stack.shape[-1]):
        mask = stack[..., k].astype(bool)
        _, cs = components(mask)
        cs = sorted(cs, key=lambda x: x[1], reverse=True)
        per_slice.append(cs)
    multi = [k for k, cs in enumerate(per_slice) if len(cs) > 1]
    if not multi:
        return False, [], "single_component_volume"
    single_fraction = sum(len(cs) == 1 for cs in per_slice) / max(len(per_slice), 1)
    if single_fraction < 0.80 or len(multi) > max(8, int(len(per_slice) * 0.40)):
        return False, [], f"ambiguous_topology_single_fraction={single_fraction:.2f}"

    actions = []
    for k in multi:
        dominant = per_slice[k][0]
        _, area, cx, cy = dominant
        neighbours = []
        for j in (k - 1, k + 1):
            if 0 <= j < len(per_slice) and per_slice[j]:
                neighbours.append(per_slice[j][0])
        supported = False
        support_details = []
        for _, narea, ncx, ncy in neighbours:
            distance = float(((cx - ncx) ** 2 + (cy - ncy) ** 2) ** 0.5)
            ratio = max(area, narea) / max(min(area, narea), 1)
            support_details.append({"neighbor_area": narea, "distance": round(distance, 2), "area_ratio": round(ratio, 2)})
            if distance <= 30 and ratio <= 4:
                supported = True
        if not supported:
            return False, [], f"dominant_component_not_supported_slice={k + 1}"
        actions.append({"slice_index": k + 1, "component_areas": [x[1] for x in per_slice[k]],
                        "kept_area": area, "support": support_details})
    return True, actions, f"single_fraction={single_fraction:.2f};multi_slices={len(multi)}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", choices=["dry-run", "apply"], default="dry-run")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run = root / "label_iteration/cmr_case_matching_runs" / stamp
    run.mkdir(parents=True, exist_ok=True)
    issue_root = root / "labels/check/cmr_issue_slices"
    issue_root.mkdir(parents=True, exist_ok=True)
    backup_root = root / "label_iteration/cmr_case_matching_backups" / stamp
    records = []
    changed_volumes = 0

    for patient_dir in sorted((root / "images/cmr").iterdir()):
        if not patient_dir.is_dir():
            continue
        for image_path in sorted(patient_dir.glob("*.nii.gz")):
            label_path = root / "labels/cmr/cmr_Myo_mask" / patient_dir.name / image_path.name
            if not label_path.exists():
                continue
            _, image_raw = load(image_path)
            label_nii, label_raw = load(label_path)
            image_stack, _ = as_stack(image_raw)
            label_stack, label_axis = as_stack(label_raw)
            matched, actions, reason = volume_match(label_stack)
            if not matched:
                continue
            records.append({"patient": patient_dir.name, "volume": image_path.name,
                            "reason": reason, "slices": actions, "status": "candidate"})
            if args.mode != "apply":
                continue
            backup = backup_root / label_path.relative_to(root)
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(label_path, backup)
            new_stack = label_stack.astype(bool).copy()
            for action in actions:
                k = action["slice_index"] - 1
                mask = label_stack[..., k].astype(bool)
                lab, cs = components(mask)
                keep_id = max(cs, key=lambda x: x[1])[0]
                new_stack[..., k] = lab == keep_id
                if args.render:
                    check = issue_root / f"{patient_dir.name}_{image_path.name[:-7]}_s{k + 1:02d}_case_largest.png"
                    render(image_stack[..., k], new_stack[..., k], check,
                           f"{patient_dir.name} | {image_path.name} | slice {k + 1} | matched keep-largest")
            restored = restore_stack(new_stack.astype(np.uint8), label_raw, label_axis)
            out = nib.Nifti1Image(restored, label_nii.affine, label_nii.header)
            out.set_data_dtype(np.uint8)
            nib.save(out, str(label_path))
            records[-1].update(status="applied", backup=str(backup))
            changed_volumes += 1

    (run / "case_matches.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (run / "case_matches.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["patient", "volume", "reason", "slices", "status", "backup"])
        writer.writeheader()
        for row in records:
            writer.writerow({"patient": row["patient"], "volume": row["volume"], "reason": row["reason"],
                             "slices": ",".join(str(x["slice_index"]) for x in row["slices"]),
                             "status": row["status"], "backup": row.get("backup", "")})
    summary = {"mode": args.mode, "candidate_volumes": len(records), "changed_volumes": changed_volumes,
               "run": str(run), "rule": "dominant component supported by adjacent layers in predominantly single-component volume"}
    (run / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
