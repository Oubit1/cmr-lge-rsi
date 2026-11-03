#!/usr/bin/env python3
"""Run the integrated CMR/LGE models and place predictions in standard labels.

Input must already be standardized NIfTI under ROOT/images/cmr and ROOT/images/lge.
The source images are never modified. Predictions are first written to a
timestamped run directory, then copied into the standard label directories.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]
VENDOR = HERE / "vendor"
MODELS = HERE / "models"


def has_nifti(path: Path) -> bool:
    return path.is_dir() and any(p.is_file() and p.name.endswith((".nii", ".nii.gz")) for p in path.rglob("*"))


def run(cmd: list[str], cwd: Path, dry_run: bool):
    print("$", " ".join(str(x) for x in cmd))
    if not dry_run: subprocess.run(cmd, cwd=str(cwd), check=True)


def copy_masks(source: Path, target: Path, source_subdir: str = "masks") -> int:
    source_root = source / source_subdir
    masks = sorted(p for p in source_root.rglob("*") if p.is_file() and p.name.endswith((".nii", ".nii.gz")))
    count = 0
    for path in masks:
        relative = path.relative_to(source_root)
        dst = target / relative; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, dst); count += 1
    return count


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--classification-gate-threshold", type=float, default=0.1)
    ap.add_argument("--skip-cmr", action="store_true")
    ap.add_argument("--skip-lge-myo", action="store_true")
    ap.add_argument("--skip-lge-mi", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--run-qc", action="store_true")
    args = ap.parse_args(); root = args.root.resolve(); stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cmr = root / "images/cmr"; lge = root / "images/lge"
    if not args.skip_cmr and not has_nifti(cmr): raise RuntimeError(f"No CMR NIfTI files under {cmr}")
    if (not args.skip_lge_myo or not args.skip_lge_mi) and not has_nifti(lge): raise RuntimeError(f"No LGE NIfTI files under {lge}")
    run_root = root / "label_iteration/annotation_runs" / stamp
    py = sys.executable
    commands = []
    if not args.skip_cmr:
        commands.append(([
            py, str(VENDOR / "automated_annotation_of_cine/main_multicenter.py"), "predict",
            "--weights", str(MODELS / "cmr_myo_best_model.pt"), "--input", str(cmr),
            "--output-dir", str(run_root / "cmr_myo"), "--device", args.device, "--threshold", str(args.threshold), "--visualize"
        ], VENDOR / "automated_annotation_of_cine"))
    if not args.skip_lge_myo:
        commands.append(([
            py, str(VENDOR / "automated_annotation_of_lge/main.py"), "predict",
            "--weights", str(MODELS / "lge_myo_best_model.pt"), "--input", str(lge),
            "--output-dir", str(run_root / "lge_myo"), "--device", args.device, "--threshold", str(args.threshold), "--visualize"
        ], VENDOR / "automated_annotation_of_lge"))
    if not args.skip_lge_mi:
        commands.append(([
            py, str(VENDOR / "automated_annotation_of_lge_mi/main_myo_prior.py"), "predict",
            "--weights", str(MODELS / "lge_mi_myo_prior_cls_best_model.pt"),
            "--myo-weights", str(MODELS / "lge_myo_best_model.pt"), "--input", str(lge),
            "--output-dir", str(run_root / "lge_mi"), "--device", args.device,
            "--threshold", str(args.threshold), "--classification-gate-threshold", str(args.classification_gate_threshold), "--visualize"
        ], VENDOR / "automated_annotation_of_lge_mi"))
    if args.dry_run:
        for cmd, cwd in commands: run(cmd, cwd, True)
        print(json.dumps({"dry_run": True, "commands": len(commands), "run_root": str(run_root)}, ensure_ascii=False, indent=2)); return
    run_root.mkdir(parents=True, exist_ok=True)
    for cmd, cwd in commands: run(cmd, cwd, False)
    copied = {}
    if not args.skip_cmr: copied["cmr_myo"] = copy_masks(run_root / "cmr_myo", root / "labels/cmr/cmr_Myo_mask", "predictions")
    if not args.skip_lge_myo: copied["lge_myo"] = copy_masks(run_root / "lge_myo", root / "labels/lge_checked/lge_Myo_labels")
    if not args.skip_lge_mi: copied["lge_mi"] = copy_masks(run_root / "lge_mi", root / "labels/lge_checked/lge_MI_labels")
    summary = {"timestamp": stamp, "root": str(root), "run_root": str(run_root), "weights": {"cmr_myo": str(MODELS / "cmr_myo_best_model.pt"), "lge_myo": str(MODELS / "lge_myo_best_model.pt"), "lge_mi": str(MODELS / "lge_mi_myo_prior_cls_best_model.pt")}, "copied_masks": copied}
    (run_root / "annotation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.run_qc:
        run([py, str(HERE / "scripts/run_lge_qc_workflow.py"), "--root", str(root), "--mode", "audit"], HERE, False)
        run([py, str(HERE / "tools/cmr_qc.py"), "--root", str(root), "--mode", "audit"], HERE, False)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
