#!/usr/bin/env python3
"""Run the complete annotation + LGE/CMR QC workflow for a standardized dataset.

The pipeline only applies rules that have explicit high-confidence evidence.  All
other candidates remain in the final manifests for review.  Every child command
is recorded in ``label_iteration/full_pipeline_runs/<timestamp>/``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]
TOOLS = HERE / "tools"


def run_step(run_dir: Path, name: str, args: list[str], env: dict[str, str], *, defer_log: bool = False) -> None:
    command = [sys.executable, *args]
    log = run_dir / f"{name}.log"
    # Intake requires an empty destination. Spool its log outside the dataset
    # until it exits, including on failure, then retain it at the usual path.
    with (tempfile.TemporaryFile(mode="w+b") if defer_log
          else log.open("wb")) as handle:
        handle.write(("$ " + " ".join(command) + "\n\n").encode("utf-8"))
        handle.flush()
        try:
            result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, env=env)
        finally:
            if defer_log:
                run_dir.mkdir(parents=True, exist_ok=True)
                handle.seek(0)
                with log.open("wb") as destination:
                    shutil.copyfileobj(handle, destination)
    if result.returncode:
        raise RuntimeError(f"step {name} failed; inspect {log}")


def newest_manifest(root: Path) -> Path | None:
    candidates = sorted((root / "label_iteration" / "cmr_qc_runs").glob("*/issue_manifest.csv"))
    return candidates[-1] if candidates else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--input", type=Path, help="raw NIfTI/DICOM root; converted into --root before annotation")
    ap.add_argument("--device", default="auto", help="auto, cpu, cuda, or a CUDA device")
    ap.add_argument("--model-registry", type=Path, help="Use the explicitly selected segmentation versions")
    ap.add_argument("--orchestrator-root", type=Path, help="Controller checkout providing the registry adapter")
    ap.add_argument("--skip-annotation", action="store_true")
    ap.add_argument("--audit-only", action="store_true", help="audit an existing dataset without annotation or label changes")
    ap.add_argument("--no-render", action="store_true", help="skip large CMR check PNG rendering")
    ap.add_argument("--render-intermediate", action="store_true",
                    help="render every intermediate repair candidate (slow; final review rendering is always kept)")
    ap.add_argument("--workers", type=int, default=16, help="parallel workers for full-cohort CMR audit and check rendering")
    ap.add_argument("--max-rounds", type=int, default=3, help="maximum learned-rule repair rounds")
    ap.add_argument("--finalize-dataset", action="store_true",
                    help="move unusable boundary/rejected paired bundles into recoverable quarantine")
    ap.add_argument("--reject-manifest", type=Path, action="append", default=[],
                    help="additional manually rejected patient/volume CSV for final quarantine")
    args = ap.parse_args()
    if args.audit_only and args.finalize_dataset:
        ap.error("--finalize-dataset cannot be combined with --audit-only")
    if args.audit_only and args.input:
        ap.error("--input cannot be combined with --audit-only; audit an existing --root")
    if bool(args.model_registry) != bool(args.orchestrator_root):
        ap.error("--model-registry and --orchestrator-root must be supplied together")

    root = args.root.resolve()
    if args.input and root.exists() and any(root.iterdir()):
        ap.error(f"refusing to overwrite non-empty output: {root}")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / "label_iteration" / "full_pipeline_runs" / stamp
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    steps: list[str] = []
    try:
        if args.input:
            run_step(run_dir, "00_prepare_dataset", [str(TOOLS / "prepare_dataset.py"), "--input", str(args.input.resolve()), "--output", str(root)], env, defer_log=True)
            steps.append("prepare_dataset")
    finally:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "command.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n", encoding="utf-8")
    if not args.audit_only and not args.skip_annotation:
        model_args = (["--model-registry", str(args.model_registry.resolve()),
                       "--orchestrator-root", str(args.orchestrator_root.resolve())]
                      if args.model_registry else [])
        run_step(run_dir, "01_annotation", [str(TOOLS / "auto_annotate_dataset.py"), "--root", str(root), "--device", args.device, *model_args], env)
        steps.append("annotation")

    # CMR has its own dedicated audit in the next step.  Keep the LGE workflow
    # modality-scoped here to avoid scanning/rendering every cine frame twice.
    # In apply mode these images would be stale as soon as repairs start. Keep
    # the initial manifests but defer PNG generation to the final review pass.
    initial_no_render = args.no_render or not args.audit_only
    run_step(run_dir, "02_initial_lge_audit", [str(HERE / "scripts/run_lge_qc_workflow.py"), "--root", str(root), "--mode", "audit", "--skip-cmr", *(["--no-render"] if initial_no_render else [])], env)
    run_step(run_dir, "03_initial_cmr_audit", [str(TOOLS / "cmr_qc.py"), "--root", str(root), "--mode", "audit", *(["--no-render"] if initial_no_render else [])], env)
    run_step(run_dir, "03b_lge_casebook_dry_run", [str(TOOLS / "casebook_qc.py"), "--root", str(root), "--mode", "dry-run"], env)
    steps += ["initial_lge_audit", "initial_cmr_audit", "lge_casebook_dry_run"]

    if not args.audit_only:
        # These matchers contain the generalized rules learned from reviewed cases.
        # Each matcher writes its own backup, manifest and check images.
        matchers = [
            ("04_cmr_apex_base", "match_cmr_apex_base_rule.py"),
            ("05_cmr_lge_reference", "match_cmr_to_lge_rule.py"),
            ("06_cmr_lge_shape", "match_cmr_lge_shape_rule.py"),
            ("07_cmr_temporal_lge", "match_cmr_temporal_lge_rule.py"),
        ]
        for round_index in range(1, max(1, args.max_rounds) + 1):
            for name, script in matchers:
                prefix = f"round{round_index}_{name}"
                run_step(run_dir, prefix + "_dry_run", [str(TOOLS / script), "--root", str(root), "--mode", "dry-run"], env)
                run_step(run_dir, prefix, [str(TOOLS / script), "--root", str(root), "--mode", "apply", *(["--render"] if args.render_intermediate else [])], env)
                steps.extend([prefix + "_dry_run", prefix])

            manifest = newest_manifest(root)
            if manifest:
                prefix = f"round{round_index}_cmr_ring_gap"
                run_step(
                    run_dir,
                    prefix + "_dry_run",
                    [str(TOOLS / "match_cmr_ring_gap_rule.py"), "--root", str(root), "--mode", "dry-run", "--manifest", str(manifest)],
                    env,
                )
                run_step(
                    run_dir,
                    prefix,
                    [str(TOOLS / "match_cmr_ring_gap_rule.py"), "--root", str(root), "--mode", "apply", *(["--render"] if args.render_intermediate else []), "--manifest", str(manifest)],
                    env,
                )
                steps.extend([prefix + "_dry_run", prefix])

        # Apply only the CMR repair that is unambiguous from adjacent layers:
        # remove unsupported <=20px components.  Apex/base boundary layers,
        # multi-ring anatomy and uncertain gaps remain review candidates.
        run_step(run_dir, "08_cmr_safe_repair", [str(TOOLS / "cmr_qc.py"), "--root", str(root), "--mode", "apply-safe", "--no-render"], env)
        steps.append("cmr_safe_repair")
        run_step(run_dir, "09_lge_safe_repair", [str(HERE / "scripts/run_lge_qc_workflow.py"), "--root", str(root), "--mode", "apply-safe", "--no-render", "--skip-cmr"], env)
        run_step(run_dir, "09b_lge_casebook_safe_repair", [str(TOOLS / "casebook_qc.py"), "--root", str(root), "--mode", "apply-safe"], env)
        run_step(run_dir, "09c_clean_myo_topology", [str(TOOLS / "clean_lge_myo_topology.py"), "--root", str(root), "--mode", "apply-safe", "--max-noise-px", "100", "--max-ratio", "0.12"], env)
        run_step(run_dir, "09d_merge_nearby_mi", [str(TOOLS / "merge_nearby_mi_components.py"), "--root", str(root), "--mode", "apply-safe", "--max-added-px", "30"], env)
        run_step(run_dir, "09e_remove_small_unsupported_mi", [str(TOOLS / "remove_small_unsupported_mi.py"), "--root", str(root), "--mode", "apply-safe", "--max-tiny-px", "40", "--max-area-ratio", "0.15"], env)
        # Shared CMR/LGE Myo rule: infer the patient's robust LV-centre
        # trajectory from the majority of valid layers/frames, then remove an
        # unsupported competing component only when one component uniquely
        # agrees with that trajectory. LGE MI is clipped to repaired Myo.
        run_step(run_dir, "09f_myo_trajectory_dry_run", [str(TOOLS / "myo_stack_trajectory_qc.py"), "--root", str(root), "--modality", "both", "--mode", "dry-run", "--workers", str(args.workers)], env)
        run_step(run_dir, "09g_myo_trajectory_apply", [str(TOOLS / "myo_stack_trajectory_qc.py"), "--root", str(root), "--modality", "both", "--mode", "apply", "--workers", str(args.workers)], env)
        steps.extend(["lge_safe_repair", "lge_casebook_safe_repair", "clean_myo_topology", "merge_nearby_mi", "remove_small_unsupported_mi", "myo_trajectory_dry_run", "myo_trajectory_apply"])

    run_step(run_dir, "10_final_lge_audit", [str(HERE / "scripts/run_lge_qc_workflow.py"), "--root", str(root), "--mode", "audit", "--skip-cmr"], env)
    run_step(run_dir, "11_final_cmr_audit", [str(TOOLS / "cmr_qc.py"), "--root", str(root), "--mode", "audit", *([] if not args.no_render else ["--no-render"])], env)
    steps += ["final_lge_audit", "final_cmr_audit"]

    # Convert the final residual lists into the four active review pools. CMR
    # cine frame indices are temporal; spatial boundary status is determined
    # from the patient-level ordered CMR files.
    run_step(run_dir, "12_boundary_middle_triage", [str(TOOLS / "isolate_boundary_issue_slices.py"), "--root", str(root), *([] if args.audit_only else ["--remove-source-cmr"])], env)
    run_step(run_dir, "13_temporal_hole_dry_run", [str(TOOLS / "repair_cmr_temporal_missing_holes.py"), "--root", str(root), "--mode", "dry-run"], env)
    if not args.audit_only:
        run_step(run_dir, "13b_temporal_hole_apply", [str(TOOLS / "repair_cmr_temporal_missing_holes.py"), "--root", str(root), "--mode", "apply"], env)
        steps.append("temporal_hole_apply")
    run_step(run_dir, "13c_temporal_open_ring_dry_run", [str(TOOLS / "repair_cmr_temporal_open_ring.py"), "--root", str(root), "--mode", "dry-run"], env)
    if not args.audit_only:
        run_step(run_dir, "13d_temporal_open_ring_apply", [str(TOOLS / "repair_cmr_temporal_open_ring.py"), "--root", str(root), "--mode", "apply"], env)
        steps.append("temporal_open_ring_apply")
        # Both temporal repairs change labels after the first audit. Rebuild
        # the residual manifest so resolved frames cannot survive as stale
        # issue rows, then repeat patient-level boundary/middle triage.
        run_step(run_dir, "13e_post_temporal_cmr_audit", [str(TOOLS / "apply_cmr_qc_fast.py"), "--root", str(root), "--mode", "audit", "--workers", str(args.workers)], env)
        run_step(run_dir, "13f_post_temporal_triage", [str(TOOLS / "isolate_boundary_issue_slices.py"), "--root", str(root), "--remove-source-cmr"], env)
        steps.extend(["post_temporal_cmr_audit", "post_temporal_triage"])
    run_step(run_dir, "14_supported_small_area_dry_run", [str(TOOLS / "suppress_supported_cmr_small_area.py"), "--root", str(root), "--mode", "dry-run"], env)
    if not args.audit_only:
        run_step(run_dir, "14b_supported_small_area_apply", [str(TOOLS / "suppress_supported_cmr_small_area.py"), "--root", str(root), "--mode", "apply"], env)
        steps.append("supported_small_area_apply")
    steps += ["boundary_middle_triage", "temporal_hole_dry_run", "temporal_open_ring_dry_run", "supported_small_area_dry_run"]

    # Re-scan after all repairs. This final pass never edits masks: its
    # unresolved rows become the shared CMR/LGE trajectory review pool.
    run_step(run_dir, "14c_final_myo_trajectory_audit", [str(TOOLS / "myo_stack_trajectory_qc.py"), "--root", str(root), "--modality", "both", "--mode", "dry-run", "--workers", str(args.workers)], env)
    trajectory_manifests = sorted((root / "label_iteration" / "myo_trajectory_runs").glob("*/trajectory_candidates.csv"))
    if trajectory_manifests and not args.no_render:
        trajectory_output = root / "labels/check/myo_trajectory_review_slices"
        run_step(run_dir, "14d_render_myo_trajectory_review", [str(TOOLS / "render_myo_trajectory_candidates.py"), "--root", str(root),
                 "--manifest", str(trajectory_manifests[-1]), "--output", str(trajectory_output), "--clean"], env)
        steps.append("render_myo_trajectory_review")
    steps.append("final_myo_trajectory_audit")

    # Final review images use modality-specific verified conventions.
    if not args.no_render:
        for modality in ("cmr", "lge"):
            for group in ("boundary", "middle"):
                directory = root / "labels/check" / f"{modality}_{group}_issue_slices"
                manifest = directory / "issue_manifest.csv"
                if manifest.exists():
                    name = f"15_render_{modality}_{group}"
                    run_step(run_dir, name, [str(TOOLS / "render_check_panels.py"), "--root", str(root), "--modality", modality,
                                             "--manifest", str(manifest), "--output", str(directory), "--clean",
                                             "--workers", str(args.workers)], env)
                    steps.append(name)

    # Final dataset contract: every retained relative level must have a CMR
    # image+Myo and LGE image+Myo+MI. Boundary/rejected levels are always
    # handled as complete bundles; audit is unconditional, movement requires
    # the explicit --finalize-dataset flag and remains recoverable.
    pair_args = [str(TOOLS / "finalize_paired_dataset.py"), "--root", str(root),
                 "--mode", "audit", "--lge-label-set", "checked"]
    for manifest in args.reject_manifest:
        pair_args += ["--reject-manifest", str(manifest.resolve())]
    run_step(run_dir, "16_pairing_and_quarantine_audit", pair_args, env)
    steps.append("pairing_and_quarantine_audit")
    if args.finalize_dataset:
        pair_args[pair_args.index("audit")] = "quarantine"
        run_step(run_dir, "16b_finalize_paired_dataset", pair_args, env)
        steps.append("finalize_paired_dataset")

    summary = {"root": str(root), "run": str(run_dir), "audit_only": args.audit_only,
               "finalize_dataset": args.finalize_dataset, "steps": steps,
               "final_cmr_manifest": str(newest_manifest(root) or "")}
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
