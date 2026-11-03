#!/usr/bin/env python3
"""Predict into a new candidate folder using a verified selected model version.

This never copies predictions to production labels; use the existing QC/apply
controllers afterwards. Config and checkpoint are selected together.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.segmentation_registry import resolve_active, resolve_candidate

ENTRYPOINTS = {"cmr_myo": "automated_annotation_of_cine/main_multicenter.py",
               "lge_myo": "automated_annotation_of_lge/main.py",
               "lge_mi": "automated_annotation_of_lge_mi/main_myo_prior.py"}


def predict_command(registry: Path, vendor: Path, target: str, images: Path,
                    output: Path, *, device: str = "cpu", model_id: str | None = None) -> tuple[list[str], dict]:
    active = resolve_candidate(registry, model_id) if model_id else resolve_active(registry, target)
    if not active:
        raise ValueError("no_selected_model")
    if active["target"] != target:
        raise ValueError("model_target_mismatch")
    images, output = images.resolve(), output.resolve()
    if output == images or output in images.parents or images in output.parents or output.exists():
        raise ValueError("prediction_requires_new_separate_output")
    # Existing vendor uses the selected config + weights, not its default config.
    entry = vendor.resolve() / ENTRYPOINTS[target]
    if not entry.is_file():
        raise ValueError("vendor_entrypoint_missing")
    a = active["artifacts"]
    from app.segmentation_training import sha256_file
    if active["kind"] == "trained":
        expected = active["training_record"].get("vendor_sources_sha256")
        sources = {p.relative_to(entry.parent).as_posix(): sha256_file(p)
                   for p in sorted(entry.parent.rglob("*.py"))}
        actual = hashlib.sha256(json.dumps(sources, sort_keys=True, separators=(",", ":"),
                                          ensure_ascii=True, allow_nan=False).encode()).hexdigest()
        if not expected or actual != expected:
            raise ValueError("trained_vendor_sources_changed")
    model_file = entry.parent / {"cmr_myo": "cine_seg/model.py", "lge_myo": "lge_seg/model.py",
                                 "lge_mi": "lge_mi_prior_seg/model.py"}[target]
    if not model_file.is_file() or sha256_file(model_file) != a["architecture"]["sha256"]:
        raise ValueError("selected_architecture_requires_matching_inference_adapter")
    command = [sys.executable, str(entry), "--config", a["config"]["path"], "predict",
               "--weights", a["weights"]["path"], "--input", str(images),
               "--output-dir", str(output), "--device", device]
    if target == "lge_mi":
        # MI dependency belongs to its approved model config. Never silently use
        # whichever Myo happens to be current after this MI model was evaluated.
        config = json.loads(Path(a["config"]["path"]).read_text())
        dep = config.get("myocardium_model", {}).get("weights")
        digest = config.get("myocardium_model", {}).get("sha256")
        if not dep or not digest or sha256_file(Path(dep)) != digest:
            raise ValueError("mi_myo_dependency_not_pinned")
        command += ["--myo-weights", dep]
    return command, {"target": target, "model_id": active["candidate_id"],
        "dataset_sha256": active["dataset_sha256"], "weights_sha256": a["weights"]["sha256"],
        "status": "PLANNED", "production_labels_written": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--registry", type=Path, required=True)
    p.add_argument("--vendor", type=Path, required=True)
    p.add_argument("--target", choices=ENTRYPOINTS, required=True)
    p.add_argument("--model-id", help="Evaluate an immutable candidate without activating it")
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    command, receipt = predict_command(args.registry, args.vendor, args.target, args.images, args.output,
                                       device=args.device, model_id=args.model_id)
    if args.execute:
        args.output.mkdir(parents=True, exist_ok=False)
        try:
            subprocess.run(command, check=True)
            receipt["status"] = "PREDICTED_PENDING_QC"
        except BaseException:
            receipt["status"] = "FAILED"
            raise
        finally:
            (args.output / "model_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
