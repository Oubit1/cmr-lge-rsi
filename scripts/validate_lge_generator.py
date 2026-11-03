"""Exercise a domain generator's file/identity contract on synthetic copies only."""
from __future__ import annotations
import argparse
import json
import sys
import tempfile
from pathlib import Path
import numpy as np
import nibabel as nib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.lge_execution import producer, sha, validate_mask


def validate(root: Path) -> dict:
    reports = []
    with tempfile.TemporaryDirectory(prefix="lge-contract-") as tmp:
        base = Path(tmp)
        for stage, target, shape in ((s,t,shape) for s,t in (("lge_myo","myo"),("lge_mi","mi"))
                                    for shape in ((24,24),(24,24,1),(24,24,3))):
            fixture = base / (stage + "_" + str(len(shape)) + "_" + str(shape[-1]))
            fixture.mkdir()
            sources = {}
            for name in ("image", "label", "mi" if target == "myo" else "myo"):
                values = np.indices(shape).sum(axis=0).astype(np.float32) if name == "image" else np.zeros(shape, dtype=np.uint8)
                if name != "image":
                    values[8:16, 8:16] = 1
                    if name == "label":
                        values[1:3, 1:3] = 1  # exercise the output-file protocol if a repair is proposed
                path = fixture / (name + ".nii.gz")
                nib.save(nib.Nifti1Image(values, np.eye(4)), str(path))
                sources[name] = path
            row = {"case_id": "synthetic_case", "sources": sources,
                   "hashes": {key: sha(path) for key, path in sources.items()}}
            work = fixture / "work"
            work.mkdir()
            candidates = producer(root, [row], work, stage, sha(root / "scripts/lge_candidate_generator.py"))
            if not candidates:
                raise ValueError("generator_omitted_fixture_decision")
            for item in candidates:
                if item.get("reason") in {"missing_snapshot_paths", "snapshot_file_not_found"}:
                    raise ValueError("generator_input_contract_unhandled:" + item["reason"])
                if item.get("decision") not in {"repair", "accept_existing", "needs_review"} or not item.get("reason"):
                    raise ValueError("invalid_fixture_decision")
                if item["decision"] == "repair":
                    from app.lge_execution import contained
                    validate_mask(sources["label"], contained(work / "outputs", item["candidate"]), work / "validated.nii.gz")
            reports.append({"stage": stage, "shape": list(shape), "decisions": len(candidates)})
    return {"passed": True, "fixtures": reports, "production_writes": 0,
            "scope": "generator_interface_only_not_clinical_accuracy"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate(ROOT)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))
