"""Bridge the existing reviewed-pair quality scorer to frozen RSI evaluation.

Only calibration/evaluation artifacts are produced. Medical label files are
never changed. Gold provenance comes from the existing configured datasets;
controlled defects remain explicitly distinct from clinical adjudications.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.rsi_kernel import canonical_hash
from app.rsi_runtime import write_record


TARGET_STAGES = {"cmr_myo": "cmr_paired", "lge_myo": "lge_myo", "lge_mi": "lge_mi"}
TEXT_FIELDS = {"dataset", "tier", "target", "case_id", "relative_id", "failure"}
BOOL_FIELDS = {"valid", "reviewed", "image_exists", "label_exists", "myo_exists",
               "boundary_slice", "sandwich_gap", "sparse_phase"}
OMIT_FIELDS = {"score", "grade", "confidence", "flags", "subscores"}
SCORER_NUMERIC_FIELDS = {
    "area", "area_fraction", "binary_fraction", "finite_fraction", "affine_match",
    "dynamic_range", "largest_component_fraction", "extra_after_two_fraction",
    "tiny_component_fraction", "max_second_component_fraction", "holes", "components",
    "boundary_gradient_ratio", "small_hole_fraction", "border_frame_fraction",
    "temporal_dice", "temporal_dice_p05", "temporal_centroid_jump", "temporal_centroid_jump_max",
    "temporal_area_cv", "isolated_area_jump", "component_mode_mismatch_fraction",
    "hole_mode_mismatch_fraction", "empty_frame_fraction", "mi_containment", "mi_signal_z",
    "interslice_score", "compactness", "inside_outside_z",
}


def finite_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def load_cached_gold(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            if raw.get("tier") != "gold" or raw.get("valid", "").lower() != "true":
                continue
            record = {}
            for key, value in raw.items():
                if key in OMIT_FIELDS:
                    continue
                if key in TEXT_FIELDS:
                    record[key] = value
                elif key in BOOL_FIELDS:
                    record[key] = value.lower() == "true"
                else:
                    try:
                        record[key] = float(value)
                    except (ValueError, TypeError):
                        record[key] = math.nan
            if record.get("target") in TARGET_STAGES:
                records.append(record)
    return records


def patient_split(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Deterministic patient split shared by all targets from the same center."""
    by_center = defaultdict(set)
    for row in records:
        by_center[row["dataset"]].add(row["case_id"])
    heldout = set()
    for center, cases in sorted(by_center.items()):
        if len(cases) < 2:
            raise ValueError("at least two gold patients required for center: " + center)
        ordered = sorted(cases, key=lambda case: canonical_hash({"center": center, "patient": case}))
        heldout.update(ordered[:max(1, len(ordered) // 5)])
    train = [r for r in records if r["case_id"] not in heldout]
    validation = [r for r in records if r["case_id"] in heldout]
    if {r["case_id"] for r in train} & {r["case_id"] for r in validation}:
        raise ValueError("patient split overlap")
    return train, validation


def solver_features(record: dict) -> dict:
    # Preserve existing measured numeric fields, including null for missing
    # measurements. Remove identifiers and cached outputs/acceptance metadata.
    result = {key: float(record.get(key, math.nan)) for key in sorted(SCORER_NUMERIC_FIELDS)}
    result.update(valid=bool(record.get("valid")), target=record["target"],
                  sparse_phase=bool(record.get("sparse_phase", False)),
                  myo_exists=bool(record.get("myo_exists", False)),
                  dataset="frozen_reference", case_id="anonymous", relative_id="anonymous",
                  tier="gold", reviewed=True, failure="")
    return finite_json(result)


def archive_asset(folder: Path, source: Path) -> tuple[str, str]:
    content = source.read_bytes()
    sha = hashlib.sha256(content).hexdigest()
    relative = "assets/" + sha + (".nii.gz" if source.name.endswith(".nii.gz") else ".nii")
    target = folder / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.write_bytes(content)
    return relative, sha


def raw_gold_probes(root: Path, validation: list[dict], folder: Path, limit: int = 32) -> tuple[list[dict], dict, dict, dict]:
    """Freeze existing CMR pairs for extraction+scoring historical replay."""
    import yaml
    config = yaml.safe_load((root / "configs/quality_scoring_datasets.yaml").read_text(encoding="utf-8"))
    sources = {r["name"]: r["targets"].get("cmr_myo") for r in config["datasets"] if r.get("tier") == "gold"}
    records, references, hashes, skipped = [], {}, {}, Counter()
    ordered = sorted((r for r in validation if r["target"] == "cmr_myo"),
                     key=lambda r: canonical_hash({"case": r["case_id"], "relative": r["relative_id"]}))
    for row in ordered:
        if len(records) >= limit:
            break
        source = sources.get(row["dataset"])
        if not source:
            skipped["config_missing"] += 1
            continue
        relative = Path(row["relative_id"])
        label_rel = relative.relative_to(source["relative_prefix"]) if source.get("relative_prefix") else relative
        image, label = Path(source["image_dir"]) / relative, Path(source["label_dir"]) / label_rel
        if not image.is_file() or not label.is_file():
            skipped["source_missing"] += 1
            continue
        pair = {}
        for key, asset in (("image", image), ("label", label)):
            rel, sha = archive_asset(folder, asset)
            pair[key], hashes[rel] = rel, sha
        sample_id = "q_probe_" + canonical_hash({"case": row["case_id"], "relative": row["relative_id"]})[:20]
        records.append({"id": sample_id, "target": "cmr_myo", "raw_pair": pair})
        references[sample_id] = {"kind": "raw_gold"}
    return records, references, hashes, {"included": len(records), "skipped": dict(skipped)}


def existing_cmr_controls(root: Path, gold: list[dict], heldout_patients: set[str],
                          asset_folder: Path | None = None) -> tuple[list[dict], dict, dict]:
    """Use existing known injected defects, never generate new domain rules."""
    from app.quality_scoring import PairSpec, extract_pair
    import nibabel as nib
    import numpy as np
    path = root / "runtime/eval_set/v1/synthetic/manifest.jsonl"
    if not path.is_file():
        return [], {}, {"status": "UNAVAILABLE", "included": 0}
    import yaml
    config = yaml.safe_load((root / "configs/quality_scoring_datasets.yaml").read_text(encoding="utf-8"))
    sources = {r["name"]: r["targets"].get("cmr_myo") for r in config["datasets"] if r.get("tier") == "gold"}
    source_index = {}
    for cached in gold:
        source = sources.get(cached["dataset"])
        if cached["target"] != "cmr_myo" or not source:
            continue
        relative = Path(cached["relative_id"])
        relative = relative.relative_to(source["relative_prefix"]) if source.get("relative_prefix") else relative
        exact = str((Path(source["label_dir"]) / relative).resolve())
        source_index.setdefault(exact, []).append(cached)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    records, references, skipped, asset_hashes = [], {}, Counter(), {}
    included = 0
    for index, row in enumerate(rows):
        clean_label = Path(row["clean_label"])
        matches = source_index.get(str(clean_label.resolve()), [])
        cases = {r["case_id"] for r in matches}
        if len(cases) != 1:
            skipped["ambiguous_or_uncached_source"] += 1
            continue
        patient = next(iter(cases))
        if patient not in heldout_patients:
            skipped["calibration_patient"] += 1
            continue
        clean_mask = np.asanyarray(nib.load(str(clean_label)).dataobj) > .5
        injected_mask = np.asanyarray(nib.load(row["synthetic_label"]).dataobj) > .5
        same = clean_mask.shape == injected_mask.shape and np.array_equal(clean_mask, injected_mask)
        if (row["class"] == "true_small") != same:
            skipped["control_not_matching_declared_change"] += 1
            continue
        ids = []
        extracted = []
        for name, label in (("clean", clean_label), ("injected", Path(row["synthetic_label"]))):
            record = extract_pair(PairSpec(
                dataset="frozen_reference", tier="gold", target="cmr_myo", image=Path(row["image"]),
                label=label, myo=None, relative_id="anonymous", case_id="anonymous", slice_position=0, reviewed=True))
            if not record.get("valid"):
                break
            sample_id = "q_" + canonical_hash({"control": index, "side": name})[:20]
            ids.append(sample_id)
            extracted.append({"id": sample_id, "features": solver_features(record)})
        if len(extracted) != 2:
            skipped["invalid_source_pair"] += 1
            continue
        records.extend(extracted)
        references[ids[0]] = {"kind": "counterpart"}
        references[ids[1]] = {"kind": "small_control" if row["class"] == "true_small" else "defect",
                              "counterpart_id": ids[0], "defect_class": row["class"]}
        included += 1
        if asset_folder is not None:
            raw_ids = []
            for name, label in (("clean", clean_label), ("injected", Path(row["synthetic_label"]))):
                pair = {}
                for key, asset in (("image", Path(row["image"])), ("label", label)):
                    relative, sha = archive_asset(asset_folder, asset)
                    pair[key], asset_hashes[relative] = relative, sha
                raw_id = "q_raw_" + canonical_hash({"control": index, "side": name})[:20]
                raw_ids.append(raw_id)
                records.append({"id": raw_id, "target": "cmr_myo", "raw_pair": pair})
            references[raw_ids[0]] = {"kind": "counterpart"}
            references[raw_ids[1]] = {"kind": "small_control" if row["class"] == "true_small" else "defect",
                                      "counterpart_id": raw_ids[0], "defect_class": "raw_" + row["class"]}
    return records, references, {"status": "AVAILABLE" if references else "UNAVAILABLE",
                                 "included": included, "skipped": dict(skipped), "asset_hashes": asset_hashes,
                                 "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def existing_lge_controls(root: Path, heldout: list[dict], target: str,
                          per_class: int = 12) -> tuple[list[dict], dict, dict]:
    """Replay existing perturbation tools on temporary heldout GT copies.

    A changed copy is a controlled reference mismatch, not a new clinical
    adjudication. No temporal perturbation is transferred to LGE.
    """
    import random
    import tempfile
    import nibabel as nib
    import numpy as np
    import yaml
    from app.quality_scoring import PairSpec, extract_pair, _assert_original
    from scripts.build_synthetic_eval import defect
    config = yaml.safe_load((root / "configs/quality_scoring_datasets.yaml").read_text(encoding="utf-8"))
    sources = {dataset["name"]: dataset["targets"].get(target)
               for dataset in config["datasets"] if dataset.get("tier") == "gold"}
    classes = ("fragmentation", "border_clip") if target == "lge_mi" else ("fragmentation", "hole_fill", "border_clip")
    counters, skipped, records, references, provenance = Counter(), Counter(), [], {}, []
    ordered = sorted(heldout, key=lambda row: canonical_hash({"patient": row["case_id"], "slice": row["relative_id"]}))
    with tempfile.TemporaryDirectory(prefix="rsi-lge-reference-controls-") as temp:
        for row in ordered:
            if all(counters[cls] >= per_class for cls in classes):
                break
            source = sources.get(row["dataset"])
            if not source:
                skipped["source_config_missing"] += 1
                continue
            relative = Path(row["relative_id"])
            label_relative = relative.relative_to(source["relative_prefix"]) if source.get("relative_prefix") else relative
            image, label = Path(source["image_dir"]) / relative, Path(source["label_dir"]) / label_relative
            myo = Path(source["myo_dir"]) / label_relative if source.get("myo_dir") else None
            assets = [image, label] + ([myo] if myo else [])
            for asset in assets:
                _assert_original(asset)
            if not all(asset.is_file() for asset in assets):
                skipped["source_pair_missing"] += 1
                continue
            before = [hashlib.sha256(asset.read_bytes()).hexdigest() for asset in assets]
            original = nib.load(str(label))
            raw = np.asanyarray(original.dataobj)
            mask = np.squeeze(raw) > .5
            if mask.ndim != 2:
                skipped["non_2d_lge"] += 1
                continue
            common = dict(dataset="frozen_reference", tier="gold", target=target,
                          image=image, myo=myo, relative_id="anonymous", case_id="anonymous",
                          slice_position=row["slice_position"], reviewed=True)
            clean = extract_pair(PairSpec(label=label, **common))
            clean["interslice_score"] = row.get("interslice_score", math.nan)
            if not clean.get("valid"):
                skipped["invalid_reference"] += 1
                continue
            source_id = canonical_hash({"target": target, "patient": row["case_id"], "relative": row["relative_id"]})[:20]
            clean_id = "q_clean_" + source_id
            generated, ref_rows = [], {}
            for cls in classes:
                if counters[cls] >= per_class:
                    continue
                rng = random.Random(int(source_id[:12], 16))
                changed = defect(mask, cls, rng)
                if np.array_equal(changed, mask):
                    skipped[cls + "_unchanged"] += 1
                    continue
                temporary = Path(temp) / (cls + ".nii.gz")
                nib.save(nib.Nifti1Image(changed.astype(np.uint8).reshape(raw.shape), original.affine), str(temporary))
                extracted = extract_pair(PairSpec(label=temporary, **common))
                extracted["interslice_score"] = row.get("interslice_score", math.nan)
                if not extracted.get("valid"):
                    skipped["invalid_perturbation"] += 1
                    continue
                sample_id = "q_" + canonical_hash({"source": source_id, "class": cls})[:20]
                generated.append({"id": sample_id, "features": solver_features(extracted)})
                ref_rows[sample_id] = {"kind": "defect", "counterpart_id": clean_id, "defect_class": cls}
            if before != [hashlib.sha256(asset.read_bytes()).hexdigest() for asset in assets]:
                raise RuntimeError("reference source changed during feature extraction")
            if generated:
                records.extend([{"id": clean_id, "features": solver_features(clean)}, *generated])
                references.update({clean_id: {"kind": "counterpart"}, **ref_rows})
                counters.update(ref["defect_class"] for ref in ref_rows.values())
                provenance.append({"source_id": source_id, "asset_sha256": before})
    return records, references, {"status": "AVAILABLE" if counters else "UNAVAILABLE",
                                 "included": sum(counters.values()), "class_counts": dict(counters),
                                 "skipped": dict(skipped), "source_assets_sha256": canonical_hash(provenance),
                                 "perturbation_tool_sha256": hashlib.sha256((root / "scripts/build_synthetic_eval.py").read_bytes()).hexdigest(),
                                 "production_writes": False,
                                 "meaning": "Controlled mismatches to existing gold masks; not clinical defect labels."}


def _prepare_contents(root: Path, version: str, folder: Path, final_folder: Path) -> dict:
    """Freeze existing measurements; no provider calls or production writes."""
    from app.quality_scoring import build_profile
    cache = root / "artifacts/image_label_quality_v1/pair_scores.csv"
    gold = load_cached_gold(cache)
    if not gold:
        raise ValueError("no existing gold pair measurements")
    train, validation = patient_split(gold)
    profile = finite_json(build_profile(train))
    heldout = {r["case_id"] for r in validation}
    raw_probes, raw_references, raw_hashes, raw_summary = raw_gold_probes(root, validation, folder)
    controls, control_references, controls_summary = existing_cmr_controls(root, gold, heldout, folder)
    raw_hashes.update(controls_summary.pop("asset_hashes", {}))
    result = {"schema_version": "rsi-quality-readiness-v1", "version": version,
              "source_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(), "targets": {},
              "patient_overlap": 0, "controlled_defects": controls_summary}
    bundles, contracts = {}, {}
    for target, stage in TARGET_STAGES.items():
        selected = [r for r in validation if r["target"] == target]
        if not selected or not any(r["target"] == target for r in train):
            raise ValueError("missing training or heldout records for " + target)
        records, references = [], {}
        for index, row in enumerate(selected):
            sample_id = "q_" + canonical_hash({"target": target, "index": index})[:20]
            records.append({"id": sample_id, "features": solver_features(row)})
            references[sample_id] = {"kind": "gold", "center": row["dataset"]}
        lge_summary = None
        if target == "cmr_myo":
            records.extend(controls)
            references.update(control_references)
            records.extend(raw_probes)
            references.update(raw_references)
        else:
            lge_records, lge_references, lge_summary = existing_lge_controls(root, selected, target)
            records.extend(lge_records)
            references.update(lge_references)
        data = {"schema_version": "reviewed-pair-features-v1", "target": target,
                "profile": profile, "records": records, "references": references}
        if target == "cmr_myo":
            data["asset_hashes"] = raw_hashes
        metrics = {
            "gold_retention": {"direction": "higher", "report_field": "gold_retention"},
            "scorable_fraction": {"direction": "higher", "report_field": "scorable_fraction"},
            "control_ranking": {"direction": "higher", "report_field": "control_ranking"},
        }
        classes = sorted({ref["defect_class"] for ref in references.values() if ref["kind"] == "defect"})
        for cls in classes:
            metrics[cls + "_recall"] = {"direction": "higher", "report_field": "defect_recall." + cls}
        if any(ref["kind"] == "small_control" for ref in references.values()):
            metrics["small_control_retention"] = {"direction": "higher", "report_field": "small_control_retention"}
        if any(ref["kind"] == "raw_gold" for ref in references.values()):
            metrics["raw_gold_retention"] = {"direction": "higher", "report_field": "raw_gold_retention"}
        contract = {"schema_version": "rsi-benchmark-v1", "benchmark_id": version + ":" + target,
                    "scope": "quality_scorer_reference_preservation" if not classes else "quality_scorer_with_existing_defects",
                    "module": "app.quality_scoring", "scorer": "reviewed_quality_scores_v1", "target": target,
                    "input_path": (final_folder / (target + ".json")).relative_to(root).as_posix(),
                    "input_sha256": canonical_hash(data), "metrics": metrics,
                    "pass_score": 70, "defect_drop": 5, "timeout_seconds": 180,
                    "require_discrimination": True,
                    "applies_to": ["app/quality_scoring.py"],
                    "negative_controls_available": bool(classes),
                    "limitations": ["Gold source designation follows the existing reviewed-pair configuration; patient groups use persisted case_id.",
                                    "CMR has hashed raw extraction probes; LGE extraction/context and calibration changes require another contract.",
                                    "Controlled perturbation detection is not clinical defect precision.",
                                    "Repeated benchmark selection may overfit; later operator sampling remains independent."]}
        bundles[target], contracts[stage] = data, contract
        result["targets"][target] = {"calibration_pairs": sum(r["target"] == target for r in train),
                                      "validation_pairs": len(selected),
                                      "validation_patients": len({r["case_id"] for r in selected}),
                                      "calibration_patients": len({r["case_id"] for r in train if r["target"] == target}),
                                      "negative_classes": classes,
                                      "controlled_defects": lge_summary if target != "cmr_myo" else controls_summary,
                                      "raw_extraction_probes": raw_summary if target == "cmr_myo" else None,
                                      "contract_path": "configs/rsi_benchmarks/" + stage + ".json",
                                      "input_sha256": contract["input_sha256"]}
    for target, data in bundles.items():
        write_record(folder / (target + ".json"), data)
    write_record(folder / "contracts.json", contracts)
    write_record(folder / "readiness.json", result)
    return result


def activate(root: Path, version: str) -> None:
    folder = root / "runtime/rsi/benchmarks" / version
    contracts = json.loads((folder / "contracts.json").read_text(encoding="utf-8"))
    if set(contracts) != set(TARGET_STAGES.values()):
        raise ValueError("incomplete benchmark generation")
    for stage, contract in contracts.items():
        if stage not in TARGET_STAGES.values():
            raise ValueError("invalid benchmark stage")
        source = root / contract["input_path"]
        if not source.resolve().is_relative_to(folder.resolve()) or canonical_hash(json.loads(source.read_text(encoding="utf-8"))) != contract["input_sha256"]:
            raise ValueError("benchmark activation hash mismatch")
    write_record(root / "configs/rsi_benchmarks/active.json",
                 {"version": version, "generation_sha256": canonical_hash(contracts), "contracts": contracts})


def prepare(root: Path, version: str = "reviewed-pairs-v4") -> dict:
    import os
    import tempfile
    if not version or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in version):
        raise ValueError("invalid benchmark version")
    folder = root / "runtime/rsi/benchmarks" / version
    if folder.exists():
        raise FileExistsError("benchmark versions are immutable; choose a new version")
    folder.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=str(folder.parent)) as temporary:
        result = _prepare_contents(root, version, Path(temporary), folder)
        os.replace(temporary, folder)
    activate(root, version)
    return result


def benchmark_status(root: Path) -> dict:
    from app.rsi_runtime import benchmark_contracts
    targets = {}
    try:
        contracts = benchmark_contracts(root)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"targets": {}, "all_three_registered": False, "status": "UNAVAILABLE", "reason": str(exc)[:200]}
    for target, stage in TARGET_STAGES.items():
        if stage not in contracts:
            continue
        try:
            contract = contracts[stage]
            if not isinstance(contract, dict):
                raise ValueError("benchmark contract is not an object")
            if contract.get("scorer") != "reviewed_quality_scores_v1":
                continue
            data_path = (root / contract["input_path"]).resolve()
            if not data_path.is_relative_to((root / "runtime/rsi/benchmarks").resolve()):
                raise ValueError("benchmark data outside frozen directory")
            cached = _cached_input_status(str(data_path), data_path.stat().st_mtime_ns,
                                          data_path.stat().st_size, contract["input_sha256"])
            targets[target] = {"benchmark_id": contract["benchmark_id"], "scope": contract["scope"],
                               "input_verified": cached["input_verified"],
                               "applies_to": contract["applies_to"],
                               "negative_controls_available": contract["negative_controls_available"],
                               "heldout_gold_pairs": cached["heldout_gold_pairs"]}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            targets[target] = {"status": "UNAVAILABLE", "reason": str(exc)[:200], "input_verified": False}
    latest_path = root / "runtime/rsi/quality_verification_latest.json"
    try:
        latest = json.loads(latest_path.read_text(encoding="utf-8")) if latest_path.exists() else None
        if latest is not None and not isinstance(latest, dict):
            raise ValueError("verification summary is not an object")
    except (OSError, ValueError) as exc:
        latest = {"status": "UNAVAILABLE", "reason": str(exc)[:200]}
    if latest:
        latest = {key: value for key, value in latest.items() if key != "report_path"}
    weaknesses = []
    if latest:
        measurements = latest.get("candidate_metrics", {})
        if not isinstance(measurements, dict):
            measurements = {}
            latest = {"status": "UNAVAILABLE", "reason": "invalid verification metrics"}
        for metric, value in measurements.items():
            if (metric.endswith("_recall") or metric.endswith(".gold_retention")) and isinstance(value, (int, float)) and value < 1:
                weaknesses.append({"metric": metric, "value": value,
                                   "scope": "measured_reference_benchmark", "component": "app/quality_scoring.py"})
    weaknesses.sort(key=lambda item: item["value"])
    return {"schema_version": "rsi-quality-status-v1", "targets": targets,
            "all_three_registered": len(targets) == 3 and all(t["input_verified"] for t in targets.values()),
            "latest_verification": latest, "measured_weaknesses": weaknesses}


@lru_cache(maxsize=16)
def _cached_input_status(path: str, mtime_ns: int, size: int, expected_hash: str) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("references"), dict):
        raise ValueError("invalid frozen reference schema")
    if not all(isinstance(ref, dict) and isinstance(ref.get("kind"), str) for ref in data["references"].values()):
        raise ValueError("invalid frozen reference entries")
    return {"input_verified": canonical_hash(data) == expected_hash,
            "heldout_gold_pairs": sum(r["kind"] == "gold" for r in data["references"].values()),
            "calibration_profile": data.get("profile", {})}


def owner_context(root: Path, stage: str) -> dict:
    from app.rsi_runtime import benchmark_contracts
    target = next((key for key, value in TARGET_STAGES.items() if value == stage), None)
    if target is None:
        return {"status": "UNAVAILABLE", "reason": "no_scoring_target_for_stage"}
    contracts = benchmark_contracts(root)
    if stage not in contracts:
        return {"status": "UNAVAILABLE", "reason": "benchmark_contract_missing"}
    contract = contracts[stage]
    source = (root / contract["input_path"]).resolve()
    if not source.is_relative_to((root / "runtime/rsi/benchmarks").resolve()):
        raise ValueError("benchmark data outside frozen directory")
    cached = _cached_input_status(str(source), source.stat().st_mtime_ns, source.stat().st_size, contract["input_sha256"])
    if not cached["input_verified"]:
        raise ValueError("benchmark input hash mismatch")
    return {"status": "AVAILABLE", "scope": contract["scope"], "target": target,
            "calibration_profile": {"targets": {target: cached["calibration_profile"].get("targets", {}).get(target, {})}},
            "acceptance": "Paired score_record comparison on all three targets; improve at least one metric, "
                          "preserve others, distinguish controlled mismatches, pass existing full regression and independent review.",
            "private_holdout_records_supplied": False,
            "current_evaluation": benchmark_status(root).get("latest_verification")}


def verify(root: Path, output: Path, *, baseline_ref: str | None = None,
           candidate_root: Path | None = None, history_root: Path | None = None) -> dict:
    """Offline historical replay. This does not create a role/action output."""
    import shutil
    import subprocess
    from app.database import utcnow
    from app.rsi_runtime import freeze_parent, paired_evaluation
    from experiments.llm_maintenance import _copy_source
    output.mkdir(parents=True, exist_ok=False)
    _copy_source(root, output / "orchestrator")
    if baseline_ref:
        old = subprocess.run(["git", "show", baseline_ref + ":app/quality_scoring.py"],
                             cwd=history_root or root, check=True, capture_output=True)
        (output / "orchestrator/app/quality_scoring.py").write_bytes(old.stdout)
    freeze_parent(output, root)
    shutil.copy2((candidate_root or root) / "app/quality_scoring.py", output / "orchestrator/app/quality_scoring.py")
    report = paired_evaluation(root, output, "lge_myo", changed_files=["app/quality_scoring.py"])
    write_record(output / "quality_pair_report.json", report)
    summary = {key: report.get(key) for key in (
        "decision", "reasons", "benchmark_hash", "baseline_metrics", "candidate_metrics",
        "improved_metrics", "regressed_metrics", "scope")}
    summary.update(created_at=utcnow(), baseline_ref=baseline_ref or "current_source",
                   report_path=str(output / "quality_pair_report.json"),
                   evaluation_type="offline_framework_verification", production_writes=False,
                   model_calls=0)
    write_record(root / "runtime/rsi/quality_verification_latest.json", summary)
    return summary


def score_quality_outputs(data: dict, report: dict, contract: dict) -> dict:
    """Independent scoring of numeric outputs on a fixed private cohort."""
    references, scores = data["references"], report.get("scores")
    if not isinstance(scores, dict) or set(scores) - set(references):
        raise ValueError("quality_output_cohort_mismatch")
    valid = {}
    for sample_id, value in scores.items():
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 100:
            continue
        valid[sample_id] = value
    gold_ids = [key for key, ref in references.items() if ref["kind"] == "gold"]
    if not gold_ids:
        raise ValueError("empty_gold_validation_cohort")
    retained = sum(valid.get(key, -1) >= contract["pass_score"] for key in gold_ids)
    grouped = defaultdict(list)
    ranking = []
    small = []
    for key, ref in references.items():
        if ref["kind"] not in {"defect", "small_control"}:
            continue
        counterpart = ref["counterpart_id"]
        available = key in valid and counterpart in valid
        delta = valid[counterpart] - valid[key] if available else None
        if ref["kind"] == "defect":
            grouped[ref["defect_class"]].append(available and delta >= contract["defect_drop"])
            ranking.append((1.0 if delta > 0 else .5 if delta == 0 else 0.0) if available else 0.0)
        else:
            small.append(available and delta < contract["defect_drop"])
    result = {"gold_retention": retained / len(gold_ids),
              "scorable_fraction": len(valid) / len(references),
              "gold_pairs": len(gold_ids), "gold_below_threshold": len(gold_ids) - retained,
              "defect_recall": {key: sum(values) / len(values) for key, values in grouped.items()},
              "output_errors": report.get("errors", 0)}
    result["control_ranking"] = sum(ranking) / len(ranking) if ranking else None
    result["discrimination_ready"] = bool(ranking) and result["control_ranking"] > .5
    raw_gold = [key for key, ref in references.items() if ref["kind"] == "raw_gold"]
    if raw_gold:
        result["raw_gold_retention"] = sum(valid.get(key, -1) >= contract["pass_score"] for key in raw_gold) / len(raw_gold)
    if small:
        result["small_control_retention"] = sum(small) / len(small)
    return result
