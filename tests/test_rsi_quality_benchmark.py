import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.rsi_kernel import canonical_hash
from app.rsi_quality_benchmark import (benchmark_status, load_cached_gold, patient_split,
                                     prepare, score_quality_outputs, solver_features)
from app.rsi_runtime import freeze_parent, paired_evaluation, selection_summary, write_record


class QualityBridgeTests(unittest.TestCase):
    def test_patient_split_shared_across_targets(self):
        records = [{"dataset": center, "case_id": center + patient, "target": target}
                   for center in ("a", "b") for patient in ("1", "2", "3", "4", "5")
                   for target in ("cmr_myo", "lge_myo", "lge_mi")]
        training, heldout = patient_split(records)
        self.assertFalse({r["case_id"] for r in training} & {r["case_id"] for r in heldout})
        self.assertEqual(6, len(heldout))
        reversed_train, reversed_test = patient_split(list(reversed(records)))
        self.assertEqual({r["case_id"] for r in heldout}, {r["case_id"] for r in reversed_test})
        self.assertEqual({r["case_id"] for r in training}, {r["case_id"] for r in reversed_train})

    def test_cached_inputs_exclude_quasi_outputs_and_identifiers(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "cache.csv"
            source.write_text("tier,valid,target,dataset,case_id,relative_id,score,grade,temporal_area_cv\n"
                              "gold,True,lge_myo,a,private,private-path,99,A,nan\n"
                              "quasi_gold,True,lge_myo,a,other,other,99,A,0\n")
            rows = load_cached_gold(source)
            self.assertEqual(1, len(rows))
            features = solver_features(rows[0])
            self.assertNotIn("score", features)
            self.assertNotIn("private", json.dumps(features))
            self.assertIsNone(features["temporal_area_cv"])
            other = solver_features({**rows[0], "reviewed": True, "dataset": "synthetic", "tier": "quasi"})
            self.assertEqual(features, other)

    def test_fixed_denominator_missing_outputs_and_constant_score_guard(self):
        data = {"references": {"g1": {"kind": "gold"}, "g2": {"kind": "gold"},
                               "c": {"kind": "counterpart"},
                               "d": {"kind": "defect", "counterpart_id": "c", "defect_class": "fragmentation"}}}
        contract = {"pass_score": 70, "defect_drop": 5}
        scores = score_quality_outputs(data, {"scores": {"g1": 80}}, contract)
        self.assertEqual(.5, scores["gold_retention"])
        self.assertEqual(.25, scores["scorable_fraction"])
        self.assertEqual(0, scores["defect_recall"]["fragmentation"])
        constant = score_quality_outputs(data, {"scores": {key: 100 for key in data["references"]}}, contract)
        self.assertEqual(0, constant["defect_recall"]["fragmentation"])
        self.assertFalse(constant["discrimination_ready"])
        self.assertEqual(.5, constant["control_ranking"])

    def test_real_subprocess_cross_target_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, workspace = Path(tmp) / "root", Path(tmp) / "workspace"
            root.mkdir()
            write_record(root / "configs/rsi.json", {"mode": "observe"})
            module = workspace / "orchestrator/app/quality_scoring.py"
            module.parent.mkdir(parents=True)
            (module.parent / "__init__.py").touch()
            module.write_text("def score_record(r,p):\n    return {'score':r['value']}\n")
            freeze_parent(workspace, root)
            for target, stage in (("cmr_myo", "cmr_paired"), ("lge_myo", "lge_myo"), ("lge_mi", "lge_mi")):
                data = {"profile": {}, "records": [{"id": "g", "features": {"value": 65 if target != "lge_mi" else 80, "target": target}},
                                                     {"id": "c", "features": {"value": 80, "target": target}},
                                                     {"id": "d", "features": {"value": 30, "target": target}}],
                        "references": {"g": {"kind": "gold"}, "c": {"kind": "counterpart"},
                                       "d": {"kind": "defect", "counterpart_id": "c", "defect_class": "test_control"}}}
                source = root / "runtime/rsi/benchmarks/test" / (target + ".json")
                write_record(source, data)
                write_record(root / "configs/rsi_benchmarks" / (stage + ".json"), {
                    "schema_version": "rsi-benchmark-v1", "benchmark_id": "test:" + target,
                    "scope": "test_only", "target": target, "module": "app.quality_scoring",
                    "scorer": "reviewed_quality_scores_v1", "input_path": source.relative_to(root).as_posix(),
                    "input_sha256": canonical_hash(data), "pass_score": 70, "defect_drop": 5,
                    "applies_to": ["app/quality_scoring.py"],
                    "metrics": {"retention": {"direction": "higher", "report_field": "gold_retention"}}})
            module.write_text("def score_record(r,p):\n    return {'score':r['value']+10}\n")
            report = paired_evaluation(root, workspace, "lge_myo")
            self.assertEqual("PROMOTE", report["decision"])
            self.assertEqual(3, len(report["suites"]))
            module.write_text("def score_record(r,p):\n    return {'score':0 if r['target']=='lge_mi' else r['value']+10}\n")
            self.assertNotEqual("PROMOTE", paired_evaluation(root, workspace, "lge_myo")["decision"])
            not_applicable = paired_evaluation(root, workspace, "lge_myo", changed_files=["scripts/other.py"])
            self.assertTrue(not_applicable["scope_not_applicable"])
            dependency_changed = paired_evaluation(root, workspace, "lge_myo", changed_files=["app/__init__.py"])
            self.assertIn("suites", dependency_changed)
            module.write_text("def score_record(r,p):\n    return {'score':100}\n")
            self.assertNotEqual("PROMOTE", paired_evaluation(root, workspace, "lge_myo")["decision"])

    def test_prepare_freezes_existing_gold_and_rejects_version_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "artifacts/image_label_quality_v1/pair_scores.csv"
            cache.parent.mkdir(parents=True)
            with cache.open("w", newline="") as handle:
                fields = ["tier", "valid", "target", "dataset", "case_id", "relative_id", "area_fraction"]
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for target in ("cmr_myo", "lge_myo", "lge_mi"):
                    for patient in range(5):
                        writer.writerow(dict(tier="gold", valid="True", target=target, dataset="test",
                                             case_id="case" + str(patient), relative_id="slice", area_fraction=.1))
            with patch("app.rsi_quality_benchmark.existing_cmr_controls", return_value=([], {}, {"status": "UNAVAILABLE"})), \
                 patch("app.rsi_quality_benchmark.existing_lge_controls", return_value=([], {}, {"status": "UNAVAILABLE"})), \
                 patch("app.rsi_quality_benchmark.raw_gold_probes", return_value=([], {}, {}, {"included": 0})):
                report = prepare(root)
            self.assertEqual(0, report["patient_overlap"])
            self.assertTrue(benchmark_status(root)["all_three_registered"])
            for target in report["targets"].values():
                self.assertEqual(1, target["validation_pairs"])
                self.assertEqual(4, target["calibration_pairs"])
            with self.assertRaises(FileExistsError):
                prepare(root)

    def test_broken_artifacts_report_unavailable_instead_of_crashing_board(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_record(root / "configs/rsi_benchmarks/lge_myo.json", {
                "scorer": "reviewed_quality_scores_v1", "input_path": "runtime/rsi/benchmarks/missing.json",
                "input_sha256": "missing"})
            latest = root / "runtime/rsi/quality_verification_latest.json"
            latest.parent.mkdir(parents=True)
            latest.write_text("{broken")
            state = benchmark_status(root)
            self.assertFalse(state["all_three_registered"])
            self.assertEqual("UNAVAILABLE", state["targets"]["lge_myo"]["status"])
            self.assertEqual("UNAVAILABLE", state["latest_verification"]["status"])

    def test_per_record_outputs_are_not_sent_to_role_handoffs(self):
        full = {"decision": "PROMOTE", "baseline_metrics": {"m": 1}, "suites": {"lge_myo": {
            "target": "lge_myo", "baseline_metrics": {"m": 1}, "candidate_metrics": {"m": 2},
            "baseline_report": {"scores": {"PRIVATE_RECORD": "x" * 100000}},
            "candidate_report": {"scores": {"PRIVATE_RECORD": "x" * 100000}}}}}
        summary = json.dumps(selection_summary(full))
        self.assertNotIn("PRIVATE_RECORD", summary)
        self.assertLess(len(summary), 3000)

    def test_registry_is_one_hash_pinned_generation(self):
        from app.rsi_runtime import benchmark_contracts, scorer_dependency_paths
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contracts = {"lge_myo": {"benchmark_id": "new"}}
            write_record(root / "configs/rsi_benchmarks/lge_myo.json", {"benchmark_id": "old"})
            write_record(root / "configs/rsi_benchmarks/active.json", {
                "contracts": contracts, "generation_sha256": canonical_hash(contracts)})
            self.assertEqual(contracts, benchmark_contracts(root))
            write_record(root / "configs/rsi_benchmarks/active.json", {
                "contracts": contracts, "generation_sha256": "bad"})
            with self.assertRaisesRegex(ValueError, "generation_hash"):
                benchmark_contracts(root)
            module = root / "app/quality_scoring.py"
            module.parent.mkdir()
            module.write_text("from builtins import __import__ as load\n")
            self.assertTrue(scorer_dependency_paths(root, "app.quality_scoring")[1])
            module.write_text("import importlib.util as util\n")
            self.assertTrue(scorer_dependency_paths(root, "app.quality_scoring")[1])

    def test_evaluator_cannot_read_unsupplied_private_reference(self):
        from app.rsi_runtime import _measure
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            private = root / "references.json"
            private.write_text("private")
            code = root / "solver"
            code.mkdir()
            (code / "candidate.py").write_text(
                "from pathlib import Path\n"
                "def score_record(r,p):\n"
                " try:\n  Path(__file__).parent.parent.joinpath('references.json').read_text()\n"
                " except PermissionError:\n  return {'score':80}\n"
                " return {'score':0}\n")
            result = _measure(code, {"module": "candidate", "scorer": "reviewed_quality_scores_v1"},
                              json.dumps({"records": [{"id": "g", "features": {}}], "profile": {}}))
            self.assertEqual(80, result["scores"]["g"])

    def test_raw_assets_are_verified_before_extraction(self):
        import hashlib
        from app.rsi_runtime import _paired_evaluation_single
        with tempfile.TemporaryDirectory() as tmp:
            root, workspace = Path(tmp) / "root", Path(tmp) / "workspace"
            root.mkdir()
            write_record(root / "configs/rsi.json", {"mode": "observe"})
            module = workspace / "orchestrator/app/quality_scoring.py"
            module.parent.mkdir(parents=True)
            (module.parent / "__init__.py").touch()
            module.write_text("class PairSpec:\n def __init__(self,**kwargs): self.__dict__.update(kwargs)\n"
                              "def extract_pair(spec): return {'valid':True,'area':float(spec.label.read_text())}\n"
                              "def score_record(r,p): return {'score':r['area']}\n")
            freeze_parent(workspace, root)
            folder = root / "runtime/rsi/benchmarks/raw"
            (folder / "assets").mkdir(parents=True)
            (folder / "assets/image.nii").write_text("image")
            (folder / "assets/label.nii").write_text("80")
            data = {"profile": {}, "records": [{"id": "g", "target": "cmr_myo",
                    "raw_pair": {"image": "assets/image.nii", "label": "assets/label.nii"}}],
                    "references": {"g": {"kind": "gold"}},
                    "asset_hashes": {"assets/" + name: hashlib.sha256((folder / "assets" / name).read_bytes()).hexdigest()
                                     for name in ("image.nii", "label.nii")}}
            write_record(folder / "input.json", data)
            write_record(root / "configs/rsi_benchmarks/cmr_paired.json", {
                "schema_version": "rsi-benchmark-v1", "benchmark_id": "raw-test", "target": "cmr_myo",
                "module": "app.quality_scoring", "scorer": "reviewed_quality_scores_v1",
                "input_path": "runtime/rsi/benchmarks/raw/input.json", "input_sha256": canonical_hash(data),
                "pass_score": 70, "defect_drop": 5,
                "metrics": {"r": {"direction": "higher", "report_field": "gold_retention"}}})
            report = _paired_evaluation_single(root, workspace, "cmr_paired")
            self.assertEqual(1, report["candidate_metrics"]["r"])
            (folder / "assets/label.nii").write_text("90")
            with self.assertRaisesRegex(ValueError, "frozen_reference_asset_changed"):
                _paired_evaluation_single(root, workspace, "cmr_paired")


if __name__ == "__main__":
    unittest.main()
