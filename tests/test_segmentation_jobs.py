import copy
import hashlib
import json
import threading
import unittest
import nibabel as nib
import numpy as np
from pathlib import Path
from unittest.mock import patch

from tests import test_paired_dataset as fixtures
from app.paired_dataset import audit_pairs, freeze_release
from app import segmentation_jobs as jobs
from app.segmentation_jobs import compile_job, freeze_assignment_contract, UnboundAssignmentsWarning
from app.segmentation_training import sha256_file
from scripts.segment_from_registry import predict_command


class ReleaseTrainingIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.PairedDatasetTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        evidence = list(f.evidence)
        for index, patient in enumerate(("SYNTHETIC_B", "SYNTHETIC_C")):
            key = patient + "/level.nii.gz"
            paths = f.make_bundle(key)
            for role in ("cmr_image", "lge_image"):
                image = nib.load(paths[role])
                values = np.asarray(image.dataobj).copy() + index + 1
                nib.save(nib.Nifti1Image(values, image.affine), paths[role])
            evidence.extend(f.make_evidence(key))
        freeze_release(f.dataset, audit_pairs(f.dataset, evidence), f.destination)
        self.vendor = f.base / "vendor"
        (self.vendor / "automated_annotation_of_lge").mkdir(parents=True)
        (self.vendor / "automated_annotation_of_lge/main.py").write_text("# synthetic adapter source")
        self.config = f.base / "approved.json"
        self.config.write_text(json.dumps({"model": {"base_channels": 32},
            "training": {"epochs": 1, "seed": 42}, "visualization": {}}))
        self.args = dict(target="lge_myo", assignments={"SYNTHETIC_A": "train", "SYNTHETIC_B": "validation", "SYNTHETIC_C": "test"},
            config=self.config, vendor_root=self.vendor, output=f.base / "candidate", resources={"max_seconds": 10, "max_epochs": 1, "device": "cpu"})
        self.contract_path = f.base / "patient_assignments.json"
        self.args["assignment_contract"] = freeze_assignment_contract(self.contract_path, self.args["assignments"])

    def reexport(self):
        f = self.fixture
        original = json.loads((f.destination / "private/manifest.json").read_text())
        evidence = [e for row in original["bundles"] for records in row["evidence"].values() for e in records]
        destination = f.base / "release2"
        freeze_release(f.dataset, audit_pairs(f.dataset, evidence), destination)
        return destination

    def test_release_compiles_fixed_patient_groups(self):
        job = compile_job(self.fixture.destination, **self.args)
        self.assertEqual({k:len(v) for k,v in job["splits"].items()}, {"train":1,"validation":1,"test":1})
        self.assertEqual(len(job["records"]), 3)
        self.assertNotIn("SYNTHETIC_A", json.dumps(job))

    def test_reexport_preserves_patient_assignments(self):
        first = compile_job(self.fixture.destination, **self.args)
        # Re-export has new public pseudonyms; original private group is stable.
        again = self.reexport()
        second=compile_job(again,**self.args)
        self.assertEqual(first["splits"],second["splits"])

    def test_new_job_requires_persisted_binding_and_never_implicitly_creates_one(self):
        self.args.pop("assignment_contract")
        before = set(self.fixture.base.iterdir())
        with self.assertRaisesRegex(ValueError, "assignment_contract_required"):
            compile_job(self.fixture.destination, **self.args)
        self.assertEqual(before, set(self.fixture.base.iterdir()))
        self.assertFalse(self.args["output"].exists())

    def test_explicit_legacy_opt_out_retains_job_schema_but_warns_no_enforcement(self):
        bound = compile_job(self.fixture.destination, **self.args)
        self.args.pop("assignment_contract")
        with self.assertWarnsRegex(UnboundAssignmentsWarning, "not enforced"):
            legacy = compile_job(self.fixture.destination, **self.args, allow_unbound_assignments=True)
        self.assertEqual(bound, legacy)
        # Legacy opt-out intentionally makes no cross-job guarantee.
        self.args["assignments"].update(SYNTHETIC_A="test", SYNTHETIC_C="train")
        with self.assertWarns(UnboundAssignmentsWarning):
            changed = compile_job(self.fixture.destination, **self.args, allow_unbound_assignments=True)
        self.assertNotEqual(changed["splits"], bound["splits"])

    def test_same_release_second_job_cannot_reshuffle_stable_patient_ids(self):
        first = compile_job(self.fixture.destination, **self.args)
        before = self.contract_path.read_bytes()
        self.args["output"] = self.fixture.base / "second-job"
        self.args["assignments"].update(SYNTHETIC_A="test", SYNTHETIC_C="train")
        with self.assertRaisesRegex(ValueError, "patient_assignment_contract_mismatch"):
            compile_job(self.fixture.destination, **self.args)
        self.assertEqual(before, self.contract_path.read_bytes())
        self.assertEqual(first["splits"]["train"], [hashlib.sha256(b"SYNTHETIC_A").hexdigest()])
        self.assertFalse(self.args["output"].exists())

    def test_new_release_cannot_reshuffle_same_original_patient(self):
        compile_job(self.fixture.destination, **self.args)
        again = self.reexport()
        self.args["assignments"].update(SYNTHETIC_A="validation", SYNTHETIC_B="train")
        with self.assertRaisesRegex(ValueError, "patient_assignment_contract_mismatch"):
            compile_job(again, **self.args)

    def test_other_target_obeys_same_contract(self):
        first = compile_job(self.fixture.destination, **self.args)
        vendor = self.vendor / "automated_annotation_of_cine"
        vendor.mkdir()
        (vendor / "main_multicenter.py").write_text("# synthetic adapter")
        self.args["target"] = "cmr_myo"
        second = compile_job(self.fixture.destination, **self.args)
        self.assertEqual(first["splits"], second["splits"])
        self.args["assignments"].update(SYNTHETIC_A="test", SYNTHETIC_C="train")
        with self.assertRaisesRegex(ValueError, "patient_assignment_contract_mismatch"):
            compile_job(self.fixture.destination, **self.args)

    def test_contract_may_cover_more_patients_than_explicit_job_plan(self):
        path = self.fixture.base / "larger-contract.json"
        self.args["assignment_contract"] = freeze_assignment_contract(path, {**self.args["assignments"], "SYNTHETIC_FUTURE": "test"})
        job = compile_job(self.fixture.destination, **self.args)
        self.assertEqual(len(job["records"]), 3)
        self.assertNotIn(hashlib.sha256(b"SYNTHETIC_FUTURE").hexdigest(), json.dumps(job))

    def test_unknown_patient_and_explicit_opt_out_cannot_bypass_supplied_contract(self):
        self.args["assignments"]["SYNTHETIC_FUTURE"] = "train"
        with self.assertRaisesRegex(ValueError, "patient_assignment_contract_mismatch"):
            compile_job(self.fixture.destination, **self.args, allow_unbound_assignments=True)

    def test_first_plan_registration_is_idempotent_and_never_overwrites(self):
        before = (self.contract_path.read_bytes(), self.contract_path.stat().st_mtime_ns)
        reversed_assignments = dict(reversed(list(self.args["assignments"].items())))
        self.assertEqual(self.args["assignment_contract"], freeze_assignment_contract(self.contract_path, reversed_assignments))
        for changes in ({"SYNTHETIC_A": "test"}, {"SYNTHETIC_FUTURE": "train"}):
            with self.subTest(changes=list(changes)):
                with self.assertRaisesRegex(ValueError, "assignment_contract_conflict"):
                    freeze_assignment_contract(self.contract_path, {**self.args["assignments"], **changes})
        self.assertEqual(before, (self.contract_path.read_bytes(), self.contract_path.stat().st_mtime_ns))
        self.assertEqual(list(self.fixture.base.glob(".assignment-contract-*")), [])

    def test_stale_hash_missing_file_and_bad_binding_fail_closed(self):
        binding = dict(self.args["assignment_contract"])
        for bad in ({**binding, "sha256": "0" * 64},
                    {**binding, "path": str(self.fixture.base / "missing.json")},
                    {**binding, "path": "relative.json"},
                    {**binding, "approved": True}):
            with self.subTest(binding_keys=list(bad)):
                self.args["assignment_contract"] = bad
                with self.assertRaises(ValueError):
                    compile_job(self.fixture.destination, **self.args, allow_unbound_assignments=True)

    def test_changed_contract_cannot_be_reused_even_when_current_job_subset_unchanged(self):
        self.contract_path.chmod(0o600)
        data = json.loads(self.contract_path.read_text())
        data["assignments"]["SYNTHETIC_FUTURE"] = "test"
        self.contract_path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            compile_job(self.fixture.destination, **self.args)

    def test_contract_change_during_compilation_is_detected(self):
        validate = jobs.validate_job
        def drift(job):
            result = validate(job)
            self.contract_path.chmod(0o600)
            with self.contract_path.open("ab") as f:
                f.write(b" ")
            return result
        with patch.object(jobs, "validate_job", side_effect=drift):
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                compile_job(self.fixture.destination, **self.args)

    def test_duplicate_json_keys_and_unrecognized_identity_contract_rejected(self):
        valid = self.contract_path.read_bytes()
        invalid_identity = json.loads(valid)
        invalid_identity["patient_identity"] = "different-key-convention"
        duplicate = valid.replace(b'"assignments":{', b'"assignments":{"SYNTHETIC_A":"test",', 1)
        self.contract_path.chmod(0o600)
        for content in (duplicate, json.dumps(invalid_identity).encode()):
            self.contract_path.write_bytes(content)
            self.args["assignment_contract"]["sha256"] = sha256_file(self.contract_path)
            with self.assertRaises(ValueError):
                compile_job(self.fixture.destination, **self.args)

    def test_invalid_assignment_values_never_publish_contract(self):
        path = self.fixture.base / "invalid.json"
        for assignments in ({}, {"": "train"}, {" padded ": "train"}, {"SYNTHETIC_A": "automatic"}, {"SYNTHETIC_A": True}):
            with self.subTest(assignments=assignments):
                with self.assertRaisesRegex(ValueError, "invalid_patient_assignment"):
                    freeze_assignment_contract(path, assignments)
                self.assertFalse(path.exists())

    def test_conflicting_concurrent_first_plans_have_only_one_winner(self):
        path = self.fixture.base / "concurrent.json"
        plans = [dict(self.args["assignments"]), {**self.args["assignments"], "SYNTHETIC_A": "test"}]
        barrier, results, errors = threading.Barrier(2), [], []
        link = jobs.os.link
        def synchronized_link(source, destination):
            barrier.wait(timeout=10)
            return link(source, destination)
        def register(plan):
            try:
                results.append(freeze_assignment_contract(path, plan))
            except Exception as error:
                errors.append(error)
        with patch.object(jobs.os, "link", side_effect=synchronized_link):
            threads = [threading.Thread(target=register, args=(plan,)) for plan in plans]
            for thread in threads: thread.start()
            for thread in threads: thread.join(timeout=15)
        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(len(results), 1)
        self.assertEqual(len(errors), 1)
        self.assertEqual(str(errors[0]), "assignment_contract_conflict")
        self.assertEqual(results[0]["sha256"], sha256_file(path))
        self.assertEqual(list(self.fixture.base.glob(".assignment-contract-*")), [])

    def test_source_mapping_and_contract_unchanged_by_compilation(self):
        private = self.fixture.destination / "private/manifest.json"
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (private, self.contract_path)}
        args = copy.deepcopy(self.args)
        compile_job(self.fixture.destination, **self.args)
        self.assertEqual(self.args, args)
        for path, state in before.items(): self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), state)

    def test_job_and_frozen_cohort_use_same_explicit_private_patient_keys(self):
        import yaml
        from app.quality_scoring import load_specs
        from app.segmentation_cohort import freeze_cohort
        f = self.fixture
        config = f.base / "gold.yaml"
        config.write_text(yaml.safe_dump({"datasets": [{"name": "synthetic", "tier": "gold", "targets": {
            "lge_myo": {"image_dir": str(f.dataset / "images/lge"),
                        "label_dir": str(f.dataset / "labels/lge_checked/lge_Myo_labels")}}}]}))
        specs, _ = load_specs(config)
        plan = {"schema": "segmentation-cohort-plan-v1", "owner_role": "lge_myo_specialist",
                "config_sha256": sha256_file(config), "selections": []}
        for spec in specs:
            key = spec.relative_id.split("/", 1)[0]
            plan["selections"].append({"dataset": spec.dataset, "target": spec.target,
                "relative_id": spec.relative_id, "case_id": spec.case_id, "patient_key": key,
                "split": self.args["assignments"][key]})
        package = freeze_cohort(config, plan, f.base / "cohort")
        job = compile_job(f.destination, **self.args)
        for split, binding in package["cohorts"]["lge_myo"].items():
            cases = json.loads(Path(binding["path"]).read_text())["cases"]
            self.assertEqual({row["patient_id"] for row in cases}, set(job["splits"][split]))

    def test_missing_assignment_rejected(self):
        self.args["assignments"].pop("SYNTHETIC_C")
        with self.assertRaisesRegex(ValueError,"patient_split_missing"):
            compile_job(self.fixture.destination,**self.args)

    def test_changed_release_is_not_training_input(self):
        path=self.fixture.destination/"manifest.json"
        path.chmod(0o600)
        path.write_text("{}")
        with self.assertRaisesRegex(ValueError,"release_manifest_changed"):
            compile_job(self.fixture.destination,**self.args)

    def test_inference_uses_bound_config_and_separate_output(self):
        model=self.vendor/"automated_annotation_of_lge/lge_seg/model.py"
        model.parent.mkdir();model.write_text("# synthetic network")
        active={"candidate_id":"version-a","target":"lge_myo","kind":"baseline","dataset_sha256":"dataset-a","artifacts":{
            "config":{"path":str(self.config)},"weights":{"path":"/private/model.pt","sha256":"weight-sha"},
            "architecture":{"sha256":sha256_file(model)}}}
        with patch("scripts.segment_from_registry.resolve_active",return_value=active):
            command,receipt=predict_command(self.fixture.base/"registry",self.vendor,"lge_myo",self.fixture.dataset,
                                           self.fixture.base/"predictions")
            self.assertIn(str(self.config),command)
            self.assertFalse(receipt["production_labels_written"])
            with self.assertRaisesRegex(ValueError,"new_separate_output"):
                predict_command(self.fixture.base/"registry",self.vendor,"lge_myo",self.fixture.dataset,
                                self.fixture.dataset/"labels")
            model.write_text("# changed network")
            with self.assertRaisesRegex(ValueError,"matching_inference_adapter"):
                predict_command(self.fixture.base/"registry",self.vendor,"lge_myo",self.fixture.dataset,
                                self.fixture.base/"predictions")


if __name__ == "__main__":
    unittest.main()
