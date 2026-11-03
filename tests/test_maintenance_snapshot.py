from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments.codex_maintenance import _baseline_hashes, _copy_source
from experiments.llm_maintenance import (_copy_source as _copy_llm_source,
                                         _run_lge_consumer_contract_probe,
                                         _required_target_sources,
                                         _with_lge_consumer_contract, _previous_owner_attempt, _owner_model_config, _test_framework_diagnostics)


class MaintenanceSnapshotTests(unittest.TestCase):
    def test_genuine_runtime_repair_uses_configured_alternate_without_changing_owner_permissions(self):
        cfg={'provider':'same','model':'original','runtime_repair_models':['alternate'],'can_write_snapshot':True}
        self.assertEqual('original',_owner_model_config(cfg,{})['model'])
        actual=_owner_model_config(cfg,{'runtime_failure_feedback':{'failure':True}})
        self.assertEqual('alternate',actual['model']);self.assertTrue(actual['can_write_snapshot'])
        self.assertEqual('original',cfg['model'])
        self.assertIn('shape',_test_framework_diagnostics('np.array_equal(matrix,1) AssertionError'))
        self.assertEqual('',_test_framework_diagnostics('other exception'))

    def test_runtime_repair_inherits_only_same_action_code_without_installing_it(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);previous=root/'action_earlier';current=root/'action_current'
            (previous/'orchestrator/scripts').mkdir(parents=True);current.mkdir()
            (previous/'orchestrator/scripts/candidate.py').write_text('prior unpromoted implementation')
            (previous/'maintenance_manifest.json').write_text(json.dumps({'result':{'tests_passed':False,
                'changed_files':['scripts/candidate.py','scripts/../../outside.py']}}))
            result=_previous_owner_attempt(root,{'action_id':'action','runtime_failure_feedback':{'error':True}},current)
            self.assertEqual(['scripts/candidate.py'],[f['path'] for f in result['files']])
            self.assertEqual('prior unpromoted implementation',result['files'][0]['content'])
            self.assertEqual([],list(current.iterdir()))
            self.assertEqual({},_previous_owner_attempt(root,{'action_id':'different','runtime_failure_feedback':{'error':True}},current))
            self.assertEqual({},_previous_owner_attempt(root,{'action_id':'action'},current))

    def test_lge_candidate_owner_gets_snapshot_contract_not_direct_writer_permission(self):
        result = _with_lge_consumer_contract({"control_plane_requirements": {
            "required_target_path": "scripts/lge_candidate_generator.py"}})
        contract = result["control_plane_requirements"]["lge_candidate_contract"]
        self.assertIn("Controller alone", contract["boundary"])
        self.assertIn("repair", contract["decisions"])
        self.assertIn("output_dir", result["control_plane_requirements"]["required_interface"])

    def test_skill_snapshot_allowlist_excludes_models_and_vendor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            target = root / "target"
            for name in ("tools", "models", "vendor"):
                (source / name).mkdir(parents=True)
                (source / name / "content.bin").write_bytes(b"content")
            (source / "SKILL.md").write_text("rules\n", encoding="utf-8")

            _copy_source(source, target, include=("SKILL.md", "tools"))

            self.assertTrue((target / "SKILL.md").is_file())
            self.assertTrue((target / "tools/content.bin").is_file())
            self.assertFalse((target / "models").exists())
            self.assertFalse((target / "vendor").exists())

    def test_baseline_hashes_are_owner_scoped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tools").mkdir()
            (root / "tools/rule.py").write_text("VALUE = 1\n", encoding="utf-8")
            hashes = _baseline_hashes(root, "skill")
            self.assertEqual(list(hashes), ["skill/tools/rule.py"])
            self.assertEqual(len(hashes["skill/tools/rule.py"]), 64)

    def test_provider_snapshot_uses_same_skill_allowlist(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            target = root / "target"
            for name in ("tools", "models", "vendor", "archive"):
                (source / name).mkdir(parents=True)
                (source / name / "content.bin").write_bytes(b"content")
            _copy_llm_source(source, target, include=("tools",))
            self.assertTrue((target / "tools/content.bin").is_file())
            self.assertFalse((target / "models").exists())
            self.assertFalse((target / "vendor").exists())
            self.assertFalse((target / "archive").exists())

    def test_named_scoring_source_larger_than_old_cap_is_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "app/quality_scoring.py"
            path.parent.mkdir()
            path.write_text("#" + "x" * 27000)
            sources = _required_target_sources({"control_plane_requirements": {
                "required_target_path": "app/quality_scoring.py"}}, root)
            self.assertEqual(path.read_text(), sources["app/quality_scoring.py"])

    def test_required_target_source_is_supplied_without_path_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "scripts/shadow_spatial_boundary.py"
            target.parent.mkdir(parents=True)
            target.write_text("def evaluate(dataset):\n    return {}\n", encoding="utf-8")
            action = {"control_plane_requirements": {
                "required_target_path": "scripts/shadow_spatial_boundary.py",
            }}
            self.assertEqual(_required_target_sources(action, root), {
                "scripts/shadow_spatial_boundary.py": target.read_text(encoding="utf-8"),
            })
            action["control_plane_requirements"]["required_target_path"] = "scripts/../secret.py"
            self.assertEqual(_required_target_sources(action, root), {})

    def test_lge_consumer_owner_receives_immutable_interface_contract(self):
        action = {"control_plane_requirements": {
            "required_target_path": "scripts/lge_feedback_loop.py",
            "required_interface": "old prose",
        }}
        normalized = _with_lge_consumer_contract(action)
        requirements = normalized["control_plane_requirements"]
        self.assertEqual(requirements["required_interface"],
                         "run_evaluation(data: dict) -> dict")
        self.assertIn("holdout_contract", requirements["lge_consumer_contract"])
        self.assertIn("test_contract", requirements["lge_consumer_contract"])
        self.assertNotIn("lge_consumer_contract", action["control_plane_requirements"])

    def test_lge_contract_probe_rejects_invented_available_holdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "scripts/lge_feedback_loop.py"
            target.parent.mkdir(parents=True)
            target.write_text(
                "def run_evaluation(data):\n"
                "    return {\n"
                "      'source_modality': 'lge', 'workflow_stage': data['workflow_stage'],\n"
                "      'input_source_sha256': data['source_sha256'],\n"
                "      'lge_myo_consumer_ready': True, 'lge_myo_pending': 2,\n"
                "      'lge_myo_work': 0, 'lge_myo_resolved': 0,\n"
                "      'frozen_split_status': 'AVAILABLE', 'holdout_excluded': 0,\n"
                "      'label_write': False, 'production_writes': 0}\n",
                encoding="utf-8",
            )
            passed, output = _run_lge_consumer_contract_probe(root)
            self.assertFalse(passed)
            self.assertIn("frozen_unavailable", output)

    def test_lge_contract_probe_accepts_pure_unavailable_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "scripts/lge_feedback_loop.py"
            target.parent.mkdir(parents=True)
            target.write_text(
                "def run_evaluation(data):\n"
                "    n = len(data['records'])\n"
                "    return {\n"
                "      'source_modality': 'lge', 'workflow_stage': data['workflow_stage'],\n"
                "      'input_source_sha256': data['source_sha256'],\n"
                "      'lge_myo_consumer_ready': False, 'lge_myo_pending': n,\n"
                "      'lge_myo_work': 0, 'lge_myo_resolved': 0,\n"
                "      'frozen_split_status': 'UNAVAILABLE', 'holdout_excluded': 0,\n"
                "      'label_write': False, 'production_writes': 0}\n",
                encoding="utf-8",
            )
            passed, output = _run_lge_consumer_contract_probe(root)
            self.assertTrue(passed, output)

    def test_lge_contract_probe_rejects_runtime_file_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "scripts/lge_feedback_loop.py"
            target.parent.mkdir(parents=True)
            target.write_text(
                "from pathlib import Path\n"
                "def run_evaluation(data):\n"
                "    Path('runtime').mkdir()\n"
                "    Path('runtime/latest.json').write_text('{}')\n"
                "    n = len(data['records'])\n"
                "    return {\n"
                "      'source_modality': 'lge', 'workflow_stage': data['workflow_stage'],\n"
                "      'input_source_sha256': data['source_sha256'],\n"
                "      'lge_myo_consumer_ready': False, 'lge_myo_pending': n,\n"
                "      'lge_myo_work': 0, 'lge_myo_resolved': 0,\n"
                "      'frozen_split_status': 'UNAVAILABLE', 'holdout_excluded': 0,\n"
                "      'label_write': False, 'production_writes': 0}\n",
                encoding="utf-8",
            )
            passed, output = _run_lge_consumer_contract_probe(root)
            self.assertFalse(passed)
            self.assertIn("pure_no_files_written", output)


if __name__ == "__main__":
    unittest.main()
