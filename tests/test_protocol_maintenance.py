import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from app import lge_protocol as protocol
from app.control_checkpoint import save_journal
from contracts.models import SupervisorAction,FailureAnalysis
from contracts.compat import model_validate
from agents.governance import enforce_action_ownership
from agents.shadow_analysis import select_analysis_roles,filter_workflow_actions
from experiments.llm_maintenance import _with_lge_consumer_contract,_required_target_sources
from scripts import autopilot

class ProtocolMaintenanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        (self.root/'configs').mkdir()
    def policy(self,**extra):
        cfg=json.loads(json.dumps(protocol.DEFAULT))
        if 'input_aliases' in extra:cfg['input_aliases'].update(extra.pop('input_aliases'))
        cfg.update(extra)
        (self.root/protocol.CONFIG).write_text(json.dumps(cfg))
        return cfg
    def action(self):
        return model_validate(SupervisorAction,{'action_id':'protocol','owner_role':'workflow_maintainer',
            'action_class':'framework_repair','objective':'repair '+protocol.CONFIG,'acceptance_metrics':['lge_protocol_contract'],
            'priority':1,'max_attempts':2,'auto_execute':True})
    def test_aliases_cannot_replace_identity_decision_budget_or_source_locations(self):
        for key in ('decision','case_id','source_sha256','candidate'):
            cfg=json.loads(json.dumps(protocol.DEFAULT));cfg['input_aliases'][key]='label'
            with self.assertRaises(ValueError):protocol.validate(cfg)
        cfg=json.loads(json.dumps(protocol.DEFAULT));cfg['input_aliases']['other']='../../labels'
        with self.assertRaises(ValueError):protocol.validate(cfg)
        with self.assertRaises(ValueError):protocol.validate({**protocol.DEFAULT,'pass_score':0})
        with self.assertRaises(ValueError):protocol.validate({**protocol.DEFAULT,'max_attempts':100})
        cfg=json.loads(json.dumps(protocol.DEFAULT));cfg['input_aliases']['snapshot_image_path']='label'
        with self.assertRaises(ValueError):protocol.validate(cfg)
    def test_new_aliases_only_forward_supplied_snapshot_paths(self):
        cfg=self.policy(input_aliases={'image_filepath':'image','mask_filepath':'label','paired':'myo'})
        paths={'image':'/snapshot/image','label':'/snapshot/label'}
        mapped=protocol.input_aliases(self.root,paths)
        self.assertEqual(paths['image'],mapped['image_filepath'])
        self.assertEqual(paths['label'],mapped['mask_filepath'])
        self.assertNotIn('paired',mapped)
        self.policy(output_candidate_aliases=['candidate_filename','output_file'])
        self.assertEqual('safe.nii.gz',protocol.output_name(self.root,{'output_file':'safe.nii.gz'}))
        with self.assertRaises(ValueError):protocol.output_name(self.root,{'candidate':'a','output_file':'b'})
    def test_framework_action_has_distinct_owner_and_does_not_need_clinical_stage_repair(self):
        action=self.action();kept,blocked=enforce_action_ownership([action])
        self.assertEqual(1,len(kept));self.assertFalse(blocked)
        wrong=action.model_copy(update={'owner_role':'qc_code_optimizer'}) if hasattr(action,'model_copy') else action.copy(update={'owner_role':'qc_code_optimizer'})
        self.assertTrue(enforce_action_ownership([wrong])[1])
        workflow={'stage':'lge_myo','integration_required':True,'lge_generator_required':True,'protocol_repair_required':True}
        self.assertEqual([action],filter_workflow_actions([action],workflow)[0])
        workflow['protocol_repair_required']=False
        self.assertFalse(filter_workflow_actions([action],workflow)[0])
    def test_protocol_fault_routes_only_to_config_maintainer(self):
        analysis=object()  # Protocol evidence takes precedence without clinical interpretation.
        self.assertEqual(([],['workflow_maintainer']),select_analysis_roles({'protocol_repair':{'required':True}},analysis))
    def test_framework_owner_gets_exact_json_and_existing_producer_as_read_only_context(self):
        self.policy()
        req=autopilot.owner_execution_requirements({'action_class':'framework_repair'},None)
        contract=_with_lge_consumer_contract({'control_plane_requirements':req})
        self.assertIn('protocol_contract',contract['control_plane_requirements'])
        self.assertIn(protocol.CONFIG,_required_target_sources(contract,self.root))
    def test_scope_gate_blocks_any_medical_or_control_kernel_edit_before_running_code(self):
        with patch.object(autopilot,'command') as command:
            report=autopilot.validate_workspace(self.root,['app/quality_scoring.py'],action={'action_class':'framework_repair','owner_role':'workflow_maintainer'})
        self.assertFalse(report['passed']);command.assert_not_called()
        with patch.object(autopilot,'command') as command:
            report=autopilot.validate_workspace(self.root,[protocol.CONFIG],action={'action_class':'isolated_code_experiment','owner_role':'qc_code_optimizer'})
        self.assertFalse(report['passed']);command.assert_not_called()

    def test_candidate_protocol_validation_cannot_be_disabled_by_extra_config_fields(self):
        self.policy()
        self.assertFalse(protocol.repair_status(self.root)['required'])
        config=json.loads((self.root/protocol.CONFIG).read_text());config['skip_candidate_validation']=True
        (self.root/protocol.CONFIG).write_text(json.dumps(config))
        self.assertTrue(protocol.repair_status(self.root)['required'])

    def test_normal_daemon_proposes_one_protocol_repair_campaign_not_a_paid_loop(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from app.database import Database
        from app.autonomy_feedback import record_execution
        config=self.policy()
        (self.root/'configs/autonomy.json').write_text(json.dumps({'enabled':True,'dataset':str((self.root/'data').resolve())}))
        record_execution(self.root,{'execution_mode':'execute','workflow_stage':'lge_myo',
            'input_source_sha256':'input','calibration_id':'profile','generator':{'run_id':1,'action_id':'g'},
            'outcomes':[{'case_id':'opaque','status':'NEEDS_REVIEW','execution_error':True,'reason':'generator_failure:generator_protocol_missing_candidate_filename'}]})
        db=Database(self.root/'runtime/workflow.sqlite');db.initialize()
        pilot=autopilot.Autopilot.__new__(autopilot.Autopilot);pilot.db=db;pilot.dataset=(self.root/'data').resolve()
        pilot.state={};pilot.event=Mock();pilot.analyze_new_snapshot=Mock()
        with patch.object(autopilot,'ROOT',self.root):
            self.assertTrue(pilot.service_protocol_repair())
            self.assertTrue(pilot.service_protocol_repair())
        pilot.analyze_new_snapshot.assert_called_once()

    def test_changed_protocol_waits_for_consumer_evidence_before_another_paid_proposal(self):
        from app.autonomy_feedback import record_execution
        from app.rsi_kernel import canonical_hash
        old=self.policy()
        record_execution(self.root,{'execution_mode':'execute','workflow_stage':'lge_myo',
            'input_source_sha256':'input','calibration_id':'profile','generator':{'run_id':1,'action_id':'g','protocol_sha256':canonical_hash(old)},
            'outcomes':[{'case_id':'opaque','status':'NEEDS_REVIEW','execution_error':True,'reason':'generator_failure:generator_protocol_missing_candidate_filename'}]})
        self.assertTrue(protocol.repair_status(self.root)['required'])
        self.policy(output_candidate_aliases=['candidate_filename','additional_alias'])
        status=protocol.repair_status(self.root)
        self.assertFalse(status['required']);self.assertTrue(status['pending_verification'])
