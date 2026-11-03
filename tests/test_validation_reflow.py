import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock,patch
import scripts.autopilot as pilot
from experiments.llm_maintenance import _previous_owner_attempt

class ReflowTests(unittest.TestCase):
    def setup_case(self,root):
        w=root/'runtime/maintenance/run_000001/action_attempt';w.mkdir(parents=True)
        (w/'rsi_selection.json').write_text(json.dumps({'benchmark_id':'reviewed-generator-v1','reasons':['no_quality_improvement'],'child_variant_id':'v1'}))
        (w/'autopilot_validation.json').write_text(json.dumps({'checks':[{'returncode':0}]}))
        c=pilot.Autopilot.__new__(pilot.Autopilot);c.dataset=root/'dataset';c.db=Mock();c.event=Mock()
        c.db.list_runs.return_value=[{'run_id':1,'processed_path':str(c.dataset)}]
        c.db.run_details.return_value={'maintenance_actions':[{'action_id':'action','owner_role':'qc_code_optimizer','status':'REJECTED','artifact_path':str(w),'payload_json':json.dumps({'max_attempts':3})}]}
        c.process_actions=Mock();return c,w
    def test_measured_rejection_returns_to_original_action(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);c,w=self.setup_case(root);c.claim_bounded_owner_repair=Mock(return_value=True)
            with patch.object(pilot,'ROOT',root),patch('app.autonomy_control.policy',return_value={'enabled':True,'dataset':str(c.dataset)}),patch('agents.governance.original_action_approved',return_value=True):
                self.assertTrue(c.resume_validation_owner())
            self.assertEqual(c.claim_bounded_owner_repair.call_args.args[2]['selection']['child_variant_id'],'v1')
            c.process_actions.assert_called_once_with(1)
    def test_exhausted_budget_or_failed_tests_never_dispatch(self):
        for tests_ok in [True,False]:
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);c,w=self.setup_case(root);c.claim_bounded_owner_repair=Mock(return_value=False)
                if not tests_ok:(w/'autopilot_validation.json').write_text(json.dumps({'checks':[{'returncode':1}]}))
                with patch.object(pilot,'ROOT',root),patch('app.autonomy_control.policy',return_value={'enabled':True,'dataset':str(c.dataset)}),patch('agents.governance.original_action_approved',return_value=True):
                    self.assertFalse(c.resume_validation_owner())
                c.process_actions.assert_not_called()
    def test_passing_tests_but_rejected_quality_snapshot_is_inherited(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old=root/'action_old';(old/'orchestrator/scripts').mkdir(parents=True)
            (old/'orchestrator/scripts/example.py').write_text('value=1')
            (old/'maintenance_manifest.json').write_text(json.dumps({'result':{'tests_passed':True,'ready_for_validation':True,'changed_files':['scripts/example.py']}}))
            (old/'rsi_selection.json').write_text(json.dumps({'decision':'RETAIN_PARENT'}))
            inherited=_previous_owner_attempt(root,{'action_id':'action','runtime_failure_feedback':{'failure_type':'quality'}},root/'action_new')
            self.assertEqual(inherited['files'][0]['content'],'value=1')

    def test_original_handoffs_do_not_make_rejected_action_executable(self):
        from agents.governance import original_action_approved,agent_action_authorized
        db=Mock();db.agent_inbox.return_value=[{'sender_role':'supervisor','message_type':'assigned_action'}, {'sender_role':'independent_reviewer','message_type':'action_review','payload':{'decision':'accept_experiment'}}]
        row={'status':'REJECTED','owner_role':'qc_code_optimizer','action_id':'action'}
        payload={'review_approved':True,'action_class':'isolated_code_experiment'}
        self.assertTrue(original_action_approved(db,1,row,payload))
        self.assertFalse(agent_action_authorized(db,1,row,payload))
