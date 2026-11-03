import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
from app.control_checkpoint import save_journal as _save_journal
from app import autonomy_control as control
from app.autonomy_feedback import record_execution,feedback_summary,actual_execution
from scripts import autopilot

def save_journal(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    _save_journal(path,value)

class AutonomyControlTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.dataset=self.root/'data'
        save_journal(self.root/'configs/autonomy.json',{'schema_version':'autonomy-v1','enabled':True,'dataset':str(self.dataset.resolve()),
            'restart_cooldown':120,'max_service_restarts':3})

    def test_operator_pause_prevents_service_recovery_and_any_paid_tick(self):
        save_journal(self.root/'runtime/control/operator_state.json',{'paused':True})
        runner=Mock()
        self.assertEqual('PAUSED',control.ensure_consumer(self.root,dataset=self.dataset,runner=runner)['status'])
        runner.assert_not_called()
        pilot=autopilot.Autopilot.__new__(autopilot.Autopilot);pilot.state={};pilot.dataset=self.dataset
        pilot.db=Mock();pilot.event=Mock();pilot.analyze_new_snapshot=Mock();pilot.process_actions=Mock()
        with patch.object(autopilot,'ROOT',self.root),patch.object(autopilot,'global_promotion_preflight',return_value={}):pilot.tick()
        pilot.analyze_new_snapshot.assert_not_called();pilot.process_actions.assert_not_called()

    def test_recovery_has_persistent_cooldown_and_attempt_limit(self):
        runner=Mock(side_effect=lambda *a,**kw:SimpleNamespace(returncode=1))
        first=control.ensure_consumer(self.root,dataset=self.dataset,runner=runner,now=1000)
        self.assertEqual('RECOVERY_FAILED',first['status']);self.assertEqual(2,runner.call_count)
        self.assertEqual('RECOVERY_WAIT',control.ensure_consumer(self.root,dataset=self.dataset,runner=runner,now=1010)['status'])
        self.assertEqual(2,runner.call_count)
        control.ensure_consumer(self.root,dataset=self.dataset,runner=runner,now=1200)
        control.ensure_consumer(self.root,dataset=self.dataset,runner=runner,now=1400)
        self.assertEqual('REPAIR_REQUIRED',control.ensure_consumer(self.root,dataset=self.dataset,runner=runner,now=1600)['status'])
        self.assertEqual(6,runner.call_count)
        self.assertTrue(all(control.SERVICE in args[0] for args,kw in runner.call_args_list))

    def test_real_healthy_heartbeat_resets_only_consecutive_restart_budget(self):
        save_journal(self.root/control.HEALTH,{'status':'READY','updated_at':1000,'execute':True,'pid':os.getpid()})
        save_journal(self.root/'runtime/autonomy/service_recovery.json',{'attempts':3,'last_attempt':900})
        runner=Mock()
        self.assertTrue(control.ensure_consumer(self.root,dataset=self.dataset,runner=runner,now=1001)['ready'])
        self.assertEqual(0,control.read(self.root/'runtime/autonomy/service_recovery.json')['attempts'])
        runner.assert_not_called()

    def test_paused_heartbeat_does_not_turn_into_a_false_offline_alarm(self):
        save_journal(self.root/'runtime/control/operator_state.json',{'paused':True})
        save_journal(self.root/control.HEALTH,{'status':'PAUSED','updated_at':1,'execute':True})
        self.assertEqual('PAUSED',control.consumer_state(self.root,now=999999)['status'])

    def test_bad_control_json_surfaces_error_without_breaking_board_status(self):
        (self.root/'configs/autonomy.json').write_text('{invalid')
        status=control.autonomy_status(self.root)
        self.assertEqual('UNAVAILABLE',status['status'])
        self.assertEqual('CONTROL_ERROR',status['consumer']['status'])

    def test_processing_grace_does_not_restart_long_running_valid_job(self):
        save_journal(self.root/control.HEALTH,{'status':'RUNNING','updated_at':1000,'execute':True,'pid':os.getpid()})
        self.assertEqual('RUNNING',control.consumer_state(self.root,now=1800)['status'])
        self.assertEqual('STALE',control.consumer_state(self.root,now=2300)['status'])
        self.assertTrue(control.ensure_consumer(self.root,dataset=self.dataset,runner=Mock(),now=1800)['ready'])

    def test_dataset_mismatch_cannot_start_production_service_from_tests_or_snapshot(self):
        runner=Mock()
        self.assertEqual('DISABLED',control.ensure_consumer(self.root,dataset=self.root/'other',runner=runner)['status'])
        runner.assert_not_called()

    def test_normal_tick_reopens_existing_runtime_failure_without_new_analysis_or_budget(self):
        from app.database import Database
        from app import lge_execution as engine
        db=Database(self.root/'runtime/workflow.sqlite');db.initialize()
        run=db.create_run(db.register_dataset('fixture',self.dataset),'analysis',{})
        payload={'action_id':'g','owner_role':'qc_code_optimizer','action_class':'isolated_code_experiment',
                 'auto_execute':True,'review_approved':True,'max_attempts':3}
        db.save_maintenance_action(run,payload,status='PROMOTED')
        db.update_maintenance_action(run,'g',status='PROMOTED',artifact_path=self.root/'prior')
        release={'ready':True,'run_id':run,'action_id':'g','module_sha256':'test-module'}
        report={'execution_mode':'execute','production_writes':0,'generator':release,'outcomes':[{'reason':'generator_failure:synthetic','case_id':'opaque'}]}
        save_journal(self.root/'runtime/autonomy/last_execution.json',report)
        pilot=autopilot.Autopilot.__new__(autopilot.Autopilot);pilot.dataset=self.dataset;pilot.db=db
        pilot.state={'paused_issue_fingerprint':'unchanged'};pilot.event=Mock()
        pilot.process_actions=Mock(return_value={});pilot.analyze_new_snapshot=Mock()
        with patch.object(autopilot,'ROOT',self.root),patch.object(autopilot,'global_promotion_preflight',return_value={}), \
             patch.object(control,'ensure_consumer',return_value={'ready':True,'status':'READY'}), \
             patch.object(engine,'governed_generator',return_value=release),patch.object(autopilot,'action_attempts',return_value=1), \
             patch.object(autopilot,'agent_action_authorized',return_value=True):
            pilot.tick()
        pilot.process_actions.assert_called_once_with(run);pilot.analyze_new_snapshot.assert_not_called()
        self.assertEqual(3,db.get_maintenance_action(run,'g')['payload']['max_attempts'])
        ledger=control.read(self.root/f'runtime/autonomy/repair_run_{run:06d}.json')
        self.assertEqual(1,ledger['steps']);self.assertEqual('ACTIONS',ledger['status'])

class FeedbackTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.release={'ready':True,'run_id':1,'action_id':'g','module_sha256':'v1'}
    def result(self,rows,mode='execute'):
        return {'execution_mode':mode,'workflow_stage':'lge_myo','generator':self.release,
                'input_source_sha256':'source','calibration_id':'profile','outcomes':rows}
    def accepted(self):
        row={'case_id':'opaque','status':'CHECKED','transaction_id':'a'*32,'after_hashes':{'label':'sha'},
             'score_before':65,'score_after':90,'label_write':True}
        save_journal(self.root/('runtime/lge_execution/transactions/'+'a'*32+'.json'),{'state':'COMMITTED','receipt':row})
        return row
    def test_preview_never_becomes_learning_credit(self):
        self.assertFalse(record_execution(self.root,self.result([self.accepted()],mode='preview'))['recorded'])
        self.assertEqual(0,feedback_summary(self.root)['episodes'])
    def test_committed_application_gets_credit_once_and_replay_is_deduplicated(self):
        row=self.accepted();result=self.result([row]);first=record_execution(self.root,result)
        self.assertEqual(1,first['accepted']);self.assertTrue(first['recorded'])
        self.assertFalse(record_execution(self.root,result)['recorded'])
        # Another episode containing the same transaction cannot create extra credit.
        result['input_source_sha256']='new-input';record_execution(self.root,result)
        summary=feedback_summary(self.root)
        self.assertEqual(1,summary['role_credit'][0]['accepted'])
        self.assertEqual(25,summary['role_credit'][0]['scored_gain'])
        self.assertEqual(0,summary['recent'][0]['new_case_exposures'])
        self.assertEqual('NOT_ESTABLISHED',summary['generalization_status'])
        self.assertNotIn('outcomes',summary['recent'][0])
    def test_uncommitted_or_model_claimed_success_does_not_get_credit(self):
        row=self.accepted();(self.root/('runtime/lge_execution/transactions/'+'a'*32+'.json')).unlink()
        record_execution(self.root,self.result([row]))
        summary=feedback_summary(self.root)
        self.assertEqual([],summary['role_credit']);self.assertEqual('UNVERIFIED_RESULTS',summary['recent'][0]['kind'])
    def test_execution_failure_survives_preview_and_is_not_a_medical_failure(self):
        result=self.result([{'case_id':'opaque','status':'NEEDS_REVIEW','reason':'generator_failure:synthetic','execution_error':True}])
        record_execution(self.root,result)
        save_journal(self.root/'runtime/lge_iterations/latest.json',{'last_execution':{'execution_mode':'preview'}})
        self.assertEqual(result,actual_execution(self.root))
        summary=feedback_summary(self.root)['recent'][0]
        self.assertEqual('EXECUTION_FAILURE',summary['kind']);self.assertEqual(0,summary['domain_rejected'])


class PauseDuringApplyTests(unittest.TestCase):
    def test_paused_consumer_does_not_run_evaluation_or_label_writers(self):
        from scripts import check_feedback_loop as loop
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);save_journal(root/'runtime/control/operator_state.json',{'paused':True})
            with patch.object(loop,'ROOT',root),patch.object(loop,'tick') as scan,patch.object(loop,'run_lge_consumer_once') as consume,patch.object(loop,'run_current_auto_apply') as apply:
                loop.run_consumers(root/'dataset',auto_apply=True)
            scan.assert_not_called();consume.assert_not_called();apply.assert_not_called()
