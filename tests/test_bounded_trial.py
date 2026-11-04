import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from scripts import autopilot as a

class BoundedTrialTests(unittest.TestCase):
    def pilot(self):
        p=a.Autopilot.__new__(a.Autopilot)
        p.dataset=Path('/synthetic')
        p.db=Mock()
        p.event=Mock()
        p.analyze_new_snapshot=Mock(return_value=999)
        p.process_actions=Mock(return_value={})
        p.db.run_details.return_value={'maintenance_actions':[]}
        return p

    def test_one_analysis_and_terminal_trial_cannot_repeat(self):
        with tempfile.TemporaryDirectory() as d, patch.object(a,'global_promotion_preflight',return_value={}):
            p=self.pilot(); path=Path(d)/'trial.json'
            p.bounded_run(path)
            p.analyze_new_snapshot.assert_called_once_with(None)
            p.process_actions.assert_called_once_with(999)
            self.assertEqual(json.loads(path.read_text())['status'],'COMPLETE')
            with self.assertRaises(ValueError): p.bounded_run(path)
            self.assertEqual(p.analyze_new_snapshot.call_count,1)

    def test_interrupted_analysis_never_rebuys(self):
        with tempfile.TemporaryDirectory() as d:
            p=self.pilot(); path=Path(d)/'trial.json'
            a.write_json(path,{'dataset':str(p.dataset),'status':'STARTING'})
            with self.assertRaises(ValueError): p.bounded_run(path)
            p.analyze_new_snapshot.assert_not_called()

    def test_resume_only_same_run_with_persisted_step_bound(self):
        with tempfile.TemporaryDirectory() as d:
            p=self.pilot(); path=Path(d)/'trial.json'
            a.write_json(path,{'dataset':str(p.dataset),'status':'ACTIONS','run_id':998,'steps':11})
            p.db.run_details.return_value={'maintenance_actions':[{'action_id':'x','status':'APPROVED','payload_json':json.dumps({'auto_execute':True})}]}
            p.bounded_run(path)
            p.analyze_new_snapshot.assert_not_called()
            p.process_actions.assert_called_once_with(998)
            self.assertEqual(json.loads(path.read_text())['status'],'STEP_LIMIT')

    def test_recovery_stops_before_paid_call(self):
        with tempfile.TemporaryDirectory() as d, patch.object(a,'global_promotion_preflight',return_value={'status':'RECOVERY_REQUIRED'}):
            p=self.pilot()
            with self.assertRaises(RuntimeError): p.bounded_run(Path(d)/'trial.json')
            p.analyze_new_snapshot.assert_not_called()

    def test_real_runtime_crash_requeues_same_action_without_another_analysis(self):
        from app.database import Database
        from app import lge_execution as engine
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); p=self.pilot()
            p.db=Database(root/'workflow.sqlite');p.db.initialize()
            run=p.db.create_run(p.db.register_dataset('test',root/'data'),'analysis',{})
            p.db.save_maintenance_action(run,{'action_id':'generator','owner_role':'qc_code_optimizer',
                'action_class':'isolated_code_experiment','auto_execute':True,'review_approved':True,'max_attempts':3},status='PROMOTED')
            p.db.update_maintenance_action(run,'generator',status='PROMOTED',artifact_path=root/'old')
            release={'ready':True,'run_id':run,'action_id':'generator','module_sha256':'source'}
            a.write_json(root/'runtime/lge_iterations/latest.json',{'last_execution':{
                'execution_mode':'execute','production_writes':0,'generator':release,
                'outcomes':[{'reason':'generator_failure:synthetic_runtime_error'}]}})
            record={'run_id':run,'status':'COMPLETE','steps':2}
            with patch.object(a,'ROOT',root),patch.object(a,'action_attempts',return_value=1),patch.object(engine,'governed_generator',return_value=release):
                self.assertTrue(p.reopen_bounded_runtime_failure(record))
                self.assertFalse(p.reopen_bounded_runtime_failure(record))
            row=p.db.get_maintenance_action(run,'generator')
            self.assertEqual('APPROVED',row['status']);self.assertIsNone(row['artifact_path'])
            self.assertTrue(row['payload']['review_approved'])
            self.assertEqual(1,len(record['runtime_repair_signatures']))
            self.assertEqual('runtime_execution_failure',p.db.agent_inbox(run,'qc_code_optimizer')[0]['message_type'])
            p.analyze_new_snapshot.assert_not_called()

    def test_runtime_repair_cannot_exceed_original_attempt_limit(self):
        from app import lge_execution as engine
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=self.pilot();release={'run_id':999,'action_id':'x','module_sha256':'source'}
            p.db.run_details.return_value={'maintenance_actions':[{'action_id':'x','status':'PROMOTED','payload_json':json.dumps({'max_attempts':1})}]}
            a.write_json(root/'runtime/lge_iterations/latest.json',{'last_execution':{'execution_mode':'execute','production_writes':0,
                'generator':release,'outcomes':[{'reason':'generator_failure:synthetic'}]}})
            with patch.object(a,'ROOT',root),patch.object(a,'action_attempts',return_value=1),patch.object(engine,'governed_generator',return_value=release):
                self.assertFalse(p.reopen_bounded_runtime_failure({'run_id':999,'status':'COMPLETE'}))
            p.db.connect.assert_not_called()

    def test_failed_runtime_repair_snapshot_uses_only_remaining_original_attempts(self):
        from app.database import Database
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=self.pilot();p.db=Database(root/'workflow.sqlite');p.db.initialize()
            run=p.db.create_run(p.db.register_dataset('test',root/'data'),'analysis',{})
            work=root/'runtime/maintenance'/f'run_{run:06d}'/'snapshot';work.mkdir(parents=True)
            p.db.save_maintenance_action(run,{'action_id':'g','owner_role':'qc_code_optimizer','action_class':'isolated_code_experiment',
                'max_attempts':3,'runtime_failure_feedback':{'actual_runtime_error':True},'review_approved':True},status='REJECTED')
            p.db.update_maintenance_action(run,'g',status='REJECTED',artifact_path=work)
            a.write_json(work/'autopilot_validation.json',{'passed':False,'checks':[{'returncode':1,'stderr':'synthetic failing interface fixture'}]})
            record={'run_id':run,'steps':4,'status':'COMPLETE'}
            with patch.object(a,'ROOT',root),patch.object(a,'action_attempts',return_value=2),patch('agents.governance.original_action_approved',return_value=True):
                self.assertTrue(p.reopen_bounded_validation_failure(record))
            self.assertEqual('APPROVED',p.db.get_maintenance_action(run,'g')['status'])
            self.assertIn('synthetic failing interface fixture',p.db.agent_inbox(run,'qc_code_optimizer')[0]['payload']['failure_reasons'][0])
            p.analyze_new_snapshot.assert_not_called()

    def test_first_validation_failure_reflows_only_with_original_review(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=self.pilot()
            work=root/'runtime/maintenance/run_000999/attempt';work.mkdir(parents=True)
            a.write_json(work/'autopilot_validation.json',{'passed':False,'checks':[{'returncode':1,'stderr':'fixture conflict'}]})
            p.db.run_details.return_value={'maintenance_actions':[{'action_id':'x','owner_role':'qc_code_optimizer',
                'status':'REJECTED','artifact_path':str(work),'payload_json':json.dumps({'max_attempts':3})}]}
            p.claim_bounded_owner_repair=Mock(return_value=True)
            with patch.object(a,'ROOT',root),patch('agents.governance.original_action_approved',return_value=False):
                self.assertFalse(p.reopen_bounded_validation_failure({'run_id':999}))
            p.claim_bounded_owner_repair.assert_not_called()
            with patch.object(a,'ROOT',root),patch('agents.governance.original_action_approved',return_value=True):
                self.assertTrue(p.reopen_bounded_validation_failure({'run_id':999}))
            self.assertEqual(p.claim_bounded_owner_repair.call_args.args[2]['failure_reasons'],['fixture conflict'])

    def test_validation_repair_continues_without_rebuying_analysis(self):
        with tempfile.TemporaryDirectory() as d, patch.object(a,'global_promotion_preflight',return_value={}):
            p=self.pilot();path=Path(d)/'trial.json'
            def reflow(record):
                if record['steps']==1:
                    record['status']='ACTIONS'
                    return True
                return False
            p.reopen_bounded_validation_failure=Mock(side_effect=reflow)
            p.bounded_run(path)
            self.assertEqual(p.process_actions.call_count,2)
            p.analyze_new_snapshot.assert_called_once_with(None)
            self.assertEqual(json.loads(path.read_text())['status'],'COMPLETE')

    def test_failed_executor_test_cannot_be_overridden_by_ready_boolean(self):
        self.assertFalse(a.owner_result_ready_for_validation({'ready_for_validation':True,'tests_passed':False},{}))

    def test_explicit_budget_extension_is_once_only_and_does_not_call_models(self):
        from app.database import Database
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=self.pilot();p.db=Database(root/'workflow.sqlite');p.db.initialize()
            run=p.db.create_run(p.db.register_dataset('test',root/'data'),'analysis',{})
            work=root/'runtime/maintenance'/f'run_{run:06d}'/'failed';work.mkdir(parents=True)
            p.db.save_maintenance_action(run,{'action_id':'g','owner_role':'qc_code_optimizer','action_class':'isolated_code_experiment',
                'max_attempts':1,'runtime_failure_feedback':{'actual_runtime_error':True},'review_approved':True},status='HUMAN_REQUIRED')
            p.db.update_maintenance_action(run,'g',status='HUMAN_REQUIRED',artifact_path=work)
            a.write_json(work/'maintenance_manifest.json',{'result':{'tests_passed':False}})
            (work/'test_output.log').write_text('synthetic executor test failure')
            record=root/'trial.json';a.write_json(record,{'run_id':run,'steps':2,'status':'COMPLETE'})
            with patch.object(a,'ROOT',root),patch.object(a,'action_attempts',return_value=1):
                p.extend_bounded_owner_budget(record)
                with self.assertRaises(ValueError):p.extend_bounded_owner_budget(record)
            self.assertEqual(2,p.db.get_maintenance_action(run,'g')['payload']['max_attempts'])
            self.assertEqual('APPROVED',p.db.get_maintenance_action(run,'g')['status'])
            p.analyze_new_snapshot.assert_not_called();p.process_actions.assert_not_called()
