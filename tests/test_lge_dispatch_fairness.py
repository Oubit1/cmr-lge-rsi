import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from datetime import datetime,timezone
from scripts.run_lge_feedback_consumer import run_execution
from app.lge_execution import SCHEMA
from app.control_checkpoint import save_journal
from scripts import autopilot

class DispatchTests(unittest.TestCase):
    def run_queue(self,myo,mi,previous=None):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);output=root/'latest.json';executed=[]
            if previous:save_journal(output,{'schema_version':SCHEMA,'last_execution':{'workflow_stage':previous}})
            def batch(root,dataset,stage,data,**kw):
                if not kw.get('status_only'):executed.append(stage)
                n=myo if stage=='lge_myo' else mi
                return {stage+'_auto_complete':False,stage+'_consumer_ready':True,
                        stage+'_auto_pending':n,stage+'_dispatchable':n,stage+'_pending':n,
                        stage+'_work':256,stage+'_resolved':33,'receipt_revision':stage,
                        'workflow_stage':stage,'production_writes':0,'outcomes':[]}
            with patch('scripts.run_lge_feedback_consumer.evaluation_input',side_effect=lambda root,workflow_stage:{'status':'AVAILABLE','workflow_stage':workflow_stage}),patch('app.lge_execution.run_batch',side_effect=batch):
                run_execution(root,root,output,root/'audit.jsonl',execute=False)
            return executed,json.loads(output.read_text())
    def test_myo_manual_residuals_do_not_block_mi_work(self):
        executed,result=self.run_queue(0,10)
        self.assertEqual(executed,['lge_mi'])
        self.assertFalse(result['lge_myo_auto_complete'])
        self.assertEqual(result['lge_myo_work'],256)
    def test_ready_queues_alternate_without_starvation(self):
        self.assertEqual(self.run_queue(10,10)[0],['lge_myo'])
        self.assertEqual(self.run_queue(10,10,'lge_myo')[0],['lge_mi'])
        self.assertEqual(self.run_queue(10,10,'lge_mi')[0],['lge_myo'])
    def test_no_dispatchable_work_never_replays_parked_items(self):
        self.assertEqual(self.run_queue(0,0)[0],[])
    def test_pilot_waits_for_mi_consumer_even_when_myo_needs_improvement(self):
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'configs').mkdir();(root/'configs/lge_execution.json').write_text('{}')
            c=autopilot.Autopilot.__new__(autopilot.Autopilot);c.dataset=root/'dataset';c.db=Mock();c.state={};c.event=Mock()
            c.service_protocol_repair=Mock(return_value=False);c.resume_runtime_owner=Mock(return_value=False);c.resume_validation_owner=Mock(return_value=False);c.analyze_new_snapshot=Mock()
            dependency={'stage':'lge_myo','lge_generator_required':True,'lge_dispatchable_pending':10,
                        'lge_execution_published_at':datetime.now(timezone.utc).isoformat()}
            with patch.object(autopilot,'ROOT',root),patch.object(autopilot,'global_promotion_preflight',return_value={}),patch('app.autonomy_control.operator_paused',return_value=False),patch('app.autonomy_control.ensure_consumer',return_value={'ready':True}),patch.object(autopilot,'issue_fingerprint',return_value=('new',{})),patch.object(autopilot,'dependent_workflow_status',return_value=dependency):c.tick()
            c.analyze_new_snapshot.assert_not_called()
            self.assertEqual(c.event.call_args.args[0],'WAITING_FOR_LGE_CONSUMER')
