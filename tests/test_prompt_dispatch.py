import asyncio,json,tempfile,unittest
from pathlib import Path
from unittest.mock import Mock,patch
from pydantic import BaseModel
from app.prompt_dispatch import grant,reserve,launch
from app.workflow_budget import arm,suspend,read as budget_read
from providers.openai_compatible import OpenAICompatibleProvider
from scripts.analyze_manual_edit_queue import claim_one

JOB='20261028T010101_prompt_012345abcd'
class Response(BaseModel):
    value:int
class PromptDispatchTests(unittest.TestCase):
    def job(self,root,job=JOB):
        folder=root/'prompt_jobs'/job;folder.mkdir(parents=True)
        (folder/'local_request.json').write_text(json.dumps({'job_id':job,'status':'QUEUED'}))
        (folder/'agent_request.json').write_text('{}')
        return folder
    def test_launch_is_single_job_and_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.job(root);spawn=Mock(return_value=Mock(pid=123))
            launch(root,root,JOB,popen=spawn);launch(root,root,JOB,popen=spawn)
            self.assertEqual(spawn.call_count,1)
            command=spawn.call_args.args[0]
            self.assertEqual(command[-2:],['--job-id',JOB])
            self.assertNotIn('--watch',command)
    def test_job_claim_cannot_drain_other_jobs_or_edit_learning(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.job(root);other='20261028T010100_prompt_abcdef0123';other_folder=self.job(root,other)
            edits=root/'edits/sample';edits.mkdir(parents=True);(edits/'agent_request.json').write_text('{}')
            claimed=claim_one(root,JOB)
            self.assertEqual(claimed[2].parent.name,JOB)
            self.assertTrue((other_folder/'agent_request.json').exists())
            self.assertTrue((edits/'agent_request.json').exists())
    def test_explicit_prompt_has_separate_bounded_requests_while_loop_paused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);work=root/'work';folder=self.job(work);grant(work,JOB)
            arm(root);suspend(root,reason='operator_pause')
            async def post(*args):return {'choices':[{'message':{'content':'{"value":1}'}}]}
            with patch.dict('os.environ',{'TEST_PROMPT_KEY':'test','CMR_LGE_AGENT_CALL_LOG':str(root/'calls.jsonl')}),patch.object(OpenAICompatibleProvider,'_post',staticmethod(post)):
                provider=OpenAICompatibleProvider('test','TEST_PROMPT_KEY')
                async def invoke():return await provider.generate_structured(system_prompt='test',input_data={'_budget_root':str(root),'_request_budget_callback':lambda:reserve(work,JOB)},output_schema=Response,model='test')
                self.assertEqual(asyncio.run(invoke()).value,1)
                asyncio.run(invoke())
                with self.assertRaisesRegex(RuntimeError,'上限'):asyncio.run(invoke())
            self.assertEqual(budget_read(root)['used']['requests'],0)
            self.assertEqual(json.loads((folder/'interactive_budget.json').read_text())['used'],2)
    def test_bad_job_path_never_spawns(self):
        with tempfile.TemporaryDirectory() as tmp:
            spawn=Mock()
            with self.assertRaises(ValueError):launch(Path(tmp),Path(tmp),'../../escape',popen=spawn)
            spawn.assert_not_called()

    def test_watch_restart_does_not_steal_live_interactive_claim(self):
        from scripts.analyze_manual_edit_queue import recover_stale_claims
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=self.job(root)
            (folder/'agent_request.json').rename(folder/'agent_request.running')
            with patch('app.prompt_dispatch.worker_alive',return_value=True):
                self.assertEqual(recover_stale_claims(root,stale_seconds=0),0)
            self.assertTrue((folder/'agent_request.running').exists())
            with patch('app.prompt_dispatch.worker_alive',return_value=False):
                self.assertEqual(recover_stale_claims(root,stale_seconds=0),1)

    def test_watch_skips_interactive_before_budget_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=self.job(root)
            local=json.loads((folder/'local_request.json').read_text());local['execution_scope']='interactive_prompt';(folder/'local_request.json').write_text(json.dumps(local))
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=2) as pool:
                watch=pool.submit(claim_one,root);explicit=pool.submit(claim_one,root,JOB)
                self.assertIsNone(watch.result())
                self.assertEqual(explicit.result()[2].parent,folder)
            self.assertTrue((folder/'claim.json').exists())
            self.assertFalse((folder/'agent_request.json').exists())
    def test_status_does_not_treat_legacy_owner_as_failed_launcher(self):
        from app.manual_annotation import prompt_repair_status
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=self.job(root)
            (folder/'agent_request.json').rename(folder/'agent_request.running')
            (folder/'local_request.json').write_text(json.dumps({'status':'PROCESSING'}))
            (folder/'dispatch.json').write_text(json.dumps({'pid':99999999,'launched_at':__import__('time').time()}))
            with patch('app.prompt_dispatch.worker_alive',return_value=False):
                self.assertEqual(prompt_repair_status(root,JOB)['status'],'PROCESSING')
    def test_completed_result_wins_exit_race_during_poll(self):
        from app.manual_annotation import prompt_repair_status
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=self.job(root)
            (folder/'local_request.json').write_text(json.dumps({'status':'PROCESSING'}))
            (folder/'dispatch.json').write_text(json.dumps({'pid':99999999,'launched_at':__import__('time').time()}))
            def finish(_):
                (folder/'local_request.json').write_text(json.dumps({'status':'NO_CHANGE'}));return False
            with patch('app.prompt_dispatch.worker_alive',side_effect=finish):
                result=prompt_repair_status(root,JOB)
            self.assertEqual(result['status'],'NO_CHANGE')
            self.assertIsNone(result['error'])
    def test_terminal_result_cannot_regress_to_stalled(self):
        from app.manual_annotation import prompt_repair_status
        import os,time
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=self.job(root)
            (folder/'agent_request.json').rename(folder/'agent_request.running')
            os.utime(folder/'agent_request.running',(time.time()-300,time.time()-300))
            (folder/'local_request.json').write_text(json.dumps({'status':'MANUAL_REQUIRED'}))
            self.assertEqual(prompt_repair_status(root,JOB)['status'],'MANUAL_REQUIRED')
    def test_owner_identity_supports_watch_process_without_job_cli_argument(self):
        from app.prompt_dispatch import worker_alive
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=self.job(root)
            (folder/'agent_request.json').rename(folder/'agent_request.running')
            (folder/'claim.json').write_text(json.dumps({'pid':42,'job_id':JOB,'process_start':'123'}))
            with patch('pathlib.Path.read_bytes',return_value=b'python\x00scripts/analyze_manual_edit_queue.py\x00--watch'),patch('app.prompt_dispatch._process_start',return_value='123'):
                self.assertTrue(worker_alive(folder))
            with patch('pathlib.Path.read_bytes',return_value=b'python\x00scripts/analyze_manual_edit_queue.py\x00--watch'),patch('app.prompt_dispatch._process_start',return_value='different'):
                self.assertFalse(worker_alive(folder))
    def test_stale_source_is_rejected_before_model_call(self):
        from scripts.analyze_manual_edit_queue import process_one
        from app.manual_annotation import TARGET_DIRS
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=self.job(root);dataset=root/'dataset'
            source=dataset/TARGET_DIRS[('lge','myo')]/'synthetic'/'slice.nii.gz';source.parent.mkdir(parents=True);source.write_bytes(b'new source')
            (folder/'local_request.json').write_text(json.dumps({'status':'QUEUED','dataset_root':str(dataset),'source_label':str(source),'source_hash':'old hash'}))
            with patch('scripts.analyze_manual_edit_queue.plan_prompt_repair') as model,patch('scripts.analyze_manual_edit_queue.time.sleep'):
                process_one(root,root/'providers',root/'agents',job_id=JOB)
            model.assert_not_called()
            self.assertEqual(json.loads((folder/'local_request.json').read_text())['status'],'ERROR')
            self.assertEqual(source.read_bytes(),b'new source')

    def test_recovery_preserves_live_terminal_claim_and_archives_dead_terminal_request(self):
        from scripts.analyze_manual_edit_queue import recover_stale_claims
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=self.job(root)
            (folder/'agent_request.json').rename(folder/'agent_request.running')
            (folder/'local_request.json').write_text(json.dumps({'status':'NO_CHANGE'}))
            with patch('app.prompt_dispatch.worker_alive',return_value=True):
                self.assertEqual(recover_stale_claims(root,stale_seconds=0),0)
                self.assertTrue((folder/'agent_request.running').exists())
            with patch('app.prompt_dispatch.worker_alive',return_value=False):
                self.assertEqual(recover_stale_claims(root,stale_seconds=0),0)
            self.assertTrue((folder/'agent_request.json').exists())
            self.assertIsNone(claim_one(root,JOB))
