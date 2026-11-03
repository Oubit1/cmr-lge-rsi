import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from app.workflow_budget import arm,reserve,read,finish,BudgetExhausted

class BudgetTests(unittest.TestCase):
    def test_limits_survive_restart_and_denial_pauses(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);arm(root,analyses=2,requests=2,seconds=30,now=100)
            reserve('requests',root,now=101);reserve('requests',root,now=102)
            with self.assertRaises(BudgetExhausted):reserve('requests',root,now=103)
            self.assertEqual(read(root)['used']['requests'],2)
            self.assertTrue(json.loads((root/'runtime/control/operator_state.json').read_text())['paused'])
            with self.assertRaises(BudgetExhausted):reserve('analyses',root,now=104)
            finish(root,reason='requests_limit')
            with self.assertRaises(BudgetExhausted):reserve('requests',root,now=105)
            (root/'runtime/control/operator_state.json').write_text('{"paused":false}')
            reserve('requests',root,now=106)
    def test_deadline_before_first_request_and_no_silent_rearm(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);arm(root,seconds=3,now=100)
            with self.assertRaises(ValueError):arm(root)
            with self.assertRaises(BudgetExhausted):reserve(None,root,now=103)
            self.assertEqual(read(root)['used']['requests'],0)
    def test_analysis_cap_leaves_last_round_actions_until_next_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);arm(root,analyses=1,seconds=30,now=100)
            reserve('analyses',root,now=101);reserve('requests',root,now=102)
            self.assertEqual(read(root)['status'],'RUNNING')
            with self.assertRaises(BudgetExhausted):reserve('analyses',root,now=103)
    def test_schema_retry_cannot_pass_http_limit(self):
        from contracts.models import FailureAnalysis
        from providers.openai_compatible import OpenAICompatibleProvider
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);arm(root,requests=1)
            calls=[]
            async def post(*args):
                calls.append(1)
                return {'choices':[{'message':{'content':'{}'}}]}
            with patch.dict('os.environ',{'BUDGET_TEST_KEY':'test','CMR_LGE_AGENT_CALL_LOG':str(root/'calls.jsonl')}),patch.object(OpenAICompatibleProvider,'_post',staticmethod(post)):
                provider=OpenAICompatibleProvider('test','BUDGET_TEST_KEY')
                with self.assertRaises(BudgetExhausted):
                    asyncio.run(provider.generate_structured(system_prompt='test',input_data={'_budget_root':str(root)},output_schema=FailureAnalysis,model='test'))
            self.assertEqual(len(calls),1)
    def test_concurrent_reservations_never_exceed_cap(self):
        from concurrent.futures import ThreadPoolExecutor
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);arm(root,requests=3)
            def attempt(_):
                try:reserve('requests',root);return 1
                except BudgetExhausted:return 0
            with ThreadPoolExecutor(max_workers=8) as pool:success=sum(pool.map(attempt,range(12)))
            self.assertEqual(success,3)
            self.assertEqual(read(root)['used']['requests'],3)

    def test_observer_uses_actual_workflow_schema(self):
        from app.database import Database
        import scripts.verify_autonomous_run as observer
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);Database(root/'runtime/workflow.sqlite').initialize()
            with patch.object(observer,'ROOT',root):
                result=observer.snapshot()
            self.assertEqual(result['invocations'],0)
            self.assertIsNone(result['latest_run'])

    def test_same_session_repair_cannot_reset_counters_or_deadline(self):
        from app.workflow_budget import resume_existing
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);arm(root,analyses=1,requests=2,seconds=30,now=100)
            reserve('analyses',root,now=101)
            with self.assertRaises(BudgetExhausted):reserve('analyses',root,now=102)
            finish(root,reason='analyses_limit')
            state=resume_existing(root,now=103)
            self.assertEqual(state['deadline'],130)
            self.assertEqual(state['used']['analyses'],1)
            with self.assertRaises(BudgetExhausted):reserve('analyses',root,now=104)
            with self.assertRaises(BudgetExhausted):resume_existing(root,now=130)

    def test_explicit_operator_start_reuses_active_allowance_and_archives_finished(self):
        from app.workflow_budget import begin_operator_session,suspend
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'configs').mkdir();(root/'configs/autonomy.json').write_text(json.dumps({'run_limits':{'analyses':3,'requests':5,'seconds':60}}))
            first=begin_operator_session(root,now=100)
            reserve('requests',root,now=101)
            again=begin_operator_session(root,now=102)
            self.assertEqual(again['used']['requests'],1);self.assertEqual(again['deadline'],160)
            suspend(root,reason='test_finished')
            second=begin_operator_session(root,now=103)
            self.assertEqual(second['used']['requests'],0)
            self.assertEqual(len(list((root/'runtime/control/budget_history').glob('*.json'))),1)
