#!/usr/bin/env python3
"""Bounded real-service verification; never calls a domain writer or a role directly."""
from __future__ import annotations
import argparse
import json
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.workflow_budget import arm, read, reserve, finish, suspend, resume_existing, BudgetExhausted
from app.autonomy_control import operator_paused
from app.control_checkpoint import save_journal

def snapshot():
    from app.autonomy_feedback import feedback_summary
    with sqlite3.connect(str(ROOT/'runtime/workflow.sqlite')) as db:
        calls=db.execute('select count(*) from agent_invocations').fetchone()[0]
        run=db.execute('select max(run_id) from workflow_runs').fetchone()[0]
    def load(rel):
        p=ROOT/rel
        return json.loads(p.read_text()) if p.exists() else {}
    state=load('runtime/autopilot_state.json')
    return {'at':time.time(),'budget':read(ROOT),'invocations':calls,'latest_run':run,
            'pilot_status':state.get('status'),'pilot_error':state.get('error'),
            'consumer':load('runtime/control/check_loop_health.json'),
            'feedback':feedback_summary(ROOT)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--analyses',type=int,default=3)
    ap.add_argument('--requests',type=int,default=30)
    ap.add_argument('--seconds',type=int,default=3600)
    ap.add_argument('--resume-existing',action='store_true')
    args=ap.parse_args()
    if not operator_paused(ROOT):raise RuntimeError('start verification from operator pause')
    output=ROOT/'runtime/autonomy/multiround_20261027'
    output.mkdir(parents=True,exist_ok=True)
    if args.resume_existing:
        baseline=json.loads((output/'baseline.json').read_text())
        if (output/'final.json').exists():
            save_journal(output/'first_stop.json',json.loads((output/'final.json').read_text()))
        resume_existing(ROOT)
    else:
        baseline=snapshot()
        save_journal(output/'baseline.json',baseline)
        arm(ROOT,analyses=args.analyses,requests=args.requests,seconds=args.seconds)
    reason='verification_error'
    from scripts.board_server import control_workflow
    try:
        result=control_workflow('start')
        save_journal(output/'start.json',result)
        while True:
            try:reserve(None,ROOT)
            except BudgetExhausted as exc:
                reason=str(exc);break
            current=snapshot()
            save_journal(output/'latest.json',current)
            with (output/'observations.jsonl').open('a') as h:h.write(json.dumps(current)+'\n')
            if operator_paused(ROOT):reason='operator_pause';break
            if current['pilot_status'] in {'ESCALATION_REQUIRED','RECOVERY_REQUIRED'}:
                reason='controller_'+current['pilot_status'];break
            time.sleep(10)
    finally:
        # Operator latch is persisted before service stop, including deadline.
        suspend(ROOT,reason=reason)
        control_workflow('pause')
        stopped=False
        for _ in range(100):
            result=subprocess.run(['systemctl','--user','is-active','cmr-lge-autopilot.service','cmr-lge-check-loop.service','cmr-lge-edit-agent.service'],capture_output=True,text=True,timeout=10)
            if not any(s in {'active','activating','deactivating','reloading'} for s in result.stdout.splitlines()):
                stopped=True;break
            time.sleep(10)
        if stopped:finish(ROOT,reason=reason)
        # Otherwise leave the exhausted lease enforced across a late caller.
        drop=Path('/run/user')/str(__import__('os').getuid())/'systemd/user/cmr-lge-check-loop.service.d/90-verification.conf'
        if drop.exists():
            drop.unlink()
            subprocess.run(['systemctl','--user','daemon-reload'],check=True,timeout=30)
        save_journal(output/'final.json',{'reason':reason,'baseline':baseline,'final':snapshot()})

if __name__=='__main__':main()
