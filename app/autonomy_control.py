"""Local service supervision; no model calls, source edits or label operations."""
from __future__ import annotations
import json
import os
import subprocess
import time
from pathlib import Path
from app.control_checkpoint import save_journal as _save_journal

HEALTH='runtime/control/check_loop_health.json'
SERVICE='cmr-lge-check-loop.service'


def save_journal(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    _save_journal(path,value)


def read(path: Path, default=None):
    try:return json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError:return default if default is not None else {}


def operator_paused(root: Path) -> bool:
    from app.workflow_budget import reserve, BudgetExhausted
    try:reserve(None,root)
    except BudgetExhausted:return True
    return read(root/'runtime/control/operator_state.json').get('paused') is True


def heartbeat(root: Path, status: str, *, execute: bool, error: str | None=None):
    save_journal(root/HEALTH,{'status':status,'pid':os.getpid(),'updated_at':time.time(),
        'execute':execute,'error':error[-1000:] if error else None})


def policy(root: Path) -> dict:
    value=read(root/'configs/autonomy.json',{'enabled':False})
    if value.get('schema_version') not in {None,'autonomy-v1'}:
        raise ValueError('invalid autonomy schema')
    return value


def consumer_state(root: Path, *, now=None) -> dict:
    stamp=time.time() if now is None else now
    health=read(root/HEALTH)
    cfg=policy(root)
    age=max(0,stamp-float(health.get('updated_at',0))) if health else None
    limit=cfg.get('processing_timeout',1200) if health.get('status')=='RUNNING' else cfg.get('heartbeat_timeout',180)
    status=health.get('status','UNOBSERVED')
    if operator_paused(root):status='PAUSED'
    elif age is not None and age>limit and status in {'RUNNING','READY'}:status='STALE'
    return {**health,'status':status,'age_seconds':round(age,1) if age is not None else None}


def ensure_consumer(root: Path, *, dataset: Path | None = None, runner=subprocess.run, now=None) -> dict:
    """Restart only the authorized consumer; bounded attempts survive controller restarts."""
    stamp=time.time() if now is None else now
    cfg=policy(root)
    if not cfg.get('enabled') or dataset is None or str(dataset.resolve()) != cfg.get('dataset'):
        return {'status':'DISABLED','ready':True}
    if operator_paused(root):return {'status':'PAUSED','ready':False}
    health=consumer_state(root,now=stamp)
    if health['status'] in {'READY','RUNNING'} and health.get('execute') is True:
        recovery_path=root/'runtime/autonomy/service_recovery.json'
        recovery=read(recovery_path)
        if recovery.get('attempts') and health.get('updated_at',0)>recovery.get('last_attempt',0):
            save_journal(recovery_path,{'attempts':0,'last_healthy':stamp})
        return {'status':health['status'],'ready':True}
    state_path=root/'runtime/autonomy/service_recovery.json'
    state=read(state_path,{'attempts':0})
    from app.lge_protocol import load as load_protocol
    from app.rsi_kernel import canonical_hash
    try:protocol_hash=canonical_hash(load_protocol(root))
    except (OSError,ValueError,TypeError):protocol_hash='invalid'
    if state.get('protocol_hash') and state['protocol_hash'] != protocol_hash and protocol_hash != 'invalid':
        state={'attempts':0,'protocol_hash':protocol_hash}
    else:state['protocol_hash']=protocol_hash
    if state.get('last_attempt') and stamp-state['last_attempt']<cfg.get('restart_cooldown',120):
        return {'status':'RECOVERY_WAIT','ready':False,'attempts':state['attempts']}
    if state['attempts']>=cfg.get('max_service_restarts',3):
        return {'status':'REPAIR_REQUIRED','ready':False,'reason':'consumer_restart_budget_exhausted'}
    try:
        observation=runner(['systemctl','--user','is-active',SERVICE],capture_output=True,text=True,timeout=10,check=False)
        active=observation.returncode==0
        first=state.setdefault('unhealthy_since',stamp)
        if active and health['status']=='UNOBSERVED' and stamp-first<cfg.get('startup_grace',30):
            save_journal(state_path,state)
            return {'status':'STARTING','ready':False}
        state.update(attempts=state['attempts']+1,last_attempt=stamp)
        save_journal(state_path,state)  # Claim before invoking systemd, including failed/ambiguous calls.
        result=runner(['systemctl','--user','restart' if active else 'start',SERVICE],capture_output=True,text=True,timeout=20,check=False)
        return {'status':'RECOVERING' if result.returncode==0 else 'RECOVERY_FAILED','ready':False,'attempts':state['attempts']}
    except (OSError,subprocess.SubprocessError) as exc:
        return {'status':'RECOVERY_FAILED','ready':False,'reason':str(exc)[-300:]}


def autonomy_status(root: Path) -> dict:
    from app.autonomy_feedback import feedback_summary
    try:
        return {'enabled':policy(root).get('enabled',False),'operator_paused':operator_paused(root),
                'consumer':consumer_state(root),'consumer_task':read(root/'runtime/control/active_consumer_task.json'),'service_recovery':read(root/'runtime/autonomy/service_recovery.json'),
                'feedback':feedback_summary(root),'run_budget':__import__('app.workflow_budget',fromlist=['read']).read(root)}
    except (OSError,ValueError,TypeError,KeyError) as exc:
        return {'status':'UNAVAILABLE','reason':str(exc)[-400:],'consumer':{'status':'CONTROL_ERROR'},
                'feedback':{'episodes':0,'recent':[],'role_credit':[],'generalization_status':'UNKNOWN'}}
