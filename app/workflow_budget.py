"""Durable operator-granted run limits, shared by all provider attempts."""
from __future__ import annotations
import json
import time
from contextlib import contextmanager
from pathlib import Path
from app.control_checkpoint import save_journal

ROOT = Path(__file__).resolve().parents[1]
REL = 'runtime/control/run_budget.json'

class BudgetExhausted(RuntimeError):
    pass

@contextmanager
def locked(root):
    import fcntl
    path=Path(root)/'runtime/control/run_budget.lock'
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        yield

def read(root=ROOT):
    path=Path(root)/REL
    if not path.exists():return {}
    return json.loads(path.read_text(encoding='utf-8'))

def arm(root=ROOT, *, analyses=3, requests=30, seconds=3600, now=None):
    stamp=time.time() if now is None else now
    if min(analyses,requests,seconds)<=0:raise ValueError('positive limits required')
    with locked(root):
        previous=read(root)
        if previous.get('active'):raise ValueError('existing budget must finish first')
        state={'schema_version':'run-budget-v1','active':True,'status':'RUNNING',
               'started_at':stamp,'deadline':stamp+seconds,'limits':{'analyses':analyses,'requests':requests},
               'used':{'analyses':0,'requests':0}}
        save_journal(Path(root)/REL,state)
    return state

def _pause(root, state, reason):
    state.update(status='EXHAUSTED',reason=reason,stopped_at=time.time())
    save_journal(Path(root)/REL,state)
    save_journal(Path(root)/'runtime/control/operator_state.json',
                 {'paused':True,'source':'run_budget','reason':reason,'updated_at':time.time()})

def reserve(kind, root=ROOT, *, now=None):
    stamp=time.time() if now is None else now
    with locked(root):
        state=read(root)
        operator=Path(root)/'runtime/control/operator_state.json'
        if kind == 'requests' and state and operator.exists() and json.loads(operator.read_text()).get('paused'):
            raise BudgetExhausted('operator_pause')
        if not state.get('active'):return
        reason=None
        if state['status']!='RUNNING':reason=state.get('reason','budget_finished')
        elif stamp>=state['deadline']:reason='deadline'
        elif kind and state['used'][kind]>=state['limits'][kind]:reason=kind+'_limit'
        if reason:
            _pause(root,state,reason)
            raise BudgetExhausted(reason)
        if kind:
            state['used'][kind]+=1
            save_journal(Path(root)/REL,state)

def finish(root=ROOT, *, reason):
    with locked(root):
        state=read(root)
        if not state:return state
        state.update(active=False,status='FINISHED',reason=reason,finished_at=time.time())
        save_journal(Path(root)/REL,state)
        return state

def suspend(root=ROOT, *, reason):
    with locked(root):
        state=read(root)
        if state.get('active'):_pause(root,state,reason)

def resume_existing(root=ROOT, *, now=None):
    """Resume approved action repair without changing original counters or deadline."""
    stamp=time.time() if now is None else now
    with locked(root):
        state=read(root)
        if (not (state.get('reason') == 'analyses_limit' or (state.get('reason') == 'deadline' and state.get('observation_extension'))) or stamp>=state.get('deadline',0)
                or state['used']['requests']>=state['limits']['requests']):
            raise BudgetExhausted('no_remaining_original_session')
        state.update(active=True,status='RUNNING',resumed_at=stamp)
        save_journal(Path(root)/REL,state)
        return state

def begin_operator_session(root=ROOT, *, now=None):
    """Only an explicit UI Start may create a new allowance; daemon restarts reuse it."""
    stamp=time.time() if now is None else now
    config_path=Path(root)/'configs/autonomy.json'
    if not config_path.exists():return None
    limits=json.loads(config_path.read_text(encoding='utf-8')).get('run_limits')
    if not limits:return None
    if any(type(limits.get(k)) is not int or limits[k]<=0 for k in ('analyses','requests','seconds')):
        raise ValueError('invalid autonomous run limits')
    with locked(root):
        state=read(root)
        if state.get('active') and state.get('status')=='RUNNING' and stamp<state.get('deadline',0):return state
        if state:
            history=Path(root)/'runtime/control/budget_history'
            history.mkdir(parents=True,exist_ok=True)
            import uuid
            save_journal(history/(str(int(stamp))+'_'+uuid.uuid4().hex[:8]+'.json'),state)
        state={'schema_version':'run-budget-v1','active':True,'status':'RUNNING','source':'operator_start',
               'started_at':stamp,'deadline':stamp+limits['seconds'],
               'limits':{k:limits[k] for k in ('analyses','requests')},'used':{'analyses':0,'requests':0}}
        save_journal(Path(root)/REL,state)
        return state
