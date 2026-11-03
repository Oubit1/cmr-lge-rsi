"""One explicit editor job, independent of paused autonomous processing."""
from __future__ import annotations
import json,os,re,subprocess,sys,time
from pathlib import Path
from contextlib import contextmanager
from app.control_checkpoint import save_journal

@contextmanager
def job_lock(folder):
    import fcntl
    with (folder/'dispatch.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX);yield

def job_folder(work_root,job_id):
    if not re.fullmatch(r'[0-9]{8}T[0-9]{6}_prompt_[a-f0-9]{10}',job_id):raise ValueError('invalid prompt job id')
    folder=Path(work_root)/'prompt_jobs'/job_id
    if not (folder/'local_request.json').is_file():raise ValueError('unknown prompt job')
    return folder

def grant(work_root,job_id):
    folder=job_folder(work_root,job_id)
    with job_lock(folder):
        path=folder/'interactive_budget.json'
        if not path.exists():save_journal(path,{'scope':'operator_prompt','limit':2,'used':0,'deadline':time.time()+600})

def reserve(work_root,job_id):
    folder=job_folder(work_root,job_id)
    with job_lock(folder):
        path=folder/'interactive_budget.json';budget=json.loads(path.read_text())
        if budget.get('scope')!='operator_prompt' or time.time()>=budget['deadline'] or budget['used']>=budget['limit']:
            raise RuntimeError('人工提示词请求已达到本次调用或时间上限，请重新提交')
        budget['used']+=1;save_journal(path,budget)

def launch(root,work_root,job_id,*,popen=subprocess.Popen):
    folder=job_folder(work_root,job_id);grant(work_root,job_id)
    with job_lock(folder):
        dispatch=folder/'dispatch.json'
        if dispatch.exists():return json.loads(dispatch.read_text())
        command=[sys.executable,'-u',str(Path(root)/'scripts/analyze_manual_edit_queue.py'),
                 '--work-root',str(work_root),'--agents',str(Path(root)/'configs/agents.aliyun-roles.yaml'),
                 '--providers',str(Path(root)/'configs/providers.yaml'),'--job-id',job_id]
        try:
            with (folder/'worker.log').open('a') as log:
                worker=popen(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            state={'pid':worker.pid,'launched_at':time.time(),'scope':'single_prompt_job'}
            save_journal(dispatch,state)
            return state
        except Exception as exc:
            local=json.loads((folder/'local_request.json').read_text());local.update(status='ERROR',terminal_error='无法启动提示词工作进程：'+str(exc)[:300])
            save_journal(folder/'local_request.json',local);raise

def _process_start(pid):
    return (Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[19]

def register_claim(folder):
    pid=os.getpid()
    save_journal(Path(folder)/'claim.json',{'pid':pid,'process_start':_process_start(pid),
                 'job_id':Path(folder).name,'claimed_at':time.time()})

def worker_alive(folder):
    folder=Path(folder)
    try:
        claim=folder/'claim.json'
        if (folder/'agent_request.running').exists() and claim.exists():
            info=json.loads(claim.read_text())
            command=(Path('/proc')/str(int(info['pid']))/'cmdline').read_bytes()
            if info.get('job_id')==folder.name and _process_start(info['pid'])==info.get('process_start') and b'analyze_manual_edit_queue.py' in command:
                return True
        info=json.loads((folder/'dispatch.json').read_text())
        command=(Path('/proc')/str(int(info['pid']))/'cmdline').read_bytes()
        return folder.name.encode() in command and b'analyze_manual_edit_queue.py' in command
    except (OSError,ValueError,KeyError,TypeError,IndexError):return False
