"""Durable, model-neutral execution feedback. Never writes or adjudicates labels."""
from __future__ import annotations
import json
import sqlite3
import hashlib
import time
from collections import Counter
from pathlib import Path
from contextlib import closing
from app.control_checkpoint import save_journal as _save_journal


def save_journal(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    _save_journal(path,value)


def _read(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError:
        return default if default is not None else {}


def _connect(root: Path):
    folder = root / 'runtime/autonomy'
    folder.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(folder / 'feedback.sqlite'), timeout=30)
    connection.row_factory = sqlite3.Row
    connection.executescript('''
    CREATE TABLE IF NOT EXISTS episodes(id TEXT PRIMARY KEY, stage TEXT NOT NULL,
        created REAL NOT NULL, body TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS exposures(stage TEXT NOT NULL, case_id TEXT NOT NULL,
        episode_id TEXT NOT NULL, PRIMARY KEY(stage,case_id));
    CREATE TABLE IF NOT EXISTS credits(transaction_id TEXT PRIMARY KEY,
        episode_id TEXT NOT NULL, run_id INTEGER, action_id TEXT,
        role TEXT, score_delta REAL, label_write INTEGER NOT NULL, model TEXT);
    ''')
    if "model" not in {r[1] for r in connection.execute("PRAGMA table_info(credits)")}:
        connection.execute("ALTER TABLE credits ADD COLUMN model TEXT")
        connection.commit()
    return connection


def _canonical(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _verified_commit(root: Path, row: dict):
    transaction = str(row.get('transaction_id') or '')
    if len(transaction) != 32 or any(c not in '0123456789abcdef' for c in transaction):
        return False
    journal = _read(root / 'runtime/lge_execution/transactions' / (transaction + '.json'))
    receipt = journal.get('receipt') or {}
    return (journal.get('state') == 'COMMITTED' and receipt.get('status') == 'CHECKED'
            and receipt.get('case_id') == row.get('case_id')
            and receipt.get('transaction_id') == transaction
            and receipt.get('after_hashes') == row.get('after_hashes'))


def record_execution(root: Path, result: dict) -> dict:
    """Consume only controller outcomes; previews never become experience/credit."""
    if result.get('execution_mode') != 'execute' or result.get('workflow_stage') not in {'lge_myo','lge_mi'}:
        return {'recorded': False, 'reason': 'not_actual_execution'}
    rows = result.get('outcomes') or []
    if not rows:
        return {'recorded': False, 'reason': 'no_new_outcomes'}
    release = result.get('generator') or {}
    identity = {key:result.get(key) for key in ('workflow_stage','input_source_sha256','calibration_id')}
    identity.update(release=release, outcomes=rows)
    stage = result['workflow_stage']
    errors = [r for r in rows if r.get('execution_error') or str(r.get('reason','')).startswith('generator_failure:')]
    rejected = [r for r in rows if r.get('status') == 'NEEDS_REVIEW' and r not in errors]
    # Older controller responses omitted transaction IDs. Recover them only
    # from an exact committed controller receipt, never from provider claims.
    for index,row in enumerate(rows):
        if row.get('status')=='CHECKED' and not row.get('transaction_id'):
            matches=[]
            for journal_path in (root/'runtime/lge_execution/transactions').glob('*.json'):
                journal=_read(journal_path);receipt=journal.get('receipt') or {}
                if (journal.get('state')=='COMMITTED' and receipt.get('release')==release
                        and all(receipt.get(key)==row.get(key) for key in ('case_id','reason','score_before','score_after','label_write'))):
                    matches.append(receipt)
            if len(matches)==1:rows[index]=matches[0]
    accepted = [r for r in rows if r.get('status') == 'CHECKED' and _verified_commit(root,r)]
    unverified = sum(r.get('status') == 'CHECKED' for r in rows) - len(accepted)
    identity['outcomes']=rows
    episode_id=_canonical(identity)
    kind = ('UNVERIFIED_RESULTS' if unverified else 'EXECUTION_FAILURE' if errors else 'USEFUL_CHANGE' if accepted else 'DOMAIN_REJECTED')
    owner = None
    owner_model = None
    db_path = root / 'runtime/workflow.sqlite'
    if db_path.is_file():
        with closing(sqlite3.connect('file:' + str(db_path) + '?mode=ro', uri=True)) as db:
            row = db.execute('SELECT owner_role FROM maintenance_actions WHERE run_id=? AND action_id=?',
                             (release.get('run_id'),release.get('action_id'))).fetchone()
            owner = row[0] if row else None
            invocation=db.execute("SELECT invocation_id,requested_model FROM agent_invocations WHERE run_id=? AND role=? AND thread_id=? AND status='SUCCEEDED' ORDER BY started_at DESC LIMIT 1",
                (release.get('run_id'),owner,f"run-{release.get('run_id')}:{release.get('action_id')}")).fetchone()
            if invocation:
                event=db.execute("SELECT actual_model FROM agent_invocation_events WHERE invocation_id=? AND status='SUCCEEDED' ORDER BY event_id DESC LIMIT 1",(invocation[0],)).fetchone()
                owner_model=event[0] if event else invocation[1]
    body = {'episode_id':episode_id, 'stage':stage, 'kind':kind, 'release':release,
            'run_id':release.get('run_id'), 'action_id':release.get('action_id'), 'owner_role':owner, 'owner_model':owner_model,
            'attempted':len(rows), 'accepted':len(accepted), 'label_writes':sum(bool(r.get('label_write')) for r in accepted),
            'domain_rejected':len(rejected), 'execution_errors':len(errors), 'unverified_acceptances':unverified,
            'reason_counts':dict(Counter(str(r.get('reason','')) for r in rejected+errors)),
            'generalization_status':'NOT_ESTABLISHED', 'outcomes':rows}
    with closing(_connect(root)) as db:
        db.execute('BEGIN IMMEDIATE')
        existing = db.execute('SELECT body FROM episodes WHERE id=?',(episode_id,)).fetchone()
        if existing:
            db.rollback()
            return {'recorded':False,'reason':'duplicate_episode','episode_id':episode_id}
        fresh = 0
        for row in rows:
            cursor = db.execute('INSERT OR IGNORE INTO exposures VALUES(?,?,?)',(stage,row['case_id'],episode_id))
            fresh += cursor.rowcount
        body['new_case_exposures'] = fresh  # Operational exposure; not an independent clinical holdout.
        db.execute('INSERT INTO episodes VALUES(?,?,?,?)',(episode_id,stage,time.time(),json.dumps(body,ensure_ascii=False)))
        for row in accepted:
            before, after = row.get('score_before'), row.get('score_after')
            delta = after-before if isinstance(before,(int,float)) and isinstance(after,(int,float)) else None
            db.execute('INSERT OR IGNORE INTO credits(transaction_id,episode_id,run_id,action_id,role,score_delta,label_write,model) VALUES(?,?,?,?,?,?,?,?)',
                (row['transaction_id'],episode_id,release.get('run_id'),release.get('action_id'),owner,delta,int(bool(row.get('label_write'))),owner_model))
        db.commit()
    # A compact durable actual outcome survives later status-only/preview reports.
    save_journal(root/'runtime/autonomy/last_execution.json',result)
    return {'recorded':True,'episode_id':episode_id,'kind':kind,'accepted':len(accepted)}


def feedback_summary(root: Path) -> dict:
    path = root/'runtime/autonomy/feedback.sqlite'
    if not path.is_file():
        return {'episodes':0,'recent':[],'role_credit':[],'generalization_status':'NOT_ESTABLISHED'}
    with closing(sqlite3.connect('file:' + str(path) + '?mode=ro',uri=True)) as db:
        db.row_factory=sqlite3.Row
        count=db.execute('SELECT count(*) FROM episodes').fetchone()[0]
        recent=[]
        for row in db.execute('SELECT body FROM episodes ORDER BY created DESC LIMIT 6'):
            body=json.loads(row['body']);body.pop('outcomes',None);recent.append(body)
        credit=[dict(r) for r in db.execute('SELECT role,model,count(*) accepted,sum(label_write) label_writes,sum(score_delta) scored_gain FROM credits GROUP BY role,model')]
    return {'episodes':count,'recent':recent,'role_credit':credit,'generalization_status':'NOT_ESTABLISHED'}


def actual_execution(root: Path) -> dict:
    recorded=_read(root/'runtime/autonomy/last_execution.json')
    # Compatibility for the first tick after deployment; reject previews.
    if not recorded:
        recorded=(_read(root/'runtime/lge_iterations/latest.json').get('last_execution') or {})
    return recorded if recorded.get('execution_mode')=='execute' else {}
