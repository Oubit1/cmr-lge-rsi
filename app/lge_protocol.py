"""Typed transport aliases. This config cannot alter label decisions or QC policy."""
from __future__ import annotations
import json
import re
from pathlib import Path

CONFIG='configs/lge_protocol_aliases.json'
TEST='tests/test_lge_protocol_aliases.py'
DEFAULT={'schema_version':'lge-protocol-v1','input_aliases':{
    'snapshot_image_path':'image','snapshot_mask_path':'label',
    'snapshot_myo_path':'myo','snapshot_mi_path':'mi'},'output_candidate_aliases':['candidate_filename']}
RESERVED={'case_id','paths','source_sha256','decision','reason','candidate','records','workflow_stage','schema_version'}


def validate(value: dict) -> dict:
    if not isinstance(value,dict) or set(value)!={'schema_version','input_aliases','output_candidate_aliases'}:
        raise ValueError('protocol_config_unknown_fields')
    if value['schema_version']!='lge-protocol-v1':raise ValueError('protocol_config_schema')
    aliases=value['input_aliases'];outputs=value['output_candidate_aliases']
    if not isinstance(aliases,dict) or len(aliases)>24 or not isinstance(outputs,list) or len(outputs)>16:
        raise ValueError('protocol_alias_limits')
    if any(aliases.get(name)!=source for name,source in DEFAULT['input_aliases'].items()):
        raise ValueError('protocol_canonical_alias_changed')
    if not set(DEFAULT['output_candidate_aliases']).issubset(outputs):
        raise ValueError('protocol_canonical_output_alias_removed')
    for name,source in aliases.items():
        if not isinstance(name,str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,63}',name) or name in RESERVED:
            raise ValueError('protocol_input_alias_name')
        if source not in {'image','label','myo','mi'}:raise ValueError('protocol_input_alias_source')
    for name in outputs:
        if not isinstance(name,str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,63}',name) or name in RESERVED:
            raise ValueError('protocol_output_alias_name')
    if len(outputs)!=len(set(outputs)):raise ValueError('protocol_duplicate_alias')
    return value


def load(root: Path) -> dict:
    path=root/CONFIG
    if not path.exists():return validate(json.loads(json.dumps(DEFAULT)))
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()) or path.stat().st_size>8192:
        raise ValueError('protocol_config_path_or_size')
    return validate(json.loads(path.read_text(encoding='utf-8')))


def input_aliases(root: Path, paths: dict) -> dict:
    return {name:paths[source] for name,source in load(root)['input_aliases'].items() if source in paths}


def output_name(root: Path, proposal: dict):
    supplied=[proposal[key] for key in ['candidate',*load(root)['output_candidate_aliases']] if proposal.get(key)]
    if any(not isinstance(value,str) for value in supplied):raise ValueError('generator_protocol_invalid_output_filename')
    if len(set(supplied))>1:raise ValueError('generator_protocol_conflicting_output_aliases')
    return supplied[0] if supplied else None


def repair_status(root: Path) -> dict:
    from app.autonomy_feedback import feedback_summary
    try:
        config=load(root);invalid=None
    except (OSError,ValueError,TypeError) as exc:
        config=None;invalid=type(exc).__name__+':'+str(exc)[-200:]
    recent=feedback_summary(root).get('recent') or []
    latest=recent[0] if recent else {}
    reasons=latest.get('reason_counts') or {}
    protocol_failure=any(any(marker in reason for marker in ('generator_protocol_', 'missing_snapshot_paths','snapshot_file_not_found')) for reason in reasons)
    pending=False
    expected=(latest.get('release') or {}).get('protocol_sha256')
    if protocol_failure and expected and config is not None:
        from app.rsi_kernel import canonical_hash
        pending=expected != canonical_hash(config)
        if pending:protocol_failure=False
    return {'required':bool(invalid or protocol_failure),'pending_verification':pending,'target':CONFIG,'configuration':config,
            'error':invalid,'failure_reasons':list(reasons)[:5] if protocol_failure else []}


def action_allowed(payload: dict) -> bool:
    return (payload.get('action_class')=='framework_repair'
            and payload.get('owner_role')=='workflow_maintainer'
            and CONFIG in str(payload.get('objective','')))
