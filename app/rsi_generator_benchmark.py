"""Frozen doctor-pair challenge for producer evolution, never a production writer.

Uses the existing heldout patient split and existing controlled perturbations.
Reference labels stay outside candidate subprocesses. Scores measure this challenge,
not clinical accuracy on the unresolved cohort.
"""
from __future__ import annotations
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import random
import numpy as np
import nibabel as nib
from app.rsi_kernel import canonical_hash,MetricSpec,compare_fitness
from app.control_checkpoint import save_journal

CONFIG='configs/rsi_generator_benchmark.json'
GENERATOR='scripts/lge_candidate_generator.py'


def _sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def _save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);save_journal(path,value)


def load(root: Path):
    cfg=json.loads((root/CONFIG).read_text(encoding='utf-8'))
    path=(root/cfg['manifest_path']).resolve()
    if not path.is_relative_to((root/'runtime/rsi/generator_benchmarks').resolve()):raise ValueError('generator_benchmark_path')
    if _sha(path)!=cfg['manifest_sha256']:raise ValueError('generator_benchmark_manifest_changed')
    data=json.loads(path.read_text(encoding='utf-8'))
    for rel,sha in data['asset_hashes'].items():
        asset=(path.parent/rel).resolve()
        if not asset.is_relative_to(path.parent) or _sha(asset)!=sha:raise ValueError('generator_benchmark_asset_changed')
    return cfg,path.parent,data


def prepare(root: Path, version='reviewed-generator-v1', per_stage=8, decision_contract=None):
    if decision_contract not in {None,'doctor-disposition-v1'}:raise ValueError('invalid_decision_contract')
    from app.rsi_quality_benchmark import load_cached_gold,patient_split,archive_asset,finite_json
    from app.lge_execution import calibration
    from app.quality_scoring import _assert_original
    from scripts.build_synthetic_eval import defect
    import yaml
    destination=root/'runtime/rsi/generator_benchmarks'/version
    if destination.exists():
        cfg,folder,data=load(root)
        return {'status':'EXISTS','version':data['version'],'records':len(data['records'])}
    destination.parent.mkdir(parents=True,exist_ok=True)
    cached=load_cached_gold(root/'artifacts/image_label_quality_v1/pair_scores.csv')
    training,heldout=patient_split(cached)
    if {r['case_id'] for r in training}&{r['case_id'] for r in heldout}:raise ValueError('generator_patient_split_overlap')
    datasets={d['name']:d for d in yaml.safe_load((root/'configs/quality_scoring_datasets.yaml').read_text())['datasets'] if d.get('tier')=='gold'}
    originals={};records=[];hashes={};profiles={};selected={};skipped=Counter()
    with tempfile.TemporaryDirectory(prefix='generator-bench-',dir=str(destination.parent)) as tmp:
        folder=Path(tmp)/'bundle';folder.mkdir()
        for stage,target in [('lge_myo','myo'),('lge_mi','mi')]:
            profiles[stage]=finite_json(calibration(root,stage)[0]);patients=set()
            rows=sorted([r for r in heldout if r['target']==stage],key=lambda r:canonical_hash({'case':r['case_id'],'relative':r['relative_id']}))
            for row in rows:
                if len(patients)>=per_stage:break
                if row['case_id'] in patients:continue
                targets=datasets[row['dataset']]['targets'];cfg=targets[stage];relative=Path(row['relative_id'])
                label_rel=relative.relative_to(cfg['relative_prefix']) if cfg.get('relative_prefix') else relative
                pair_key='myo' if target=='mi' else 'mi';pair_cfg=targets['lge_'+pair_key]
                pair_rel=relative.relative_to(pair_cfg['relative_prefix']) if pair_cfg.get('relative_prefix') else relative
                sources={'image':Path(cfg['image_dir'])/relative,'label':Path(cfg['label_dir'])/label_rel,
                         pair_key:Path(pair_cfg['label_dir'])/pair_rel}
                if not all(p.is_file() for p in sources.values()):skipped['missing_pair']+=1;continue
                for path in sources.values():_assert_original(path);originals[str(path)]=_sha(path)
                image=nib.load(str(sources['label']));raw=np.asanyarray(image.dataobj);plane=raw.squeeze()
                if plane.ndim!=2 or not (plane>0.5).any():skipped['nonempty_2d_challenge_required']+=1;continue
                assets={}
                for name,path in sources.items():
                    rel,sha=archive_asset(folder,path);assets[name]=rel;hashes[rel]=sha
                patient=canonical_hash({'dataset':row['dataset'],'case':row['case_id']})
                reference=assets['label']
                cases=['clean','fragmentation','border_clip'] + (['hole_fill'] if target=='myo' else [])
                for kind in cases:
                    selected_assets=dict(assets)
                    if kind!='clean':
                        changed=defect(plane>0.5,kind,random.Random(312))
                        if np.array_equal(changed,plane>0.5):skipped['noop_'+kind]+=1;continue
                        staged=Path(tmp)/('perturbed_'+stage+'_'+kind+'.nii.gz')
                        nib.save(image.__class__(changed.astype(raw.dtype).reshape(raw.shape),image.affine,image.header.copy()),str(staged))
                        rel,sha=archive_asset(folder,staged);selected_assets['label']=rel;hashes[rel]=sha
                    records.append({'id':'probe_'+canonical_hash({'patient':patient,'stage':stage,'kind':kind})[:20],
                                    'stage':stage,'kind':kind,'patient':patient,'sources':selected_assets,'reference':reference})
                patients.add(row['case_id'])
            selected[stage]=len(patients)
        if any(selected.get(stage,0)<2 for stage in ['lge_myo','lge_mi']):raise ValueError('insufficient_generator_gold_patients')
        for stage in ['lge_myo','lge_mi']:
            if not any(r['stage']==stage and r['kind']!='clean' for r in records):raise ValueError('generator_challenge_has_no_controls')
        if any(_sha(Path(path))!=sha for path,sha in originals.items()):raise ValueError('doctor_sources_changed_during_prepare')
        data={'schema_version':'generator-benchmark-v1','version':version,'records':records,'asset_hashes':hashes,
              'training_profiles':profiles,'patients_per_stage':selected,'patient_split':'existing_quality_heldout',
              'scope':'nonempty_doctor_masks_with_existing_controlled_defects','original_source_hashes':originals,
              'skipped':dict(skipped),'production_writes':0}
        _save(folder/'manifest.json',data)
        os.replace(folder,destination)
    cfg={'schema_version':'rsi-generator-benchmark-v1','mode':'enforce','manifest_path':str((destination/'manifest.json').relative_to(root)),
         'manifest_sha256':_sha(destination/'manifest.json'),'decision_contract':decision_contract}
    _save(root/CONFIG,cfg)
    return {'status':'PREPARED','records':len(records),'patients_per_stage':selected,'production_writes':0,'manifest_sha256':cfg['manifest_sha256']}


def _dice(a,b):
    a=np.asarray(a)>0.5;b=np.asarray(b)>0.5
    if a.shape!=b.shape:raise ValueError('generator_reference_shape')
    total=int(a.sum()+b.sum())
    return 2*int(np.logical_and(a,b).sum())/total if total else 1.0


def evaluate(root: Path, implementation: Path):
    from app.lge_execution import producer,validate_mask,contained
    cfg,folder,data=load(root);metrics={};errors=[]
    missing=not (implementation/GENERATOR).is_file()
    with tempfile.TemporaryDirectory(prefix='generator-evaluation-') as tmp:
        base=Path(tmp)
        for stage in ['lge_myo','lge_mi']:
            rows=[];entries=[r for r in data['records'] if r['stage']==stage]
            for item in entries:
                sources={name:folder/rel for name,rel in item['sources'].items()}
                rows.append({'case_id':item['id'],'sources':sources,'hashes':{name:_sha(p) for name,p in sources.items()}})
            work=base/stage;work.mkdir()
            try:
                proposals=producer(implementation,rows,work,stage,_sha(implementation/GENERATOR),data['training_profiles'][stage]) if not missing else [dict(case_id=r['case_id'],decision='accept_existing',reason='missing-parent-identity-baseline') for r in rows]
            except Exception as exc:
                proposals=[];errors.append(type(exc).__name__+':'+str(exc)[-300:])
            by_id={p['case_id']:p for p in proposals};clean=[];gains=[];by_kind=defaultdict(list);clean_changes=[];clean_acceptances=[];false_acceptances=[];failures=0
            for item,row in zip(entries,rows):
                source=row['sources']['label'];reference=np.asanyarray(nib.load(str(folder/item['reference'])).dataobj)
                before=np.asanyarray(nib.load(str(source)).dataobj);after=before
                proposal=by_id.get(item['id'])
                try:
                    if proposal is None:raise ValueError('generator_missing_decision')
                    if proposal.get('decision')=='repair':
                        staged=work/(item['id']+'_validated.nii.gz')
                        validate_mask(source,contained(work/'outputs',proposal['candidate']),staged)
                        after=np.asanyarray(nib.load(str(staged)).dataobj)
                    elif proposal.get('decision') not in {'accept_existing','needs_review'}:raise ValueError('generator_invalid_decision')
                except Exception as exc:
                    failures+=1;after=before
                    if len(errors)<6:errors.append(type(exc).__name__+':'+str(exc)[-200:])
                if item['kind']=='clean':
                    clean.append(_dice(after,reference));clean_changes.append(not np.array_equal(after>0.5,reference>0.5))
                    clean_acceptances.append(bool(proposal and proposal.get('decision')=='accept_existing' and np.array_equal(before>0.5,reference>0.5)))
                else:
                    gain=_dice(after,reference)-_dice(before,reference)
                    gains.append(gain);by_kind[item['kind']].append(gain)
                    false_acceptances.append(bool(proposal and proposal.get('decision')=='accept_existing' and not np.array_equal(before>0.5,reference>0.5)))
            metrics[stage+'.clean_dice']=sum(clean)/len(clean)
            metrics[stage+'.repair_gain']=sum(gains)/len(gains)
            metrics[stage+'.clean_changes']=sum(clean_changes)/len(clean_changes)
            metrics[stage+'.error_rate']=failures/len(entries)
            if cfg.get('decision_contract')=='doctor-disposition-v1':
                metrics[stage+'.clean_acceptance']=sum(clean_acceptances)/len(clean_acceptances)
                metrics[stage+'.defect_false_acceptance']=sum(false_acceptances)/len(false_acceptances)
            for kind,values in by_kind.items():metrics[stage+'.'+kind+'_gain']=sum(values)/len(values)
    # No output is installed anywhere; only numerical measurements leave the worker.
    return {'metrics':metrics,'record_count':len(data['records']),'missing_parent_identity':missing,'errors':errors,
            'benchmark_hash':cfg['manifest_sha256'],'scope':data['scope'],'production_writes':0}


def paired(root: Path, workspace: Path):
    cfg,folder,data=load(root)
    reports=[]
    for implementation in [workspace/'rsi_parent/orchestrator',workspace/'orchestrator']:
        driver='import sys,json;from pathlib import Path;sys.path.insert(0,sys.argv[1]);from app.rsi_generator_benchmark import evaluate;print(json.dumps(evaluate(Path(sys.argv[1]),Path(sys.argv[2])),allow_nan=False))'
        env={k:v for k,v in os.environ.items() if k in {'PATH','LANG','LC_ALL','SYSTEMROOT','TMP','TEMP'}}
        run=subprocess.run([sys.executable,'-I','-c',driver,str(root.resolve()),str(implementation.resolve())],env=env,text=True,capture_output=True,timeout=600,check=False)
        if run.returncode:raise ValueError('generator_benchmark_worker_failed:'+run.stderr[-500:])
        reports.append(json.loads(run.stdout))
    before,after=reports
    specs={name:MetricSpec(direction='lower' if name.endswith(('.error_rate','.clean_changes','.defect_false_acceptance')) else 'higher',minimum_delta=1e-8) for name in before['metrics']}
    constraints={'same_frozen_input':before['benchmark_hash']==after['benchmark_hash'],'zero_production_writes':before['production_writes']==after['production_writes']==0}
    selection=compare_fitness(baseline=before['metrics'],candidate=after['metrics'],
        baseline_benchmark_id=data['version'],candidate_benchmark_id=data['version'],
        baseline_benchmark_hash=cfg['manifest_sha256'],candidate_benchmark_hash=cfg['manifest_sha256'],metric_specs=specs,constraints=constraints).as_dict()
    result={**selection,'baseline_metrics':before['metrics'],'candidate_metrics':after['metrics'],
            'benchmark_id':data['version'],'benchmark_hash':cfg['manifest_sha256'],'constraints':constraints,
            'scope':'lge_generator_doctor_reference_challenge','workflow_stage':'lge_generator',
            'baseline_report':before,'candidate_report':after,'production_writes':0}
    _save(workspace/'generator_benchmark.json',result)
    return result


def status(root: Path):
    try:
        cfg=json.loads((root/CONFIG).read_text(encoding='utf-8'))
        path=(root/cfg['manifest_path']).resolve()
        if not path.is_relative_to((root/'runtime/rsi/generator_benchmarks').resolve()) or _sha(path)!=cfg['manifest_sha256']:
            raise ValueError('generator_benchmark_manifest_changed')
        data=json.loads(path.read_text(encoding='utf-8'))
        return {'status':'READY' ,'version':data['version'],'records':len(data['records']),
                'patients_per_stage':data['patients_per_stage'],'scope':data['scope'],'mode':cfg['mode'],'decision_contract':cfg.get('decision_contract'),
                'fitness_contract':'Keep every existing Dice, repair, clean-change and error metric non-regressing. When doctor-disposition-v1 is enabled, also compare explicit acceptance of doctor-verified clean pairs and false acceptance of controlled corrupted pairs; increased false acceptance blocks promotion. Reference answers are never supplied to the generator.'}
    except (OSError,ValueError,KeyError,TypeError) as exc:
        return {'status':'UNAVAILABLE','reason':str(exc)[-250:]}
