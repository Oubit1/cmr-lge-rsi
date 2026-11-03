"""Optional offline doctor-reference execution probe, never a domain-agent output.

Uses copied doctor pairs, existing controlled perturbations and an oracle test
candidate. All writer calls target a newly created temporary dataset. No model
calls, production label writes or fake production governance records occur.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
import nibabel as nib
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import lge_execution as engine
from app.control_checkpoint import save_journal
from app.rsi_quality_benchmark import load_cached_gold
from experiments.runtime_evaluation import evaluation_input
from scripts.build_synthetic_eval import defect
from scripts.run_lge_feedback_consumer import run_once


ORACLE = '''from pathlib import Path
import shutil
def generate_candidates(data,output_dir):
 result=[]
 for row in data['records']:
  item=dict(case_id=row['case_id'],reason='offline known-reference protocol fixture')
  if data['workflow_stage']=='lge_myo':
   name=row['case_id']+'.nii.gz';shutil.copyfile(row['paths']['reference'],str(Path(output_dir)/name))
   item.update(decision='repair',candidate=name)
  else:item.update(decision='accept_existing')
  result.append(item)
 return result
'''


def verify(source_root: Path) -> dict:
    config = yaml.safe_load((source_root/'configs/quality_scoring_datasets.yaml').read_text())
    datasets={row['name']:row for row in config['datasets'] if row.get('tier')=='gold'}
    cached=load_cached_gold(source_root/'artifacts/image_label_quality_v1/pair_scores.csv')
    candidates=[row for row in cached if row['target']=='lge_myo']
    profile,profile_id=engine.calibration(source_root,'lge_myo')
    real_queue=engine.source_queue
    source_hashes={}
    with tempfile.TemporaryDirectory(prefix='lge-reference-protocol-') as tmp:
        root=Path(tmp)/'project';dataset=Path(tmp)/'dataset'
        (root/'scripts').mkdir(parents=True);(root/'scripts/__init__.py').touch()
        (root/engine.GENERATOR).write_text(ORACLE)
        (root/'configs').mkdir()
        save_journal(root/'configs/lge_execution.json',{'schema_version':engine.SCHEMA,'dataset':str(dataset),'enabled':True})
        selected=[];patients=set()
        for row in candidates:
            patient=(row['dataset'],row['case_id'])
            if patient in patients:continue
            paths=datasets[row['dataset']]['targets']
            relative=Path(row['relative_id']);myo_cfg=paths['lge_myo'];mi_cfg=paths['lge_mi']
            label_rel=relative.relative_to(myo_cfg['relative_prefix']) if myo_cfg.get('relative_prefix') else relative
            image=Path(myo_cfg['image_dir'])/relative;myo=Path(myo_cfg['label_dir'])/label_rel
            mi=Path(mi_cfg['label_dir'])/label_rel
            if not all(path.is_file() for path in (image,myo,mi)):continue
            reference=nib.load(str(myo));raw=np.asanyarray(reference.dataobj);plane=raw.squeeze()
            if plane.ndim!=2:continue
            changed=defect(plane>0.5,'border_clip',np.random.default_rng(29))
            if np.array_equal(changed,plane>0.5):continue
            index=len(selected);rel=Path('reference_'+str(index))/'slice.nii.gz'
            local={}
            for key,folder,source in [('image','images/lge',image),('label',engine.TARGET_DIRS[('lge','myo')],myo),
                                      ('mi',engine.TARGET_DIRS[('lge','mi')],mi),('reference','test_reference',myo)]:
                path=dataset/folder/rel;path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,path);local[key]=path
            nib.save(nib.Nifti1Image(changed.astype(raw.dtype).reshape(raw.shape),reference.affine,reference.header.copy()),str(local['label']))
            test_row={'target':'myo','sources':local}
            before=engine.measure(test_row,local['label'],profile);after=engine.measure(test_row,local['reference'],profile)
            if (after['score']<70 or after['score']<=before['score']
                    or after.get('paired_mi_score',0)<before.get('paired_mi_score',0)):
                continue
            for path in (image,myo,mi):source_hashes[path]=engine.sha(path)
            selected.append(dict(patient=rel.parent.as_posix(),slice=rel.name,layer_index='1',issues='myo_controlled_border;mi_reference',evidence=''))
            patients.add(patient)
            if len(selected)==3:break
        if len(selected)!=3:raise ValueError('three suitable independent test patients unavailable')
        manifest=root/'artifacts/run_000001/evidence_snapshot/lge_issue_manifest.csv';manifest.parent.mkdir(parents=True)
        with manifest.open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(selected[0]));writer.writeheader();writer.writerows(selected)
        def queue(*args):
            rows,holdout=real_queue(*args)
            if args[2]=='lge_myo':
                for item in rows:
                    rel=item['sources']['label'].relative_to(dataset/engine.TARGET_DIRS[('lge','myo')])
                    item['sources']['reference']=dataset/'test_reference'/rel
                    item['hashes']['reference']=engine.sha(item['sources']['reference'])
            return rows,holdout
        release={'ready':True,'module_sha256':engine.sha(root/engine.GENERATOR),'test_fixture':True}
        with patch.dict(os.environ,{'CMR_LGE_CASE_SALT':'isolated-reference-probe'}), \
             patch.object(engine,'calibration',return_value=(profile,profile_id)), \
             patch.object(engine,'source_queue',side_effect=queue), \
             patch.object(engine,'governed_generator',return_value=release):
            run_once(root,dataset,execute=True)
            myo=json.loads((root/'runtime/lge_execution/lge_myo.json').read_text())
            run_once(root,dataset,execute=True)
            mi=json.loads((root/'runtime/lge_execution/lge_mi.json').read_text())
            replay=engine.run_batch(root,dataset,'lge_myo',evaluation_input(root,workflow_stage='lge_myo'),execute=True)
            results={'myo_fixture_writes':myo['production_writes'],'myo_checked':myo['lge_myo_resolved'],
                     'mi_checked':mi['lge_mi_resolved'],'replay_writes':replay['production_writes'],
                     'frozen_cases_excluded':myo['holdout_excluded']}
            if results!={'myo_fixture_writes':2,'myo_checked':2,'mi_checked':2,'replay_writes':0,'frozen_cases_excluded':1}:
                raise ValueError('unexpected isolated receipt counts:'+str(results))
    unchanged=all(engine.sha(path)==value for path,value in source_hashes.items())
    if not unchanged:raise ValueError('reference source changed')
    return {'status':'VERIFIED','evaluation_type':'offline_known_reference_protocol_probe',
            'candidate_origin':'oracle_test_reference_not_agent_reasoning','reference_pairs':3,
            'real_production_writes':0,'model_calls':0,'original_source_hashes_unchanged':unchanged,**results}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report=verify(args.root)
    args.output.parent.mkdir(parents=True,exist_ok=True);save_journal(args.output,report)
    print(json.dumps(report,indent=2))
