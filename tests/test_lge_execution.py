from __future__ import annotations
import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np

from app import lge_execution as engine
from app.control_checkpoint import save_journal
from experiments.runtime_evaluation import evaluation_input
REAL_MEASURE = engine.measure
REAL_GOVERNED = engine.governed_generator


GENERATOR = '''from pathlib import Path
import nibabel as nib
import numpy as np
def generate_candidates(data,output_dir):
 result=[]
 for row in data['records']:
  if data['workflow_stage']=='lge_mi':
   result.append(dict(case_id=row['case_id'],decision='accept_existing',reason='synthetic fixture acceptance'))
   continue
  image=nib.load(row['paths']['label']);values=np.asanyarray(image.dataobj).copy()
  values[0,0]=0
  name=row['case_id']+'.nii.gz'
  nib.save(nib.Nifti1Image(values,image.affine,image.header),str(Path(output_dir)/name))
  result.append(dict(case_id=row['case_id'],decision='repair',candidate=name,reason='synthetic fixture correction'))
 return result
'''


class LgeExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'project'
        self.dataset = Path(self.temp.name) / 'dataset'
        (self.root / 'scripts').mkdir(parents=True)
        (self.root / 'scripts/__init__.py').touch()
        (self.root / engine.GENERATOR).write_text(GENERATOR)
        (self.root / 'configs').mkdir()
        save_journal(self.root / 'configs/lge_execution.json', {'schema_version': engine.SCHEMA,
                     'dataset': str(self.dataset), 'enabled': True})
        self.labels = []
        records=[]
        for patient in range(4):
            relative = Path('synthetic_' + str(patient)) / 'slice.nii.gz'
            for name, folder in [('image','images/lge'),('myo',engine.TARGET_DIRS[('lge','myo')]),('mi',engine.TARGET_DIRS[('lge','mi')])]:
                path = self.dataset / folder / relative
                path.parent.mkdir(parents=True,exist_ok=True)
                values=np.indices((24,24)).sum(axis=0).astype(np.float32) if name=='image' else np.zeros((24,24),np.uint8)
                if name=='myo':
                    values[8:16,8:16]=1;values[0,0]=1
                    self.labels.append(path)
                if name=='mi': values[10:12,10:12]=1
                nib.save(nib.Nifti1Image(values,np.eye(4)),str(path))
            records.append(dict(patient=relative.parent.as_posix(),slice=relative.name,layer_index='1',
                                issues='myo_topology;mi_component',evidence='components=2'))
        manifest=self.root/'artifacts/run_000001/evidence_snapshot/lge_issue_manifest.csv'
        manifest.parent.mkdir(parents=True)
        with manifest.open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
        editor=self.dataset/'labels/check/manifest_backup/lge_issue_manifest.csv'
        editor.parent.mkdir(parents=True);editor.write_bytes(manifest.read_bytes())
        self.env=patch.dict(os.environ,{'CMR_LGE_CASE_SALT':'synthetic-test-salt'});self.env.start();self.addCleanup(self.env.stop)
        self.cal=patch.object(engine,'calibration',return_value=({},'synthetic-calibration'));self.cal.start();self.addCleanup(self.cal.stop)
        self.governance=patch.object(engine,'governed_generator',side_effect=lambda root: {
            'ready':True,'module_sha256':engine.sha(root/engine.GENERATOR),'run_id':1,'action_id':'fixture'})
        self.governance.start();self.addCleanup(self.governance.stop)
        self.scoring=patch.object(engine,'measure',side_effect=self.measure);self.scoring.start();self.addCleanup(self.scoring.stop)

    @staticmethod
    def measure(row,label,profile):
        values=np.asanyarray(nib.load(str(label)).dataobj)
        result={'score':75.0 if values[0,0] else 95.0}
        if row['target']=='myo' and 'mi' in row['sources']: result['paired_mi_score']=95.0
        return result

    def run_stage(self,stage='lge_myo',**kwargs):
        return engine.run_batch(self.root,self.dataset,stage,evaluation_input(self.root,workflow_stage=stage),**kwargs)

    def test_input_contract_upgrade_retries_parked_items_without_changing_domain_code(self):
        version = ['old']
        mock_release = patch.object(engine, 'governed_generator', side_effect=lambda root: {'ready': True, 'module_sha256': engine.sha(root / engine.GENERATOR),
            'run_id': 1, 'action_id': 'fixture', 'input_contract_version': version[0]})
        mock_release.start()
        self.addCleanup(mock_release.stop)
        def unavailable(root, rows, *args):
            return [{'case_id': row['case_id'], 'decision': 'needs_review', 'reason': 'missing_snapshot_paths'} for row in rows]
        with patch.object(engine, 'producer', side_effect=unavailable):
            self.assertEqual(3, len(self.run_stage(execute=True)['outcomes']))
        self.assertEqual(0, len(self.run_stage(execute=True)['outcomes']))
        version[0] = 'new'
        self.assertEqual(3, self.run_stage(execute=True)['production_writes'])

    def test_contract_probe_rejects_missing_paths_for_supplied_fixtures(self):
        from scripts.validate_lge_generator import validate
        (self.root / engine.GENERATOR).write_text("def generate_candidates(data,output_dir):\n return [dict(case_id=r['case_id'],decision='needs_review',reason='missing_snapshot_paths') for r in data['records']]\n")
        with self.assertRaisesRegex(ValueError, 'generator_input_contract_unhandled'):
            validate(self.root)

    def test_snapshot_aliases_are_real_copies_and_preserve_canonical_paths(self):
        source = GENERATOR.replace("row['paths']['label']", "row['snapshot_mask_path']")
        source = source.replace(" for row in data['records']:", " for row in data['records']:\n  assert row['snapshot_image_path']==row['paths']['image']\n  assert row['snapshot_mask_path']==row['paths']['label']\n  assert row['snapshot_mi_path']==row['paths']['mi']")
        (self.root / engine.GENERATOR).write_text(source)
        before = [engine.sha(path) for path in self.labels]
        report = self.run_stage(execute=False)
        self.assertTrue(all(r['status']=='VALIDATED' for r in report['outcomes']))
        self.assertEqual(3, len(report['outcomes']))
        self.assertEqual(before, [engine.sha(path) for path in self.labels])

    def test_producer_crash_is_pending_framework_work_not_a_human_clinical_queue(self):
        with patch.object(engine, 'producer', side_effect=ValueError('synthetic dimensional error')):
            first = self.run_stage(execute=True)
        self.assertEqual(3, first['lge_myo_execution_failed'])
        self.assertEqual(3, first['lge_myo_pending'])
        self.assertEqual(1, first['lge_myo_work'])  # frozen holdout only
        second = self.run_stage(execute=True)
        self.assertEqual([], second['outcomes'])  # no blind retry of unchanged producer
        self.assertEqual(3, second['lge_myo_execution_failed'])
        self.assertEqual(1, len(second['manual_review_items']))

    def test_candidate_filename_alias_is_normalized_before_independent_validation(self):
        (self.root / engine.GENERATOR).write_text(GENERATOR.replace("candidate=name", "candidate_filename=name"))
        report = self.run_stage(execute=True)
        self.assertEqual(3, report['production_writes'])
        self.assertEqual(3, report['lge_myo_resolved'])

    def test_missing_repair_output_file_is_a_protocol_failure_not_a_clinical_review(self):
        source = "def generate_candidates(data,output_dir):\n return [dict(case_id=r['case_id'],decision='repair',reason='synthetic') for r in data['records']]\n"
        (self.root / engine.GENERATOR).write_text(source)
        report = self.run_stage(execute=True)
        self.assertEqual(0,report['production_writes'])
        self.assertEqual(3,report['lge_myo_execution_failed'])
        self.assertEqual(1,report['lge_myo_work'])

    def test_preview_is_not_application_and_does_not_resolve_candidates(self):
        before=[engine.sha(path) for path in self.labels]
        report=self.run_stage(execute=False)
        self.assertEqual(0,report['production_writes'])
        self.assertEqual(0,report['lge_myo_resolved'])
        self.assertEqual(3,len(report['outcomes']))
        self.assertTrue(all(item['status']=='VALIDATED' for item in report['outcomes']))
        self.assertEqual(before,[engine.sha(path) for path in self.labels])

    def test_real_synthetic_files_apply_with_backup_receipts_and_idempotence(self):
        for path in self.labels: path.chmod(0o640)
        before=[engine.sha(path) for path in self.labels]
        first=self.run_stage(execute=True)
        self.assertEqual(3,first['production_writes'])
        self.assertEqual(3,first['lge_myo_resolved'])
        from app.autonomy_feedback import record_execution, feedback_summary
        experience=record_execution(self.root,first)
        self.assertEqual(3,experience['accepted'])
        self.assertEqual(3,feedback_summary(self.root)['role_credit'][0]['label_writes'])
        self.assertFalse(record_execution(self.root,first)['recorded'])
        self.assertEqual(1,first['holdout_excluded'])
        self.assertEqual(1,sum(a==engine.sha(p) for a,p in zip(before,self.labels)))
        second=self.run_stage(execute=True)
        self.assertEqual(0,second['production_writes'])
        self.assertTrue(second['lge_myo_auto_complete'])
        audit=[json.loads(line) for line in (self.root/'runtime/lge_execution/apply_audit.jsonl').read_text().splitlines()]
        self.assertEqual(3,sum(item['event']=='LGE_APPLIED' for item in audit))
        self.assertEqual(3,len(list((self.root/'runtime/lge_execution/transactions').glob('*.backup.nii.gz'))))
        if os.name!='nt':
            self.assertTrue(all(path.stat().st_mode & 0o777 == 0o640 for path in self.labels))
            self.assertTrue(all(path.stat().st_mode & 0o777 == 0o600 for path in (self.root/'runtime/lge_execution/transactions').glob('*.backup.nii.gz')))

    def test_myo_change_invalidates_previous_mi_receipts(self):
        self.run_stage(execute=True)
        self.assertEqual(3,self.run_stage('lge_mi',execute=True)['lge_mi_resolved'])
        rows,_=engine.source_queue(self.root,self.dataset,'lge_mi',evaluation_input(self.root,workflow_stage='lge_mi'))
        changed=next(row for row in rows if not row['excluded'])['sources']['myo']
        image=nib.load(str(changed));values=np.asanyarray(image.dataobj).copy();values[1,1]=1
        nib.save(nib.Nifti1Image(values,image.affine),str(changed))
        report=self.run_stage('lge_mi',status_only=True)
        self.assertEqual(2,report['lge_mi_resolved'])

    def test_driver_advances_from_receipts_and_keeps_manual_residual_visible(self):
        from scripts.run_lge_feedback_consumer import run_once
        from agents.evidence import dependent_workflow_status
        run_once(self.root,self.dataset,execute=True)
        self.assertEqual('lge_mi',dependent_workflow_status(self.dataset,self.root)['stage'])
        run_once(self.root,self.dataset,execute=True)
        self.assertEqual('paired_selection',dependent_workflow_status(self.dataset,self.root)['stage'])
        report=json.loads((self.root/'runtime/lge_iterations/latest.json').read_text())
        self.assertEqual(1,report['lge_myo_work'])
        self.assertEqual(1,report['lge_mi_work'])
        self.assertEqual(3,report['lge_myo_resolved'])
        self.assertEqual(3,report['lge_mi_resolved'])
        self.assertFalse(report['pairing_ready'])

    def test_mi_blocker_requeues_upstream_myo_even_without_old_myo_issue(self):
        from scripts.run_lge_feedback_consumer import run_once
        from agents.evidence import dependent_workflow_status
        rows,_=engine.source_queue(self.root,self.dataset,'lge_mi',evaluation_input(self.root,workflow_stage='lge_mi'))
        bad=next(row for row in rows if not row['excluded'])
        patient=bad['sources']['label'].parent.name
        manifest=self.root/'artifacts/run_000001/evidence_snapshot/lge_issue_manifest.csv'
        records=list(csv.DictReader(manifest.read_text().splitlines()))
        for record in records:
            if record['patient']==patient: record['issues']='mi_component'
        with manifest.open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
        original=self.measure
        def score(row,label,profile):
            result=original(row,label,profile)
            if row['target']=='myo' and Path(label)==bad['sources']['myo']:
                result['score']=60.0 if np.asanyarray(nib.load(str(label)).dataobj)[0,0] else 95.0
            return result
        with patch.object(engine,'measure',side_effect=score):
            run_once(self.root,self.dataset,execute=True)
            run_once(self.root,self.dataset,execute=True)
            self.assertEqual('lge_myo',dependent_workflow_status(self.dataset,self.root)['stage'])
            run_once(self.root,self.dataset,execute=True)
            run_once(self.root,self.dataset,execute=True)
        self.assertEqual('paired_selection',dependent_workflow_status(self.dataset,self.root)['stage'])

    def test_aggregate_report_cannot_skip_candidate_execution(self):
        from agents.evidence import dependent_workflow_status
        path=self.root/'runtime/lge_iterations/latest.json';path.parent.mkdir(parents=True)
        save_journal(path,{'lge_myo_consumer_ready':True,'lge_mi_consumer_ready':True,
            'lge_myo_pending':0,'lge_myo_work':0,'lge_mi_pending':0,'lge_mi_work':0,
            'pairing_ready':True,'paired_slice_count':100})
        self.assertEqual('lge_myo',dependent_workflow_status(self.dataset,self.root)['stage'])

    def test_planning_guard_accepts_generator_and_rejects_old_report_adapter(self):
        from agents.evidence import action_builds_stage_consumer, action_matches_workflow_stage
        generator={'objective':engine.GENERATOR+' generate_candidates(data, output_dir)'}
        old={'objective':'scripts/lge_feedback_loop.py runtime/lge_iterations/latest.json'}
        self.assertTrue(action_builds_stage_consumer(generator,require_generator=True))
        self.assertFalse(action_builds_stage_consumer(old,require_generator=True))
        self.assertTrue(action_matches_workflow_stage({'workflow_stage':'lge_myo',
            'objective':engine.GENERATOR+' reuse CMR topology priors for LGE Myo'},'lge_myo'))

    def test_physical_lge_progress_updates_global_fingerprint(self):
        from scripts.autopilot import issue_fingerprint
        from scripts.run_lge_feedback_consumer import run_once
        before,_=issue_fingerprint(self.dataset,self.root)
        run_once(self.root,self.dataset,execute=True)
        after,counts=issue_fingerprint(self.dataset,self.root)
        self.assertNotEqual(before,after)
        self.assertEqual(3,counts['lge_myo_resolved'])

    def test_lge_first_consumer_does_not_run_cmr_in_advance(self):
        from scripts import check_feedback_loop as loop
        with patch.object(loop,'ROOT',self.root),patch.object(loop,'tick') as cmr, \
             patch.object(loop,'run_lge_consumer_once',return_value={'status':'UNCHANGED'}) as lge:
            loop.run_consumers(self.dataset,auto_apply=True)
            cmr.assert_not_called()
            lge.assert_called_once()

    def test_scorer_generation_change_restarts_before_any_consumer(self):
        from scripts import check_feedback_loop as loop
        with patch.object(loop,'ROOT',self.root),patch.object(loop,'tick') as cmr, \
             patch.object(loop,'run_lge_consumer_once') as lge, \
             patch.object(engine,'scorer_fingerprint',return_value='new'):
            with self.assertRaises(loop.ScorerReloadRequired):
                loop.run_consumers(self.dataset,auto_apply=True,scorer_boot_digest='old')
            cmr.assert_not_called();lge.assert_not_called()

    def test_mi_repair_does_not_invalidate_unchanged_myo_but_changed_myo_invalidates_mi(self):
        row={'case_id':'x','target':'myo','hashes':{'image':'i','label':'m','mi':'updated'}}
        receipt={'status':'CHECKED','case_id':'x','calibration_id':'c','after_hashes':{'image':'i','label':'m','mi':'old'}}
        self.assertTrue(engine.receipt_current(receipt,row,'c'))
        row['target']='mi'
        self.assertFalse(engine.receipt_current(receipt,row,'c'))

    def test_new_generator_version_retries_parked_items(self):
        with patch.object(engine,'measure',return_value={'score':60.0}): self.run_stage(execute=True)
        with (self.root/engine.GENERATOR).open('a') as handle: handle.write('\n# new governed synthetic fixture version\n')
        report=self.run_stage(execute=True)
        self.assertEqual(3,report['production_writes'])

    def test_source_and_dependency_changes_are_rechecked_before_write(self):
        data=evaluation_input(self.root,workflow_stage='lge_myo')
        rows,_=engine.source_queue(self.root,self.dataset,'lge_myo',data)
        row=next(item for item in rows if not item['excluded'])
        source=row['sources']['label'];image=nib.load(str(source));values=np.asanyarray(image.dataobj).copy()
        candidate=self.root/'work/outputs/candidate.nii.gz';candidate.parent.mkdir(parents=True)
        values[0,0]=0;nib.save(nib.Nifti1Image(values,image.affine),str(candidate))
        dependency=row['sources']['mi'];dependency.write_bytes(dependency.read_bytes()+b'synthetic_changed')
        with self.assertRaisesRegex(ValueError,'stale_lge_source'):
            engine.apply_one(self.root,self.dataset,row,{'decision':'repair','candidate':'candidate.nii.gz','reason':'fixture'},
                candidate.parent.parent,{},'c','lge_myo',{},True)

    def test_rejected_candidate_is_parked_until_new_evidence(self):
        with patch.object(engine,'measure',return_value={'score':60.0}):
            first=self.run_stage(execute=True)
        self.assertEqual(0,first['production_writes'])
        self.assertTrue(all(item['status']=='NEEDS_REVIEW' for item in first['outcomes']))
        with patch.object(engine,'producer') as producer:
            self.run_stage(execute=True)
            producer.assert_not_called()

    def test_do_not_apply_without_authorized_generator_or_when_paused(self):
        with patch.object(engine,'governed_generator',return_value={'ready':False,'reason':'missing'}):
            report=self.run_stage(execute=True)
            self.assertFalse(report['lge_myo_consumer_ready'])
            self.assertEqual([],report['outcomes'])
        state=self.root/'runtime/control/operator_state.json';state.parent.mkdir(parents=True,exist_ok=True)
        save_journal(state,{'paused':True})
        with self.assertRaisesRegex(ValueError,'operator_paused'): self.run_stage(execute=True)

    def test_generator_requires_real_promoted_source_and_full_regression_record(self):
        import shutil
        from app.database import Database
        db=Database(self.root/'runtime/workflow.sqlite');db.initialize()
        dataset_id=db.register_dataset('fixture',self.dataset)
        run=db.create_run(dataset_id,'analysis',{})
        action={'action_id':'fixture','owner_role':'qc_code_optimizer','action_class':'snapshot_patch','objective':engine.GENERATOR}
        db.save_maintenance_action(run,action,status='PROMOTED')
        workspace=self.root/'runtime/workspaces/fixture'
        shutil.copytree(self.root/'scripts',workspace/'orchestrator/scripts')
        artifact=self.root/'runtime/promotions/fixture';artifact.mkdir(parents=True)
        save_journal(workspace/'rsi_promotion_journal.json',{'artifact':str(artifact),'run_id':run,'action_id':'fixture'})
        save_journal(workspace/'autopilot_validation.json',{'passed':True,'full_regression_required':True})
        save_journal(artifact/'promotion_manifest.json',{'status':'PROMOTED','promoted':[{
            'owner':'orchestrator','relative':engine.GENERATOR,'new_hash':engine.sha(self.root/engine.GENERATOR)}]})
        db.update_maintenance_action(run,'fixture',status='PROMOTED',artifact_path=workspace)
        self.assertTrue(REAL_GOVERNED(self.root)['ready'])
        with (self.root/engine.GENERATOR).open('a') as handle: handle.write('\n# unreviewed change\n')
        self.assertFalse(REAL_GOVERNED(self.root)['ready'])

    def test_human_applied_context_counts_but_staged_or_stale_review_does_not(self):
        rows,_=engine.source_queue(self.root,self.dataset,'lge_myo',evaluation_input(self.root,workflow_stage='lge_myo'))
        held=next(row for row in rows if row['excluded'])
        provenance=self.root/'runtime/manual_annotations/edits/synthetic/provenance.json'
        provenance.parent.mkdir(parents=True)
        item={'status':'STAGED','modality':'lge','target':'myo','applied_by':'synthetic_reviewer',
            'source_label':str(held['sources']['label']),'applied_hash':held['hashes']['label'],
            'applied_context':{'image':held['hashes']['image']}}
        save_journal(provenance,item)
        self.assertEqual(0,self.run_stage(status_only=True)['lge_myo_resolved'])
        item['status']='APPLIED';save_journal(provenance,item)
        self.assertEqual(1,self.run_stage(status_only=True)['lge_myo_resolved'])
        item['applied_context']['image']='stale';save_journal(provenance,item)
        self.assertEqual(0,self.run_stage(status_only=True)['lge_myo_resolved'])

    def test_crash_after_replace_recovers_receipt_without_reapplying(self):
        original=engine.audit
        failed=False
        def crash(root,event):
            nonlocal failed
            if event['event']=='LGE_APPLIED' and not failed:
                failed=True;raise OSError('simulated audit crash')
            return original(root,event)
        with patch.object(engine,'audit',side_effect=crash):
            with self.assertRaisesRegex(OSError,'simulated'): self.run_stage(execute=True)
        report=self.run_stage(execute=True)
        self.assertEqual(3,report['lge_myo_resolved'])
        lines=[json.loads(line) for line in (self.root/'runtime/lge_execution/apply_audit.jsonl').read_text().splitlines()]
        self.assertEqual(1,sum(row['event']=='LGE_APPLY_RECOVERED' for row in lines))

    def test_geometry_binary_and_path_validation(self):
        source=self.labels[0];out=self.root/'candidate.nii.gz';stage=self.root/'stage.nii.gz'
        nib.save(nib.Nifti1Image(np.zeros((4,4)),np.eye(4)),str(out))
        with self.assertRaisesRegex(ValueError,'geometry'): engine.validate_mask(source,out,stage)
        nib.save(nib.Nifti1Image(np.full((24,24),2,np.uint8),np.eye(4)),str(out))
        with self.assertRaisesRegex(ValueError,'binary'): engine.validate_mask(source,out,stage)
        with self.assertRaisesRegex(ValueError,'relative'): engine.contained(self.root,'../escape.nii.gz')

    def test_controller_reads_actual_arrays_not_proposed_scores(self):
        row={'target':'myo','sources':{'image':self.dataset/'images/lge/synthetic_0/slice.nii.gz'}}
        with patch.object(engine,'measure',side_effect=REAL_MEASURE):
            # The test stub is used for deterministic writer tests; exercise the
            # real existing scorer separately, with a synthetic profile only.
            from app.quality_scoring import PairSpec,extract_pair,build_profile,score_record
            features=extract_pair(PairSpec(dataset='synthetic',tier='gold',target='lge_myo',
                image=row['sources']['image'],label=self.labels[0],myo=None,relative_id='x',case_id='x',slice_position=0,reviewed=True))
            score=score_record(features,build_profile([features]))['score']
            self.assertTrue(np.isfinite(score))
            self.assertEqual(score,engine.measure(row,self.labels[0],build_profile([features]))['score'])


if __name__=='__main__': unittest.main()
