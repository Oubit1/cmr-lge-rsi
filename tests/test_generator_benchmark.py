import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import nibabel as nib
from app import rsi_generator_benchmark as bench
from app.rsi_kernel import canonical_hash

class GeneratorBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.folder=self.root/'runtime/rsi/generator_benchmarks/fixture';self.folder.mkdir(parents=True)
        records=[];hashes={}
        for stage in ['lge_myo','lge_mi']:
            for kind in ['clean','defect']:
                reference=np.zeros((12,12),np.uint8);reference[3:8,3:8]=1
                label=reference.copy()
                if kind=='defect':label[0,0]=1
                sources={}
                for name,values in [('image',np.indices((12,12)).sum(axis=0).astype(np.float32)),('label',label),('mi' if stage=='lge_myo' else 'myo',reference),('reference',reference)]:
                    path=self.folder/(stage+'_'+kind+'_'+name+'.nii.gz');nib.save(nib.Nifti1Image(values,np.eye(4)),str(path))
                    hashes[path.name]=bench._sha(path);sources[name]=path.name
                ref=sources.pop('reference')
                records.append({'id':stage+'_'+kind,'stage':stage,'kind':kind,'patient':'opaque','sources':sources,'reference':ref})
        data={'version':'fixture','records':records,'asset_hashes':hashes,'training_profiles':{'lge_myo':{},'lge_mi':{}},'scope':'synthetic_test_only','patients_per_stage':{'lge_myo':1,'lge_mi':1}}
        bench._save(self.folder/'manifest.json',data)
        bench._save(self.root/bench.CONFIG,{'mode':'enforce','manifest_path':str((self.folder/'manifest.json').relative_to(self.root)),
                                        'manifest_sha256':bench._sha(self.folder/'manifest.json')})
        self.impl=self.root/'implementation';(self.impl/'scripts').mkdir(parents=True);(self.impl/'scripts/__init__.py').touch()
    def generator(self,body):
        (self.impl/bench.GENERATOR).write_text(body)
    def test_frozen_reference_is_not_in_candidate_input_and_original_assets_remain_unchanged(self):
        self.generator('''from pathlib import Path
import nibabel as nib
import numpy as np
def generate_candidates(data,output_dir):
 out=[]
 for r in data['records']:
  assert 'reference' not in r and 'reference' not in r['paths']
  im=nib.load(r['paths']['label']);v=np.asanyarray(im.dataobj).copy();v[0,0]=0
  name=r['case_id']+'.nii.gz';nib.save(nib.Nifti1Image(v,im.affine,im.header),str(Path(output_dir)/name))
  out.append(dict(case_id=r['case_id'],decision='repair',candidate=name,reason='synthetic test'))
 return out
''')
        before={p.name:bench._sha(p) for p in self.folder.glob('*.nii.gz')}
        report=bench.evaluate(self.root,self.impl)
        self.assertGreater(report['metrics']['lge_myo.repair_gain'],0)
        self.assertEqual(1,report['metrics']['lge_myo.clean_dice'])
        self.assertEqual(0,report['metrics']['lge_myo.error_rate'])
        self.assertEqual(before,{p.name:bench._sha(p) for p in self.folder.glob('*.nii.gz')})
    def test_identity_parent_is_explicit_and_not_an_invented_solver(self):
        report=bench.evaluate(self.root,self.impl)
        self.assertTrue(report['missing_parent_identity'])
        self.assertEqual(0,report['metrics']['lge_mi.repair_gain'])
        self.assertEqual(1,report['metrics']['lge_mi.clean_dice'])
    def test_reference_tampering_fails_before_generator_runs(self):
        one=next(self.folder.glob('*.nii.gz'));one.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'asset_changed'):bench.evaluate(self.root,self.impl)
    def test_producer_crash_is_measured_as_error_not_false_improvement(self):
        self.generator('def generate_candidates(data,output_dir):\n raise ValueError("synthetic crash")\n')
        report=bench.evaluate(self.root,self.impl)
        self.assertEqual(1,report['metrics']['lge_myo.error_rate'])
        self.assertEqual(0,report['metrics']['lge_myo.repair_gain'])
    def test_regressing_clean_masks_is_measured_even_when_producer_claims_repair(self):
        self.generator('''from pathlib import Path
import nibabel as nib
import numpy as np
def generate_candidates(data,output_dir):
 result=[]
 for r in data['records']:
  im=nib.load(r['paths']['label']);name=r['case_id']+'.nii.gz'
  nib.save(nib.Nifti1Image(np.zeros(im.shape,np.uint8),im.affine),str(Path(output_dir)/name))
  result.append(dict(case_id=r['case_id'],decision='repair',candidate=name,reason='false progress'))
 return result
''')
        report=bench.evaluate(self.root,self.impl)
        self.assertEqual(0,report['metrics']['lge_myo.clean_dice'])
        self.assertLess(report['metrics']['lge_myo.repair_gain'],0)

    def test_decision_contract_counts_clean_acceptance_and_corrupt_false_acceptance(self):
        cfg=json.loads((self.root/bench.CONFIG).read_text());cfg['decision_contract']='doctor-disposition-v1';bench._save(self.root/bench.CONFIG,cfg)
        self.generator("def generate_candidates(data,output_dir):\n return [dict(case_id=r['case_id'],decision='accept_existing',reason='fixture') for r in data['records']]\n")
        metrics=bench.evaluate(self.root,self.impl)['metrics']
        self.assertEqual(metrics['lge_myo.clean_acceptance'],1)
        self.assertEqual(metrics['lge_myo.defect_false_acceptance'],1)
        from types import SimpleNamespace
        before=dict(metrics);before['lge_myo.clean_acceptance']=0;before['lge_myo.defect_false_acceptance']=0
        report={'metrics':before,'benchmark_hash':cfg['manifest_sha256'],'production_writes':0}
        after={**report,'metrics':metrics}
        with patch.object(bench.subprocess,'run',side_effect=[SimpleNamespace(returncode=0,stdout=json.dumps(report)),SimpleNamespace(returncode=0,stdout=json.dumps(after))]):
            selection=bench.paired(self.root,self.root/'workspace')
        self.assertEqual(selection['decision'],'RETAIN_PARENT')
        self.assertTrue(selection['regressed_metrics'])
    def test_legacy_benchmark_metric_contract_is_unchanged_without_new_opt_in(self):
        report=bench.evaluate(self.root,self.impl)
        self.assertNotIn('lge_myo.clean_acceptance',report['metrics'])
