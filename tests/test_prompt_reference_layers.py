import csv,json,tempfile,unittest
from pathlib import Path
import numpy as np
import nibabel as nib
from app.manual_annotation import frame_metadata,queue_prompt_repair,execute_prompt_repair,_resolve_plane,TARGET_DIRS

class ReferenceLayersTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)/'dataset';self.work=Path(self.tmp.name)/'work'
        patient='SYNTHETIC_ONLY'
        for number in range(1,13):
            image=np.zeros((16,16),np.float32);mask=np.zeros((16,16),np.uint8);mask[4:12,4:12]=1
            for folder,values in [('images/lge',image),(TARGET_DIRS[('lge','myo')],mask),(TARGET_DIRS[('lge','mi')],np.zeros_like(mask) if number in {9,10} else mask)]:
                path=self.root/folder/patient/(str(number)+'.nii.gz');path.parent.mkdir(parents=True,exist_ok=True)
                nib.save(nib.Nifti1Image(values,np.eye(4)),str(path))
        folder=self.root/'labels/check/manifest_backup';folder.mkdir(parents=True)
        with (folder/'lge_boundary_manifest.csv').open('w',newline='') as h:
            w=csv.DictWriter(h,fieldnames=['patient','volume','frame_index','issues']);w.writeheader()
            w.writerow({'patient':patient,'volume':'11.nii.gz','frame_index':1,'issues':'synthetic'})
            w.writerow({'patient':patient,'volume':'1.nii.gz','frame_index':1,'issues':'synthetic'})
    def queue(self,prompt):
        meta=frame_metadata(self.root,'lge_boundary_manifest',0)
        return queue_prompt_repair(self.root,self.work,{'pool':'lge_boundary_manifest','index':0,'target':'mi','source_hash':meta['source_hashes']['mi'],'prompt':prompt})
    def test_explicit_previous_layers_are_supplied_with_absolute_layer_numbers(self):
        job=self.queue('参考前面9层和10层，处理当前11层的梗死')
        request=json.loads((self.work/'prompt_jobs'/job['job_id']/'agent_request.json').read_text())
        self.assertEqual(request['reference_context']['requested_layers'],[9,10])
        self.assertEqual(request['reference_context']['missing_layers'],[])
        neighbors={n['layer_number']:n for n in request['neighbors']}
        self.assertEqual(set(neighbors),{9,10,12})
        self.assertEqual(neighbors[9]['area'],0);self.assertEqual(neighbors[10]['area'],0)
        self.assertGreater(neighbors[12]['area'],0)
        self.assertNotIn('SYNTHETIC_ONLY',json.dumps(request))
    def test_refinement_retains_original_reference_layers(self):
        first=self.queue('参考第9和10层，修改当前11层')
        folder=self.work/'prompt_jobs'/first['job_id']
        plan={'decision':'apply_operations','target':'mi','operations':[{'operation':'erase_box','x0':500,'x1':500,'y0':500,'y1':500}],'rationale':'synthetic fixture','confidence':0.9}
        execute_prompt_repair(self.work,first['job_id'],plan)
        (folder/'agent_plan.json').write_text(json.dumps({'plan':plan}))
        meta=frame_metadata(self.root,'lge_boundary_manifest',0)
        second=queue_prompt_repair(self.root,self.work,{'pool':'lge_boundary_manifest','index':0,'target':'mi','source_hash':meta['source_hashes']['mi'],'mode':'refine','previous_job_id':first['job_id']})
        request=json.loads((self.work/'prompt_jobs'/second['job_id']/'agent_request.json').read_text())
        self.assertEqual(request['reference_context']['requested_layers'],[9,10])
    def test_previous_layer_at_first_boundary_cannot_wrap_to_last(self):
        with self.assertRaises(IndexError):_resolve_plane(self.root,'lge_boundary_manifest',1,'mi',-1)
    def test_missing_reference_is_reported_not_replaced_by_another_layer(self):
        job=self.queue('参考第99层，处理第11层')
        request=json.loads((self.work/'prompt_jobs'/job['job_id']/'agent_request.json').read_text())
        self.assertEqual(request['reference_context']['missing_layers'],[99])
        self.assertNotIn(99,request['reference_context']['provided_layers'])
