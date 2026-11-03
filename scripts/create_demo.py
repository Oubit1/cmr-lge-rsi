#!/usr/bin/env python3
"""Generate geometric demo data, never copy a clinical image or label."""
import csv
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.paths import DATASET_ROOT

def create(root=DATASET_ROOT):
    import numpy as np
    import nibabel as nib
    from PIL import Image
    root=Path(root).resolve()
    if root.exists() and any(root.iterdir()):
        raise SystemExit("Dataset directory is not empty; demo refuses to overwrite it.")
    root.mkdir(parents=True,exist_ok=True)
    y,x=np.ogrid[:64,:64]
    ring=(((x-32)**2+(y-32)**2 < 20**2) & ((x-32)**2+(y-32)**2 > 12**2))
    intensity=(np.exp(-((x-32)**2+(y-32)**2)/700)*100).astype(np.float32)
    subject="DEMO_SUBJECT_001"
    for modality in ("cmr","lge"):
        image=np.repeat(intensity[...,None],3 if modality=="cmr" else 1,axis=2)
        mask=np.repeat(ring[...,None],image.shape[-1],axis=2).astype(np.uint8)
        labels={"cmr":[("labels/cmr_checked/cmr_Myo_mask",mask)],
                "lge":[("labels/lge_checked/lge_Myo_labels",mask),
                       ("labels/lge_checked/lge_MI_labels",np.zeros_like(mask))]}[modality]
        for folder,arr in [("images/"+modality,image)]+labels:
            p=root/folder/subject/"1.0.nii.gz";p.parent.mkdir(parents=True,exist_ok=True)
            nib.save(nib.Nifti1Image(arr,np.eye(4)),str(p))
    pools=("cmr_boundary_manifest","cmr_middle_manifest","lge_boundary_manifest",
           "lge_issue_manifest","lge_middle_manifest","trajectory_manifest")
    for pool in pools:
        p=root/"labels/check/manifest_backup"/(pool+".csv");p.parent.mkdir(parents=True,exist_ok=True)
        with p.open("w",newline="",encoding="utf-8") as handle:
            w=csv.DictWriter(handle,fieldnames=["patient","modality","volume","slice_index","issues","evidence"])
            w.writeheader()
            if pool in {"cmr_middle_manifest","lge_boundary_manifest"}:
                w.writerow({"patient":subject,"modality":"cmr" if pool.startswith("cmr") else "lge",
                            "volume":"1.0.nii.gz","slice_index":1,
                            "issues":"cmr_myo_excess_components" if pool.startswith("cmr") else "lge_myo_topology",
                            "evidence":"Synthetic geometry for software demonstration only"})
    for bucket in ("待筛","待处理","checked","垃圾桶"):
        for modality in ("cmr","lge"):
            p=root/"labels/check"/bucket/modality;p.mkdir(parents=True,exist_ok=True)
    p=root/"labels/check/待筛/cmr"/subject/"1.0.png";p.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray((intensity/intensity.max()*255).astype(np.uint8)).save(p)
    print("Synthetic demo created. No cloud calls, clinical data or real QC acceptance.")
if __name__=="__main__":create()
