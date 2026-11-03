#!/usr/bin/env python3
"""Fail closed on obvious private artifacts and secrets in public source.
This is a publication check, not proof that clinical data is anonymized.
"""
import re
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RULES={
    "cloud_credential": r"(?:sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[A-Z0-9]{16})",
    "private_key": r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    "private_network": r"\b(?:10\.\d+\.\d+\.\d+|172\.(?:1[6-9]|2\d|3[01])\.\d+\.\d+|192\.168\.\d+\.\d+)\b",
    "name_age_identifier": r"\b(?:[A-Z]{2,}_){1,5}\d{2,3}Y(?:_\d{8,})?\b",
}
BLOCKED={".nii",".dcm",".dicom",".npy",".npz",".db",".sqlite",".sqlite3",".pt",".pth",".ckpt",".safetensors",".log",".jsonl",".png",".jpg",".jpeg",".pdf",".zip",".gz"}
def audit(root=ROOT):
    root=Path(root)
    try:
        raw=subprocess.check_output(["git","ls-files","--cached","--others","--exclude-standard","-z"],cwd=root,stderr=subprocess.DEVNULL)
        paths=sorted(set(p for p in raw.decode().split("\0") if p))
    except (subprocess.CalledProcessError,FileNotFoundError):
        paths=[p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and not any(x in p.parts for x in (".git",".venv","__pycache__"))]
    findings=[]
    for rel in paths:
        p=root/rel
        if not p.is_file() or p.is_symlink():
            findings.append((rel,"not_regular_source"));continue
        if rel.split("/")[0] in {"runtime","artifacts","data","backups"} or p.suffix.lower() in BLOCKED or p.name in {"secrets.env", ".env", "auth.json"}:
            findings.append((rel,"private_artifact"));continue
        if p.stat().st_size>2_000_000:
            findings.append((rel,"large_file"));continue
        try:text=p.read_text(encoding="utf-8")
        except (UnicodeError,OSError):findings.append((rel,"non_text"));continue
        for name,pattern in RULES.items():
            if re.search(pattern,text):findings.append((rel,name))
    return paths,findings
if __name__=="__main__":
    paths,findings=audit()
    for rel,rule in findings:print(f"BLOCKED {rel}: {rule}")
    print(f"Public tree audit: {len(paths)} files, {len(findings)} findings")
    sys.exit(bool(findings))
