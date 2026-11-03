#!/usr/bin/env python3
"""Fail closed on obvious private artifacts and secrets in public source.
This is a publication check, not proof that clinical data is anonymized.
"""
import re
import hashlib
import json
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
MEDIA_NAMES = {"console-ui.png", "workflow-ui.jpg", "workflow-preview.gif",
               "workflow-demo.mp4", "editor-ui.jpg", "editor-preview.gif",
               "editor-demo.mp4"}

def reviewed_media(root):
    """Only exact, manually reviewed documentation assets may bypass text checks."""
    manifest = root / "docs/assets/approved-media.json"
    if not manifest.is_file() or manifest.is_symlink():
        return {}
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
        if data.get("schema") != "reviewed-ui-media-v1":
            return {}
        return {rel: item for rel, item in data.get("files", {}).items()
                if rel.startswith("docs/assets/")
                and rel.removeprefix("docs/assets/") in MEDIA_NAMES
                and re.fullmatch(r"[a-f0-9]{64}", str(item.get("sha256", "")))}
    except (OSError, ValueError, AttributeError, TypeError):
        return {}
def audit(root=ROOT):
    root=Path(root)
    try:
        raw=subprocess.check_output(["git","ls-files","--cached","--others","--exclude-standard","-z"],cwd=root,stderr=subprocess.DEVNULL)
        paths=sorted(set(p for p in raw.decode().split("\0") if p))
    except (subprocess.CalledProcessError,FileNotFoundError):
        paths=[p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and not any(x in p.parts for x in (".git",".venv","__pycache__"))]
    findings=[]
    media = reviewed_media(root)
    for rel in paths:
        p=root/rel
        if not p.is_file() or p.is_symlink():
            findings.append((rel,"not_regular_source"));continue
        if rel in media:
            item = media[rel]
            if (p.stat().st_size > 10_000_000 or p.stat().st_size != item.get("bytes")
                    or hashlib.sha256(p.read_bytes()).hexdigest() != item["sha256"]):
                findings.append((rel,"unreviewed_media_change"))
            continue
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
