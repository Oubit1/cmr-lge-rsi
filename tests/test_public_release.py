import importlib.util
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("public_audit",ROOT/"scripts/audit_public_tree.py")
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)

class PublicReleaseTests(unittest.TestCase):
    def test_blocks_sensitive_artifacts_even_when_contents_are_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/"sample.nii.gz").write_bytes(b"")
            (root/"runtime").mkdir()
            (root/"runtime/receipt.json").write_text("{}")
            _,findings=audit.audit(root)
            self.assertEqual({kind for _,kind in findings},{"private_artifact"})
            self.assertEqual(len(findings),2)

    def test_blocks_credentials_without_echoing_secret_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);secret="sk-"+"a"*32
            (root/"settings.py").write_text(secret)
            _,findings=audit.audit(root)
            self.assertEqual(findings,[("settings.py","cloud_credential")])
            self.assertNotIn(secret,str(findings))

    def test_demo_refuses_to_overwrite_existing_dataset(self):
        from scripts.create_demo import create
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/"keep.txt";p.write_text("preserve")
            with self.assertRaises(SystemExit):create(root)
            self.assertEqual(p.read_text(),"preserve")

    def test_reviewed_media_exception_is_bound_to_exact_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            folder=root/"docs/assets";folder.mkdir(parents=True)
            data=b"synthetic-review-fixture"
            image=folder/"console-ui.png";image.write_bytes(data)
            (folder/"approved-media.json").write_text(json.dumps({
                "schema":"reviewed-ui-media-v1", "files":{
                    "docs/assets/console-ui.png":{
                        "sha256":hashlib.sha256(data).hexdigest(), "bytes":len(data)
                    }
                }
            }))
            self.assertEqual(audit.audit(root)[1],[])
            image.write_bytes(b"unreviewed replacement")
            self.assertIn(("docs/assets/console-ui.png","unreviewed_media_change"),
                          audit.audit(root)[1])

    def test_manifest_cannot_allow_arbitrary_medical_or_runtime_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/"docs/assets";folder.mkdir(parents=True)
            data=b"synthetic"
            p=root/"patient.nii.gz";p.write_bytes(data)
            (folder/"approved-media.json").write_text(json.dumps({
                "schema":"reviewed-ui-media-v1", "files":{
                    "patient.nii.gz":{
                        "sha256":hashlib.sha256(data).hexdigest(), "bytes":len(data)
                    }
                }
            }))
            self.assertIn(("patient.nii.gz","private_artifact"),audit.audit(root)[1])
