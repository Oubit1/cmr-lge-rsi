import importlib.util
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
