import tempfile
import unittest
from pathlib import Path
from scripts.check_feedback_loop import pipeline_fingerprint


class CheckPipelineFingerprintTests(unittest.TestCase):
    def test_consumed_rule_changes_invalidate_cached_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'scripts').mkdir()
            rule = root / 'scripts/rules_v2.py'
            rule.write_text('version = 1')
            before = pipeline_fingerprint(root)
            rule.write_text('version = 2')
            self.assertNotEqual(before, pipeline_fingerprint(root))

    def test_unconnected_shadow_module_does_not_trigger_label_scan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            before = pipeline_fingerprint(root)
            (root / 'experiments').mkdir()
            (root / 'experiments/shadow.py').write_text('pass')
            self.assertEqual(before, pipeline_fingerprint(root))
