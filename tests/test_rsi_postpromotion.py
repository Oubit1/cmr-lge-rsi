import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.control_checkpoint import digest, save_journal
from app.rsi_postpromotion import begin_install, consumer_gate, finish_install, gate_path


class InstallationVerificationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'root'
        self.workspace = Path(self.tmp.name) / 'workspace'
        self.workspace.mkdir()
        self.source = self.root / 'app/quality_scoring.py'
        self.source.parent.mkdir(parents=True)
        self.source.write_text('old')
        self.old = digest(self.source)
        self.source.write_text('new')
        self.new = digest(self.source)
        self.artifact = self.root / 'runtime/promotions/test'
        backup = self.artifact / 'backup/orchestrator/app/quality_scoring.py'
        backup.parent.mkdir(parents=True)
        backup.write_text('old')
        self.manifest = {'status': 'PROMOTED', 'promoted': [{'owner': 'orchestrator',
            'relative': 'app/quality_scoring.py', 'old_hash': self.old, 'new_hash': self.new,
            'backup': str(backup)}]}
        save_journal(self.artifact / 'promotion_manifest.json', self.manifest)
        self.selection = {'scope': 'quality_scorer_all_targets', 'mode': 'enforce', 'decision': 'PROMOTE'}
        save_journal(self.workspace / 'rsi_selection.json', self.selection)
        save_journal(self.workspace / 'rsi_promotion_journal.json',
                     {'artifact': str(self.artifact), 'run_id': 1, 'action_id': 'a'})

    def start(self):
        self.assertTrue(begin_install(self.root, self.workspace, 1, 'a'))
        self.assertFalse(consumer_gate(self.root)['ready'])

    def test_independent_verification_opens_gate_and_is_not_repeated(self):
        self.start()
        with patch('app.rsi_postpromotion.verify_installed', return_value={'status': 'VERIFIED'}) as verify:
            self.assertEqual('VERIFIED', finish_install(self.root)['status'])
            self.assertTrue(finish_install(self.root)['ready'])
            verify.assert_called_once()
        self.assertEqual('new', self.source.read_text())

    def test_regression_restores_only_code_and_preserves_human_data(self):
        self.start()
        label = self.root / 'runtime/human_review.json'
        label.write_text('new human review')
        with patch('app.rsi_postpromotion.verify_installed', return_value={'status': 'REGRESSED', 'reason': 'measured_regression'}):
            self.assertEqual('ROLLED_BACK', finish_install(self.root)['status'])
        self.assertEqual('old', self.source.read_text())
        self.assertEqual('new human review', label.read_text())
        self.assertEqual('ROLLED_BACK', json.loads((self.artifact / 'promotion_manifest.json').read_text())['status'])

    def test_concurrent_edit_is_never_overwritten(self):
        self.start()
        self.source.write_text('unrelated user edit')
        with patch('app.rsi_postpromotion.verify_installed') as verify:
            self.assertFalse(finish_install(self.root)['ready'])
            verify.assert_not_called()
        self.assertEqual('unrelated user edit', self.source.read_text())

    def test_changed_accepted_selection_is_rejected(self):
        self.start()
        save_journal(self.workspace / 'rsi_selection.json', {'decision': 'PROMOTE'})
        self.assertIn('selection_changed', finish_install(self.root)['reason'])

    def test_interrupted_rollback_replays_and_is_idempotent(self):
        self.start()
        gate = json.loads(gate_path(self.root).read_text())
        gate['status'] = 'ROLLING_BACK'
        save_journal(gate_path(self.root), gate)
        # Simulate a crash after source replacement, before status commit.
        self.source.write_text('old')
        with patch('app.rsi_postpromotion.verify_installed') as verify:
            self.assertEqual('ROLLED_BACK', finish_install(self.root)['status'])
            self.assertTrue(finish_install(self.root)['ready'])
            verify.assert_not_called()

    def test_environment_failure_does_not_claim_a_scoring_regression(self):
        self.start()
        with patch('app.rsi_postpromotion.verify_installed', return_value={
            'status': 'RECOVERY_REQUIRED', 'reason': 'postinstall_parent_environment_changed'}):
            self.assertFalse(finish_install(self.root)['ready'])
        self.assertEqual('new', self.source.read_text())

    def test_invalid_gate_blocks_consumers(self):
        gate_path(self.root).parent.mkdir(parents=True, exist_ok=True)
        gate_path(self.root).write_text('{')
        self.assertFalse(consumer_gate(self.root)['ready'])

    def test_failed_or_unregistered_install_cannot_start_monitor(self):
        save_journal(self.workspace / 'rsi_selection.json', {'scope': 'unrelated'})
        self.assertFalse(begin_install(self.root, self.workspace, 1, 'a'))
        save_journal(self.workspace / 'rsi_selection.json', {**self.selection, 'decision': 'RETAIN_PARENT'})
        with self.assertRaisesRegex(ValueError, 'unapproved'):
            begin_install(self.root, self.workspace, 1, 'a')

    def test_pending_install_prevents_all_automatic_consumers(self):
        from scripts import check_feedback_loop as loop
        self.start()
        with patch.object(loop, 'ROOT', self.root), patch.object(loop, 'tick') as scan, \
             patch.object(loop, 'run_current_auto_apply') as apply, patch.object(loop, 'run_lge_consumer_once') as lge:
            with self.assertRaisesRegex(RuntimeError, 'SCORING_INSTALL_PENDING'):
                loop.run_consumers(self.root / 'data', auto_apply=True)
            scan.assert_not_called()
            apply.assert_not_called()
            lge.assert_not_called()

    def test_consumer_reloads_scorer_before_binding_apply_functions(self):
        from scripts import check_feedback_loop as loop
        from unittest.mock import Mock, call
        quality, writer = Mock(), Mock()
        with patch.object(loop.importlib, 'import_module', side_effect=[quality, writer]), \
             patch.object(loop.importlib, 'reload', side_effect=lambda module: module) as reload:
            loop.run_current_auto_apply(self.root, self.root / 'runtime', execute=True)
            self.assertEqual([call(quality), call(writer)], reload.call_args_list)
            writer.run.assert_called_once()

    @unittest.skipIf(__import__('os').name == 'nt', 'Unix deployed shared lock semantics')
    def test_install_waits_for_active_consumer_to_finish(self):
        import threading
        from app.rsi_postpromotion import installation_lock
        entered, attempted = threading.Event(), threading.Event()
        def writer():
            attempted.set()
            with installation_lock(self.root, exclusive=True):
                entered.set()
        with installation_lock(self.root, exclusive=False):
            thread = threading.Thread(target=writer, daemon=True)
            thread.start()
            self.assertTrue(attempted.wait(2))
            self.assertFalse(entered.wait(.1))
        self.assertTrue(entered.wait(2))
        thread.join(2)


if __name__ == '__main__':
    unittest.main()
