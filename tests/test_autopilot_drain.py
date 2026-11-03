import os
import signal
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import autopilot


class AutopilotDrainTests(unittest.TestCase):
    def test_stop_finishes_current_tick_and_does_not_start_another(self):
        events = []
        class Pilot:
            def __init__(self, args):
                pass
            def tick(self):
                events.append("started")
                os.kill(os.getpid(), signal.SIGTERM)
                events.append("finished")
            def event(self, name, **kwargs):
                events.append(name)
        handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
        try:
            with tempfile.TemporaryDirectory() as tmp, patch.object(autopilot, "ROOT", Path(tmp)), \
                 patch.object(autopilot, "Autopilot", Pilot), \
                 patch.object(autopilot, "parse_args", return_value=SimpleNamespace(
                     resume_after_fix=False, resume_paused_analysis_once=False,
                     once=False, interval=30)):
                autopilot.main()
            self.assertEqual(events, ["started", "finished", "CONTROLLER_STOPPED_AFTER_TICK"])
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
