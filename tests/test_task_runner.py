from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from app.task_runner import sanitized_environment


class TaskRunnerTests(unittest.TestCase):
    def test_cloud_secrets_are_not_inherited_by_qc_subprocess(self):
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "secret",
            "DASHSCOPE_API_KEY": "secret",
            "SAFE_SETTING": "visible",
        }, clear=True):
            env = sanitized_environment()
        self.assertNotIn("OPENAI_API_KEY", env)
        self.assertNotIn("DASHSCOPE_API_KEY", env)
        self.assertEqual(env["SAFE_SETTING"], "visible")


if __name__ == "__main__":
    unittest.main()
