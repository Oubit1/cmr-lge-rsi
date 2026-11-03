from __future__ import annotations

import unittest
from unittest.mock import patch

from app.gpu_scheduler import normalize_gpu_id, select_free_gpu


class Result:
    returncode = 0
    stdout = "0, 12000, 30\n1, 16000, 60\n2, 16000, 10\n"


class GpuSchedulerTests(unittest.TestCase):
    def test_selects_most_free_then_least_utilized(self):
        with patch("app.gpu_scheduler.subprocess.run", return_value=Result()):
            self.assertEqual(select_free_gpu(), "2")

    def test_normalize_explicit_device(self):
        self.assertEqual(normalize_gpu_id("cuda:3"), "3")
        self.assertIsNone(normalize_gpu_id("cpu"))


if __name__ == "__main__":
    unittest.main()
