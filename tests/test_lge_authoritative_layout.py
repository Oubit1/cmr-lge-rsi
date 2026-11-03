from __future__ import annotations

import unittest

from app.manual_annotation import TARGET_DIRS


class LgeAuthoritativeLayoutTests(unittest.TestCase):
    def test_editor_maintains_checked_labels(self) -> None:
        self.assertEqual(TARGET_DIRS[("lge", "myo")],
                         "labels/lge_checked/lge_Myo_labels")
        self.assertEqual(TARGET_DIRS[("lge", "mi")],
                         "labels/lge_checked/lge_MI_labels")


if __name__ == "__main__":
    unittest.main()
