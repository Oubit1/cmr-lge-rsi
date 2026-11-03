from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.artifact_registry import csv_issue_slices


class IssueSliceCountTests(unittest.TestCase):
    def test_multiple_issue_rows_count_as_one_review_slice(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "issues.csv"
            path.write_text(
                "patient,volume,slice_index,issues,check\n"
                "P1,a.nii.gz,2,gap,/check/P1_a_s02.png\n"
                "P1,a.nii.gz,2,fragment,/check/P1_a_s02.png\n"
                "P1,a.nii.gz,3,gap,/check/P1_a_s03.png\n",
                encoding="utf-8",
            )
            self.assertEqual(csv_issue_slices(path), 2)


if __name__ == "__main__":
    unittest.main()
