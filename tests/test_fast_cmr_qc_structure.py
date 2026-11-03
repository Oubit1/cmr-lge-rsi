from __future__ import annotations

import ast
import unittest
from pathlib import Path


FAST_QC = (Path(__file__).resolve().parents[1] / "skills/cmr-lge-label-qc-workflow"
           / "tools/apply_cmr_qc_fast.py")


class FastCmrQcStructureTests(unittest.TestCase):
    def test_cross_slice_passes_are_not_nested_in_per_frame_loop(self):
        tree = ast.parse(FAST_QC.read_text(encoding="utf-8"))
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == "process_single_patient")

        per_frame = next(
            node for node in ast.walk(function)
            if isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and node.target.id == "k"
            and isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Name)
            and node.iter.func.id == "range"
            and len(node.iter.args) == 1
            and isinstance(node.iter.args[0], ast.Name)
            and node.iter.args[0].id == "n_slices"
        )
        nested_full_stack_passes = [
            node for statement in per_frame.body for node in ast.walk(statement)
            if isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and node.target.id == "k"
            and isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Name)
            and node.iter.func.id == "range"
        ]
        self.assertEqual(nested_full_stack_passes, [])


if __name__ == "__main__":
    unittest.main()
