from __future__ import annotations

import unittest

from agents.implementation_review import PRODUCTION_REVIEW_POLICY


class ImplementationReviewPolicyTests(unittest.TestCase):
    def test_spatial_gate_contract_does_not_require_unavailable_truth(self):
        self.assertIn("spatial_boundary_checked_controls", PRODUCTION_REVIEW_POLICY)
        self.assertIn("any positive count", PRODUCTION_REVIEW_POLICY)
        self.assertIn("not a promotion blocker", PRODUCTION_REVIEW_POLICY)
        self.assertIn("This call itself is the independent implementation review",
                      PRODUCTION_REVIEW_POLICY)
        self.assertIn("reject failing tests", PRODUCTION_REVIEW_POLICY)


if __name__ == "__main__":
    unittest.main()
