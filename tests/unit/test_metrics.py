from __future__ import annotations

import math
import unittest

from vision_jev.eval import accuracy, brier_multiclass, negative_log_likelihood


class MetricsTest(unittest.TestCase):
    def test_reference_metrics(self) -> None:
        probs = [[0.8, 0.2], [0.4, 0.6]]
        targets = [0, 1]
        self.assertEqual(accuracy(probs, targets), 1.0)
        self.assertAlmostEqual(
            negative_log_likelihood(probs, targets), -(math.log(0.8) + math.log(0.6)) / 2
        )
        self.assertAlmostEqual(brier_multiclass(probs, targets), 0.2)


if __name__ == "__main__":
    unittest.main()
