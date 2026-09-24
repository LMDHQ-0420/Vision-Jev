"""Run with the optional train dependencies installed."""

from __future__ import annotations

import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is an optional train dependency")
class HeadsIntegrationTest(unittest.TestCase):
    def test_choice_is_permutation_equivariant(self) -> None:
        from vision_jev.model.heads import ChoiceHead

        torch.manual_seed(7)
        head = ChoiceHead(hidden_size=16, width=8, heads=2).eval()
        q = torch.randn(2, 16)
        h = torch.randn(2, 4, 16)
        mask = torch.ones(2, 4, dtype=torch.bool)
        permutation = torch.tensor([2, 0, 3, 1])
        with torch.no_grad():
            _, original = head(q, h, mask)
            _, permuted = head(q, h[:, permutation], mask[:, permutation])
        self.assertTrue(torch.allclose(original[:, permutation], permuted, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
