"""Fast checks of the covariance extracted from the unmodified upstream code."""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from run_audit import official_modules, score_prefixes, tree_coefficients


class TreeAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = os.environ.get("DP_FTRL_UPSTREAM")
        if not path:
            raise unittest.SkipTest("Set DP_FTRL_UPSTREAM to the pinned official clone")
        cls.noise_module, cls.optimizer_module = official_modules(Path(path))

    def test_exact_upstream_noise_map(self):
        for efficient in (False, True):
            with self.subTest(efficient=efficient):
                rounds = 8
                A = tree_coefficients(self.noise_module, rounds, efficient)
                draws = np.random.default_rng(31).normal(size=A.shape[1])
                cursor = 0

                def deterministic_draw(_mean, _std, shape):
                    nonlocal cursor
                    self.assertEqual(tuple(shape), (1,))
                    value = float(draws[cursor])
                    cursor += 1
                    return torch.tensor([value], dtype=torch.float64)

                cls = (self.noise_module.CummuNoiseEffTorch if efficient
                       else self.noise_module.CummuNoiseTorch)
                with patch.object(torch, "normal", side_effect=deterministic_draw):
                    tree = cls(1.0, [(1,)], "cpu")
                    observed = np.array([float(tree()[0][0]) for _ in range(rounds)])
                self.assertEqual(cursor, A.shape[1])
                # Upstream initializes its running noise tensor in float32.
                np.testing.assert_allclose(observed, A @ draws, atol=3e-7)
                np.linalg.cholesky(A @ A.T)

    def test_standard_tree_first_and_last_positions(self):
        A = tree_coefficients(self.noise_module, 8, efficient=False)
        K = A @ A.T
        self.assertEqual(K[-1, -1], 1.0)
        first = np.ones(8)
        self.assertAlmostEqual(float(first @ np.linalg.solve(K, first)), 4.0)
        last = np.r_[np.zeros(7), 1.0]
        self.assertAlmostEqual(float(last @ np.linalg.solve(K, last)), 1.0)
        scores = score_prefixes(last, 1.0, 8, K, 4.0, 1.0, [8])
        self.assertAlmostEqual(scores["tree_8"], scores["oracle_final_prefix"])

    def test_optimizer_checkpoint_identity(self):
        weight = torch.nn.Parameter(torch.zeros(2))
        optimizer = self.optimizer_module.FTRLOptimizer([weight], momentum=0.0)
        noise = torch.tensor([0.3, -0.2])
        weight.grad = torch.tensor([1.0, 2.0])
        optimizer.step((7.0, [noise]))
        torch.testing.assert_close(-7.0 * weight.detach(), weight.grad + noise)


if __name__ == "__main__":
    unittest.main()
