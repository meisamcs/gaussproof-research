import unittest

import numpy as np

from gaussproof.multi_contributor import (bounded_sparse,
                                          exact_subset_posterior, score_round)


class MultiContributorTests(unittest.TestCase):
    def test_exact_posterior_recovers_two_active_coordinates(self):
        bank = np.eye(3)
        masks = ((np.arange(8)[:, None] >> np.arange(3)) & 1).astype(float)
        posterior = exact_subset_posterior(bank, np.array([.99, .99, 0.]),
                                           variance=.005, q=.2, masks=masks)
        self.assertGreater(posterior[0], .99)
        self.assertGreater(posterior[1], .99)
        self.assertLess(posterior[2], .01)

    def test_bounded_sparse_allows_multiple_contributors(self):
        bank = np.eye(3)
        estimate = bounded_sparse(bank, np.array([1., 1., 0.]),
                                  penalty=.01, iterations=50)
        self.assertGreater(estimate[0], .9)
        self.assertGreater(estimate[1], .9)
        self.assertLess(estimate[2], .1)

    def test_identical_release_bank_for_all_estimators(self):
        bank = np.eye(2)
        masks = ((np.arange(4)[:, None] >> np.arange(2)) & 1).astype(float)
        config = dict(batch_size=2, sigma=.1, clip=1., q=.2,
                      penalty_factors=[0.])
        shift, scores = score_round(np.array([.5, .5]), bank,
                                    np.zeros(2), config, masks)
        self.assertTrue(np.allclose(shift, bank / 2))
        self.assertGreater(scores["exact"].min(), .9)
        self.assertGreater(scores["sparse_0"].min(), .9)


if __name__ == "__main__":
    unittest.main()
