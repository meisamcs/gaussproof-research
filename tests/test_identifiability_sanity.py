import unittest

import numpy as np

from scripts.identifiability_sanity import exact_bit_scores
from gaussproof.one_run_audit import joint_trajectory_score


class ExactMixtureTests(unittest.TestCase):
    def test_orthogonal_bank_factorizes(self):
        bank = np.eye(2)
        observations = np.array([[.8, .1], [1.1, -.3]])
        q, sigma = .2, .7
        projection = observations @ bank.T
        expected = np.logaddexp(np.log1p(-q), np.log(q) +
            (projection - .5) / sigma**2).sum(axis=0)
        actual = exact_bit_scores(observations, bank, q, sigma)
        joint = joint_trajectory_score(np.broadcast_to(np.eye(2), (2, 2, 2)),
                                       projection, sigma**2, q)
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        np.testing.assert_allclose(joint, expected, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
