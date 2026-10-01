import unittest

import numpy as np

from gaussproof.one_run_audit import (
    audit_guesses, conservative_epsilon_lower, joint_trajectory_score,
    paper_audit_pvalue, paper_epsilon_lower, score_step)


class OneRunAuditTests(unittest.TestCase):
    def test_fixed_extreme_guesses(self):
        bits = np.array([-1, 1, -1, 1])
        correct, guesses = audit_guesses([0., 3., 1., 2.], bits, 1, 1)
        self.assertEqual(correct, 2)
        np.testing.assert_array_equal(guesses, [-1, 1, 0, 0])

    def test_conservative_bound_requires_evidence(self):
        self.assertEqual(conservative_epsilon_lower(16, 32, 64, 1e-5, .0125), 0.)
        self.assertGreater(conservative_epsilon_lower(29, 32, 64, 1e-5, .0125), 0.)

    def test_paper_appendix_d_reference_cases(self):
        self.assertAlmostEqual(paper_audit_pvalue(75, 100, 100,
                               np.log(3), 0.), .553, places=2)
        self.assertAlmostEqual(paper_epsilon_lower(75, 100, 100,
                               0., .05), .702, places=2)
        self.assertAlmostEqual(paper_epsilon_lower(75, 100, 100,
                               1e-4, .05), .699, places=2)
        self.assertAlmostEqual(paper_epsilon_lower(75, 100, 1000,
                               1e-4, .05), .673, places=2)

    def test_sparse_and_nasr_use_same_release(self):
        cfg = dict(batch_size=2, clip=1., sigma=.5, q=.1,
                   sparse_penalty_factor=0.)
        scores = score_step(np.array([1., 0.]), np.eye(2),
                            np.zeros(2), cfg)
        self.assertAlmostEqual(scores["nasr_dot"][0], 1.)
        self.assertGreater(scores["sparse"][0], scores["sparse"][1])

    def test_joint_score_reduces_to_independent_mixture_for_orthogonal_bank(self):
        gram = np.broadcast_to(np.eye(2), (3, 2, 2)).copy()
        projections = np.array([[1., 0.], [.9, .1], [1.1, -.1]])
        q, variance = .2, .4
        expected = np.logaddexp(np.log1p(-q),
             np.log(q) + (projections - .5) / variance).sum(axis=0)
        actual = joint_trajectory_score(gram, projections, variance, q)
        np.testing.assert_allclose(actual, expected)


if __name__ == "__main__":
    unittest.main()
