import unittest

import numpy as np

from gaussproof.whitebox_gallery import (positive_simplex_projection,
                                         sparse_coefficients, score_round,
                                         temporal_posterior_mean)
from scripts.summarize_whitebox_gallery import paired_differences


class WhiteboxGalleryTests(unittest.TestCase):
    def test_sparse_decoder_constraints_and_known_signal(self):
        shifts = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        residual = np.array([.8, .01, 0.])
        coefficients = sparse_coefficients(shifts @ shifts.T, shifts @ residual,
                                           penalty=.05, iterations=80)
        self.assertTrue(np.all(coefficients >= 0))
        self.assertLessEqual(coefficients.sum(), 1 + 1e-10)
        self.assertEqual(np.argmax(coefficients), 0)
        self.assertLess(np.linalg.norm(coefficients - residual),
                        np.linalg.norm(residual))

    def test_simplex_projection_is_idempotent_and_caps_total(self):
        for vector in ([.2, -.1, .1], [2., 1., -.5], [-1., -2., -3.]):
            projected = positive_simplex_projection(vector)
            self.assertTrue(np.allclose(projected,
                                        positive_simplex_projection(projected)))
            self.assertTrue(np.all(projected >= 0))
            self.assertLessEqual(projected.sum(), 1 + 1e-10)

    def test_score_round_uses_same_release_and_bank_for_every_method(self):
        config = dict(sigma=1., clip=1., batch_size=1, q=.5,
                      penalty_factors=[0., .5])
        bank = np.array([[1., 0.], [0., 1.]])
        release = np.array([1., 0.])
        scores, errors = score_round(release, release, bank,
                                      {"matched": np.zeros(2)}, config)["matched"]
        for method in ("alignment", "gaussian_llr", "mixture_llr", "sparse_0"):
            self.assertGreater(scores[method][0], scores[method][1])
        self.assertEqual(errors["raw"], 0)

    def test_temporal_posterior_uses_identity_and_inclusion_evidence(self):
        bank = np.array([[1., 0.], [0., 1.]])
        estimate = temporal_posterior_mean(np.zeros(2), bank,
                                           np.array([8., -8.]),
                                           np.array([8., -8.]), .5, 1)
        self.assertGreater(estimate[0], .99)
        self.assertLess(estimate[1], .01)

    def test_paired_posterior_vs_sparse_mse_uses_sparse_estimate(self):
        rows = []
        for identity in (1, 2):
            for method in ("alignment", "gaussian_llr", "mixture_llr", "sparse_0"):
                rows.append(dict(world="present", distribution="balanced",
                    background="matched", steps=1, true_position=identity,
                    method=method, top1=1., same_class_top1=1., mrr=1.,
                    background_mse=.2, posterior_mse=.1,
                    gradient_mse=.15 if method == "sparse_0" else np.nan))
        config = dict(seed=1, distributions=["balanced"], prefix_steps=[1],
                      bootstrap_repetitions=50, null_identities=0)
        selected = dict(sparse="sparse_0", same_access_baseline="mixture_llr")
        output = paired_differences(rows, config, selected, [1, 2])
        contrast = next(row for row in output if row["first"] == "posterior_vs_sparse")
        self.assertAlmostEqual(contrast["difference"], .05)
        self.assertTrue(np.isfinite(contrast["ci_low"]))


if __name__ == "__main__":
    unittest.main()
