import importlib.util
import unittest

import numpy as np


@unittest.skipUnless(importlib.util.find_spec("diffusers"), "Optional diffusion dependency not installed")
class LearnedCanaryTests(unittest.TestCase):
    def test_features_and_mixture_likelihood(self):
        from gaussproof.canary_learned import FEATURE_NAMES, mixture_llr, trajectory_features
        rng = np.random.default_rng(4)
        run = dict(
            observations=rng.normal(size=(5, 11)).astype("float32"),
            fingerprints=rng.normal(size=(5, 11)).astype("float32"),
            backgrounds=rng.normal(size=(5, 11)).astype("float32"),
            included=np.asarray([0, 1, 0, 1, 0]))
        features, analytic = trajectory_features(run, sigma=1.0, clip=1.0, batch_size=8)
        self.assertEqual(features.shape, (5, len(FEATURE_NAMES)))
        self.assertTrue(np.isfinite(features).all())
        self.assertAlmostEqual(mixture_llr(analytic["round_llr"], 1.0),
                               analytic["sequence_llr"], places=5)
        self.assertGreater(mixture_llr([3.0, 3.0], .5), mixture_llr([-3.0, -3.0], .5))

    def test_model_and_sampler_shapes(self):
        import torch
        from gaussproof.canary_learned import CanaryBiLSTM, InclusionDiffusion, diffusion_schedule, diffusion_reconstruct
        features = np.zeros((3, 4, 7), dtype="float32")
        lstm = CanaryBiLSTM(7, hidden=8)
        self.assertEqual(tuple(lstm(torch.tensor(features)).shape), (3,))
        cfg = dict(diffusion_train_timesteps=10, diffusion_sampling_steps=3,
                   posterior_draws=2)
        diffusion = InclusionDiffusion(7, 4, train_timesteps=10, hidden=8)
        reconstructed = diffusion_reconstruct(diffusion, diffusion_schedule(cfg), features, cfg, seed=8)
        self.assertEqual(tuple(reconstructed.shape), (2, 3, 4, 1))


if __name__ == "__main__":
    unittest.main()
