import importlib.util
import unittest

import numpy as np


@unittest.skipUnless(importlib.util.find_spec("diffusers"), "Optional trajectory dependencies not installed")
class LongTrajectoryTests(unittest.TestCase):
    def test_prefix_scores_preserve_requested_lengths(self):
        from gaussproof.canary_audit import scores
        from gaussproof.canary_long_trajectory import prefix_scores
        rng = np.random.default_rng(9)
        run = dict(
            observations=rng.normal(size=(8, 13)).astype("float32"),
            fingerprints=rng.normal(size=(8, 13)).astype("float32"),
            backgrounds=rng.normal(size=(8, 13)).astype("float32"),
            included=np.asarray([0, 1, 0, 1, 1, 0, 0, 1]),
            endpoint_losses=np.linspace(2, 1, 8),
            endpoint_confidences=np.linspace(.1, .8, 8))
        result = prefix_scores(run, sigma=4.0, clip=1.0, batch_size=8,
                               probability=.5, lengths=[2, 4, 8])
        self.assertEqual([row["steps"] for row in result], [2, 4, 8])
        self.assertAlmostEqual(result[-1]["sequence_llr"], scores(run, 4.0, 8)["sequence_llr"])


if __name__ == "__main__":
    unittest.main()
