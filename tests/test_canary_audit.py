import importlib.util
import unittest
import numpy as np

AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(AVAILABLE, "torch is not installed")
class CanaryAuditTests(unittest.TestCase):
    def test_auc_perfect_and_reversed(self):
        from gaussproof.canary_audit import auc
        self.assertEqual(auc([0, 1, 2, 3], [0, 0, 1, 1]), 1.0)
        self.assertEqual(auc([3, 2, 1, 0], [0, 0, 1, 1]), 0.0)

    def test_score_shape_and_sequence_sum(self):
        from gaussproof.canary_audit import scores
        run = dict(observations=np.ones((3, 4)), fingerprints=np.ones((3, 4)) * 2,
                   backgrounds=np.zeros((3, 4)))
        result = scores(run, 1.0, 2)
        self.assertEqual(result["round_llr"].shape, (3,))
        self.assertAlmostEqual(result["sequence_llr"], float(result["round_llr"].sum()))

    def test_finite_zero_observation_score(self):
        from gaussproof.canary_audit import scores
        base = np.zeros((2, 3), dtype=float)
        run = dict(observations=base.copy(), fingerprints=np.ones_like(base), backgrounds=base.copy())
        self.assertTrue(np.isfinite(scores(run, .5, 2)["sequence_llr"]))
