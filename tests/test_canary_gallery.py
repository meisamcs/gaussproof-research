import importlib.util
import unittest

import numpy as np
import torch


@unittest.skipUnless(importlib.util.find_spec("diffusers"), "Optional trajectory dependencies not installed")
class CanaryGalleryTests(unittest.TestCase):
    def test_distributions_are_normalized_and_distinct(self):
        from gaussproof.canary_gallery import class_probabilities
        balanced = class_probabilities("balanced", 3)
        rare = class_probabilities("canary_rare", 3)
        dominant = class_probabilities("canary_dominant", 3)
        for values in [balanced, rare, dominant, class_probabilities("long_tail", 3)]:
            self.assertAlmostEqual(float(values.sum()), 1.0)
            self.assertTrue((values > 0).all())
        self.assertLess(rare[3], balanced[3]); self.assertGreater(dominant[3], balanced[3])

    def test_gallery_score_prefers_aligned_fingerprint(self):
        from gaussproof.canary_gallery import gallery_round_scores
        fingerprints = torch.tensor([[1., 0.], [0., 1.]])
        observation = torch.tensor([.2, 0.])
        llr, mixture = gallery_round_scores(
            observation, fingerprints, torch.zeros(2), sigma=1., clip=1.,
            batch_size=8, probability=.5)
        self.assertGreater(llr[0], llr[1]); self.assertGreater(mixture[0], mixture[1])

    def test_rank_separates_class_and_identity(self):
        from gaussproof.canary_gallery import rank_result
        gallery = np.arange(4)
        labels = torch.tensor([2, 2, 7, 7])
        images = torch.zeros(4, 1, 2, 2)
        result = rank_result(np.asarray([1., 3., 4., 2.]), 0, gallery, labels, images)
        self.assertEqual(result["rank"], 4)
        self.assertEqual(result["same_class_rank"], 2)
        self.assertEqual(result["predicted_label_match"], 0)


if __name__ == "__main__":
    unittest.main()
