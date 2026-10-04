"""Check writer separation and high-confidence LiRA signal extraction."""

import unittest

import numpy as np
import torch

from validate_nodp_mia import balanced_roles, predictions, tie_auc


class ValidationDesignTests(unittest.TestCase):
    def test_roles_are_digit_balanced_and_writer_disjoint(self):
        class FakeData:
            ids = list(range(500))

            def get(self, writer):
                x = np.zeros((10, 784), np.float32)
                x[:, 0] = writer
                return x, np.arange(10)

        roles = balanced_roles(FakeData(), per_digit=2,
                               population_per_digit=3, references=2,
                               seed=19)
        writers = [record[0] for role in roles.values() for record in role]
        self.assertEqual(len(writers), len(set(writers)))
        for name, records in roles.items():
            expected = 3 if name == "population" else 2
            self.assertEqual(np.bincount([record[2] for record in records],
                                          minlength=10).tolist(), [expected] * 10)

    def test_scaled_logit_survives_probability_rounding(self):
        model = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[40., 0.], [0., 0.]]))
        probability, raw, accuracy, scaled = predictions(
            model, [("writer", np.array([1., 0.], np.float32), 0)])
        self.assertEqual(accuracy, 1.)
        self.assertEqual(probability[0], 1.)
        self.assertEqual(raw[0], 40.)
        self.assertAlmostEqual(scaled[0], 40.)

    def test_auc_counts_ties_as_half(self):
        self.assertEqual(tie_auc([1., 2.], [1., 0.]), .875)


if __name__ == "__main__":
    unittest.main()
