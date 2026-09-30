"""Protocol invariants for the LiRA fusion and fixed-neighbor audit."""
import unittest

import numpy as np
import torch

from gaussproof.fixed_pair_audit import fixed_world, one_sided_limits
from gaussproof.models import initialize
from scripts.low_fpr_lira_fusion import analyze


class IntegrationProtocolTests(unittest.TestCase):
    def test_fusion_selection_cannot_see_holdout_scores(self):
        raw = []
        for digit in range(10):
            for index in range(3):
                for label in (0, 1):
                    raw.append(dict(q=.5, steps=16, digit=digit,
                        canary=10*digit+index, label=label,
                        trajectory_mixture=(0. if index == 0 else 2*label-1),
                        endpoint_lira_z=(2*label-1 if index == 0 else 1-2*label)))
        spec = dict(identities_per_class=3, calibration_per_class=1,
                    bootstrap_repetitions=20, bootstrap_seed=7,
                    q_values=[.5], prefix_steps=[16])
        selected = [r for r in analyze(raw, spec) if r["method"] == "fusion"][0]
        changed = [dict(r, trajectory_mixture=-1000*r["trajectory_mixture"],
                        endpoint_lira_z=-1000*r["endpoint_lira_z"])
                   if r["canary"] % 10 else dict(r) for r in raw]
        repeated = [r for r in analyze(changed, spec) if r["method"] == "fusion"][0]
        self.assertEqual(selected["trajectory_weight"], 0.)
        self.assertEqual(repeated["trajectory_weight"], 0.)

    def test_zero_participation_makes_worlds_identical(self):
        torch.set_num_threads(1)
        x = torch.rand(7, 1, 28, 28, generator=torch.Generator().manual_seed(2))
        y = torch.arange(7) % 10
        initial = initialize(11).state_dict()
        cfg = dict(source_seed=11, q=0., batch_size=2, clip=1., sigma=2.,
                   steps=2, learning_rate=.01)
        args = (initial, x, y, np.array([2, 3, 4]), np.array([0, 1]), 6, 5)
        absent = fixed_world(*args, 0, cfg, 99)
        present = fixed_world(*args, 1, cfg, 99)
        self.assertEqual(absent, present)

    def test_zero_false_positives_has_positive_uncertainty(self):
        lower, upper = one_sided_limits(0, 80, .025)
        self.assertEqual(lower, 0.)
        self.assertGreater(upper, 0.)


if __name__ == "__main__":
    unittest.main()
