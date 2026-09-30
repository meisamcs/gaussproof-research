import unittest

import numpy as np
import torch

from gaussproof.canary_audit import run_sequence
from gaussproof.models import initialize
from gaussproof.trajectory_endpoint_replication import public_roles
from scripts.summarize_trajectory_endpoint_replication import summarize


class TrajectoryEndpointReplicationTests(unittest.TestCase):
    def test_public_reference_roles_exclude_all_target_roles(self):
        splits = {"pretrain": [0, 1], "population": [3, 4],
                  "test": [7, 8], "background": [9]}
        train, query = public_roles(30, splits, 8, 5, 12)
        self.assertEqual(len(set(train) | set(query)), 13)
        self.assertTrue(set(train).isdisjoint(query))
        self.assertTrue((set(train) | set(query)).isdisjoint(
            {value for group in splits.values() for value in group}))
        again = public_roles(30, splits, 8, 5, 12)
        np.testing.assert_array_equal(train, again[0])
        np.testing.assert_array_equal(query, again[1])

    def test_endpoint_query_matches_candidate_probability(self):
        torch.manual_seed(5)
        x = torch.rand(12, 1, 28, 28)
        labels = torch.arange(12) % 10
        cfg = {"steps": 2, "clip": 1., "batch_size": 3,
               "learning_rate": .05, "decouple_inclusion_rng": True}
        run = run_sequence(initialize(3), x, labels, np.arange(2, 9),
                           np.asarray([9, 10]), 0, cfg, 1., 8, .5,
                           query_indices=[0, 11], query_steps=[1, 2])
        for step in (1, 2):
            self.assertAlmostEqual(
                run["endpoint_query_probs"][step][0],
                run["endpoint_confidences"][step - 1], places=6)
            self.assertEqual(len(run["endpoint_query_predictions"][step]), 2)
        with self.assertRaises(ValueError):
            run_sequence(initialize(3), x, labels, np.arange(2, 9),
                         np.asarray([9, 10]), 0, cfg, 1., 8, .5,
                         query_indices=[0])

    def test_endpoint_selection_uses_only_calibration_identities(self):
        rows = []
        for digit in range(10):
            for slot in range(2):
                for label in (0, 1):
                    score = label + digit * .001
                    rows.append(dict(
                        q=.5, steps=2, canary=10 * digit + slot,
                        digit=digit, label=label, appearances=label,
                        trajectory_mixture=score,
                        trajectory_sum_llr=score,
                        trajectory_raw_alignment=score,
                        endpoint_loss=score,
                        endpoint_lira_fixed=score if slot == 0 else -score,
                        endpoint_lira_z=-score if slot == 0 else score,
                        endpoint_rmia_a050_g100=score,
                        endpoint_public_accuracy=.7))
        spec = dict(q_values=[.5], prefix_steps=[2], identities_per_class=2,
                    calibration_per_class=1, bootstrap_repetitions=10,
                    bootstrap_seed=1)
        _, selections, _, _ = summarize(rows, spec)
        self.assertEqual(selections[0]["endpoint_lira_selected"],
                         "endpoint_lira_fixed")
        # Reversing every *holdout* score cannot change selection.
        for row in rows:
            if row["canary"] % 10 == 1:
                row["endpoint_lira_fixed"], row["endpoint_lira_z"] = (
                    row["endpoint_lira_z"], row["endpoint_lira_fixed"])
        _, changed, _, _ = summarize(rows, spec)
        self.assertEqual(changed[0]["endpoint_lira_selected"],
                         selections[0]["endpoint_lira_selected"])


if __name__ == "__main__":
    unittest.main()
