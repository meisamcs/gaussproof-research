"""Small mechanism and scoring checks for the independent federated pilot."""

import unittest

import numpy as np

from run_pilot import (EfficientTree, auc, calibrated_rates,
                       cross_identity_auc,
                       recover_noised_sum, server_step, simulate,
                       sparse_coordinate,
                       tree_matrix)


class MechanismTests(unittest.TestCase):
    def test_source_optimizer_fixture_and_checkpoint_inverse(self):
        # Matches the official optimizer_utils_test.py constant-gradient
        # DP-FTRLM fixture at learning rate .1 and momentum .9.
        initial = np.zeros(3)
        current = initial.copy()
        velocity = np.zeros(3)
        for step, expected in enumerate((-.1, -.29, -.561), start=1):
            previous = current.copy()
            current, velocity = server_step(
                initial, velocity, np.ones(3) * step, np.zeros(3), .1, .9)
            np.testing.assert_allclose(current, expected, atol=1e-12)
            np.testing.assert_allclose(recover_noised_sum(
                initial, current, previous, .1, .9), step, atol=1e-12)

    def test_efficient_tree_reference_values_and_covariance(self):
        # Direct small-tree values from TF Privacy 0.6's node rule:
        # node = Gaussian + (left+right)/2; weight = 1/(2-2^-depth).
        tree = EfficientTree(lambda: 1.)
        np.testing.assert_allclose([tree.next() for _ in range(4)],
                                   [1., 4 / 3, 7 / 3, 12 / 7])
        matrix = tree_matrix(16)
        covariance = matrix @ matrix.T
        self.assertGreater(np.linalg.eigvalsh(covariance)[0], 0)
        # The same linear map must predict outputs for arbitrary node draws.
        values = iter(np.arange(matrix.shape[1], dtype=float) / 11)
        replay = EfficientTree(lambda: next(values))
        np.testing.assert_allclose(
            [replay.next() for _ in range(16)],
            matrix @ (np.arange(matrix.shape[1]) / 11), atol=1e-12)

    def test_sparse_decoder_finds_known_client_column(self):
        bank = np.eye(4)
        self.assertGreater(sparse_coordinate(bank, bank[0], .01, 80), .9)
        self.assertLess(sparse_coordinate(bank, bank[1], .01, 80), .01)

    def test_calibration_and_tie_aware_auc(self):
        calibration = [dict(negative=dict(score=x)) for x in (.1, .2, .3, .4)]
        heldout = [dict(positive=dict(score=.35), negative=dict(score=.25)),
                   dict(positive=dict(score=.45), negative=dict(score=.42))]
        tpr, fpr = calibrated_rates(calibration, heldout, "score", .25)
        self.assertEqual((tpr, fpr), (1., .5))
        tpr, fpr = calibrated_rates(calibration, heldout, "score", .01)
        self.assertTrue(np.isnan(tpr) and np.isnan(fpr))
        self.assertEqual(auc([1, 1], [0, 1]), .75)

    def test_cross_identity_auc_removes_self_pair_boost(self):
        records = [dict(positive=dict(s=.11), negative=dict(s=.10)),
                   dict(positive=dict(s=.91), negative=dict(s=.90))]
        self.assertEqual(cross_identity_auc(records, "s"), .5)
        self.assertEqual(auc([.11, .91], [.10, .90]), .75)

    def test_hidden_insertion_round_does_not_enter_public_scores(self):
        rng = np.random.default_rng(6)
        ids = [f"client-{i}" for i in range(35)]
        clients = {}
        for client_id in ids:
            x = rng.normal(size=(4, 50))
            x[:, -1] = 1
            clients[client_id] = (x, rng.integers(0, 10, size=4))

        class SyntheticData:
            def get(self, key):
                return clients[key]

        kw = dict(data=SyntheticData(), candidate_id=ids[0],
                  background_ids=np.asarray(ids[1:29]).reshape(4, 7),
                  public_ids=ids[29:32], validation=ids[32:35],
                  member=False, rounds=4, clients_per_round=8,
                  sigma=2, clip=.25, client_lr=.2, server_lr=.005,
                  momentum=.9, seed=12, matrix=tree_matrix(4),
                  penalty=.0001, sparse_iterations=20)
        first = simulate(position=0, **kw)
        last = simulate(position=3, **kw)
        for score in ("rero_max", "rero_mean", "gaussproof_max",
                      "gaussproof_mean", "public_gls_max",
                      "endpoint_loss", "accuracy", "noise_check"):
            self.assertEqual(first[score], last[score], score)


if __name__ == "__main__":
    unittest.main()
